# Architecture and data flow

## Layers

```mermaid
flowchart LR
    subgraph UI["app/ (Streamlit + Plotly) — presentation only"]
        O["Overview"] --- U["Upload & validate"] --- L["Lot analysis"] --- D["Device analysis"]
        E["Evaluation"] --- R["Reports & audit"] --- X["Real-data explorer"]
    end
    subgraph CORE["drishti/ (no Streamlit imports)"]
        S["schema.validate_early_inputs"] --> A["module_a.run_module_a"]
        S --> B["module_b.Forecaster.predict"]
        A --> DE["decision.decide"]
        B --> DE
        SP["specs.SpecRegistry"] --> S
        SP --> DE
        DE --> EX["explain"] & RP["report.build_pdf"] & AU["audit.record_run"]
        RL["real.validate_real / analyze_real"]
    end
    subgraph ART["Versioned artifacts"]
        CFG["config/drishti.yaml"]
        REG["config/spec_registry/DEMO_ONLY_v1.csv"]
        MA["models/module_a: reference.json, detectors.joblib"]
        MB["models/module_b: q05/q50/q95 boosters, calibration.json, domain.json, manifest.json"]
    end
    UI --> CORE
    ART --> CORE
    AU --> LOG[("runtime/audit/audit_log.jsonl")]
```

`drishti.pipeline.run_screening()` is the single entry point used by the dashboard, tests, evaluation and
sample-report scripts, so the UI never re-implements analysis logic.

## Screening data flow (24 h decision)

```mermaid
flowchart TD
    IN["CSV: device, lot, part, parameter, unit, value_0h, value_24h, provenance"] --> V{"Validate"}
    V -- blocking error --> REF["Refuse run; actionable messages; nothing scored"]
    V -- ok --> N["Normalise: SI-prefix unit conversion, registry limits, row flags"]
    N --> GA["Peer groups: part × lot × parameter × unit × test_condition"]
    GA --> RZ["Robust z of level (24 h) and change (24 h − 0 h)"]
    RZ --> G1{"≥ min peers and MAD > 0?"}
    G1 -- no --> INS["A_INSUFFICIENT_PEERS / A_ZERO_MAD + historical evidence"]
    G1 -- yes --> DET["Robust rule; IF + ECOD fitted on historical lot-standardised data; percentile ensemble"]
    GA --> LS["Lot median vs historical lot medians → A_LOT_SHIFT / A_LOT_SPREAD"]
    N --> FB["Forecaster: exact features value_0h, value_24h"]
    FB --> CQ["q05/q50/q95 + lot-level conformal margin"]
    CQ --> IV["Interval vs registry limits"]
    FB --> DOM["Unit / domain / calibration checks"]
    INS & DET & LS & IV & DOM --> DT["Decision table R1–R6 per parameter"]
    DT --> DV["Device aggregation → PASS / FAIL / ESCALATE"]
    DV --> OUT["Dashboard · PDF · CSV/JSON · audit entry"]
```

Only readings available by 24 h enter any computation for the decision. The dashboard can reveal the hidden
48/96/168 h synthetic readings for demonstration, clearly labelled, after the decision is made.

## Training / fitting flow

```mermaid
flowchart LR
    SYN[("SYNTHETIC early inputs + 168 h targets")] --> SPLIT{"whole-lot splits"}
    SPLIT -- train 90 lots --> FA["Module A detectors"] & FBM["Module B quantile models"]
    SPLIT -- calibration 45 lots --> PA["Module A percentile grids"] & CB["Module B lot-level conformal margin"]
    SPLIT -- validation 15 lots --> TH["Ensemble threshold (2% healthy flag rate)"]
    SPLIT -- test 30 / shifted 12 / edge 12 lots --> EV["scripts/evaluate.py only"]
```

Real component data and SECOM are **never** concatenated with the synthetic data or with each other. SECOM has its
own evaluation (`scripts/evaluate.py::secom`), and real component cohorts use the separate exploratory route
(`drishti/real.py`).

## Artifacts and versioning

| Artifact | Identity recorded in each run |
|---|---|
| Module B boosters, calibration, domain | SHA-256 per file in `models/module_b/manifest.json`; `model_version_hash`; load refuses mismatching files |
| Module A reference + detectors | `reference_hash` in `models/module_a/reference.json` |
| Specification registry | `revision` column + SHA-256 of the file |
| Input | SHA-256 of the uploaded bytes |
| Code | `drishti.__version__` inside the model version string |

## Runtime files

`runtime/` (git-ignored; override with `DRISHTI_RUNTIME_DIR`) holds the audit log. Nothing else is written at
run time; reports and exports are streamed to the browser.
