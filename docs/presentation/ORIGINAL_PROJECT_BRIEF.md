> Historical idea-stage reference supplied by the team. This is not an instruction file or a description of verified implementation. See README.md, docs/MODEL_CARD.md and config/drishti.yaml for current behaviour. Some proposed thresholds, mechanisms and performance claims were changed or not implemented.

# DRISHTI: Complete Project Summary

**Team Agamemnon · SIH 2026 · PS 26170 (ISRO, Department of Space) · Theme: Smart Automation · Category: Software**

## 1. The project in one paragraph

Space-grade electronic components go through burn-in: they run hot (125 °C) for days so weak parts fail on the bench instead of in orbit. Today, parts are judged against fixed datasheet limits. That misses "latent defects": parts that stay under the limit but drift abnormally compared with their own manufacturing lot. DRISHTI does two things:
- It judges each part against its own lot.
- It predicts each part's 168-hour value from only the 0 h and 24 h readings.

Every part then gets a PASS / FAIL / ESCALATE verdict with an auditable reason.

The core example: a lot averages 10 µA. One part reads 45 µA, which is 4.5× its lot, but it passes the 50 µA datasheet limit.

## 2. What the problem statement requires

| Requirement | Detail |
|---|---|
| Module A | Dynamic, lot-relative outlier detection |
| Module B | Regression taking Value_0h + Value_24h, forecasting Value_168h, flagging early rejection if predicted drift exceeds a safety slope |
| Graded on | Anomaly detection with false negatives heavily penalised; MAE of the 168 h forecast against a hidden ground truth; explainability to a QA inspector |

## 3. Our solution

**Module A: lot-relative outlier detection**
- Robust lot statistics: median ± 6 × robust sigma, where robust sigma = 1.4826 × MAD. This is Part Average Testing (AEC-Q001) practice.
- An ML ensemble on top: Isolation Forest + ECOD.
- Split-conformal calibration turns scores into p-values with a **bounded false-alarm rate**. It needs no defect labels.

**Module B: drift forecasting**
- LightGBM with quantile/L1 loss, trained directly on MAE, the graded metric.
- Conformalized Quantile Regression (CQR) produces a prediction interval [L, U].
- A part is rejected when the upper bound U crosses the safety slope S.
- Module B may **only** see 0 h and 24 h data. This is enforced in code.

**Decision engine**
- Fuses both modules into PASS / FAIL / ESCALATE.
- If the interval straddles the limit, the part can never PASS; it escalates to extended burn-in.
- Timing: an early verdict at 24 h, and a final verdict with a candidate mechanism at 168 h.

**Mechanism attribution**
- Drift shape (slope, curvature, steps) is matched to known degradation signatures such as TDDB, NBTI and electromigration.
- It is always hedged: "pattern consistent with…", never a diagnosis.

**Explainability and reporting**
- Transparent PAT rules and decision table; SHAP explanations for ML scores.
- A PDF rejection certificate and an append-only audit log.

**Safety slope**: three derivations (standards delta limit, Arrhenius mission-based, statistical tolerance limit), taking the most conservative.

## 4. Technology stack

| Layer | Tools |
|---|---|
| Language | Python 3.11 |
| Data | pandas, numpy, SciPy |
| Anomaly detection | scikit-learn (Isolation Forest), PyOD (ECOD) |
| Forecasting | LightGBM (quantile / MAE objectives) |
| Uncertainty | Own ~50-line split-conformal + CQR module |
| Explainability | SHAP, InterpretML |
| Dashboard | Streamlit + Plotly |
| Reports | Jinja2 + WeasyPrint (PDF), JSONL audit log |
| Validation | pytest, GroupKFold by lot |
| Development | Google Colab or local Jupyter / VS Code |
| Demo hosting | Hugging Face Spaces or Streamlit Community Cloud |
| Production target | On-premise, CPU-only, air-gapped |

## 5. Datasets

| Dataset | Link | Tests | Real? | Key caveat |
|---|---|---|---|---|
| Synthetic generator (you build it) | Your own code | Full pipeline end to end | No | Parameters must be sourced, defects subtle, always labelled SYNTHETIC |
| UCI SECOM | `pip install ucimlrepo` / archive.ics.uci.edu/dataset/179/secom | Module A | Yes | Fab process data, one timepoint, anonymous features, no lot IDs (make pseudo-lots from timestamps) |
| NASA #13 MOSFET Thermal Overstress | data.phmsociety.org/nasa | Module B | Yes | Power MOSFETs, aging cycles not hours, `.mat` files, reference doc offline |
| NASA #8 IGBT | same | Module B | Yes | Only 6 devices; use leave-one-device-out |
| NASA #12/#14 Capacitors | same | Cross-family tests | Yes | Capacitors, not ICs |
| NASA GSFC Radiation Database | nepp.nasa.gov/radhome/RadDatabase/RadDataBase.html | Module A on real space parts | Yes | Radiation stress, not heat; PDF extraction; check for per-device values |
| ISRO / PS dataset | Ask your SIH SPOC | Everything | Yes | Unknown whether it exists; the "hidden ground truth" suggests it may come at the finale |

Skip: C-MAPSS (simulated), WM-811K (only for a future spatial extension), other teams' GitHub data.

## 6. APIs and tools

- **`ucimlrepo`:** one-line SECOM download.
- **PHM Society S3 links:** direct `.zip` downloads of the NASA datasets.
- **NTRS API** (ntrs.nasa.gov): search and download NASA test reports by script.
- **`scipy.io.loadmat` / `h5py`:** read NASA `.mat` files.
- **Semi-ATE-STDF** (preferred) or **PySTDF** (GPL licence): read real tester output (STDF). This answers "how does it plug into our test floor?"
- **`pdfplumber` / `camelot`:** extract tables from NASA PDF reports.
- **WebPlotDigitizer:** pull data points from plots in papers to calibrate the generator.

You don't need any paid or external web API. Everything runs locally.

## 7. Validation plan

- **Splits:** GroupKFold by lot, so the same lot never appears in both train and test. Use leave-one-device-out for NASA data.
- **Module A metrics:** AUCPR, recall at fixed precision, MCC, F-beta (β > 1), escape rate, scrap rate, escalation rate.
- **Module B metrics:** MAE (primary), pinball loss, interval coverage vs nominal, interval width.
- **Baselines to beat:**
  - Module A: static threshold → Isolation Forest alone → XGBoost.
  - Module B: last-value persistence → linear extrapolation.
- **Cost model:** escape ≫ scrap > escalate. All cost figures labelled DEMONSTRATION.
- **Headline result format:** "Static limits missed X of Y planted defects; DRISHTI caught Z" (synthetic), plus one real SECOM number.

## 8. Limitations to state openly

1. **No public burn-in data exists.** Every real dataset is an analogue, and each validates only part of the system.
2. **Synthetic results are circular.** You planted the defects, so real-data results must be the headline.
3. **Four timepoints are very few.** Mechanism attribution is weakly supported, and deep time-series models are ruled out.
4. **Early prediction is genuinely hard.** A competing team's honest escape rate for 24 h predictions was about 27.5%. Don't promise near-zero escapes.
5. **Conformal guarantees are marginal and assume exchangeability.** They bound false alarms, not escapes, and they break under a lot-to-lot process shift or a new defect type.
6. **Lot-relative screening can't see a uniformly bad lot.** It needs a historical cross-lot baseline.
7. **Small space-grade lots make MAD unstable.** You need a minimum-n rule.
8. **No real ground-truth labels.** Real latent-defect labels come from field returns, years later.
9. **Standards alignment is not certification.** DRISHTI is a decision-support prototype only.
10. **Unknown PDA interaction.** Under MIL flows, a lot is scrapped if more than 5% of it rejects, so extra DRISHTI rejects could trigger lot scrap. Decide whether ESCALATE parts count toward that.

## 9. Claims to avoid (corrected errors)

| Don't say | Say instead |
|---|---|
| "Statistically bounded escape rate / bounded false negatives" | "Calibrated false-alarm rate; escape rate measured and minimised" |
| "168 h at 125 °C per MIL-STD-883 Method 1015" | "168 h as specified in the PS". Method 1015 at 125 °C is 240 h for class S and 160 h for class B. |
| "AEC-Q100 Part Average Testing" | "AEC-Q001" |
| "Arrhenius correction of leakage readings" | Drop it or explain it carefully. Post-burn-in electrical tests are done at 25 °C. |
| "Shorter qualification cycle" | "Flagged parts leave early; the main win is catching escapes" |
| "Millisecond inference", any unmeasured number | Nothing until measured |
| "Four and a half stars" | ★★★★☆, consistently |

## 10. The toughest questions to prepare for

1. What exactly is guaranteed?
2. Space parts get 240 h, so why 168?
3. At 24 hours, what does the verdict actually use?
4. What if the whole lot is bad?
5. How does SECOM prove anything about burn-in?
6. Could your extra rejects push a lot over PDA?
7. How do you tell real drift from a bad test socket?
8. Our testers output STDF. How does it plug in?
9. Other teams already have working code. Why you?

Route physics and standards questions to your ECE member, and algorithm questions to the ML members.

## 11. Competition

At least seven public GitHub repos exist for PS 26170. Several have working prototypes with conformal intervals, "route uncertain parts to full burn-in" logic, PDF reports, test suites, and one has a live website. Your differentiators on paper are no longer unique. Your edge has to be rigour, honesty about limits, and real-data results.

## 12. Current status and priority to-do

- **Done:** architecture, method selection, documentation, deck content, datasets located.
- **Not done:** code, results, prototype.

Priority order:
1. Download SECOM; run robust-z + Isolation Forest + ECOD; get one real AUCPR. **(Highest value.)**
2. Build a minimal Streamlit page (lot band + part + verdict) and take a screenshot.
3. Build the synthetic generator with subtle, sourced defects.
4. Download NASA MOSFET data and inspect one file.
5. Screen 5–10 GSFC TID reports for per-device data.
6. Ask your SIH SPOC about a PS dataset.
7. Fill in placeholders (Team ID, names, results), move the content into the official template, export as PDF.

## 13. Submission rules (from the official template)

- **6 slides maximum**, including the title slide.
- **Points, diagrams and infographics**, not paragraphs.
- **Only the provided template**, with its idea-detail pointers unchanged.
- **Upload as PDF only.**

I can put this into a shareable doc that your whole team can edit and keep updated, if that's useful.