# Testing and verification record

## Automated tests (`python -m pytest`)

| File | Covers |
|---|---|
| `tests/test_schema.py` | required/forbidden columns incl. any `value_<t>h` with t > 24; blank vs non-numeric readings; duplicates, device in two lots, mixed parts; SI-prefix conversion and incompatible units; registry-limit mismatch; unknown part; identical readings; single provenance per upload; feature matrix = exactly `[value_0h, value_24h]` |
| `tests/test_module_a.py` | robust-z definition; zero MAD never invents a score; insufficient peers; missing rows are not peers; drifting device flagged; uniformly shifted lot detected against history; IDs don't change scores; source-flagged row gets no verdict; reference population documented |
| `tests/test_module_b.py` | exactly two features in every booster; one-to-one target join; value_24h equals the 24 h observation; IDs/limits/split don't change predictions; save → load gives identical predictions; tampered artifact refused; training used only training lots; finite-sample lot conformal margin; unsupported part, unit mismatch, missing input, out-of-domain, missing calibration; interval status |
| `tests/test_decision.py` | every rule R1–R6 incl. crossing ⇒ ESCALATE, Module A warning beats forecast PASS, observed vs predicted basis, no predicted FAIL on extrapolation or suspect readings; device aggregation and missing-parameter handling |
| `tests/test_data_provenance.py` | whole-lot split isolation (counts per split); synthetic labelling incl. illustrative lot naming; real rows have no burn-in hours or labels; public data contains only redistributable sources; SECOM dates are not lots; real route keeps channels apart, never uses controls as peers, gives AD620 no verdict, refuses dose-as-burn-in-hours and non-real rows |
| `tests/test_e2e.py` | **load demo → validate → analyse → predict → explain → export (CSV/JSON/PDF) → audit entry** incl. the named demo cases; audit chain detects an edited line; invalid upload produces no decisions; production gate refuses synthetic data |
| `tests/test_app.py` | Streamlit AppTest: overview renders; after **Load demo** every page renders without exceptions; device page shows the SYN_L159_D009 reasons; reports page builds a PDF |

CI (`.github/workflows/ci.yml`) installs `requirements.txt` on Ubuntu with **Python 3.11 and 3.12**, runs
`scripts/check_repo.py` (no large/banned files, no local-only data), regenerates the synthetic fixture from its
seed and compares it with the committed files, verifies model artifacts against their manifest hashes, and runs
the full suite. pytest scratch goes to the git-ignored `runtime/pytest`.

## Browser verification (performed 29–30 Sep 2026)

In a local browser against `streamlit run app/streamlit_app.py`:

* Overview → **Load demo**: run created (147 devices; 50 PASS, 96 ESCALATE, 1 FAIL); sidebar status updates.
* Device analysis (`SYN_L159_D009`): ESCALATE (REVIEW) by R4 with A_UNUSUAL_CHANGE, 4.1× peer median change,
  forecast inside limits, provenance section.
* Lot analysis, Evaluation (all five tabs, 9 Plotly charts, no errors), Reports & audit, Real-data explorer
  (AD620 shown as UNTRUSTED_NO_VERDICT).
* **Upload**: an injected CSV containing `value_168h` and text in a numeric column was refused with an actionable
  message (this exposed a duplicated column name in the message, since fixed); a valid 24-row CSV in **nA** was
  converted to µA, the missing reading escalated, and the run appeared in the audit log.
* Reports: **Build PDF report** produced the Download PDF / CSV / JSON / audit-log buttons; hash chain reported
  consistent.
* A bug in the Module B evaluation tab (pandas `melt` name clash) and a crash on the reports page when the audit
  log had no run entries were found this way and fixed; both are now covered by `tests/test_app.py`.

Additional checks with headless Edge (Playwright) on 30 Sep 2026:

* **Narrow viewport (390 × 844, mobile emulation):** overview, Load demo, device and lot pages render with no
  horizontal page scroll; charts, tabs and evidence stack vertically. Metric tiles stack one per row, which makes
  the lot page long but readable.
* **Real-data upload route:** a U309 file with radiation dose copied into `burn_in_hours` was refused
  ("Radiation dose must not be converted into thermal burn-in hours"); the unmodified U309 file was accepted and
  plotted against the dose axis with controls dotted.

Screenshots in `docs/screenshots/` were captured with headless Microsoft Edge via Playwright
(`scripts/take_screenshots.py`) and reviewed for readability; the sample PDF was rasterised and reviewed, which
led to fixing table-cell wrapping.

## Not verified / known gaps

* Clicking the browser **download** buttons to save files was not performed (it would write to the user's
  Downloads folder); the bytes behind them are tested (`test_e2e.py`, `test_app.py`).
* Loading a page by direct URL starts a new Streamlit session (state is per session); in-app navigation keeps
  the run. This is standard Streamlit behaviour.
* Narrow layouts were checked at one phone size only, in headless Edge; no physical devices.
* The dashboard was exercised with Python 3.11 on Windows; CI covers 3.11/3.12 on Linux for the core and
  AppTest pages, not a live browser.
* Evaluation counts for local-only real sources (NDS352, AD648, capacitor #14) cannot be reproduced from a clone
  without the dataset package.
# Migration verification — 30 September 2026

The private migration package adds a new-laptop setup helper, authenticated release restoration,
archive SHA-256/path checks, local-only data restoration and a deterministic smoke check.
Local verification on the existing Python 3.11 environment:

- Full pytest suite passed, including archive corruption, unsafe paths, changed-file preservation
  and idempotent restore checks, plus existing dashboard/PDF and screening tests.
- The original 272,302,234-byte dataset ZIP matches its delivered SHA-256. Its 68 files were
  extracted to a fresh directory; all 67 hashes listed by the package's verifier matched.
- The research context archive restores 101 files and matches its pinned SHA-256.
- `prepare_data.py --local-only` restores the full local data without rewriting tracked public data.
- `setup_project.py --skip-install` passes the smoke check: 147 devices, 50 PASS, 1 FAIL,
  96 ESCALATE; the full real-data loader returns 2,332 records.

This local check reuses installed dependencies. Clean dependency installation is also exercised
by the repository's Python 3.11/3.12 GitHub Actions jobs; their status is recorded on each commit.
No real-world screening performance or new operating-system support is inferred from these checks.

The first migration CI run passed on Python 3.12 but the Python 3.11 hygiene process aborted
at interpreter shutdown after reporting all 140 files OK. The lightweight Parquet scan now
disables reader threads and avoids pandas conversion, following the workaround documented in
[Apache Arrow issue 34314](https://github.com/apache/arrow/issues/34314). The source checks are
unchanged; this addresses a native shutdown failure rather than suppressing a failed check.
