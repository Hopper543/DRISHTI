# SIH screening demo script (≈ 6 minutes)

Setup (before the call): `.venv\Scripts\python -m streamlit run app/streamlit_app.py`, open
http://localhost:8501, keep the PDF `docs/sample_report/drishti_sample_report_SYNTHETIC.pdf` ready as backup.

| Time | Screen | Do | Say |
|---|---|---|---|
| 0:00 | Overview | Point at the yellow banner | "DRISHTI screens components at 24 h instead of waiting 168 h. Everything you see today is built on **synthetic demo data with demo limits**; we'll show real data separately and honestly." |
| 0:30 | Overview | Press **▶ Load demo** | "No preparation: five held-out synthetic lots, plus one clearly labelled illustrative lot. Every run is hashed and logged." Point at PASS/ESCALATE/FAIL counts. |
| 1:00 | Lot analysis (SYN_L159, leakage) | Show the scatter and the shaded robust band | "Each device is judged against its own lot — same part, parameter, unit and test condition — on its 24 h value and its change since 0 h, using median and MAD so outliers don't distort the reference." |
| 1:40 | Device analysis → `SYN_L159_D009` | Read the evidence card | "Leakage rose 1.11 µA in 24 h — 4.1× the lot median change. The forecast interval stays inside the 50 µA limit, so a forecast-only system would PASS it. DRISHTI escalates (rule R4): **a forecast PASS never hides a peer warning.**" Toggle *hidden readings*: "In the generator this is a planted gradual defect — shown only for the demo; the decision never sees these readings." |
| 2:30 | Device → `SYN_L152_D000`, RDS_on | Point at the error bar touching 0.5 Ω | "Normal peers, forecast 0.367 Ω, but the calibrated 90% interval (0.271–0.509 Ω) crosses the 0.5 Ω demo limit → ESCALATE for extended burn-in, not FAIL. Intervals are calibrated on whole held-out lots." |
| 3:00 | Device → `SYN_L192_D000` | | "Missing 24 h reading and a 3-device lot: no imputation and no peer verdict — escalated with the reason shown." |
| 3:15 | Device → `ILLUSTRATIVE_Q155_D000` | | "Illustrative lot: readings quantised to 0.1 V make the peer spread zero. We refuse to invent a score and escalate." |
| 3:30 | Lot analysis → `SYN_L180` | Point at the red lot-shift box | "A uniformly shifted lot looks normal to peer comparison. DRISHTI also compares the lot with historical lots — robust z 61 — and escalates the whole lot. The forecast inputs are out of the training range, so no PASS." |
| 4:10 | Evaluation → Module A / SECOM | | "Synthetic: the simple robust rule matched Isolation Forest and beat ECOD and the ensemble, so we kept the inspectable rule. Delayed defects are invisible at 24 h — we say so. On real SECOM process data our unsupervised baseline finds 0 of 23 failures; we kept that result instead of tuning on the test set." |
| 4:50 | Reports & audit | **Build PDF report**, show audit table | "Every recommendation has a PDF with evidence, and an audit entry with input hash, model version and spec revision. It is append-only with a hash chain — detectable, not tamper-proof." |
| 5:20 | Real-data explorer → reram / standby_current | | "Real NASA radiation data is analysed separately: dose is dose, never converted to burn-in hours, and there are no PASS/FAIL labels. AD620 gets no verdict because its bias assignment is unresolved." |
| 5:40 | Close | | "Next: real multi-lot burn-in data from a partner, approved limits, re-calibration, and escape/false-reject rates on untouched lots. Production mode is locked until then." |

## Questions to expect

* *Why so many ESCALATEs?* Conservative whole-lot intervals against demo limits, mainly RDS_on. ESCALATE means
  "keep burning in", which is the safe default for a 24 h decision; the rate must be tuned with real data and
  approved limits, not on our test set.
* *Is the anomaly score a failure probability?* No. It is statistical unusualness relative to peers.
* *Can it replace 168 h burn-in?* Not as shipped. It is an early-screening aid; production is gated.
* *Why not deep learning?* Two readings per parameter; inspectable statistics performed as well as the ML
  detectors on our validation lots.
