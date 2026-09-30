# Recording script using verified prototype values

Verified by running the committed demo pipeline on 30 September 2026. Values below use the
dashboard's rounding. Changes are calculated from unrounded readings. Start with **Load demo**.
This is an approximately eight-minute walkthrough, not an official SIH duration requirement.

## Opening — PPT title and architecture

"Hello, we are Team Agamemnon, and this is DRISHTI, our prototype for ISRO's problem statement on
AI-driven anomaly detection in component burn-in and screening. A component can remain within
its electrical specification while changing unusually compared with other components from the
same manufacturing lot. We examine both that relative behaviour and the predicted future
measurement, helping engineers identify cases that need closer attention.

Module A compares a component with comparable devices from its lot. Module B uses its initial
and 24-hour readings to forecast its value at 168 hours. The decision engine combines measured
values, peer behaviour, uncertainty and data quality into PASS, FAIL or ESCALATE."

## Overview

Click **Load demo**. Expected: **147 devices**, **6 lots**, **294 device/parameter records**;
**50 PASS, 1 FAIL, 96 ESCALATE**.

"The loaded demonstration contains 147 devices across six lots. Each device has two electrical
parameters. The system returns 50 early-screening passes, one observed failure and 96 escalations.
Escalation means continued testing or engineering review, not a confirmed defect. This demonstration
uses synthetic measurements and demo specification limits. Real measurements are shown separately."

## Upload & validate

"For a new batch, the engineer supplies device and lot IDs, part number, parameter, unit, test
condition, and zero-hour and 24-hour readings. Limits come from a separate registry. The model
does not invent an acceptance limit for each component. Inputs are validated before analysis."

## Device analysis: a normal PASS

Select **SYN_L150_D000**, **leakage_current**.

"Leakage rises from 9.348 to 9.527 microamps, a change of 0.179 microamps. This does not trigger
a peer warning. The 168-hour forecast is 10.99 microamps, with an interval of 0.7819 to 27.99,
inside the zero-to-50-microamp demo limits. Its supply-current parameter also passes. The overall
device receives an early-screening PASS, not completed qualification."

## Module A: unusual drift while still within limits

Select **SYN_L159_D009**, **leakage_current**; keep future readings hidden initially.

| Evidence | Displayed value |
|---|---|
| 0 h / 24 h leakage | 10.13 / 11.24 uA |
| Change / peer median change | 1.109 / 0.2678 uA |
| Peer MAD of change | 0.07226 uA |
| Robust change score / configured threshold | 7.85 / 3.5 |
| Forecast / interval | 14.42 / [2.474, 29.94] uA |
| Limit / outcome | 50 uA / ESCALATE, A_UNUSUAL_CHANGE |

"The leakage rises from 10.13 to 11.24 microamps. Its increase of 1.109 microamps is about
4.1 times the group's median increase of 0.2678. The robust change score is 7.85, above our
threshold of 3.5. Its reading remains below the 50-microamp limit, and even the forecast interval
remains below that limit. Nevertheless, the unusual measured drift causes an ESCALATE recommendation.
This is information a fixed-limit check alone would not provide."

Optionally reveal hidden readings: "The generated 168-hour endpoint is 23.10 microamps. These
later values are for evaluation and did not enter the 24-hour decision."

## Module B: uncertain future value

Select **SYN_L152_D000**, **RDS_on**.

"On-resistance increases from 0.3369 to 0.3405 ohms. Using those two inputs, the model forecasts
0.3673 ohms at 168 hours. However, the prediction interval is 0.2714 to 0.509 ohms, crossing the
0.5-ohm demo limit. The system escalates because uncertainty includes an unacceptable outcome;
it does not claim that failure is certain."

Optionally reveal the hidden endpoint: **0.3654 ohm**. Explain that it stayed below the limit,
illustrating the distinction between escalation and failure. Threshold voltage also escalates
for this device: interval **[1.537, 2.509] V** crosses **2.5 V**. Keep RDS_on selected when reading resistance.

## Observed failure

Select **SYN_L180_D020**, **leakage_current**.

"The reading rises from 48.57 to 50.79 microamps, exceeding the 50-microamp demo limit. This is
FAIL with an OBSERVED basis. A forecast cannot override a limit violation already measured."

The screen also shows an out-of-domain warning and a numerically lower forecast. Do not treat
that forecast as evidence that the observed failure can be ignored.

## Missing data and small lot

Select **SYN_L192_D000**, **leakage_current**.

"The initial reading is 10.66 microamps, but the 24-hour reading is missing. No forecast is produced;
the result is ESCALATE for insufficient evidence. The lot also contains only two valid comparison
records, below our minimum group size of eight. We do not invent the missing measurement."

## Shifted lot

Lot analysis -> **SYN_L180** -> **leakage_current**.

"If an entire lot shifts together, devices may still look normal relative to their neighbours.
This lot's median is 31.37 microamps at 24 hours and its median increase is 2.273 microamps.
The historical level score is about 61.4 against a threshold of four. The lot is flagged for
review; the device already exceeding its limit remains a FAIL."

## Optional zero-spread case

Select **ILLUSTRATIVE_Q155_D000**, **threshold_voltage**.

"The reading changes from 1.6 to 1.7 volts, but the group's median change and its MAD are both
zero because this illustrative fixture is quantised. Dividing by zero would produce an invalid
score, so the system returns insufficient evidence. This lot is explicitly illustrative."

## Real measurements

Real-data explorer -> **reram** -> **standby_current** -> **package**. Inspect **reram:4**.
The graph's unit is **mA**, not uA.

| Dose krad(Si) | mA | Equivalent uA |
|---|---|---|
| 0 | 0.0102 | 10.2 |
| 10 | 0.0104 | 10.4 |
| 20 | 0.0108 | 10.8 |
| 50 | 0.0215 | 21.5 |

"This real ReRAM measurement rises from 0.0102 to 0.0215 milliamps by 50 kilorad, equivalent to
10.2 to 21.5 microamps. These are radiation measurements, not thermal burn-in hours. The explorer
supports source-traceable investigation; it does not establish real burn-in forecasting accuracy."

## Evaluation

Evaluation -> **Module B - SYNTHETIC** -> **leakage_current**, **Ordinary test**.

"The model's mean absolute error is 1.18 microamps, compared with 2.67 for keeping the last
reading unchanged and 2.22 for linear extrapolation. This is synthetic test evidence. Performance
is weaker on shifted lots, and defects that begin after the observation window can be missed."

## Reports and closing

Reports & audit -> select **SYN_L159_D009** as focus -> **Build PDF report**.

"The engineer can export a PDF, device and parameter CSVs, and JSON results. The report presents
the drift, peer comparison, interval, limits and reason for escalation. The audit log records
the input fingerprint, model version and specification revision.

Our prototype demonstrates validated inputs, lot-relative analysis, early forecasting and an
explainable recommendation. Next comes validation on real multi-lot burn-in data with approved
specifications. Our aim is earlier engineering insight while preserving required qualification.
Thank you. We are Team Agamemnon, and this is DRISHTI."
