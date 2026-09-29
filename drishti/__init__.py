"""DRISHTI: lot-relative anomaly detection and early (24 h -> 168 h) forecasting
for electronic component burn-in screening. Screening-round prototype.

Core logic lives here and is independent of the Streamlit dashboard in app/.
"""

__version__ = "0.1.0"
EVIDENCE_NOTICE = (
    "Screening-round prototype. Synthetic results are not real defect-detection "
    "performance; specification limits are DEMO assumptions; recommendations at 24 h "
    "are early screening decisions, not qualification completion."
)
