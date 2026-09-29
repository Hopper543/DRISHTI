"""DRISHTI inspector dashboard.  Run:  streamlit run app/streamlit_app.py"""
import streamlit as st

st.set_page_config(page_title="DRISHTI · Component Screening", page_icon="🔭", layout="wide")

from ui import sidebar_status  # noqa: E402

pages = [
    st.Page("views/overview.py", title="Overview", icon=":material/home:", default=True),
    st.Page("views/upload.py", title="Upload & validate", icon=":material/upload_file:"),
    st.Page("views/lot.py", title="Lot analysis", icon=":material/scatter_plot:"),
    st.Page("views/device.py", title="Device analysis", icon=":material/memory:"),
    st.Page("views/evaluation.py", title="Evaluation", icon=":material/analytics:"),
    st.Page("views/reports.py", title="Reports & audit", icon=":material/description:"),
    st.Page("views/real_data.py", title="Real-data explorer", icon=":material/science:"),
]
nav = st.navigation(pages)
sidebar_status()
nav.run()
