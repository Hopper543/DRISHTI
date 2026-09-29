"""Headless dashboard smoke tests (Streamlit AppTest): load demo, render every
page, no exceptions. Browser-level checks are documented in docs/TESTING.md."""
import pytest
from streamlit.testing.v1 import AppTest

from conftest import ROOT

APP = str(ROOT / "app" / "streamlit_app.py")
PAGES = ["views/upload.py", "views/lot.py", "views/device.py", "views/evaluation.py", "views/reports.py",
         "views/real_data.py"]


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception
    at.button(key="sidebar_demo").click().run()
    assert not at.exception
    assert at.session_state["run"].ok
    return at


def test_overview_before_demo():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert any("DRISHTI" in t.value for t in at.title)


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_with_demo(app, page):
    app.switch_page(page).run()
    assert not app.exception, [e.value for e in app.exception]


def test_device_page_shows_named_case(app):
    app.switch_page("views/device.py").run()
    app.selectbox[0].set_value("SYN_L159_D009").run()
    text = " ".join(m.value for m in app.markdown)
    assert "A_UNUSUAL_CHANGE" in text and "ESCALATE" in text


def test_reports_page_builds_pdf_and_logs(app):
    app.switch_page("views/reports.py").run()
    next(b for b in app.button if b.label == "Build PDF report").click().run()
    assert not app.exception
    assert app.session_state["pdf"][1][:5] == b"%PDF-"
