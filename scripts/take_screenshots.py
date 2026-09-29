"""Capture dashboard screenshots with headless Microsoft Edge (Playwright).

    streamlit run app/streamlit_app.py            # in another terminal
    python scripts/take_screenshots.py [--url http://localhost:8501] [--channel msedge]

Dev-only dependency: pip install playwright (uses the installed Edge/Chrome via
--channel; no browser download). Output: docs/screenshots/*.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"


def settle(page, ms=2500):
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(ms)
    try:
        page.wait_for_selector("[data-testid='stStatusWidget']", state="detached", timeout=30000)
    except Exception:  # noqa: BLE001 - widget may never appear for fast reruns
        pass
    page.wait_for_timeout(800)


def nav(page, label):
    page.get_by_test_id("stSidebarNav").get_by_text(label, exact=True).click()
    settle(page)


def choose(page, index, text):
    box = page.get_by_test_id("stSelectbox").nth(index)
    box.click()
    page.keyboard.type(text)
    page.keyboard.press("Enter")
    settle(page)


def shot(page, name):
    OUT.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(OUT / f"{name}.png"))
    print("saved", name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--channel", default="msedge")
    a = ap.parse_args()
    with sync_playwright() as p:
        b = p.chromium.launch(channel=a.channel, headless=True)
        page = b.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        page.goto(a.url)
        settle(page, 4000)
        page.get_by_test_id("stSidebar").get_by_role("button", name="▶ Load demo").click()
        page.get_by_text("Current run").wait_for(timeout=120000)
        settle(page)
        shot(page, "01_overview")
        nav(page, "Upload & validate")
        shot(page, "02_upload_validate")
        nav(page, "Lot analysis")
        shot(page, "03_lot_analysis_L159")
        nav(page, "Device analysis")
        page.mouse.wheel(0, 250)
        page.wait_for_timeout(800)
        shot(page, "04_device_SYN_L159_D009")
        choose(page, 0, "SYN_L152_D000")
        page.get_by_role("tab", name="RDS_on · ESCALATE").click()
        settle(page, 1500)
        page.mouse.wheel(0, 250)
        page.wait_for_timeout(800)
        shot(page, "05_device_SYN_L152_D000_interval")
        choose(page, 0, "ILLUSTRATIVE_Q155_D000")
        shot(page, "06_device_zero_mad")
        nav(page, "Lot analysis")
        choose(page, 0, "SYN_L180")
        shot(page, "07_lot_shift_L180")
        nav(page, "Evaluation")
        shot(page, "08_evaluation_module_a")
        page.get_by_role("tab", name="SECOM · REAL process data").click()
        settle(page, 1500)
        shot(page, "09_evaluation_secom")
        nav(page, "Reports & audit")
        page.get_by_role("button", name="Build PDF report").click()
        settle(page, 4000)
        shot(page, "10_reports_audit")
        nav(page, "Real-data explorer")
        shot(page, "11_real_data_explorer")
        b.close()


if __name__ == "__main__":
    main()
