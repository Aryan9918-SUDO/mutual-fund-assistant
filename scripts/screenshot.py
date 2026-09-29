"""Capture UI screenshots for the README using Playwright + the system Chrome.

Usage: python scripts/screenshot.py   (Streamlit must be running on :8601)
"""
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)
BASE = "http://localhost:8601"

_Q1 = "What%20is%20the%20expense%20ratio%20of%20HDFC%20Flexi%20Cap%20Fund%3F"
SHOTS = [
    ("answer.png", f"{BASE}/?q={_Q1}", "text=Total Expense Ratio"),
    ("refusal.png", f"{BASE}/?q=Should%20I%20buy%20HDFC%20Flexi%20Cap%20Fund%3F",
     "text=facts-only assistant"),
    ("landing.png", BASE, "text=Facts-Only MF Assistant"),
]


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 900, "height": 1150}, device_scale_factor=2)
        for name, url, wait_for in SHOTS:
            page.goto(url, wait_until="networkidle")
            try:
                page.wait_for_selector(wait_for, timeout=20000)
            except Exception:
                pass
            time.sleep(2)  # let expanders/animations settle
            page.screenshot(path=str(OUT / name), full_page=True)
            print("saved", name)
        browser.close()


if __name__ == "__main__":
    main()
