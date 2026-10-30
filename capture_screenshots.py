"""
Start the Streamlit app and capture screenshots with Playwright, driving the locally
installed Microsoft Edge / Google Chrome (no browser download).

Usage:  python capture_screenshots.py [--port 8599]
Output: results/screenshots/ui_*.png
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request

from playwright.sync_api import sync_playwright

import config as C

OUT = C.RESULTS_DIR / "screenshots"


def wait_for_server(url: str, timeout: float = 90) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(url + "/_stcore/health", timeout=2)
            return
        except OSError:
            time.sleep(1)
    raise TimeoutError(f"Streamlit did not start at {url}")


def launch_browser(p):
    for channel in ("msedge", "chrome", None):
        try:
            return p.chromium.launch(channel=channel) if channel else p.chromium.launch()
        except Exception:
            continue
    raise RuntimeError("No Chromium-based browser available for Playwright")


def full_page(page, path, width: int) -> None:
    """Streamlit scrolls inside its own container, so grow the viewport to the content height."""
    height = page.evaluate("document.querySelector('[data-testid=\"stMain\"]').scrollHeight")
    page.set_viewport_size({"width": width, "height": max(900, height + 40)})
    page.wait_for_timeout(800)
    page.screenshot(path=str(path))
    print(f"saved {path.name}")


def translate(page, sentence: str, method: str) -> None:
    page.get_by_label("English sentence").fill(sentence)
    page.locator(".st-key-decoding label").filter(has_text=method).first.click()
    page.locator(".st-key-translate_btn button").click()
    page.locator(".hint", has_text=sentence).wait_for(timeout=120_000)
    page.wait_for_timeout(1200)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8599)
    args = parser.parse_args()
    url = f"http://localhost:{args.port}"
    OUT.mkdir(parents=True, exist_ok=True)

    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true",
         "--server.port", str(args.port)],
        cwd=C.ROOT_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_server(url)
        with sync_playwright() as p:
            browser = launch_browser(p)
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            page.goto(url)
            page.locator(".pill.ok").wait_for(timeout=120_000)

            translate(page, "A little girl in a pink dress is climbing a tree.", "Beam Search")
            full_page(page, OUT / "ui_01_full_page_beam.png", 1440)

            page.set_viewport_size({"width": 1440, "height": 1000})
            translate(page, "Two dogs are playing in the snow.", "Greedy Search")
            page.evaluate("document.querySelector('[data-testid=stMain]').scrollTo(0, 0)")
            page.wait_for_timeout(500)
            page.screenshot(path=str(OUT / "ui_02_translate_greedy.png"))
            print("saved ui_02_translate_greedy.png")

            page.set_viewport_size({"width": 390, "height": 900})        # phone width
            page.wait_for_timeout(1500)
            full_page(page, OUT / "ui_03_mobile.png", 390)
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=20)


if __name__ == "__main__":
    main()
