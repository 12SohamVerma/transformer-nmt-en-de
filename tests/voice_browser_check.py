"""
Browser check for the voice features (needs Playwright + Microsoft Edge or Google Chrome).

The browser's speech recogniser is replaced by a stub that "hears" a fixed sentence and
speechSynthesis.speak is recorded, so the full path is exercised without a microphone:
mic button -> Python callback -> local Transformer translation -> Listen buttons -> clipboard.

Run:  python tests/voice_browser_check.py
"""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from translate import Translator  # noqa: E402

PORT = 8596
SPOKEN = "Two dogs are playing in the snow."
STUBS = """
window.__spoken = [];
class FakeRecognition {
  start() {
    setTimeout(() => {
      if (this.onresult) this.onresult({ results: [[{ transcript: window.__heard }]] });
      if (this.onend) this.onend();
    }, 300);
  }
  stop() { if (this.onend) this.onend(); }
  abort() {}
}
window.SpeechRecognition = FakeRecognition;
window.webkitSpeechRecognition = FakeRecognition;
window.__realVoices = () => speechSynthesis.getVoices().map((v) => `${v.name} [${v.lang}]${v.localService ? "" : " (online)"}`);
speechSynthesis.speak = (u) => {
  window.__spoken.push({ text: u.text, lang: u.lang });
  if (u.onstart) u.onstart();
  setTimeout(() => { if (u.onend) u.onend(); }, 200);
};
"""


def main() -> int:
    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true", "--server.port", str(PORT)],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    failures = []
    try:
        for _ in range(90):
            try:
                urllib.request.urlopen(f"http://localhost:{PORT}/_stcore/health", timeout=2)
                break
            except OSError:
                time.sleep(1)
        expected = Translator(ROOT / "checkpoints" / "small_best.pt").translate(SPOKEN, 4, 0.6).translation

        with sync_playwright() as p:
            browser = None
            for channel in ("msedge", "chrome", None):
                try:
                    browser = p.chromium.launch(channel=channel) if channel else p.chromium.launch()
                    break
                except Exception:
                    continue
            context = browser.new_context(viewport={"width": 1440, "height": 1000},
                                          permissions=["clipboard-read", "clipboard-write"])
            context.add_init_script(STUBS)
            page = context.new_page()
            page.goto(f"http://localhost:{PORT}")
            page.locator(".st-key-voice_mic button").wait_for(timeout=120_000)
            page.wait_for_timeout(1500)
            voices = page.evaluate("window.__realVoices()")
            print("Voices available in this browser:")
            for v in voices:
                print("   ", v)
            if not any("[de" in v for v in voices):
                print("   (no German voice found - German will be read with a default voice)")

            # 1) voice input -> transcript in the text box -> translated by the model
            page.evaluate(f"window.__heard = {SPOKEN!r}")
            page.locator(".st-key-voice_mic button").click()
            page.wait_for_function("t => document.querySelector('textarea').value === t", arg=SPOKEN, timeout=30_000)
            page.wait_for_function("t => (document.querySelector('.tgt-box .txt') || {}).innerText === t",
                                   arg=expected, timeout=60_000)
            print(f"[ok] voice input  : heard {SPOKEN!r} -> translated {expected!r}")

            # 2) Listen buttons read the right text in the right language
            page.locator(".st-key-speak_tgt button").click()
            page.locator(".st-key-speak_src button").click()
            page.wait_for_timeout(600)
            spoken = page.evaluate("window.__spoken")
            if {"text": expected, "lang": "de-DE"} not in spoken:
                failures.append(f"German Listen spoke {spoken}")
            if {"text": SPOKEN, "lang": "en-US"} not in spoken:
                failures.append(f"English Listen spoke {spoken}")
            print(f"[{'ok' if not failures else 'FAIL'}] voice output : {spoken}")

            # 3) Copy button puts the translation on the clipboard
            page.locator(".st-key-copy_tgt button").click()
            page.wait_for_timeout(500)
            clip = page.evaluate("navigator.clipboard.readText()")
            if clip != expected:
                failures.append(f"clipboard holds {clip!r}")
            print(f"[{'ok' if clip == expected else 'FAIL'}] copy         : {clip!r}")
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=20)

    for f in failures:
        print("FAIL:", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
