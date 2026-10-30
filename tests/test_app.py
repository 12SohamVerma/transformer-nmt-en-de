"""
Headless end-to-end test of the Streamlit app (streamlit.testing AppTest).
Requires a trained checkpoint and tokenizer; skipped otherwise.

Run:  python tests/test_app.py
"""

from __future__ import annotations

import html
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config as C  # noqa: E402

CHECKPOINT = C.CHECKPOINT_DIR / "small_best.pt"
HAS_MODEL = CHECKPOINT.exists() and C.SPM_MODEL_PATH.exists()
SENTENCE = "Two dogs are playing in the snow."


@unittest.skipUnless(HAS_MODEL, "no trained checkpoint / tokenizer")
class TestStreamlitApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from streamlit.testing.v1 import AppTest
        from translate import Translator

        cls.reference = Translator(CHECKPOINT)
        cls.at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()

    def page_html(self) -> str:
        return " ".join(m.value for m in self.at.markdown)

    def translate(self, method: str, text: str = SENTENCE) -> None:
        self.at.text_area(key="src_text").set_value(text)
        self.at.radio[0].set_value(method)
        next(b for b in self.at.button if b.label == "Translate").click()
        self.at.run()
        self.assertFalse(self.at.exception, self.at.exception)

    def test_1_page_loads_with_model_and_metrics(self):
        self.assertFalse(self.at.exception, self.at.exception)
        page = self.page_html()
        self.assertIn("Neural Machine Translation", page)
        self.assertIn("Model loaded", page)
        self.assertIn("37.09", page)          # read from results/metrics/small_metrics.json
        self.assertIn("6.00", page)           # test perplexity

    def test_2_greedy_matches_model(self):
        self.translate(self.at.radio[0].options[0])            # Greedy Search
        expected = self.reference.translate(SENTENCE, beam_size=1).translation
        self.assertIn(html.escape(expected), self.page_html())
        self.assertIn("Greedy search", self.page_html())

    def test_3_beam_matches_model(self):
        self.translate(self.at.radio[0].options[1])            # Beam Search (4, 0.6)
        expected = self.reference.translate(SENTENCE, beam_size=4, alpha=0.6).translation
        self.assertIn(html.escape(expected), self.page_html())
        self.assertIn("Beam search", self.page_html())
        self.assertIn("k = 4, α = 0.6", self.page_html())

    def test_4_example_button_fills_input(self):
        self.at.button(key="example_1").click()
        self.at.run()
        self.assertEqual(self.at.text_area(key="src_text").value, "Two dogs are playing in the snow.")

    def test_5_empty_input_shows_friendly_warning(self):
        self.translate(self.at.radio[0].options[0], text="   ")
        self.assertTrue(any("Please enter an English sentence" in w.value for w in self.at.warning))


if __name__ == "__main__":
    unittest.main(verbosity=2)
