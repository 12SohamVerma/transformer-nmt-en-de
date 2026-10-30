"""
Streamlit UI for the from-scratch Transformer NMT model (English -> German, Multi30k).

Run:  streamlit run app.py          (or: python -m streamlit run app.py)

Translation uses the project's own Translator (translate.py): it loads
checkpoints/small_best.pt and the SentencePiece tokenizer, and calls the greedy and
beam-search decoders in models/decoding.py. All metrics are read from results/ -
nothing shown here is hard-coded. No external translation service is used.
"""

from __future__ import annotations

import html
import json

import streamlit as st

import config as C

CHECKPOINT = C.CHECKPOINT_DIR / "small_best.pt"
BEAM_SIZE, ALPHA = 4, 0.6
GREEDY, BEAM = "Greedy Search", "Beam Search"
REPO_URL = "https://github.com/12SohamVerma/transformer-nmt-en-de"
PAPER_URL = "https://arxiv.org/abs/1706.03762"
EXAMPLES = [
    "A man is riding a bicycle down the street.",
    "Two dogs are playing in the snow.",
    "A woman is cooking in the kitchen.",
    "Children are sitting at a table.",
    "A red car is parked by the building.",
]
MORE_EXAMPLES = [
    "A little girl in a pink dress is climbing a tree.",
    "A woman is reading a book on a bench in the park.",
    "Three children are playing soccer on a green field.",
    "A group of people are standing in front of a building.",
    "A man in a blue shirt is standing on a ladder.",
]

st.set_page_config(page_title="Transformer AI · English → German", page_icon="🔤", layout="wide",
                   initial_sidebar_state="expanded")

# --------------------------------------------------------------------------
# Icons (Lucide-style line icons, inline SVG)
# --------------------------------------------------------------------------
_ICON_PATHS = {
    "home": '<path d="M3 10.5 12 3l9 7.5"/><path d="M5 9.5V21h14V9.5"/><path d="M9.5 21v-6h5v6"/>',
    "languages": '<path d="m5 8 6 6"/><path d="m4 14 6-6 2-3"/><path d="M2 5h12"/><path d="M7 2h1"/>'
                 '<path d="m22 22-5-10-5 10"/><path d="M14 18h6"/>',
    "box": '<path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 '
           '2 0l7-4A2 2 0 0 0 21 16Z"/><path d="m3.3 7 8.7 5 8.7-5"/><path d="M12 22V12"/>',
    "chart": '<path d="M3 3v18h18"/><path d="M8 17v-6"/><path d="M13 17V6"/><path d="M18 17v-3"/>',
    "file": '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v5h6"/>'
            '<path d="M16 13H8"/><path d="M16 17H8"/><path d="M10 9H8"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/>'
             '<path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "torch": '<path d="M12 3 7.6 7.4a6.2 6.2 0 1 0 8.8 0"/><circle cx="15.6" cy="5.8" r="1" fill="currentColor"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>'
                '<path d="M3 12c0 1.66 4 3 9 3s9-1.34 9-3"/>',
    "sparkles": '<path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9Z"/><path d="M19 3v4"/>'
                '<path d="M21 5h-4"/><path d="M5 17v3"/><path d="M6.5 18.5h-3"/>',
    "gear": '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 '
            '2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 '
            '9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 '
            '0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 '
            '2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 '
            '1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 '
            '0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1Z"/>',
    "zap": '<path d="M13 2 3 14h9l-1 8 10-12h-9l1-8Z"/>',
    "list": '<path d="M8 6h13"/><path d="M8 12h13"/><path d="M8 18h13"/><path d="M3 6h.01"/><path d="M3 12h.01"/>'
            '<path d="M3 18h.01"/>',
    "check": '<circle cx="12" cy="12" r="10"/><path d="m8.5 12 2.5 2.5 4.5-5"/>',
    "chevron": '<path d="m6 9 6 6 6-6"/>',
    "award": '<circle cx="12" cy="8" r="6"/><path d="M15.48 12.89 17 22l-5-3-5 3 1.52-9.11"/>',
    "type": '<path d="M4 7V4h16v3"/><path d="M9 20h6"/><path d="M12 4v16"/>',
    "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "down": '<path d="m22 17-8.5-8.5-5 5L2 7"/><path d="M16 17h6v-6"/>',
    "cpu": '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/>'
           '<path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>',
    "layers": '<path d="m12 2 10 5-10 5L2 7Z"/><path d="m2 17 10 5 10-5"/><path d="m2 12 10 5 10-5"/>',
    "github": '<path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 '
              '0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.4 5.4 0 0 0 4 9c0 '
              '3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65S8.93 17.38 9 18v4"/><path d="M9 18c-4.51 2-5-2-7-2"/>',
    "book": '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 '
            '3-3h7z"/>',
    "cap": '<path d="M22 10 12 5 2 10l10 5 10-5Z"/><path d="M6 12v5c3 3 9 3 12 0v-5"/>',
    "code": '<path d="m16 18 6-6-6-6"/><path d="m8 6-6 6 6 6"/>',
    "external": '<path d="M15 3h6v6"/><path d="M10 14 21 3"/><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 '
                '2 0 0 1 2-2h6"/>',
    "copy": '<rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 '
            '2 0 0 1 2 2v1"/>',
    "scissors": '<circle cx="6" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M20 4 8.12 15.88"/>'
                '<path d="M14.47 14.48 20 20"/><path d="M8.12 8.12 12 12"/>',
    "wave": '<path d="M2 12c2-4 4-4 6 0s4 4 6 0 4-4 6 0"/><path d="M2 18c2-4 4-4 6 0s4 4 6 0 4-4 6 0"/>',
    "arrow": '<path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>',
}


def icon(name: str, size: int = 20, stroke: float = 1.9) -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">{_ICON_PATHS[name]}</svg>')


LOGO = """<svg width="54" height="54" viewBox="0 0 48 48" fill="none">
<defs><linearGradient id="lg" x1="0" y1="0" x2="48" y2="48"><stop offset="0" stop-color="#2563eb"/>
<stop offset="1" stop-color="#7c3aed"/></linearGradient></defs>
<g stroke="url(#lg)" stroke-width="2.4" stroke-linecap="round">
<path d="M24 6 9 15v18l15 9 15-9V15Z"/><path d="M24 6v12M9 15l10 6M39 15l-10 6M9 33l10-6M39 33l-10-6M24 42V30"/>
<path d="M19 21h10M19 27h10M19 21v6M29 21v6"/></g>
<g fill="#fff" stroke="url(#lg)" stroke-width="2.4"><circle cx="24" cy="6" r="3.2"/><circle cx="9" cy="15" r="3.2"/>
<circle cx="39" cy="15" r="3.2"/><circle cx="9" cy="33" r="3.2"/><circle cx="39" cy="33" r="3.2"/>
<circle cx="24" cy="42" r="3.2"/><circle cx="24" cy="24" r="4"/></g></svg>"""

FLAG_UK = """<svg viewBox="0 0 60 30" preserveAspectRatio="xMidYMid slice"><clipPath id="uk-s"><path d="M0,0v30h60V0z"/>
</clipPath><clipPath id="uk-t"><path d="M30,15h30v15zv15H0zH0V0zV0h30z"/></clipPath><g clip-path="url(#uk-s)">
<path d="M0,0v30h60V0z" fill="#012169"/><path d="M0,0 60,30M60,0 0,30" stroke="#fff" stroke-width="6"/>
<path d="M0,0 60,30M60,0 0,30" clip-path="url(#uk-t)" stroke="#C8102E" stroke-width="4"/>
<path d="M30,0v30M0,15h60" stroke="#fff" stroke-width="10"/><path d="M30,0v30M0,15h60" stroke="#C8102E"
stroke-width="6"/></g></svg>"""
FLAG_DE = """<svg viewBox="0 0 5 3" preserveAspectRatio="xMidYMid slice"><rect width="5" height="3" fill="#000"/>
<rect y="1" width="5" height="2" fill="#DD0000"/><rect y="2" width="5" height="1" fill="#FFCE00"/></svg>"""

# --------------------------------------------------------------------------
# Theme
# --------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
:root {
  --blue: #2563eb; --indigo: #4f46e5; --purple: #7c3aed; --ink: #111633; --text: #334155; --muted: #6b7280;
  --line: #e6e8f2; --card: #ffffff; --bg: #f7f8fc; --soft: #eef2ff; --soft2: #f5f3ff;
  --grad: linear-gradient(90deg, #2563eb 0%, #7c3aed 100%);
  --shadow: 0 1px 2px rgba(17,22,51,.04), 0 4px 18px rgba(79,70,229,.06);
}
.stApp *:not([data-testid="stIconMaterial"]):not(code):not(pre) {
  font-family: 'Plus Jakarta Sans', 'Segoe UI', -apple-system, Roboto, sans-serif;}
.stApp {background: var(--bg);}
header[data-testid="stHeader"] {background: transparent;}
.block-container {max-width: 1280px; padding: 1.4rem 2.2rem 3rem;}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {background: #fff; border-right: 1px solid var(--line);
  width: 270px !important; min-width: 270px !important;}
section[data-testid="stSidebar"] .block-container, [data-testid="stSidebarUserContent"] {padding: 0 1rem 1rem;}
.sb {display: flex; flex-direction: column; min-height: calc(100vh - 6rem);}
.brand {text-align: center; padding: .2rem 0 1.4rem; border-bottom: 1px solid var(--line); margin-bottom: 1rem;}
.brand-name {font-weight: 800; letter-spacing: .08em; color: var(--ink); font-size: 1.05rem; margin-top: .5rem;}
.brand-sub {color: var(--text); font-size: .82rem; margin-top: .1rem;}
.nav a {display: flex; align-items: center; gap: .85rem; padding: .72rem .95rem; margin: .18rem 0;
  border-radius: 12px; color: #1f2937 !important; text-decoration: none !important; font-weight: 500;
  font-size: .95rem; border-left: 3px solid transparent; transition: background .15s;}
.nav a svg {color: #374151; flex: none;}
.nav a span {white-space: nowrap;}
.nav a:hover {background: var(--soft);}
.nav a:focus, .nav:not(:focus-within) a.default {background: var(--soft); color: var(--indigo) !important;
  border-left-color: var(--indigo); outline: none;}
.nav a:focus svg, .nav:not(:focus-within) a.default svg {color: var(--indigo);}
.sb-status {margin-top: auto; border: 1px solid var(--line); border-radius: 16px; padding: .9rem 1rem;
  box-shadow: var(--shadow); background: #fff;}
.sb-status .row {display: flex; align-items: center; gap: .75rem; padding: .62rem 0;
  border-bottom: 1px solid var(--line);}
.sb-status .row:last-child {border-bottom: none;}
.sb-status .k {color: var(--muted); font-size: .74rem;}
.sb-status .v {color: var(--ink); font-size: .88rem; font-weight: 500;}
.dot {width: 12px; height: 12px; border-radius: 50%; background: #10b981; box-shadow: 0 0 0 4px #d1fae5; flex: none;
  margin: 0 .3rem;}
.dot.off {background: #ef4444; box-shadow: 0 0 0 4px #fee2e2;}
.sb-status .ok {color: #047857; font-weight: 700; font-size: .92rem;}
.sb-status .bad {color: #b91c1c; font-weight: 700; font-size: .92rem;}
.sb-status .mono {color: #4f46e5; font-size: .76rem;}

/* ---------- hero ---------- */
.hero {position: relative; overflow: hidden; border-radius: 22px; padding: 2rem 2.4rem 1.8rem;
  background: radial-gradient(1200px 300px at 85% 0%, #ede9fe 0%, transparent 60%),
              linear-gradient(120deg, #f8faff 0%, #eef2ff 55%, #f3e8ff 100%);
  border: 1px solid #e7e9fb;}
.eyebrow {color: var(--indigo); font-weight: 700; font-size: .78rem; letter-spacing: .05em; margin: 0 0 .7rem;}
.hero h1 {font-size: clamp(2rem, 3.6vw, 3.1rem); line-height: 1.07; font-weight: 800; color: var(--ink);
  margin: 0; padding: 0; letter-spacing: -.02em;}
.hero h1 .grad {background: var(--grad); -webkit-background-clip: text; background-clip: text; color: transparent;}
.hero .sub {color: #1f2937; font-size: 1.2rem; margin: .7rem 0 1.2rem;}
.chips {display: flex; flex-wrap: wrap; gap: .7rem;}
.chip {display: inline-flex; align-items: center; gap: .55rem; background: #fff; border: 1px solid var(--line);
  border-radius: 999px; padding: .5rem 1rem; font-size: .88rem; font-weight: 500; color: #1f2937;
  box-shadow: var(--shadow);}
.chip svg {color: var(--indigo);}
.chip.red svg {color: #ef4444;}
.paper-stack {position: absolute; right: 2.6rem; top: 1.6rem; width: 230px; height: 165px;}
.paper-stack .sheet {position: absolute; inset: 0; border-radius: 18px; border: 1px solid rgba(255,255,255,.8);
  background: linear-gradient(135deg, rgba(255,255,255,.95), rgba(237,233,254,.85));
  box-shadow: 0 12px 30px rgba(124,58,237,.18);}
.paper-stack .s3 {transform: translate(34px, 18px); background: linear-gradient(135deg, #dbeafe, #c7d2fe); opacity: .8;}
.paper-stack .s2 {transform: translate(18px, 9px); background: linear-gradient(135deg, #e0e7ff, #ddd6fe); opacity: .9;}
.paper-stack .s1 {padding: 1.3rem 1.4rem;}
.paper-stack .ptitle {color: #4338ca; font-weight: 700; font-size: .98rem; line-height: 1.3; width: 60%;}
.paper-stack .pauth {color: var(--muted); font-size: .82rem; position: absolute; bottom: 1.2rem; left: 1.4rem;}
.paper-stack .lines {position: absolute; right: 1.2rem; top: 1.3rem; width: 52px;}
.paper-stack .lines span {display: block; height: 7px; border-radius: 4px; background: #c4b5fd; margin-bottom: 9px;
  opacity: .75;}
@media (max-width: 1100px) {.paper-stack {display: none;}}

/* ---------- sections ---------- */
.sec {display: flex; align-items: baseline; flex-wrap: wrap; gap: .4rem 1.2rem; margin: 2.3rem 0 1rem;
  scroll-margin-top: 1rem;}
.sec h2 {font-size: 1.55rem; font-weight: 800; color: var(--ink); margin: 0; padding: 0 0 0 .8rem;
  border-left: 5px solid var(--purple); line-height: 1.15; border-radius: 2px;}
.sec.blue h2 {border-left-color: var(--blue);}
.sec .desc {color: var(--muted); font-size: .95rem;}
.sec .right {margin-left: auto;}
.sec.stacked .desc {order: 3; flex-basis: 100%;}
.pill {display: inline-flex; align-items: center; gap: .45rem; border-radius: 999px; padding: .45rem .95rem;
  font-size: .85rem; font-weight: 600;}
.pill.ok {background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0;}
.pill.bad {background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca;}

.grid {display: grid; gap: 1rem;}
.cols-3 {grid-template-columns: repeat(3, minmax(0, 1fr));}
.cols-5 {grid-template-columns: repeat(5, minmax(0, 1fr));}
@media (max-width: 1000px) {.cols-5 {grid-template-columns: repeat(3, minmax(0, 1fr));}}
@media (max-width: 700px) {.cols-3, .cols-5 {grid-template-columns: repeat(2, minmax(0, 1fr));}}
@media (max-width: 460px) {.cols-3 {grid-template-columns: 1fr;}}

.card {background: var(--card); border: 1px solid var(--line); border-radius: 16px; box-shadow: var(--shadow);}
.icard {display: flex; align-items: center; gap: 1.1rem; padding: 1rem 1.2rem;}
.ibubble {width: 62px; height: 62px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
  flex: none;}
.ibubble.sm {width: 44px; height: 44px;}
.t-indigo {background: #eef2ff; color: #4f46e5;} .t-blue {background: #eff6ff; color: #2563eb;}
.t-red {background: #fff1f2; color: #ef4444;} .t-green {background: #ecfdf5; color: #059669;}
.t-purple {background: #f5f3ff; color: #7c3aed;} .t-amber {background: #fffbeb; color: #d97706;}
.klabel {color: var(--muted); font-size: .74rem; font-weight: 600; letter-spacing: .07em; text-transform: uppercase;}
.kvalue {color: var(--ink); font-size: 1.04rem; font-weight: 700; line-height: 1.3; margin-top: .2rem;}

/* ---------- translate panels ---------- */
.st-key-src_card, .st-key-tgt_card {background: #fff; border: 1px solid var(--line); border-radius: 18px;
  padding: 1.2rem 1.4rem 1.4rem; box-shadow: var(--shadow); gap: .8rem;}
.lang {display: flex; align-items: center; gap: .8rem; font-size: 1.05rem; color: var(--ink);}
.lang b {font-weight: 700;}
.lang .flag {width: 40px; height: 40px; border-radius: 50%; overflow: hidden; flex: none;
  box-shadow: 0 0 0 4px #f1f2f9;}
.lang .flag svg {width: 100%; height: 100%; display: block;}
.lang .chev {margin-left: auto; color: #374151;}
.st-key-src_card textarea {font-size: 1.05rem !important; line-height: 1.55; color: var(--ink);}
.st-key-src_card [data-testid="stTextAreaRootElement"], .st-key-src_card [data-testid="stTextAreaRootElement"] > div
  {background: #fff !important; border-radius: 12px;}
.st-key-src_card [data-testid="stTextAreaRootElement"] {border: 1px solid var(--line);}
.st-key-src_card textarea {background: #fff !important;}

.st-key-decoding [data-testid="stElementContainer"], .st-key-decoding [data-testid="stRadio"] {width: 100% !important;}
.st-key-decoding [role="radiogroup"] {display: grid; grid-template-columns: 1fr 1fr; gap: .8rem; width: 100%;}
.st-key-decoding [role="radiogroup"] > div {border: 1px solid var(--line); border-radius: 12px; padding: .8rem .95rem;
  background: #fff; transition: all .15s; margin: 0;}
.st-key-decoding [role="radiogroup"] > div:hover {border-color: #c7d2fe;}
.st-key-decoding [role="radiogroup"] > div[data-selected="true"] {border: 1.5px solid #8b5cf6;
  background: linear-gradient(135deg, #eef2ff, #f5f3ff);}
.st-key-decoding [data-testid="stRadioOption"] p {font-weight: 700; color: var(--ink); font-size: .93rem;}
.st-key-decoding [data-selected="true"] [data-testid="stRadioOption"] p {color: #4338ca;}
.st-key-decoding [data-testid="stRadioCaption"] {padding-left: 1.75rem;}
.st-key-decoding [data-testid="stRadioCaption"] p {color: var(--muted) !important; font-size: .8rem !important;}
@media (max-width: 520px) {.st-key-decoding [role="radiogroup"] {grid-template-columns: 1fr;}}
.st-key-decoding label[data-testid="stWidgetLabel"] p {font-weight: 600; color: var(--ink);}

.st-key-translate_btn button {background: var(--grad); border: none; border-radius: 12px; min-height: 3.1rem;
  box-shadow: 0 8px 20px rgba(79,70,229,.28);}
.st-key-translate_btn button p {font-size: 1.08rem; font-weight: 700;}
.st-key-translate_btn button::after {content: "›"; position: absolute; right: 1.6rem; font-size: 1.5rem;
  line-height: 1; opacity: .85;}
.st-key-translate_btn button:hover {filter: brightness(1.06);}

.tgt-box {background: linear-gradient(135deg, #f5f7ff, #f5f3ff); border-radius: 14px; padding: 1.3rem 1.4rem;
  min-height: 112px; display: flex; align-items: center;}
.tgt-box .txt {color: #1e1b4b; font-size: clamp(1.25rem, 2.1vw, 1.75rem); font-weight: 650; line-height: 1.38;}
.tgt-box .ph {color: var(--muted); font-size: 1rem;}
.details-h {display: flex; align-items: center; gap: .5rem; font-weight: 700; color: var(--ink); font-size: .95rem;
  margin: .3rem 0 .1rem;}
.details-h svg {color: var(--indigo);}
.dgrid {display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .7rem;}
@media (max-width: 560px) {.dgrid {grid-template-columns: 1fr;}}
.dcard {display: flex; align-items: center; gap: .7rem; border: 1px solid var(--line); border-radius: 12px;
  padding: .75rem .8rem; background: #fbfbff;}
.dcard svg {color: var(--indigo); flex: none;}
.dcard .a {font-weight: 700; color: var(--ink); font-size: .86rem;}
.dcard .b {color: var(--muted); font-size: .78rem;}
.hint {color: var(--muted); font-size: .82rem;}

/* ---------- examples ---------- */
.ex-h {display: flex; align-items: center; gap: .55rem; flex-wrap: wrap;}
.ex-h svg {color: var(--purple);}
.ex-h b {color: var(--ink); font-size: 1.02rem;}
.ex-h span {color: var(--muted); font-size: .85rem; margin-left: .6rem;}
div[class*="st-key-example_"] button, div[class*="st-key-more_example_"] button {background: #fff;
  border: 1px solid var(--line); border-radius: 14px; min-height: 3.6rem; box-shadow: var(--shadow); width: 100%;}
div[class*="st-key-example_"] button p, div[class*="st-key-more_example_"] button p {white-space: normal;
  font-size: .88rem; color: #1f2937; line-height: 1.4;}
div[class*="st-key-example_"] button:hover, div[class*="st-key-more_example_"] button:hover {border-color: #a5b4fc;
  background: #f8f9ff;}
.st-key-more_examples button {border: none; background: transparent; box-shadow: none; padding: 0;}
.st-key-more_examples button p {color: var(--indigo); font-weight: 600; font-size: .88rem;}
.st-key-more_examples {display: flex; justify-content: flex-end;}

/* ---------- performance / architecture / paper / about ---------- */
.metric {padding: 1rem 1.1rem;}
.metric .top {display: flex; align-items: center; gap: .6rem; margin-bottom: .55rem;}
.metric .num {font-size: 1.75rem; font-weight: 800; color: var(--ink); letter-spacing: -.01em;}
.metric .sm {color: var(--muted); font-size: .78rem; margin-top: .15rem;}
.metric.hl {background: linear-gradient(135deg, #eef2ff, #f5f3ff); border-color: #c7d2fe;}
.metric.hl .num {background: var(--grad); -webkit-background-clip: text; background-clip: text; color: transparent;}
.note {display: flex; gap: .7rem; background: #fffbeb; border: 1px solid #fde68a; color: #78350f; border-radius: 14px;
  padding: .85rem 1.1rem; font-size: .88rem; margin-top: 1rem; align-items: flex-start;}
.note svg {flex: none; color: #d97706; margin-top: .1rem;}
.st-key-img_loss, .st-key-img_attn {background: #fff; border: 1px solid var(--line); border-radius: 16px;
  padding: 1rem 1.1rem; box-shadow: var(--shadow);}
.imgcap {font-weight: 700; color: var(--ink); font-size: .95rem; display: flex; align-items: center; gap: .5rem;}
.imgcap svg {color: var(--indigo);}

.flow {display: flex; align-items: stretch; gap: .45rem;}
.flow .step {flex: 1 1 0; min-width: 0; padding: 1rem .5rem; text-align: center; display: flex; flex-direction: column;
  align-items: center; gap: .45rem;}
.flow .step .nm {font-weight: 700; color: var(--ink); font-size: .86rem; line-height: 1.25;}
.flow .step .ds {color: var(--muted); font-size: .72rem; line-height: 1.3;}
.flow .step.io {background: var(--grad); border: none;}
.flow .step.io .nm, .flow .step.io .ds {color: #fff;}
.flow .step.io .ibubble {background: rgba(255,255,255,.18); color: #fff;}
.flow .arr {align-self: center; color: #a5b4fc; flex: none;}
@media (max-width: 860px) {.flow {flex-direction: column;} .flow .arr {transform: rotate(90deg);}}

.prose {padding: 1.3rem 1.5rem; color: var(--text); font-size: .93rem; line-height: 1.65;}
.prose h3 {font-size: 1.08rem; color: var(--ink); margin: 0 0 .6rem; font-weight: 750; display: flex; gap: .55rem;
  align-items: center; padding: 0;}
.prose h3 svg {color: var(--indigo);}
.prose ul {margin: .3rem 0 0; padding-left: 1.1rem;} .prose li {margin: .25rem 0;}
.prose a {color: var(--indigo); font-weight: 600; text-decoration: none;}
.prose code {background: #f1f2f9; border-radius: 6px; padding: .1rem .35rem; font-size: .82rem; color: #3730a3;}
.cite {background: #f8f9ff; border: 1px dashed #c7d2fe; border-radius: 12px; padding: .8rem 1rem; font-size: .86rem;
  color: #1f2937; margin: .8rem 0;}
table.pt {width: 100%; border-collapse: collapse; font-size: .9rem; margin-top: .4rem;}
table.pt th {text-align: left; color: var(--muted); font-weight: 600; font-size: .76rem; text-transform: uppercase;
  letter-spacing: .05em; padding: .5rem .4rem; border-bottom: 1px solid var(--line);}
table.pt td {padding: .62rem .4rem; border-bottom: 1px solid var(--line); color: var(--ink);}
table.pt td.n {font-weight: 700; color: #4338ca;}
.btnlink {display: inline-flex; align-items: center; gap: .45rem; background: var(--grad); color: #fff !important;
  border-radius: 10px; padding: .55rem 1rem; font-weight: 600; font-size: .88rem; text-decoration: none !important;
  margin-top: .9rem;}
.footer {color: var(--muted); font-size: .82rem; text-align: center; margin-top: 3rem; border-top: 1px solid var(--line);
  padding-top: 1.1rem;}
@media (max-width: 640px) {.block-container {padding: 1rem 1rem 2rem;} .hero {padding: 1.5rem 1.3rem;}}
</style>
"""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def render(markup: str, target=st) -> None:
    """Render an HTML snippet (lines are stripped so Markdown never treats them as code blocks)."""
    target.markdown("\n".join(line.strip() for line in markup.strip().splitlines() if line.strip()),
                    unsafe_allow_html=True)


def esc(text) -> str:
    return html.escape(str(text))


@st.cache_resource(show_spinner="Loading the trained Transformer checkpoint…")
def load_translator():
    from translate import Translator   # project's own inference code (tokenizer + decoders)
    return Translator(CHECKPOINT)


def get_translator():
    try:
        return load_translator(), None
    except Exception as exc:          # missing checkpoint / tokenizer, vocabulary mismatch, ...
        return None, str(exc)


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def use_example(sentence: str) -> None:
    st.session_state.src_text = sentence


def run_translation(translator, text: str, method: str) -> dict:
    beam = BEAM_SIZE if method == BEAM else 1           # beam size 1 = greedy decoding
    out = translator.translate(text, beam_size=beam, alpha=ALPHA)
    return {"source": out.source, "translation": out.translation, "method": method,
            "ms": out.seconds * 1000, "tokens": len(out.tokens)}


def copy_button(text: str) -> None:
    """Small iframe button that copies the translation to the clipboard."""
    payload = json.dumps(text).replace("</", "<\\/")
    st.iframe(f"""
<style>
body {{margin:0; display:flex; justify-content:flex-end; align-items:center; height:42px;}}
button {{all:unset; cursor:pointer; display:inline-flex; align-items:center; gap:7px; color:#4f46e5;
  font:600 13.5px 'Plus Jakarta Sans','Segoe UI',system-ui,sans-serif; border:1px solid #c7d2fe; background:#fff;
  border-radius:10px; padding:7px 14px;}}
button:hover {{background:#eef2ff;}}
</style>
<button id="copy">{icon("copy", 16)}<span>Copy</span></button>
<script>
const text = {payload};
const btn = document.getElementById('copy'), label = btn.querySelector('span');
btn.addEventListener('click', async () => {{
  let ok = false;
  try {{ await navigator.clipboard.writeText(text); ok = true; }}
  catch (e) {{
    const ta = document.createElement('textarea'); ta.value = text; document.body.appendChild(ta); ta.select();
    try {{ ok = document.execCommand('copy'); }} catch (e2) {{ ok = false; }}
    ta.remove();
  }}
  label.textContent = ok ? 'Copied' : 'Copy failed';
  setTimeout(() => label.textContent = 'Copy', 1600);
}});
</script>""", height=44)


def section(anchor: str, title: str, desc: str = "", right: str = "", blue: bool = False) -> None:
    """Section heading with a coloured bar; with a right-hand element the description moves below."""
    classes = "sec" + (" blue" if blue else "") + (" stacked" if right else "")
    render(f'<div id="{anchor}" class="{classes}"><h2>{esc(title)}</h2>'
           f'<span class="desc">{esc(desc)}</span>{f"<span class=right>{right}</span>" if right else ""}</div>')


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
render(CSS)
translator, load_error = get_translator()
metrics = load_json(C.METRICS_DIR / "small_metrics.json")
history = load_json(C.METRICS_DIR / "small_history.json")
cfg = translator.model_config if translator else C.MODEL_CONFIGS["small"]   # config stored in the checkpoint
best_epoch = translator.checkpoint_info.get("epoch") if translator else (metrics or {}).get("best_epoch")

if "src_text" not in st.session_state:
    st.session_state.src_text = EXAMPLES[4]
if "result" not in st.session_state and translator is not None:     # show a real translation on first load
    st.session_state.result = run_translation(translator, st.session_state.src_text, BEAM)

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
nav = [("overview", "home", "Overview"), ("translate", "languages", "Translate"),
       ("architecture", "box", "Model Architecture"), ("performance", "chart", "Performance"),
       ("paper", "file", "Research Paper"), ("about", "info", "About Project")]
status_row = (f'<div class="row"><span class="dot"></span><div><div class="ok">Model Loaded</div>'
              f'<div class="mono">{esc(CHECKPOINT.name)}</div></div></div>' if translator else
              '<div class="row"><span class="dot off"></span><div><div class="bad">Model Unavailable</div>'
              '<div class="mono">checkpoint not found</div></div></div>')
render(f"""
<div class="sb">
  <div class="brand">{LOGO}<div class="brand-name">TRANSFORMER AI</div>
  <div class="brand-sub">Neural Machine Translation</div></div>
  <div class="nav">{"".join(f'<a href="#{a}" class="{"default" if a == "overview" else ""}">{icon(i, 21)}'
                            f'<span>{esc(t)}</span></a>' for a, i, t in nav)}</div>
  <div class="sb-status">
    {status_row}
    <div class="row"><span style="color:#ef4444">{icon("torch", 22)}</span>
      <div><div class="k">Framework</div><div class="v">PyTorch</div></div></div>
    <div class="row"><span style="color:#374151">{icon("database", 22)}</span>
      <div><div class="k">Dataset</div><div class="v">Multi30k</div></div></div>
    <div class="row"><span style="color:#374151">{icon("gear", 22)}</span>
      <div><div class="k">Model</div><div class="v">Transformer {esc(cfg.name.capitalize())}
      {f"· epoch {best_epoch}" if best_epoch else ""}</div></div></div>
  </div>
</div>
""", st.sidebar)

# --------------------------------------------------------------------------
# Hero
# --------------------------------------------------------------------------
render(f"""
<div class="hero">
  <p class="eyebrow">COMPUTATIONAL LINGUISTICS &amp; NLP LAB · EXPERIMENTS 9 &amp; 10</p>
  <h1>Transformer-Based<br><span class="grad">Neural Machine Translation</span></h1>
  <p class="sub">Implementation of <em>Attention Is All You Need</em> (2017)</p>
  <div class="chips">
    <span class="chip">{icon("languages", 18)}English → German</span>
    <span class="chip">{icon("code", 18)}Built from scratch in PyTorch</span>
    <span class="chip red">{icon("torch", 18)}Runs locally · no external APIs</span>
  </div>
  <div class="paper-stack"><div class="sheet s3"></div><div class="sheet s2"></div>
    <div class="sheet s1"><div class="ptitle">Attention<br>Is All You Need<br>(2017)</div>
      <div class="lines"><span></span><span></span><span></span><span></span><span></span></div>
      <div class="pauth">Vaswani et al.</div></div></div>
</div>
""")

# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------
section("overview", "Project Overview", "Key details about the research paper and our implementation")
overview = [("file", "t-indigo", "Research paper", "Attention Is All You Need (2017)"),
            ("users", "t-blue", "Authors", "Vaswani et al."),
            ("torch", "t-red", "Framework", "PyTorch"),
            ("target", "t-green", "Application", "English-to-German Machine Translation"),
            ("database", "t-blue", "Dataset", "Multi30k"),
            ("box", "t-purple", "Architecture", "Transformer Encoder–Decoder")]
render('<div class="grid cols-3">' + "".join(
    f'<div class="card icard"><div class="ibubble {tone}">{icon(ic, 28)}</div>'
    f'<div><div class="klabel">{esc(k)}</div><div class="kvalue">{esc(v)}</div></div></div>'
    for ic, tone, k, v in overview) + "</div>")

# --------------------------------------------------------------------------
# Translate
# --------------------------------------------------------------------------
pill = (f'<span class="pill ok">{icon("check", 17)}Model loaded · checkpoints/{esc(CHECKPOINT.name)}</span>'
        if translator else f'<span class="pill bad">{icon("info", 17)}Model unavailable</span>')
section("translate", "Translate English → German",
        "Type an English sentence or pick an example. The translation is generated using our trained "
        "Transformer model.", right=pill, blue=True)

left, right = st.columns(2, gap="medium")
with left:
    with st.container(key="src_card"):
        render(f'<div class="lang"><span class="flag">{FLAG_UK}</span><span><b>English</b> (Source Language)'
               f'</span><span class="chev">{icon("chevron", 20)}</span></div>')
        st.text_area("English sentence", key="src_text", height=128, max_chars=500, label_visibility="collapsed",
                     placeholder="Type an English sentence, e.g. A man is riding a bicycle down the street.")
        with st.container(key="decoding"):
            decoding = st.radio(
                "Decoding method", [GREEDY, BEAM], index=1, key="decoding_method",
                captions=["Select highest probability token at each step",
                          f"Beam size: {BEAM_SIZE}, α = {ALPHA}"],
                help="Greedy picks the most likely next word. Beam search keeps the 4 best partial translations "
                     "and applies length penalty α = 0.6 (the paper's settings).")
        with st.container(key="translate_btn"):
            clicked = st.button("Translate", type="primary", icon=":material/auto_awesome:", width="stretch")

        if clicked:
            text = st.session_state.src_text.strip()
            if not text:
                st.warning("Please enter an English sentence to translate.", icon="✏️")
            elif translator is None:
                st.error("The trained model could not be loaded, so translation is unavailable right now. "
                         "Make sure `checkpoints/small_best.pt` and `tokenizer/spm_bpe.model` exist "
                         f"(run `python prepare_data.py` and `python train.py`).\n\nDetails: {load_error}",
                         icon="⚠️")
            else:
                try:
                    with st.spinner("Translating with the Transformer…"):
                        st.session_state.result = run_translation(translator, text, decoding)
                except Exception as exc:
                    st.error(f"Something went wrong during translation: {exc}", icon="⚠️")

with right:
    res = st.session_state.get("result")
    with st.container(key="tgt_card"):
        head, copy_col = st.columns([4, 1.2], vertical_alignment="center")
        render(f'<div class="lang"><span class="flag">{FLAG_DE}</span><span><b>German</b> (Target Language)'
               f'</span></div>', head)
        with copy_col:
            if res and res["translation"]:
                copy_button(res["translation"])
        if res:
            if res["method"] == GREEDY:
                m_title, m_sub = "Greedy search", "argmax at each step"
            else:
                m_title, m_sub = "Beam search", f"k = {BEAM_SIZE}, α = {ALPHA}"
            render(f"""
            <div class="tgt-box"><div class="txt">{esc(res['translation']) or '<em>(empty output)</em>'}</div></div>
            <div class="details-h">{icon("chart", 18)}Translation Details</div>
            <div class="dgrid">
              <div class="dcard">{icon("gear", 22)}<div><div class="a">{m_title}</div><div class="b">{m_sub}</div></div></div>
              <div class="dcard">{icon("zap", 22)}<div><div class="a">{res['ms']:.0f} ms</div>
                <div class="b">Inference time (CPU)</div></div></div>
              <div class="dcard">{icon("list", 22)}<div><div class="a">{res['tokens']} tokens</div>
                <div class="b">Generated length</div></div></div>
            </div>
            <div class="hint">Input: “{esc(res['source'])}”</div>""")
        else:
            render('<div class="tgt-box"><div class="ph">Your German translation will appear here.</div></div>'
                   '<div class="hint">The model works best on short, descriptive sentences like Multi30k '
                   'image captions.</div>')

ex_head, ex_more = st.columns([4, 1], vertical_alignment="center")
render(f'<div class="ex-h">{icon("sparkles", 19)}<b>Try an example</b><span>Click on any example to fill the '
       f'input box</span></div>', ex_head)
with ex_more:
    with st.popover("More examples →", key="more_examples"):
        for i, sentence in enumerate(MORE_EXAMPLES):
            st.button(sentence, key=f"more_example_{i}", on_click=use_example, args=(sentence,), width="stretch")
for col, (i, sentence) in zip(st.columns(len(EXAMPLES)), enumerate(EXAMPLES)):
    col.button(sentence, key=f"example_{i}", on_click=use_example, args=(sentence,), width="stretch")

# --------------------------------------------------------------------------
# Architecture
# --------------------------------------------------------------------------
section("architecture", "Model Architecture", "How a sentence flows through our Transformer")
steps = [("languages", "English Input", "source sentence", True), ("scissors", "Tokenization", "SentencePiece BPE", False),
         ("wave", "Positional Encoding", "embedding × √d + sin/cos", False),
         ("layers", "Encoder", "self-attention + FFN", False), ("layers", "Decoder", "masked + cross-attention", False),
         ("chart", "Linear + Softmax", "word probabilities", False), ("target", "German Output", "translated sentence", True)]
arrow = f'<span class="arr">{icon("arrow", 18, 2.2)}</span>'
render('<div class="flow">' + arrow.join(
    f'<div class="card step{" io" if io else ""}"><div class="ibubble sm t-indigo">{icon(ic, 22)}</div>'
    f'<div class="nm">{esc(nm)}</div><div class="ds">{esc(ds)}</div></div>' for ic, nm, ds, io in steps) + "</div>")

arch = [("layers", "t-indigo", "Encoder layers", cfg.num_encoder_layers),
        ("layers", "t-purple", "Decoder layers", cfg.num_decoder_layers),
        ("box", "t-blue", "Embedding dimension", cfg.d_model),
        ("users", "t-green", "Attention heads", cfg.num_heads),
        ("cpu", "t-amber", "Feed-forward dimension", cfg.d_ff),
        ("activity", "t-red", "Dropout", cfg.dropout)]
render('<div class="grid cols-3" style="margin-top:1rem">' + "".join(
    f'<div class="card icard"><div class="ibubble sm {tone}">{icon(ic, 22)}</div>'
    f'<div><div class="klabel">{esc(k)}</div><div class="kvalue">{esc(v)}</div></div></div>'
    for ic, tone, k, v in arch) + "</div>")
base = C.MODEL_CONFIGS["base"]
render(f'<div class="note">{icon("info", 18)}<div>This is the <b>Small configuration</b> of the original '
       f'Transformer, sized for CPU training. The paper\'s base model uses {base.num_encoder_layers} + '
       f'{base.num_decoder_layers} layers, d_model = {base.d_model}, {base.num_heads} heads and d_ff = {base.d_ff}; '
       f'that configuration is also implemented in this project (<code>python train.py --config base</code>).'
       f'</div></div>')

# --------------------------------------------------------------------------
# Performance
# --------------------------------------------------------------------------
section("performance", "Model Performance",
        f"Multi30k test results · test_2016_flickr (1,000 held-out sentence pairs) · best checkpoint "
        f"(epoch {best_epoch})", blue=True)
if metrics is None:
    st.info("Evaluation results not found. Run `python evaluate.py` to generate "
            "`results/metrics/small_metrics.json`.")
else:
    bleu, losses, timing = metrics["bleu"], metrics["losses"], metrics["inference_time"]
    beam_bleu = next(v for k, v in bleu.items() if k.startswith("beam") and k != "beam_detail")
    train_min = history["total_train_time_sec"] / 60 if history else None
    cards = [
        ("award", "t-indigo", "BLEU · Beam Search", f"{beam_bleu:.2f}", "SacreBLEU, beam 4, α 0.6", True),
        ("award", "t-purple", "BLEU · Greedy", f"{bleu['greedy']:.2f}", "SacreBLEU", False),
        ("type", "t-blue", "chrF", f"{bleu['chrf_beam']:.2f}", "character n-gram F-score", False),
        ("activity", "t-green", "Perplexity", f"{losses['test_ppl']:.2f}", "test set, per sub-word token", False),
        ("down", "t-indigo", "Training Loss", f"{losses['train_loss_at_best_epoch']:.3f}", f"epoch {best_epoch}", False),
        ("down", "t-purple", "Validation Loss", f"{losses['val_loss']:.3f}", "1,014 sentences", False),
        ("down", "t-blue", "Test Loss", f"{losses['test_loss']:.3f}", "1,000 sentences", False),
        ("cpu", "t-amber", "Parameters", f"{metrics['num_trainable_parameters'] / 1e6:.2f}M", "trainable", False),
        ("clock", "t-green", "Training Time", f"{train_min:.0f} min" if train_min else "–", "15 epochs on CPU", False),
        ("zap", "t-red", "Inference Latency", f"≈{timing['single_sentence_greedy_latency_ms_mean']:.0f} ms",
         f"per sentence, greedy (beam ≈{timing['beam_ms_per_sentence']:.0f} ms)", False),
    ]
    render('<div class="grid cols-5">' + "".join(
        f'<div class="card metric{" hl" if hl else ""}"><div class="top"><div class="ibubble sm {tone}">'
        f'{icon(ic, 20)}</div><div class="klabel">{esc(label)}</div></div><div class="num">{esc(value)}</div>'
        f'<div class="sm">{esc(sub)}</div></div>' for ic, tone, label, value, sub, hl in cards) + "</div>")
    render(f'<div class="note">{icon("info", 18)}<div>Losses are label-smoothed cross-entropy (ε = 0.1); '
           'perplexity = exp(un-smoothed test cross-entropy). These are <b>Multi30k</b> results and are not '
           "comparable to the paper's WMT 2014 scores (27.3 / 28.4 BLEU), which were not reproduced here.</div></div>")

    g1, g2 = st.columns(2, gap="medium")
    for col, key, png, title in [(g1, "img_loss", "small_loss_curve.png", "Training vs validation loss"),
                                 (g2, "img_attn", "small_cross_attention.png", "Learned encoder–decoder attention")]:
        path = C.GRAPHS_DIR / png
        if path.exists():
            with col, st.container(key=key):
                render(f'<div class="imgcap">{icon("chart", 18)}{esc(title)}</div>')
                st.image(str(path), width="stretch")

# --------------------------------------------------------------------------
# Research paper
# --------------------------------------------------------------------------
section("paper", "Research Paper", "The work this project implements")
p1, p2 = st.columns([1.35, 1], gap="medium")
render(f"""
<div class="card prose">
  <h3>{icon("book", 20)}Attention Is All You Need</h3>
  <div class="cite">A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser,
  I. Polosukhin. <em>Advances in Neural Information Processing Systems 30</em> (NeurIPS 2017). arXiv:1706.03762.</div>
  The paper introduced the <b>Transformer</b>, a sequence-to-sequence model that drops recurrence and convolution
  entirely and relies only on attention. Every word can attend to every other word in a single step, so whole
  sentences are processed in parallel.
  <ul>
    <li><b>Scaled dot-product attention:</b> softmax(QKᵀ / √d<sub>k</sub>) V</li>
    <li><b>Multi-head attention:</b> h parallel attention heads over different learned subspaces</li>
    <li><b>Sinusoidal positional encoding</b> to inject word order</li>
    <li><b>Encoder–decoder stacks</b> with residual connections and layer normalisation</li>
    <li><b>Training recipe:</b> Adam with warm-up learning-rate schedule, dropout, label smoothing</li>
  </ul>
  <a class="btnlink" href="{PAPER_URL}" target="_blank" rel="noopener">{icon("external", 16)}Read on arXiv</a>
</div>""", p1)
render(f"""
<div class="card prose">
  <h3>{icon("award", 20)}Results reported in the paper</h3>
  BLEU on the WMT 2014 newstest2014 benchmarks (Table 2 of the paper):
  <table class="pt"><tr><th>Task</th><th>Transformer (base)</th><th>Transformer (big)</th></tr>
  <tr><td>English → German</td><td class="n">27.3</td><td class="n">28.4</td></tr>
  <tr><td>English → French</td><td class="n">38.1</td><td class="n">41.8</td></tr></table>
  <div class="note" style="margin-top:1rem">{icon("info", 18)}<div>These are the authors' numbers on WMT 2014
  (~4.5M sentence pairs, 8 GPUs). This project trains on Multi30k (29k pairs, laptop CPU), so its scores
  measure a different task and are not a reproduction of the paper.</div></div>
</div>""", p2)

# --------------------------------------------------------------------------
# About
# --------------------------------------------------------------------------
section("about", "About Project", "What was built, with which tools, and how to run it", blue=True)
render(f"""
<div class="grid cols-3">
  <div class="card prose"><h3>{icon("cap", 20)}Academic context</h3>
    Mini-project for <b>Experiments 9 &amp; 10</b> of the Computational Linguistics and Natural Language
    Processing Laboratory: implement a research paper end-to-end, train it on real data and evaluate it honestly.
    <ul><li>All 15 Transformer components written by hand (no <code>nn.Transformer</code>)</li>
    <li>28 unit tests + overfit sanity check</li><li>Academic report and viva questions included</li></ul></div>
  <div class="card prose"><h3>{icon("layers", 20)}Tech stack</h3>
    <ul><li><b>PyTorch</b> – model, training, inference</li>
    <li><b>SentencePiece</b> – joint 8k BPE vocabulary</li>
    <li><b>SacreBLEU</b> – BLEU and chrF evaluation</li>
    <li><b>Streamlit</b> – this interface</li>
    <li><b>Multi30k</b> – 29,000 / 1,014 / 1,000 sentence pairs</li></ul></div>
  <div class="card prose"><h3>{icon("code", 20)}Run it yourself</h3>
    <ul><li><code>pip install -r requirements.txt</code></li><li><code>python prepare_data.py</code></li>
    <li><code>python train.py --config small</code></li><li><code>python evaluate.py</code></li>
    <li><code>streamlit run app.py</code></li></ul>
    <a class="btnlink" href="{REPO_URL}" target="_blank" rel="noopener">{icon("github", 16)}Source code on GitHub</a></div>
</div>""")

render(f"""<div class="footer">Vaswani et al., <em>Attention Is All You Need</em>, NeurIPS 2017 · Multi30k
(Elliott et al., 2016) · All translations are produced locally by the trained checkpoint
<code>{esc(CHECKPOINT.name)}</code>.</div>""")
