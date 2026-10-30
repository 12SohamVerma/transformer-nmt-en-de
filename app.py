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
    "plus_circle": '<circle cx="12" cy="12" r="10"/><path d="M12 8v8M8 12h8"/>',
    "network": '<circle cx="5" cy="6" r="2"/><circle cx="19" cy="6" r="2"/><circle cx="12" cy="12" r="2"/>'
               '<circle cx="6" cy="19" r="2"/><circle cx="18" cy="18" r="2"/>'
               '<path d="M6.7 7.2 10.5 10.8M17.3 7.2l-3.8 3.6M10.6 13.4l-3.2 4M13.4 13.4l3.2 3.2"/>',
    "search": '<circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/>',
    "mask": '<path d="M3 7c0-1 1-2 3-2 3 0 4 1 6 1s3-1 6-1c2 0 3 1 3 2 0 6-4 11-9 11S3 13 3 7Z"/>'
            '<path d="M7.5 10.5c.8-.6 1.7-.6 2.5 0M14 10.5c.8-.6 1.7-.6 2.5 0M9.5 14.5c1.5 1 3.5 1 5 0"/>',
    "link": '<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/>'
            '<path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>',
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
.st-key-img_loss, .st-key-img_attn {background: #fff; border: 1px solid var(--line); border-radius: 16px;
  padding: 1rem 1.1rem; box-shadow: var(--shadow);}
.imgcap {font-weight: 700; color: var(--ink); font-size: .95rem; display: flex; align-items: center; gap: .5rem;}
.imgcap svg {color: var(--indigo);}
.graph {width: 100%; height: auto; display: block; margin-top: .6rem; border-radius: 8px;}


.prose {padding: 1.3rem 1.5rem; color: var(--text); font-size: .93rem; line-height: 1.65;}
.prose .ph3 {font-size: 1.08rem; color: var(--ink); margin: 0 0 .7rem; font-weight: 750; display: flex;
  gap: .6rem; align-items: center;}
.prose .ph3 svg {color: var(--indigo); flex: none;}
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
.arch-chips {margin: -.3rem 0 1rem;}
.arch-wrap {padding: 1rem; overflow-x: auto;}
.arch-wrap svg {display: block; width: 100%; min-width: 980px; height: auto;}
.arch-wrap text {font-family: 'Plus Jakarta Sans', 'Segoe UI', sans-serif;}
.fine {color: var(--muted); font-size: .8rem; margin-top: .6rem;}
.meta {color: var(--muted); font-size: .85rem; margin: -.2rem 0 .8rem;}
.tags {display: flex; flex-wrap: wrap; gap: .45rem;}
.tags span {background: var(--soft); color: #3730a3; border-radius: 999px; padding: .3rem .75rem; font-size: .82rem;
  font-weight: 600;}
@media (max-width: 640px) {
  .st-key-src_head [data-testid="stHorizontalBlock"], .st-key-tgt_head [data-testid="stHorizontalBlock"] {
    flex-wrap: wrap; gap: .5rem;}
  .st-key-src_head [data-testid="stColumn"], .st-key-tgt_head [data-testid="stColumn"] {
    flex: 1 1 calc(50% - .5rem) !important; width: calc(50% - .5rem) !important; min-width: 0 !important;}
  .st-key-src_head [data-testid="stColumn"]:first-child, .st-key-tgt_head [data-testid="stColumn"]:first-child {
    flex-basis: 100% !important; width: 100% !important;}
}
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


@st.cache_data(show_spinner=False)
def png_data_uri(path: str) -> str:
    """Embed result graphs inline: Streamlit's media files can 404 if a rerun interrupts the first page load."""
    import base64
    from pathlib import Path
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


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


# --------------------------------------------------------------------------
# Voice and clipboard buttons (Streamlit v2 components, run in the page itself).
# Speech recognition and speech synthesis are provided by the user's browser;
# translation itself always runs on the local Transformer.
# --------------------------------------------------------------------------
_BUTTON_CSS = """
:host {display: block;}
button {all: unset; box-sizing: border-box; cursor: pointer; display: inline-flex; align-items: center;
  justify-content: center; gap: 7px; width: 100%; padding: 8px 10px; border-radius: 10px; white-space: nowrap;
  font: 600 13.5px 'Plus Jakarta Sans', 'Segoe UI', system-ui, sans-serif; color: #4f46e5; background: #fff;
  border: 1px solid #c7d2fe; transition: background .15s;}
button:hover {background: #eef2ff;}
button:disabled {cursor: not-allowed; color: #9ca3af; border-color: #e5e7eb; background: #f9fafb;}
button.on {color: #dc2626; border-color: #fecaca; background: #fef2f2;}
.dot {display: none; width: 8px; height: 8px; border-radius: 50%; background: #dc2626;}
button.on .dot {display: inline-block;}
"""
_MIC_ICON = ('<path d="M12 2a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10a7 7 0 0 1-14 0"/>'
             '<path d="M12 17v5"/><path d="M8 22h8"/>')
_SPEAKER_ICON = ('<path d="M11 5 6 9H2v6h4l5 4V5Z"/><path d="M15.5 8.5a5 5 0 0 1 0 7"/>'
                 '<path d="M19 5a10 10 0 0 1 0 14"/>')


def _button_html(icon_paths: str, label: str, title: str) -> str:
    svg = (f'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
           f'stroke-linecap="round" stroke-linejoin="round">{icon_paths}</svg>')
    return f'<button type="button" title="{title}">{svg}<span class="dot"></span><span class="lbl">{label}</span></button>'


VOICE_INPUT = st.components.v2.component(
    "voice_input",
    html=_button_html(_MIC_ICON, "Speak", "Speak an English sentence (browser speech recognition)"),
    css=_BUTTON_CSS,
    js="""
export default function ({ parentElement, setTriggerValue }) {
  const btn = parentElement.querySelector("button");
  const label = btn.querySelector(".lbl");
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Recognition) {
    btn.disabled = true;
    btn.title = "Voice input needs Google Chrome or Microsoft Edge";
    return;
  }
  let rec = null;
  btn.onclick = () => {
    if (rec) { rec.stop(); return; }
    let problem = "";
    rec = new Recognition();
    rec.lang = "en-US";
    rec.interimResults = false;
    rec.maxAlternatives = 1;
    rec.onresult = (e) => {
      const text = Array.from(e.results).map((r) => r[0].transcript).join(" ").trim();
      if (text) setTriggerValue("transcript", text);
    };
    rec.onerror = (e) => {
      problem = e.error === "not-allowed" ? "Mic blocked" : e.error === "no-speech" ? "No speech" : "Try again";
    };
    rec.onend = () => {
      rec = null;
      btn.classList.remove("on");
      label.textContent = problem || "Speak";
      if (problem) setTimeout(() => { if (!rec) label.textContent = "Speak"; }, 2000);
    };
    btn.classList.add("on");
    label.textContent = "Listening…";
    try { rec.start(); } catch (err) { rec = null; btn.classList.remove("on"); label.textContent = "Speak"; }
  };
  return () => { if (rec) rec.abort(); };
}
""",
)

VOICE_OUTPUT = st.components.v2.component(
    "voice_output",
    html=_button_html(_SPEAKER_ICON, "Listen", "Read aloud"),
    css=_BUTTON_CSS,
    js="""
export default function ({ parentElement, data }) {
  const btn = parentElement.querySelector("button");
  const label = btn.querySelector(".lbl");
  const synth = window.speechSynthesis;
  const text = (data && data.text) || "";
  const lang = (data && data.lang) || "en-US";
  btn.disabled = !synth || !text;
  if (!synth) { btn.title = "Speech output is not supported in this browser"; return; }
  synth.getVoices();                                   // voices load asynchronously in some browsers
  const pickVoice = () => {
    const voices = synth.getVoices();
    return voices.find((v) => v.lang === lang && v.localService) || voices.find((v) => v.lang === lang)
      || voices.find((v) => v.lang.startsWith(lang.slice(0, 2))) || null;
  };
  btn.onclick = () => {
    if (btn.classList.contains("on")) { synth.cancel(); return; }
    synth.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = lang;
    utterance.rate = 0.95;
    const voice = pickVoice();
    if (voice) utterance.voice = voice;
    utterance.onstart = () => { btn.classList.add("on"); label.textContent = "Stop"; };
    utterance.onend = utterance.onerror = () => { btn.classList.remove("on"); label.textContent = "Listen"; };
    synth.speak(utterance);
  };
}
""",
)

COPY_TEXT = st.components.v2.component(
    "copy_text",
    html=_button_html(_ICON_PATHS["copy"], "Copy", "Copy the translation"),
    css=_BUTTON_CSS,
    js="""
export default function ({ parentElement, data }) {
  const btn = parentElement.querySelector("button");
  const label = btn.querySelector(".lbl");
  const text = (data && data.text) || "";
  btn.disabled = !text;
  btn.onclick = async () => {
    let ok = false;
    try { await navigator.clipboard.writeText(text); ok = true; }
    catch (e) {
      const area = document.createElement("textarea");
      area.value = text;
      document.body.appendChild(area);
      area.select();
      try { ok = document.execCommand("copy"); } catch (e2) { ok = false; }
      area.remove();
    }
    label.textContent = ok ? "Copied" : "Copy failed";
    setTimeout(() => { label.textContent = "Copy"; }, 1600);
  };
}
""",
)


def on_voice_transcript() -> None:
    """Spoken English fills the input box and is translated on this rerun."""
    text = ((st.session_state.get("voice_mic") or {}).get("transcript") or "").strip()
    if text:
        st.session_state.src_text = text[:500]
        st.session_state.voice_translate = True


def section(anchor: str, title: str, desc: str = "", right: str = "", blue: bool = False) -> None:
    """Section heading with a coloured bar; with a right-hand element the description moves below."""
    classes = "sec" + (" blue" if blue else "") + (" stacked" if right else "")
    render(f'<div id="{anchor}" class="{classes}"><h2>{esc(title)}</h2>'
           f'<span class="desc">{esc(desc)}</span>{f"<span class=right>{right}</span>" if right else ""}</div>')


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
render(CSS)

# --------------------------------------------------------------------------
# Page loader mascot: shown on a session's first run (while the model loads),
# faded out by the style rendered at the very end of the script.
# --------------------------------------------------------------------------
LOADER = """
<style>
.page-loader {position: fixed; inset: 0; z-index: 1000000; display: flex; flex-direction: column; align-items: center;
  justify-content: center; gap: .4rem; text-align: center; padding: 1rem;
  background: radial-gradient(900px 420px at 50% 28%, #ede9fe 0%, transparent 62%),
              linear-gradient(120deg, #f8faff 0%, #eef2ff 55%, #f3e8ff 100%);
  transition: opacity .5s ease .9s, visibility 0s linear 1.4s;
  animation: loader-failsafe .5s ease 25s forwards;}
.page-loader svg {width: 210px; height: auto; overflow: visible;}
.page-loader .ttl {font-weight: 800; font-size: 1.35rem; color: #111633; margin-top: .4rem; letter-spacing: -.01em;}
.page-loader .ttl b {background: linear-gradient(90deg, #2563eb, #7c3aed); -webkit-background-clip: text;
  background-clip: text; color: transparent;}
.page-loader .ttl i {display: inline-block; width: 1.4em; text-align: left; font-style: normal; color: #7c3aed;}
.page-loader .ttl i::after {content: ""; animation: loader-dots 1.2s steps(4) infinite;}
.page-loader .sub {color: #6b7280; font-size: .92rem;}
.page-loader .bot {animation: loader-bob 1.6s ease-in-out infinite;}
.page-loader .shadow {transform-box: fill-box; transform-origin: center; animation: loader-shadow 1.6s ease-in-out infinite;}
.page-loader .eye {transform-box: fill-box; transform-origin: center; animation: loader-blink 3.4s infinite;}
.page-loader .bulb {animation: loader-glow 1.1s ease-in-out infinite alternate;}
.page-loader .hd {animation: loader-wave 1.6s ease-in-out infinite;}
.page-loader .hello {animation: loader-say 2.6s infinite;}
.page-loader .hallo {animation: loader-say 2.6s infinite 1.3s; opacity: 0;}
@keyframes loader-bob {50% {transform: translateY(-7px);}}
@keyframes loader-shadow {50% {transform: scaleX(.82); opacity: .35;}}
@keyframes loader-blink {0%, 90%, 100% {transform: scaleY(1);} 94% {transform: scaleY(.1);}}
@keyframes loader-glow {from {opacity: .55;} to {opacity: 1;}}
@keyframes loader-wave {0%, 100% {opacity: .25;} 30% {opacity: 1;}}
@keyframes loader-say {0%, 42% {opacity: 1;} 50%, 92% {opacity: 0;} 100% {opacity: 1;}}
@keyframes loader-dots {0% {content: "";} 25% {content: ".";} 50% {content: "..";} 75% {content: "...";}}
@keyframes loader-failsafe {to {opacity: 0; visibility: hidden;}}
@media (prefers-reduced-motion: reduce) {.page-loader * {animation: none !important;}
  .page-loader .hallo {opacity: 0;}}
</style>
<div class="page-loader" role="status" aria-live="polite">
<svg viewBox="0 0 200 180" aria-hidden="true">
<defs><linearGradient id="botGrad" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#3b82f6"/>
<stop offset="1" stop-color="#7c3aed"/></linearGradient></defs>
<ellipse class="shadow" cx="90" cy="172" rx="36" ry="5" fill="#c7d2fe"/>
<g class="bot">
<line x1="90" y1="34" x2="90" y2="17" stroke="#6366f1" stroke-width="4" stroke-linecap="round"/>
<circle class="bulb" cx="90" cy="12" r="7" fill="url(#botGrad)"/>
<rect x="29" y="62" width="13" height="32" rx="6.5" fill="#6366f1"/>
<rect x="138" y="62" width="13" height="32" rx="6.5" fill="#6366f1"/>
<rect x="38" y="34" width="104" height="84" rx="30" fill="url(#botGrad)"/>
<rect x="50" y="49" width="80" height="54" rx="20" fill="#eef2ff"/>
<ellipse class="eye" cx="74" cy="73" rx="6" ry="8" fill="#1e1b4b"/>
<ellipse class="eye" cx="106" cy="73" rx="6" ry="8" fill="#1e1b4b"/>
<circle cx="76" cy="70" r="2" fill="#fff"/><circle cx="108" cy="70" r="2" fill="#fff"/>
<circle cx="62" cy="87" r="4.5" fill="#fbcfe8"/><circle cx="118" cy="87" r="4.5" fill="#fbcfe8"/>
<path d="M80 88 Q90 97 100 88" fill="none" stroke="#312e81" stroke-width="3" stroke-linecap="round"/>
<rect x="60" y="121" width="60" height="38" rx="14" fill="#ffffff" stroke="#c7d2fe" stroke-width="2"/>
<circle class="hd" cx="75" cy="133" r="3.2" fill="#93c5fd" style="animation-delay: 0s"/>
<circle class="hd" cx="85" cy="133" r="3.2" fill="#a5b4fc" style="animation-delay: .1s"/>
<circle class="hd" cx="95" cy="133" r="3.2" fill="#a5b4fc" style="animation-delay: .2s"/>
<circle class="hd" cx="105" cy="133" r="3.2" fill="#c4b5fd" style="animation-delay: .3s"/>
<circle class="hd" cx="75" cy="147" r="3.2" fill="#c4b5fd" style="animation-delay: .4s"/>
<circle class="hd" cx="85" cy="147" r="3.2" fill="#d8b4fe" style="animation-delay: .5s"/>
<circle class="hd" cx="95" cy="147" r="3.2" fill="#f0abfc" style="animation-delay: .6s"/>
<circle class="hd" cx="105" cy="147" r="3.2" fill="#f9a8d4" style="animation-delay: .7s"/>
</g>
<g>
<rect x="128" y="2" width="68" height="30" rx="12" fill="#ffffff" stroke="#c7d2fe" stroke-width="1.5"/>
<path d="M140 31 L134 43 L152 31" fill="#ffffff" stroke="#c7d2fe" stroke-width="1.5" stroke-linejoin="round"/>
<rect x="139" y="29" width="14" height="4" fill="#ffffff"/>
<text class="hello" x="162" y="22" text-anchor="middle" font-size="14" font-weight="700" fill="#2563eb">Hello!</text>
<text class="hallo" x="162" y="22" text-anchor="middle" font-size="14" font-weight="700" fill="#7c3aed">Hallo!</text>
</g>
</svg>
<div class="ttl">Warming up the <b>Transformer</b><i></i></div>
<div class="sub">Loading the trained checkpoint · English → German</div>
</div>
"""
first_run = "booted" not in st.session_state
# Later runs keep an empty element in the loader's slot: shifting every element after it would make
# Streamlit re-mount the voice components and duplicate their buttons.
render(LOADER if first_run else '<span class="page-loader-slot" hidden></span>')

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
        "Type an English sentence or pick an example.", right=pill, blue=True)

left, right = st.columns(2, gap="medium")
with left:
    with st.container(key="src_card"):
        with st.container(key="src_head"):
            head, listen_col, mic_col = st.columns([4, 1.25, 1.35], vertical_alignment="center")
            render(f'<div class="lang"><span class="flag">{FLAG_UK}</span><span><b>English</b> (Source Language)'
                   f'</span></div>', head)
            with listen_col:
                VOICE_OUTPUT(key="speak_src", data={"text": st.session_state.src_text, "lang": "en-US"})
            with mic_col:
                VOICE_INPUT(key="voice_mic", on_transcript_change=on_voice_transcript)
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

        if clicked or st.session_state.pop("voice_translate", False):
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
        with st.container(key="tgt_head"):
            head, listen_col, copy_col = st.columns([4, 1.25, 1.2], vertical_alignment="center")
            render(f'<div class="lang"><span class="flag">{FLAG_DE}</span><span><b>German</b> (Target Language)'
                   f'</span></div>', head)
            translation = res["translation"] if res else ""
            with listen_col:
                VOICE_OUTPUT(key="speak_tgt", data={"text": translation, "lang": "de-DE"})
            with copy_col:
                COPY_TEXT(key="copy_tgt", data={"text": translation})
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
            </div>""")
        else:
            render('<div class="tgt-box"><div class="ph">Your German translation will appear here.</div></div>')

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
@st.cache_data(show_spinner=False)
def example_translation(sentence: str) -> str:
    return load_translator().translate(sentence, beam_size=BEAM_SIZE, alpha=ALPHA).translation


def _svg_icon(name: str, x: float, y: float, size: float, color: str, sw: float = 1.9) -> str:
    return (f'<svg x="{x}" y="{y}" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" '
            f'stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round">{_ICON_PATHS[name]}</svg>')


def _text(x, y, s, size=14, weight=500, fill="#111633", anchor="start") -> str:
    return (f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}">{esc(s)}</text>')


def _arrow_up(x, y_from, y_to, color="#64748b") -> str:
    return (f'<line x1="{x}" y1="{y_from}" x2="{x}" y2="{y_to + 7}" stroke="{color}" stroke-width="1.6"/>'
            f'<polygon points="{x - 4.5},{y_to + 8} {x + 4.5},{y_to + 8} {x},{y_to}" fill="{color}"/>')


def _head_right(x, y, color="#64748b") -> str:
    return f'<polygon points="{x - 8},{y - 4.5} {x - 8},{y + 4.5} {x},{y}" fill="{color}"/>'


_BOX_STYLES = {"io": ("#f5f7ff", "#dfe3f5", "#2563eb"), "an": ("#f1efff", "#d9d3fb", "#4f46e5"),
               "ffn": ("#eef4ff", "#cfdcfb", "#2563eb"), "attn": ("#fff4ec", "#fbd5b8", "#ea580c"),
               "cross": ("#fff0f1", "#fbc8cf", "#e11d48"), "purple": ("#f5f3ff", "#ddd6fe", "#7c3aed")}


def _box(x, y, w, h, kind, ic, title, sub=(), title_color="#111633", sub_color="#64748b") -> str:
    fill, stroke, icol = _BOX_STYLES[kind]
    s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}"/>'
    if kind == "an":                                   # centred "(+) Add & Norm"
        cx = x + w / 2
        return s + _svg_icon("plus_circle", cx - 52, y + h / 2 - 9, 18, icol) + \
            _text(cx - 28, y + h / 2 + 5, title, 14, 650)
    s += _svg_icon(ic, x + 20, y + h / 2 - 13, 26, icol)
    lines = list(sub)
    ty = y + h / 2 + 5 - 9 * len(lines)
    s += _text(x + 62, ty, title, 14.5, 700, title_color)
    for i, line in enumerate(lines):
        s += _text(x + 62, ty + 18 + 16 * i, line, 12, 450, sub_color)
    return s


def _quoted(text: str, width: int = 38) -> list[str]:
    import textwrap
    lines = textwrap.wrap(text, width) or [""]
    lines[0], lines[-1] = "“" + lines[0], lines[-1] + "”"
    return lines


def architecture_svg(cfg, n_params: int, vocab: int, src: str, tgt: str) -> str:
    """Encoder-decoder diagram drawn from the loaded model's real configuration."""
    E, D = 265, 825                                      # centre lines of the encoder / decoder columns
    p = ['<svg viewBox="0 0 1500 960" xmlns="http://www.w3.org/2000/svg" role="img" '
         'aria-label="Transformer encoder-decoder architecture">',
         '<defs><linearGradient id="archHead" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#2563eb"/>'
         '<stop offset="1" stop-color="#7c3aed"/></linearGradient></defs>']

    # ----- encoder column -----
    p.append('<rect x="30" y="362" width="420" height="328" rx="16" fill="#fbfbff" stroke="#e6e8f2"/>')
    p.append('<rect x="30" y="362" width="420" height="44" rx="12" fill="url(#archHead)"/>')
    p.append(_text(240, 391, f"Encoder × {cfg.num_encoder_layers}", 19, 750, "#ffffff", "middle"))
    p.append(_box(100, 424, 330, 40, "an", "", "Add & Norm"))
    p.append(_box(100, 476, 330, 52, "ffn", "network", "Feed Forward Network", ["Linear → ReLU → Linear"]))
    p.append(_box(100, 548, 330, 40, "an", "", "Add & Norm"))
    p.append(_box(100, 600, 330, 52, "attn", "search", "Multi-Head Self-Attention"))
    p.append(_box(80, 715, 370, 60, "io", "wave", "Embedding + Positional Encoding",
                  ["Token embeddings + sinusoidal positions"]))
    p.append(_box(80, 795, 370, 55, "purple", "file", "SentencePiece BPE", ["Subword tokenization"]))
    p.append(_box(80, 870, 370, 80, "io", "file", "English input sentence", _quoted(src), "#1e3a8a"))
    for a, b in [(870, 850), (795, 775), (715, 652), (600, 588), (548, 528), (476, 464), (424, 406), (362, 336)]:
        p.append(_arrow_up(E, a, b))
    for y0, y1 in [(666, 568), (538, 444)]:            # residual connections
        p.append(f'<path d="M{E} {y0} H70 V{y1} H92" fill="none" stroke="#64748b" stroke-width="1.6"/>')
        p.append(_head_right(100, y1))

    # ----- decoder column -----
    p.append('<rect x="590" y="240" width="420" height="450" rx="16" fill="#fbfbff" stroke="#e6e8f2"/>')
    p.append('<rect x="590" y="240" width="420" height="44" rx="12" fill="url(#archHead)"/>')
    p.append(_text(800, 269, f"Decoder × {cfg.num_decoder_layers}", 19, 750, "#ffffff", "middle"))
    p.append(_box(660, 300, 330, 40, "an", "", "Add & Norm"))
    p.append(_box(660, 352, 330, 52, "ffn", "network", "Feed Forward Network", ["Linear → ReLU → Linear"]))
    p.append(_box(660, 424, 330, 40, "an", "", "Add & Norm"))
    p.append(_box(660, 476, 330, 52, "cross", "link", "Cross-Attention", ["(Encoder-Decoder Attention)"],
                  "#9f1239", "#be123c"))
    p.append(_box(660, 548, 330, 40, "an", "", "Add & Norm"))
    p.append(_box(660, 600, 330, 52, "attn", "mask", "Masked Multi-Head Attention"))
    p.append(_box(640, 10, 370, 80, "io", "file", "German output sentence", _quoted(tgt), "#1e3a8a"))
    p.append(_box(640, 105, 370, 55, "purple", "chart", "Softmax", [f"Token probabilities ({vocab:,} tokens)"]))
    p.append(_box(640, 172, 370, 55, "io", "network", "Linear", ["Project to vocabulary dimension"]))
    p.append(_box(640, 715, 370, 60, "io", "wave", "Embedding + Positional Encoding",
                  ["Token embeddings + sinusoidal positions"]))
    shifted = "<BOS> " + " ".join(tgt.split()[:4]) + " …"
    p.append(_box(640, 795, 370, 80, "io", "file", "German input (shifted right)", [shifted], "#1e3a8a"))
    for a, b in [(795, 775), (715, 652), (600, 588), (548, 528), (476, 464), (424, 404), (352, 340), (300, 284),
                 (240, 227), (172, 160), (105, 90)]:
        p.append(_arrow_up(D, a, b))
    for y0, y1 in [(666, 568), (538, 444), (414, 320)]:
        p.append(f'<path d="M{D} {y0} H630 V{y1} H652" fill="none" stroke="#64748b" stroke-width="1.6"/>')
        p.append(_head_right(660, y1))

    # ----- encoder output -> cross-attention (drawn over the decoder panel) -----
    p.append('<path d="M430 444 C 520 444, 560 502, 652 502" fill="none" stroke="#7c3aed" stroke-width="2.4"/>')
    p.append(_head_right(660, 502, "#7c3aed"))
    for i, line in enumerate(["Encoder", "Outputs", "(K, V)"]):
        p.append(_text(518, 400 + 16 * i, line, 13, 650, "#334155", "middle"))

    # ----- multi-head attention detail -----
    p.append('<path d="M990 502 C 1018 502, 1022 330, 1050 330" fill="none" stroke="#a5b4fc" stroke-width="1.6" '
             'stroke-dasharray="5 5"/>')
    p.append('<rect x="1050" y="10" width="440" height="460" rx="16" fill="#ffffff" stroke="#e6e8f2"/>')
    p.append('<rect x="1072" y="34" width="5" height="24" rx="2" fill="#2563eb"/>')
    p.append(_text(1088, 53, f"Multi-Head Attention (h = {cfg.num_heads})", 18, 750))
    p.append(_text(1266, 86, f"Split into {cfg.num_heads} heads", 12.5, 500, "#374151", "middle"))
    p.append(_text(1072, 272, "Input", 12.5, 500, "#374151", "middle"))
    p.append(_text(1072, 288, "(X)", 12.5, 500, "#374151", "middle"))
    for cy, letter, color in [(160, "Q", "#2563eb"), (276, "K", "#7c3aed"), (392, "V", "#db2777")]:
        p.append(f'<path d="M1094 276 H1100 V{cy} H1104" fill="none" stroke="#64748b" stroke-width="1.5"/>')
        p.append(_head_right(1112, cy))
        p.append(f'<rect x="1112" y="{cy - 20}" width="72" height="40" rx="9" fill="#eef4ff" stroke="#cfdcfb"/>')
        p.append(_text(1148, cy + 5, "Linear", 13, 600, "#1e293b", "middle"))
        p.append(_text(1197, cy + 6, letter, 17, 800, color, "middle"))
        p.append(f'<line x1="1206" y1="{cy}" x2="1212" y2="{cy}" stroke="#64748b" stroke-width="1.5"/>')
    p.append('<rect x="1218" y="95" width="96" height="362" rx="10" fill="none" stroke="#a5b4fc" '
             'stroke-dasharray="5 4"/>')
    heads = cfg.num_heads
    step = 352 / heads
    lo, hi = (0xbf, 0xdb, 0xfe), (0xfb, 0xcf, 0xe8)            # blue-200 -> pink-200
    for i in range(heads):
        t = i / max(heads - 1, 1)
        col = "#" + "".join(f"{round(a + (b - a) * t):02x}" for a, b in zip(lo, hi))
        y = 100 + i * step
        p.append(f'<rect x="1228" y="{y:.1f}" width="76" height="{step - 10:.1f}" rx="7" fill="{col}"/>')
        p.append(_text(1266, round(y + (step - 10) / 2 + 4.5, 1), f"Head {i + 1}", 12.5, 600, "#312e81", "middle"))
    p.append('<path d="M1304 121 H1326 V431 H1304" fill="none" stroke="#64748b" stroke-width="1.5"/>')
    p.append('<line x1="1326" y1="276" x2="1336" y2="276" stroke="#64748b" stroke-width="1.5"/>')
    p.append(_head_right(1344, 276))
    p.append('<rect x="1344" y="236" width="70" height="80" rx="10" fill="#eef2ff" stroke="#c7d2fe"/>')
    for i, line in enumerate(["Concat", "+", "Linear"]):
        p.append(_text(1379, 262 + 16 * i, line, 12.5, 600, "#1e293b", "middle"))
    p.append('<line x1="1414" y1="276" x2="1472" y2="276" stroke="#64748b" stroke-width="1.5"/>')
    p.append(_head_right(1480, 276))
    for i, line in enumerate(["Multi-head", "attention", "output"]):
        p.append(_text(1450, 236 + 13 * i, line, 10.5, 500, "#475569", "middle"))

    # ----- configuration card -----
    p.append('<rect x="1050" y="490" width="440" height="460" rx="16" fill="#ffffff" stroke="#e6e8f2"/>')
    p.append(_svg_icon("gear", 1072, 512, 24, "#4f46e5"))
    p.append(_text(1106, 531, f"Model Configuration ({cfg.name.capitalize()})", 17, 750))
    rows = [("Encoder layers", cfg.num_encoder_layers), ("Decoder layers", cfg.num_decoder_layers),
            ("Embedding dimension (d_model)", cfg.d_model), ("Attention heads (h)", cfg.num_heads),
            ("Feed forward dimension (d_ff)", cfg.d_ff), ("Dropout", cfg.dropout),
            ("Vocabulary (BPE)", f"{vocab:,}"), ("Trainable parameters", f"{n_params / 1e6:.2f}M")]
    for i, (k, v) in enumerate(rows):
        y = 576 + 46 * i
        p.append(_text(1072, y, k, 14, 450, "#334155"))
        p.append(_text(1468, y, str(v), 14, 700, "#111633", "end"))
        if i < len(rows) - 1:
            p.append(f'<line x1="1072" y1="{y + 18}" x2="1468" y2="{y + 18}" stroke="#eef0f6"/>')
    p.append("</svg>")
    return "".join(p)


section("architecture", "Model Architecture",
        f"Attention Is All You Need (2017) · {cfg.name.capitalize()} configuration")
render(f"""<div class="chips arch-chips">
<span class="chip">{icon("languages", 18)}English → German</span>
<span class="chip">{icon("box", 18)}Transformer Encoder-Decoder</span>
<span class="chip">{icon("database", 18)}Trained on Multi30k</span>
<span class="chip red">{icon("torch", 18)}Built from scratch in PyTorch</span></div>""")
ARCH_SRC = "Three children are playing soccer on a green field."
arch_tgt = example_translation(ARCH_SRC) if translator else "(model unavailable)"
n_params = translator.num_parameters if translator else (metrics or {}).get("num_trainable_parameters", 0)
vocab = translator.sp.get_piece_size() if translator else C.VOCAB_SIZE
render(f'<div class="card arch-wrap">{architecture_svg(cfg, n_params, vocab, ARCH_SRC, arch_tgt)}</div>')

# --------------------------------------------------------------------------
# Performance
# --------------------------------------------------------------------------
section("performance", "Model Performance", "Multi30k test set · 1,000 held-out sentences", blue=True)
if metrics is None:
    st.info("Evaluation results not found. Run `python evaluate.py`.")
else:
    bleu, losses, timing = metrics["bleu"], metrics["losses"], metrics["inference_time"]
    beam_bleu = next(v for k, v in bleu.items() if k.startswith("beam") and k != "beam_detail")
    train_min = history["total_train_time_sec"] / 60 if history else None
    cards = [
        ("award", "t-indigo", "BLEU · Beam", f"{beam_bleu:.2f}", "k = 4, α = 0.6", True),
        ("award", "t-purple", "BLEU · Greedy", f"{bleu['greedy']:.2f}", "SacreBLEU", False),
        ("type", "t-blue", "chrF", f"{bleu['chrf_beam']:.2f}", "character F-score", False),
        ("activity", "t-green", "Perplexity", f"{losses['test_ppl']:.2f}", "test set", False),
        ("down", "t-indigo", "Training Loss", f"{losses['train_loss_at_best_epoch']:.3f}", f"epoch {best_epoch}", False),
        ("down", "t-purple", "Validation Loss", f"{losses['val_loss']:.3f}", "1,014 sentences", False),
        ("down", "t-blue", "Test Loss", f"{losses['test_loss']:.3f}", "1,000 sentences", False),
        ("cpu", "t-amber", "Parameters", f"{metrics['num_trainable_parameters'] / 1e6:.2f}M", "trainable", False),
        ("clock", "t-green", "Training Time", f"{train_min:.0f} min" if train_min else "–", "15 epochs · CPU", False),
        ("zap", "t-red", "Latency", f"≈{timing['single_sentence_greedy_latency_ms_mean']:.0f} ms",
         "per sentence · CPU", False),
    ]
    render('<div class="grid cols-5">' + "".join(
        f'<div class="card metric{" hl" if hl else ""}"><div class="top"><div class="ibubble sm {tone}">'
        f'{icon(ic, 20)}</div><div class="klabel">{esc(label)}</div></div><div class="num">{esc(value)}</div>'
        f'<div class="sm">{esc(sub)}</div></div>' for ic, tone, label, value, sub, hl in cards) + "</div>")
    render('<div class="fine">Not comparable to the paper\'s WMT 2014 scores.</div>')

    g1, g2 = st.columns(2, gap="medium")
    for col, key, png, title in [(g1, "img_loss", "small_loss_curve.png", "Training vs validation loss"),
                                 (g2, "img_attn", "small_cross_attention.png", "Learned cross-attention")]:
        path = C.GRAPHS_DIR / png
        if path.exists():
            with col, st.container(key=key):
                render(f'<div class="imgcap">{icon("chart", 18)}{esc(title)}</div>'
                       f'<img class="graph" alt="{esc(title)}" src="{png_data_uri(str(path))}">')

# --------------------------------------------------------------------------
# Research paper
# --------------------------------------------------------------------------
section("paper", "Research Paper", "Vaswani et al. · NeurIPS 2017")
p1, p2 = st.columns([1.2, 1], gap="medium")
render(f"""
<div class="card prose">
  <div class="ph3">{icon("book", 20)}Attention Is All You Need</div>
  <div class="meta">Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin · arXiv:1706.03762</div>
  <div class="tags"><span>Self-attention</span><span>Multi-head attention</span><span>Positional encoding</span>
  <span>Encoder–decoder</span><span>No recurrence</span></div>
  <a class="btnlink" href="{PAPER_URL}" target="_blank" rel="noopener">{icon("external", 16)}Read on arXiv</a>
</div>""", p1)
render(f"""
<div class="card prose">
  <div class="ph3">{icon("award", 20)}Paper results · WMT 2014 BLEU</div>
  <table class="pt"><tr><th>Task</th><th>Base</th><th>Big</th></tr>
  <tr><td>English → German</td><td class="n">27.3</td><td class="n">28.4</td></tr>
  <tr><td>English → French</td><td class="n">38.1</td><td class="n">41.8</td></tr></table>
</div>""", p2)

# --------------------------------------------------------------------------
# About
# --------------------------------------------------------------------------
section("about", "About Project", blue=True)
render(f"""
<div class="grid cols-3">
  <div class="card prose"><div class="ph3">{icon("cap", 20)}Lab project</div>
    <div class="meta">Computational Linguistics &amp; NLP Lab · Experiments 9 &amp; 10</div></div>
  <div class="card prose"><div class="ph3">{icon("layers", 20)}Tech stack</div>
    <div class="tags"><span>PyTorch</span><span>SentencePiece</span><span>SacreBLEU</span><span>Streamlit</span></div></div>
  <div class="card prose"><div class="ph3">{icon("code", 20)}Source code</div>
    <a class="btnlink" href="{REPO_URL}" target="_blank" rel="noopener">{icon("github", 16)}View on GitHub</a></div>
</div>""")

render('<div class="footer">Vaswani et al., 2017 · Multi30k · runs locally on the trained checkpoint</div>')

if first_run:                       # page fully rendered: fade the loader mascot out
    st.session_state.booted = True
    render("<style>.page-loader {opacity: 0; visibility: hidden; pointer-events: none;}</style>")
