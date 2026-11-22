"""Look and feel: obsidian + violet theme, bento cards, small HTML helpers."""
from html import escape

import streamlit as st

VIOLET, EMERALD, AMBER = "#A854F6", "#10b881", "#f59f0b"
MUTED = "#8b8b94"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, [class*="css"], h1, h2, h3, h4, button, input {{ font-family: 'Inter', sans-serif !important; }}
:root {{ --fs-xs: 12px; --fs-sm: 13px; --fs-md: 14px; --fs-body: 15px; --fs-h3: 16px; --fs-h2: 24px; --fs-h1: 30px; }}
.stApp p, .stApp li {{ font-size: var(--fs-body); font-weight: 400; }}
h1, h2, h3, h4 {{ font-weight: 500 !important; font-style: normal; text-wrap: balance; }}
h2 {{ font-size: var(--fs-h2) !important; line-height: 1.2 !important; letter-spacing: -0.015em; margin-top: 8px !important; }}
h3 {{ font-size: var(--fs-h3) !important; line-height: 1.3 !important; }}
em, i {{ font-style: normal; }}
.section {{ height: 40px; }}
.stApp {{
  background:
    radial-gradient(900px 500px at 85% -10%, rgba(168,84,246,0.16), transparent 60%),
    radial-gradient(700px 400px at -10% 110%, rgba(168,84,246,0.10), transparent 60%),
    #000000;
}}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2rem; max-width: 1280px; }}

/* sidebar */
[data-testid="stSidebar"] {{ background: #0b0b0d; border-right: 1px solid #222222; min-width: 250px; max-width: 250px; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{
  padding: 10px 14px; border-radius: 999px; margin: 2px 0; width: 100%;
  transition: background .15s;
}}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: #18181b; }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
  background: {VIOLET}; box-shadow: 0 0 18px rgba(168,84,246,0.55);
}}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {{ color: #fff; font-weight: 500; }}
[data-testid="stSidebar"] [role="radiogroup"] label > div > div:first-child:not([data-testid]) {{ display: none; }}

/* bento cards */
.card {{ background: #161616; border: 1px solid #242424; border-radius: 24px; padding: 20px 22px;
        animation: up .5s ease-out both; height: 100%; font-size: 14px; }}
.card b {{ font-size: var(--fs-md); font-weight: 500; }}
.card .muted {{ font-size: var(--fs-sm); }}
.card.dark {{ background: #111111; }}
.card.hero {{ border-radius: 28px; padding: 32px 34px;
             background: linear-gradient(135deg, rgba(168,84,246,0.18), rgba(17,17,17,1) 55%); }}
.kpi-label {{ color: {MUTED}; font-size: var(--fs-sm); margin-bottom: 6px; }}
.kpi-value {{ font-size: 28px; font-weight: 500; color: #fff; font-variant-numeric: tabular-nums; line-height: 1.1; letter-spacing: -0.02em; }}
.kpi-sub {{ font-size: var(--fs-xs); margin-top: 6px; }}
.h-hero {{ font-size: var(--fs-h1); font-weight: 500; color: #fff; line-height: 1.15; letter-spacing: -0.025em;
           margin: 0 0 12px; text-wrap: balance; }}
.muted {{ color: {MUTED}; font-size: var(--fs-md); }}
.badge {{ display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: var(--fs-xs); font-weight: 500; }}
.badge.fail {{ background: rgba(245,159,11,.15); color: #fbbf24; border: 1px solid rgba(245,159,11,.35); }}
.badge.pass {{ background: rgba(16,184,129,.15); color: #34d399; border: 1px solid rgba(16,184,129,.35); }}
.badge.info {{ background: rgba(168,84,246,.15); color: #c084fc; border: 1px solid rgba(168,84,246,.35); }}
.row {{ display: flex; justify-content: space-between; align-items: center; gap: 10px;
       padding: 10px 0; border-bottom: 1px solid #222; }}
.row:last-child {{ border-bottom: none; }}
.bar {{ height: 10px; border-radius: 999px; background: #222; overflow: hidden; }}
.bar > div {{ height: 100%; border-radius: 999px; }}
.formula {{ font-family: ui-monospace, monospace; font-size: 11px; color: {MUTED}; }}
.cm {{ display: grid; grid-template-columns: 110px 1fr 1fr; gap: 8px; }}
.cm div {{ padding: 14px; border-radius: 16px; text-align: center; }}
.cm .h {{ color: {MUTED}; font-size: 12px; padding: 6px; }}
.cm .ok {{ background: rgba(168,84,246,.18); border: 1px solid rgba(168,84,246,.4); }}
.cm .bad {{ background: rgba(245,159,11,.12); border: 1px solid rgba(245,159,11,.35); }}
.cm b {{ font-size: 24px; font-weight: 500; display: block; color: #fff; letter-spacing: -0.02em; }}
.author {{ position: fixed; bottom: 18px; width: 210px; color: {MUTED}; font-size: 12px;
          border-top: 1px solid #222; padding-top: 10px; }}

/* widgets */
.stButton button, .stDownloadButton button {{ border-radius: 999px; border: 1px solid #2a2a2a; }}
.stButton button p, .stDownloadButton button p {{ font-size: var(--fs-md) !important; }}
.stButton button[kind="primary"] {{ background: {VIOLET}; border: none; box-shadow: 0 0 16px rgba(168,84,246,.45); }}
[data-testid="stDataFrame"] {{ border: 1px solid #242424; border-radius: 16px; overflow: hidden; }}
.stTabs [aria-selected="true"] {{ color: #c084fc; }}
div[data-testid="stForm"] {{ background: #111111; border: 1px solid #242424; border-radius: 24px; }}
div[role="radiogroup"][aria-label="Start from an example"] label {{
  border: 1px solid #242424; border-radius: 18px; padding: 12px 16px; background: #111;
}}
div[role="radiogroup"][aria-label="Start from an example"] label:has(input:checked) {{
  border-color: {VIOLET}; box-shadow: 0 0 14px rgba(168,84,246,.4);
}}

@keyframes up {{ from {{ opacity: 0; transform: translateY(8px); }} to {{ opacity: 1; transform: none; }} }}
@media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; }} }}
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)


def html(s):
    st.markdown(s, unsafe_allow_html=True)


def card(inner, cls=""):
    html(f'<div class="card {cls}">{inner}</div>')


def kpi(label, value, sub="", color=EMERALD):
    card(f'<div class="kpi-label">{escape(label)}</div><div class="kpi-value">{escape(value)}</div>'
         f'<div class="kpi-sub" style="color:{color}">{escape(sub)}</div>')


def badge(pred):
    cls = "fail" if pred == "Fail" else "pass"
    text = "Likely Fail" if pred == "Fail" else "On track"
    return f'<span class="badge {cls}">{text}</span>'


def bar(share, color):
    return f'<div class="bar"><div style="width:{share * 100:.0f}%;background:{color}"></div></div>'


def rings(items):
    """Concentric ring gauges. items = [(label, 0-1 value, color)]."""
    svg, legend = [], []
    for i, (lab, v, col) in enumerate(items):
        r = 70 - i * 16
        c = 2 * 3.1416 * r
        svg.append(f'<circle cx="90" cy="90" r="{r}" stroke="#222" stroke-width="10" fill="none"/>'
                   f'<circle cx="90" cy="90" r="{r}" stroke="{col}" stroke-width="10" fill="none" '
                   f'stroke-linecap="round" stroke-dasharray="{v * c:.1f} {c:.1f}" transform="rotate(-90 90 90)"/>')
        legend.append(f'<div class="row"><span class="muted"><span style="color:{col}">●</span> {escape(lab)}</span>'
                      f'<b style="color:#fff">{v:.1%}</b></div>')
    return (f'<div style="display:flex;gap:18px;align-items:center;flex-wrap:wrap">'
            f'<svg width="180" height="180" viewBox="0 0 180 180" role="img" aria-label="Model score rings">'
            f'{"".join(svg)}</svg><div style="flex:1;min-width:160px">{"".join(legend)}</div></div>')