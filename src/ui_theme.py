"""Visual helpers for the Streamlit UI: global CSS, page headers, KPI cards, chips, status badges.

Only app-controlled strings are rendered as HTML, and every dynamic value is escaped.
Resume text is never rendered through these helpers.
"""
from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root { --primary:#4F46E5; --primary-soft:#EEF2FF; --ink:#0F172A; --body:#334155; --muted:#64748B;
        --line:#E2E8F0; --bg:#F8FAFC; --card:#FFFFFF; --radius:14px;
        --shadow:0 1px 2px rgba(15,23,42,.04), 0 4px 16px rgba(15,23,42,.05); }
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select {
  font-family:'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif !important; }
.stApp { background: var(--bg); }

/* hide Streamlit chrome (Deploy button, menu, header bar, footer) */
header[data-testid="stHeader"] { display:none; }
[data-testid="stToolbar"], [data-testid="stDecoration"], footer { display:none !important; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }
h1, h2, h3, h4 { color: var(--ink); letter-spacing:-0.015em; font-weight:700; }
h2 { font-size:1.35rem !important; } h3 { font-size:1.15rem !important; } h4 { font-size:1.02rem !important; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] { background:#0F172A; border-right:1px solid #1E293B; }
section[data-testid="stSidebar"] .block-container, section[data-testid="stSidebar"] > div { padding-top:1.2rem; }
section[data-testid="stSidebar"] * { color:#CBD5E1; }
.brand { display:flex; align-items:center; gap:.7rem; margin:.2rem 0 1.4rem; padding:0 .25rem; }
.brand-logo { width:40px; height:40px; border-radius:12px; display:flex; align-items:center; justify-content:center;
  background:linear-gradient(135deg,#6366F1,#06B6D4); font-weight:800; color:#fff !important; font-size:1rem; }
.brand-name { font-weight:700; font-size:1.05rem; color:#F8FAFC !important; line-height:1.15; }
.brand-sub { font-size:.74rem; color:#94A3B8 !important; }
.side-label { font-size:.68rem; font-weight:600; text-transform:uppercase; letter-spacing:.09em;
  color:#64748B !important; margin:1.2rem .35rem .45rem; }

/* sidebar radio -> menu items */
section[data-testid="stSidebar"] div[data-testid="stRadio"], section[data-testid="stSidebar"] div[data-testid="stRadio"] > div { width:100%; }
section[data-testid="stSidebar"] div[role="radiogroup"] { gap:.2rem; width:100%; align-items:stretch; }
section[data-testid="stSidebar"] div[role="radiogroup"] > div { width:100%; display:block; }
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] {
  margin:0; padding:.6rem .75rem; border-radius:10px; width:100%; cursor:pointer; transition:background .15s; }
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div:first-child { display:none; }
section[data-testid="stSidebar"] label[data-testid="stRadioOption"]:hover { background:#1E293B; }
section[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] {
  background:linear-gradient(90deg, rgba(99,102,241,.30), rgba(99,102,241,.06)); box-shadow: inset 3px 0 0 #818CF8; }
section[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] p {
  color:#FFFFFF !important; font-weight:600; }
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] p { font-size:.93rem; margin:0; }
.int-row { display:flex; justify-content:space-between; align-items:center; padding:.32rem .35rem; font-size:.83rem; }
.pill { font-size:.68rem; font-weight:700; padding:.12rem .5rem; border-radius:999px; letter-spacing:.03em; }
.pill.live { background:rgba(16,185,129,.16); color:#34D399 !important; }
.pill.mock { background:rgba(245,158,11,.16); color:#FBBF24 !important; }
.side-note { font-size:.75rem; color:#94A3B8 !important; padding:.4rem .35rem; line-height:1.4; }
section[data-testid="stSidebar"] .stButton button { background:#1E293B; border:1px solid #334155; color:#E2E8F0; width:100%; }

/* ---------- page header ---------- */
.page-head { margin:0 0 1.4rem; padding-bottom:1.1rem; border-bottom:1px solid var(--line); }
.page-head .eyebrow { font-size:.75rem; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:var(--primary); }
.page-head h1 { margin:.25rem 0 .3rem; font-size:1.85rem; padding:0; }
.page-head p { margin:0; color:var(--muted); font-size:.98rem; }
.hero { position:relative; overflow:hidden; border-radius:20px; padding:2rem 2.2rem; margin-bottom:1.6rem; color:#fff;
  background: radial-gradient(1200px 300px at 100% 0%, rgba(6,182,212,.35), transparent 60%),
              linear-gradient(120deg, #312E81 0%, #4F46E5 60%, #6366F1 100%); box-shadow:0 12px 32px rgba(79,70,229,.22); }
.hero h1 { color:#fff; margin:0 0 .35rem; font-size:2rem; padding:0; }
.hero p { margin:0; color:#E0E7FF; font-size:1.02rem; max-width:640px; }
.hero .eyebrow { font-size:.75rem; font-weight:600; letter-spacing:.1em; text-transform:uppercase; color:#A5B4FC; margin-bottom:.4rem; }

/* ---------- cards / kpis ---------- */
.card { background:var(--card); border:1px solid var(--line); border-radius:var(--radius); padding:1.2rem 1.3rem;
  box-shadow:var(--shadow); margin-bottom:1rem; }
.card h3 { margin:0 0 .35rem; font-size:1.12rem; }
.card .meta { color:var(--muted); font-size:.87rem; margin-bottom:.65rem; }
.card.selected { border-color:#C7D2FE; box-shadow:0 0 0 3px rgba(99,102,241,.12), var(--shadow); }
.section-title { font-size:.72rem; font-weight:600; text-transform:uppercase; letter-spacing:.08em; color:var(--muted);
  margin:.75rem 0 .4rem; }
.kpis { display:grid; grid-template-columns:repeat(var(--n, 4), minmax(0, 1fr)); gap:.8rem; margin:.2rem 0 1.4rem; }
@media (max-width: 900px) { .kpis { grid-template-columns:repeat(2, minmax(0, 1fr)); } }
.kpi { background:var(--card); border:1px solid var(--line); border-radius:var(--radius); padding:.95rem 1.05rem;
  box-shadow:var(--shadow); border-top:3px solid var(--accent, #6366F1); }
.kpi .label { font-size:.74rem; font-weight:600; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; }
.kpi .value { font-size:1.65rem; font-weight:700; color:var(--ink); margin-top:.25rem; line-height:1.1; }
.kpi .value.small { font-size:1.05rem; margin-top:.45rem; }
.notice { border-radius:12px; padding:.85rem 1rem; margin:.4rem 0 1rem; font-size:.9rem; border:1px solid; }
.notice.warn { background:#FFFBEB; border-color:#FDE68A; color:#92400E; }
.notice.info { background:#F0F9FF; border-color:#BAE6FD; color:#075985; }
.notice ul { margin:.35rem 0 0 1.1rem; padding:0; } .notice li { margin:.15rem 0; }
.notice b { color:inherit; }

/* chips + badges */
.chip { display:inline-block; padding:.22rem .7rem; margin:0 .35rem .4rem 0; border-radius:999px; font-size:.8rem; font-weight:600; }
.chip.req { background:var(--primary-soft); color:#3730A3; border:1px solid #C7D2FE; }
.chip.pref { background:#F1F5F9; color:#334155; border:1px solid var(--line); }
.badge { display:inline-flex; align-items:center; gap:.35rem; padding:.28rem .75rem; border-radius:999px; font-weight:600; font-size:.82rem; }
.badge::before { content:""; width:7px; height:7px; border-radius:50%; background:currentColor; opacity:.85; }
.badge.green { background:#DCFCE7; color:#166534; } .badge.red { background:#FEE2E2; color:#991B1B; }
.badge.amber { background:#FEF3C7; color:#92400E; } .badge.blue { background:#DBEAFE; color:#1E40AF; }
.badge.gray { background:#F1F5F9; color:#475569; }

/* widgets */
div[data-testid="stForm"], div[data-testid="stExpander"] details { background:var(--card); border:1px solid var(--line) !important;
  border-radius:var(--radius) !important; box-shadow:var(--shadow); }
div[data-testid="stForm"] { padding:1.3rem 1.4rem; }
.stButton button, .stFormSubmitButton button { border-radius:10px; font-weight:600; padding:.5rem 1rem; }
.stButton button[kind="primary"], .stFormSubmitButton button[kind="primaryFormSubmit"] {
  background:linear-gradient(135deg,#4F46E5,#6366F1); border:none; box-shadow:0 4px 12px rgba(79,70,229,.25); }
div[data-testid="stFileUploaderDropzone"] { border:2px dashed #C7D2FE; background:#F8FAFF; border-radius:12px; padding:1.2rem; }
div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; box-shadow:var(--shadow); }
.stTabs [data-baseweb="tab-list"] { gap:.4rem; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"] { padding:.55rem .9rem; border-radius:10px 10px 0 0; font-weight:600; }
div[data-testid="stAlert"] { border-radius:12px; }
</style>
"""

_STATUS = {
    "ELIGIBLE": ("green", "Eligible"), "NOT_ELIGIBLE": ("red", "Not eligible"),
    "REVIEW_REQUIRED": ("amber", "Needs review"), "PENDING": ("gray", "Pending"),
    "PROCESSING": ("blue", "Processing"), "ERROR": ("red", "Error"),
    "SCREENING": ("gray", "Screening"), "APPROVED": ("blue", "Approved"),
    "INTERVIEW_SCHEDULED": ("blue", "Interview scheduled"), "INTERVIEWED": ("blue", "Interviewed"),
    "SELECTED": ("green", "Selected"), "REJECTED": ("red", "Rejected"), "SUPERSEDED": ("gray", "Superseded"),
    "CONFIRMED": ("green", "Confirmed"), "PENDING_CONFIRMATION": ("amber", "Pending"),
    "CANCELLED": ("gray", "Cancelled"), "COMPLETED": ("blue", "Completed"),
    "SENT": ("green", "Sent"), "FAILED": ("red", "Failed"), "SENDING": ("amber", "Sending"), "NOT_SENT": ("gray", "Not sent"),
}
_STATUS_ICON = {"ELIGIBLE": "🟢", "NOT_ELIGIBLE": "🔴", "REVIEW_REQUIRED": "🟠", "ERROR": "⛔", "PENDING": "⚪",
                "PROCESSING": "🔵", "CONFIRMED": "🟢", "CANCELLED": "⚪", "COMPLETED": "🔵",
                "PENDING_CONFIRMATION": "🟠", "SENT": "🟢", "FAILED": "🔴", "SENDING": "🟠", "NOT_SENT": "⚪",
                "APPROVED": "🔵", "INTERVIEW_SCHEDULED": "🔵", "INTERVIEWED": "🔵", "SELECTED": "🟢",
                "REJECTED": "🔴", "SCREENING": "⚪", "SUPERSEDED": "⚪"}
_KPI_ACCENT = {"green": "#10B981", "red": "#EF4444", "amber": "#F59E0B", "blue": "#3B82F6", "indigo": "#6366F1",
               "gray": "#94A3B8"}


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def sidebar_brand(subtitle: str) -> None:
    st.sidebar.markdown(
        f"""<div class="brand"><div class="brand-logo">AR</div>
        <div><div class="brand-name">AI Recruiter</div><div class="brand-sub">{escape(subtitle)}</div></div></div>""",
        unsafe_allow_html=True)


def sidebar_label(text: str) -> None:
    st.sidebar.markdown(f'<div class="side-label">{escape(text)}</div>', unsafe_allow_html=True)


def sidebar_integrations(items: list[tuple[str, bool]]) -> None:
    rows = "".join(f'<div class="int-row"><span>{escape(name)}</span>'
                   f'<span class="pill {"mock" if mock else "live"}">{"MOCK" if mock else "LIVE"}</span></div>'
                   for name, mock in items)
    st.sidebar.markdown(rows, unsafe_allow_html=True)


def sidebar_note(text: str) -> None:
    st.sidebar.markdown(f'<div class="side-note">{escape(text)}</div>', unsafe_allow_html=True)


def hero(title: str, subtitle: str = "", eyebrow: str = "") -> None:
    eb = f'<div class="eyebrow">{escape(eyebrow)}</div>' if eyebrow else ""
    html(f'<div class="hero">{eb}<h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>')


def page_header(title: str, subtitle: str = "", eyebrow: str = "") -> None:
    eb = f'<div class="eyebrow">{escape(eyebrow)}</div>' if eyebrow else ""
    html(f'<div class="page-head">{eb}<h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>')


def kpis(items: list[tuple[str, object, str]], small: bool = False) -> None:
    """items: (label, value, accent color name). `small` for text values."""
    cards = "".join(
        f'<div class="kpi" style="--accent:{_KPI_ACCENT.get(color, color)}"><div class="label">{escape(label)}</div>'
        f'<div class="value{" small" if small else ""}">{escape(str(value))}</div></div>'
        for label, value, color in items)
    html(f'<div class="kpis" style="--n:{len(items)}">{cards}</div>')


def notice(title: str, lines: list[str], kind: str = "warn") -> None:
    items = "".join(f"<li>{escape(line)}</li>" for line in lines)
    html(f'<div class="notice {kind}"><b>{escape(title)}</b><ul>{items}</ul></div>')


def chips(names: list[str], kind: str = "req") -> str:
    if not names:
        return '<span class="chip pref">none</span>'
    return "".join(f'<span class="chip {kind}">{escape(n)}</span>' for n in names)


def badge(status: str | None) -> str:
    color, label = _STATUS.get(status or "", ("gray", (status or "-").replace("_", " ").title()))
    return f'<span class="badge {color}">{escape(label)}</span>'


def status_icon(status: str | None) -> str:
    """Plain-text status with a colored dot, for use inside dataframes."""
    label = _STATUS.get(status or "", (None, (status or "-").replace("_", " ").title()))[1]
    return f"{_STATUS_ICON.get(status or '', '•')} {label}"


def table(rows: list[dict] | pd.DataFrame, **kwargs) -> None:
    """Dataframe with empty values shown as an em dash instead of None/NaN."""
    # dtype=object keeps whole numbers as "3" (not "3.0") when a column also contains None
    df = rows.astype(object) if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows, dtype=object)
    df = df.where(pd.notna(df), "—")
    st.dataframe(df, hide_index=True, width="stretch", **kwargs)


def job_card(title: str, min_exp: int, mandatory: list[str], preferred: list[str], selected: bool = False) -> None:
    html(f"""<div class="card {'selected' if selected else ''}">
        <h3>{escape(title)}</h3>
        <div class="meta">Minimum experience · <b>{int(min_exp)} year{'s' if int(min_exp) != 1 else ''}</b></div>
        <div class="section-title">Required skills</div>{chips(mandatory, 'req')}
        <div class="section-title">Nice to have</div>{chips(preferred, 'pref')}
        </div>""")
