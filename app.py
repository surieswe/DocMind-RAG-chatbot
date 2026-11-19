# SQLite fix for Linux servers (Streamlit Cloud). Harmless on Windows.
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import io
import sys
import time
import html
import math
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))

import streamlit as st
import streamlit.components.v1 as components
from pypdf import PdfReader
from pipeline import (
    SCORE_THRESHOLD,
    build_retriever,
    build_chain_engine,
    build_graph_engine,
)
from config import GOOGLE_API_KEY

# Total pages allowed across all uploaded files
MAX_PAGES = 100

# Height (px) of the scrolling chat box. Lower it if your screen is short,
# raise it if you have room, so the whole app fits without page scrolling.
CHAT_HEIGHT = 360

# App name: the logo shows MAIN in white and ACCENT in blue
APP_NAME_MAIN = "Docu"
APP_NAME_ACCENT = "Mind"

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(
    page_title=f"{APP_NAME_MAIN}{APP_NAME_ACCENT} · Chat with your documents",
    page_icon="▶️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Theme: dark navy with blue accent, WCAG AA contrast
#   Background  #0F171E   Surface #1A242F   Raised #232F3E
#   Text        #FFFFFF   (~17:1 on bg)
#   Muted text  #B0BEC9   (~9:1 on bg, ~8:1 on surface)
#   Accent blue #00A8E1   used with DARK text #0F171E (~7:1)
# ------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #0F171E;
    --surface: #1A242F;
    --raised: #232F3E;
    --line: rgba(255, 255, 255, 0.14);
    --text: #FFFFFF;
    --muted: #B0BEC9;
    --blue: #00A8E1;
    --blue-hover: #3FC1F0;
    --on-blue: #0F171E;
    --body: 'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif;
}

/* ---------- Base ---------- */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    font-family: var(--body) !important;
}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: var(--bg) !important;
    color: var(--text) !important;
}
.block-container {
    padding-top: 2.6rem !important;
    padding-bottom: 5rem !important;
    max-width: 960px !important;
}
header[data-testid="stHeader"] {background: transparent !important; height: 2.4rem !important;}

/* Hide Streamlit's menu, footer and deploy button, but keep the sidebar arrows */
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stStatusWidget"],
[data-testid="stMainMenu"], [data-testid="stAppDeployButton"], .stDeployButton {
    visibility: hidden !important;
}
[data-testid="stExpandSidebarButton"], [data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] {
    visibility: visible !important;
    color: var(--text) !important;
}

[data-testid="stCaptionContainer"], small {color: var(--muted) !important;}
h1, h2, h3, h4 {color: var(--text) !important;}
hr, [data-testid="stDivider"] {border-color: var(--line) !important; margin: .6rem 0 !important;}

/* ---------- Top bar ---------- */
.pv-topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 0 .8rem 0;
    animation: fadeDown .45s ease both;
}
.pv-logo {
    font-size: 1.7rem; font-weight: 800; letter-spacing: -.02em;
    color: var(--text); line-height: 1;
}
.pv-logo b {color: var(--blue); font-weight: 800;}
.pv-status {
    display: inline-flex; align-items: center; gap: .5rem;
    font-size: .8rem; font-weight: 600; color: var(--muted);
    background: var(--surface); border: 1px solid var(--line);
    padding: .35rem .8rem; border-radius: 999px;
}
.pv-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #3DDC84; animation: ping 2s infinite;
}
.pv-dot.idle {background: #8696A5; animation: none;}

/* ---------- Hero (shown only before documents are loaded) ---------- */
.pv-hero {
    position: relative; overflow: hidden;
    border-radius: 14px; padding: 1.5rem 1.7rem; margin-bottom: 1rem;
    background:
        radial-gradient(circle at 90% 10%, rgba(0,168,225,.28) 0%, transparent 55%),
        linear-gradient(135deg, #1A242F, #0F171E);
    border: 1px solid var(--line);
    animation: fadeUp .5s cubic-bezier(.2,.7,.2,1) both;
}
.pv-badge {
    display: inline-block; background: var(--blue); color: var(--on-blue);
    font-size: .68rem; font-weight: 800; letter-spacing: .1em; text-transform: uppercase;
    padding: .22rem .55rem; border-radius: 4px; margin-bottom: .6rem;
}
.pv-hero h1 {
    font-size: 2.1rem !important; line-height: 1.1 !important; font-weight: 800 !important;
    letter-spacing: -.02em; margin: 0 0 .4rem 0 !important; padding: 0 !important;
}
.pv-hero p {font-size: .98rem; line-height: 1.5; color: var(--muted); max-width: 560px; margin: 0;}

.pv-section {font-size: 1.05rem; font-weight: 700; color: var(--text); margin: .9rem 0 .6rem 0;}

/* ---------- How-it-works cards ---------- */
.pv-card {
    background: var(--surface); border: 1px solid var(--line); border-radius: 12px;
    padding: 1rem 1rem; min-height: 150px;
    transition: transform .25s cubic-bezier(.2,.7,.2,1), border-color .25s, background .25s;
    animation: fadeUp .55s cubic-bezier(.2,.7,.2,1) both;
}
.pv-card:hover {transform: translateY(-3px); border-color: var(--blue); background: var(--raised);}
.pv-card .num {font-size: 1.4rem; font-weight: 800; color: var(--blue); margin-bottom: .25rem; line-height: 1;}
.pv-card h4 {font-size: 1rem !important; font-weight: 700 !important; margin: 0 0 .25rem 0 !important; padding: 0 !important;}
.pv-card p {font-size: .86rem; color: var(--muted); line-height: 1.45; margin: 0;}

/* Clickable Upload card */
.pv-card-upload {cursor: pointer;}
.pv-card-upload:focus-visible {outline: 2px solid var(--blue); outline-offset: 2px;}
.pv-card-hint {margin-top: .5rem; font-size: .78rem; font-weight: 700; color: var(--blue);}

.pv-tip {
    margin-top: .8rem; font-size: .84rem; color: var(--muted);
    background: var(--surface); border-left: 3px solid var(--blue);
    padding: .55rem .85rem; border-radius: 6px;
}

/* ---------- File chips ---------- */
.pv-chips {display: flex; flex-wrap: wrap; gap: .4rem; margin: 0 0 .6rem 0;}
.pv-chip {
    background: var(--raised); color: var(--text); border: 1px solid var(--line);
    padding: .28rem .7rem; border-radius: 999px; font-size: .78rem; font-weight: 600;
    animation: popIn .35s ease both;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {background: var(--surface) !important; border-right: 1px solid var(--line) !important;}

/* Shrink the empty header area at the top of the sidebar */
[data-testid="stSidebarHeader"] {
    height: 2.2rem !important; min-height: 0 !important;
    padding: 0 .5rem !important; margin: 0 !important;
}
[data-testid="stSidebarUserContent"] {padding-top: 0 !important;}

/* Give every block its own space so nothing overlaps or gets clipped */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap: .9rem !important;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stElementContainer"] {
    margin: 0 !important; overflow: visible !important;
}

.pv-label {
    font-size: .7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: .12em; color: #E6EDF2;
    line-height: 1.5 !important; padding: .1rem 0 .35rem 0; margin: 0 !important;
}
.pv-note {
    font-size: .78rem; color: var(--muted); background: var(--raised);
    border-radius: 6px; padding: .4rem .65rem;
    line-height: 1.45 !important; margin: 0 !important;
}
.pv-note b {color: var(--text);}

/* File uploader */
[data-testid="stFileUploaderDropzone"] {
    background: var(--bg) !important;
    border: 1.5px dashed rgba(255,255,255,.3) !important;
    border-radius: 10px !important;
    padding: .7rem !important;
    transition: border-color .2s;
}
[data-testid="stFileUploaderDropzone"]:hover {border-color: var(--blue) !important;}
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p {color: var(--muted) !important;}
[data-testid="stFileUploader"] button {
    background: var(--raised) !important; color: var(--text) !important;
    border: 1px solid var(--line) !important; border-radius: 6px !important;
}

/* Engine segmented control (3 options) */
[data-testid="stSidebar"] div[role="radiogroup"] {
    background: var(--bg); border: 1px solid var(--line);
    padding: 4px; border-radius: 10px; gap: 4px !important;
    display: flex; flex-wrap: nowrap;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label {
    flex: 1; justify-content: center; border-radius: 7px;
    padding: .35rem .2rem !important; margin: 0 !important; transition: background .2s;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {display: none;}
[data-testid="stSidebar"] div[role="radiogroup"] > label p {
    font-weight: 700 !important; font-size: .76rem !important; color: var(--muted) !important;
    white-space: nowrap;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {background: var(--blue);}
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p {color: var(--on-blue) !important;}
.pv-engine-desc {
    font-size: .76rem; color: var(--muted);
    line-height: 1.5 !important; padding-bottom: .2rem;
}

/* ---------- Buttons ---------- */
.stButton, [data-testid="stButton"] {width: 100% !important;}
.stButton > button, [data-testid="stButton"] > button {
    width: 100% !important; border-radius: 8px !important; font-weight: 700 !important;
    font-size: .9rem !important; padding: .5rem .9rem !important;
    border: 1px solid transparent !important;
    transition: transform .15s ease, background .2s, box-shadow .2s !important;
}
.stButton > button:hover {transform: translateY(-1px);}
.stButton > button:active {transform: scale(.98);}
.stButton > button:focus-visible {outline: 2px solid #FFFFFF !important; outline-offset: 2px;}
.stButton > button[kind="primary"] {
    background: var(--blue) !important; color: var(--on-blue) !important;
    box-shadow: 0 4px 14px rgba(0,168,225,.28);
}
.stButton > button[kind="primary"] p {color: var(--on-blue) !important;}
.stButton > button[kind="primary"]:hover {background: var(--blue-hover) !important;}
.stButton > button[kind="primary"]:disabled {
    background: #2B3947 !important; color: #9FB0BF !important; box-shadow: none;
}
.stButton > button[kind="primary"]:disabled p {color: #9FB0BF !important;}
.stButton > button[kind="secondary"] {
    background: var(--raised) !important; color: var(--text) !important; border-color: var(--line) !important;
}
.stButton > button[kind="secondary"] p {color: var(--text) !important;}
.stButton > button[kind="secondary"]:hover {
    background: #2E3C4D !important; border-color: rgba(255,255,255,.3) !important;
}

/* ---------- Progress, status, alerts ---------- */
[data-testid="stProgress"] > div > div > div > div {background: var(--blue) !important;}
[data-testid="stProgress"] p {color: var(--muted) !important;}
[data-testid="stStatus"], [data-testid="stExpander"] {
    background: var(--surface) !important; border: 1px solid var(--line) !important; border-radius: 10px !important;
}
[data-testid="stAlert"] {border-radius: 8px !important;}

/* ---------- Sidebar stats ---------- */
.pv-stats {display: grid; grid-template-columns: 1fr 1fr; gap: .5rem;}
.pv-stat {
    background: var(--raised); border-radius: 8px; padding: .5rem .4rem;
    text-align: center; border-top: 3px solid var(--blue); animation: popIn .4s ease both;
}
.pv-stat .v {font-size: 1.4rem; font-weight: 800; line-height: 1; color: var(--text);}
.pv-stat .l {font-size: .66rem; color: var(--muted); text-transform: uppercase; letter-spacing: .1em; margin-top: .2rem;}

/* ---------- Chat box (scrolls inside, page stays fixed) ---------- */
[data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--line) !important;
    border-radius: 12px !important;
    background: var(--surface);
}

.pv-user-row {display: flex; justify-content: flex-end; margin: .35rem 0; animation: slideIn .3s ease both;}
.pv-user {
    background: var(--blue); color: var(--on-blue);
    padding: .65rem 1rem; border-radius: 14px 14px 4px 14px;
    max-width: 78%; font-size: .95rem; line-height: 1.5; font-weight: 500;
}

[data-testid="stChatMessage"] {
    background: transparent !important; padding: .2rem 0 !important; animation: slideIn .3s ease both;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    background: var(--raised); border: 1px solid var(--line);
    padding: .7rem 1rem; border-radius: 14px 14px 14px 4px;
    line-height: 1.6; color: var(--text);
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {color: var(--text) !important;}
[data-testid="stChatMessage"] [data-testid="stCaptionContainer"] {
    font-size: .72rem !important; padding-left: .3rem; color: var(--muted) !important;
}

.pv-empty {text-align: center; padding: 1.2rem .5rem .6rem; color: var(--muted);}
.pv-empty h3 {font-size: 1.3rem !important; font-weight: 700 !important; margin: 0 0 .2rem 0 !important; padding: 0 !important;}
.pv-empty div {font-size: .88rem;}

/* ---------- Chat input ---------- */
[data-testid="stChatInput"] {
    background: var(--surface) !important;
    border: 1px solid rgba(255,255,255,.22) !important;
    border-radius: 12px !important;
    transition: border-color .2s, box-shadow .2s;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 3px rgba(0,168,225,.3) !important;
}
[data-testid="stChatInput"] textarea {color: var(--text) !important; background: transparent !important;}
[data-testid="stChatInput"] textarea::placeholder {color: #9FB0BF !important;}
[data-testid="stBottomBlockContainer"], [data-testid="stBottom"] {background: var(--bg) !important;}

/* ---------- Animations ---------- */
@keyframes fadeUp   {from {opacity: 0; transform: translateY(12px);} to {opacity: 1; transform: none;}}
@keyframes fadeDown {from {opacity: 0; transform: translateY(-8px);} to {opacity: 1; transform: none;}}
@keyframes popIn    {from {opacity: 0; transform: scale(.94);} to {opacity: 1; transform: none;}}
@keyframes slideIn  {from {opacity: 0; transform: translateY(6px);} to {opacity: 1; transform: none;}}
@keyframes ping {
    0%   {box-shadow: 0 0 0 0 rgba(61,220,132,.6);}
    70%  {box-shadow: 0 0 0 8px rgba(61,220,132,0);}
    100% {box-shadow: 0 0 0 0 rgba(61,220,132,0);}
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: .01ms !important; animation-iteration-count: 1 !important;
        transition-duration: .01ms !important;
    }
}
@media (max-width: 768px) {
    .pv-hero h1 {font-size: 1.7rem !important;}
    .pv-user {max-width: 90%;}
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

ENGINES = {
    "LangGraph": "Checks whether your documents really contain the answer, and says "
                 "'not found' instead of guessing.",
    "LangChain": "Simple retrieve-then-answer pipeline. Always answers from the closest passages.",
    "Compare": "Runs both engines on your question and shows the answers side by side, "
               "with time and similarity score. Uses 2 model calls per question.",
}

STARTERS = [
    "What are the main topics covered?",
    "Give me a short summary.",
    "List the key terms and definitions.",
]

# ------------------------------------------------------------------
# Session state (survives Streamlit reruns)
# ------------------------------------------------------------------
defaults = {"retriever": None, "messages": [], "docs_info": None, "pending": None}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def set_pending(text: str):
    st.session_state.pending = text


def friendly_error(e: Exception) -> str:
    msg = str(e)
    if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
        return ("⚠️ The AI service rate limit was reached. Please wait a minute and try again, "
                "or switch to a different model in your `.env` file.")
    if "NOT_FOUND" in msg or "404" in msg:
        return "⚠️ The configured model isn't available. Update `LLM_MODEL` in your `.env` file."
    return f"⚠️ Something went wrong: {msg}"


def stream_text(text: str):
    # Small typing effect for a smoother feel
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)


def count_pages(f) -> int:
    """Pages in a PDF; for TXT files, roughly 3000 characters = 1 page."""
    data = f.getvalue()  # getvalue() doesn't move the file's read position
    if f.name.lower().endswith(".pdf"):
        return len(PdfReader(io.BytesIO(data)).pages)
    return max(1, math.ceil(len(data) / 3000))


def run_engine(build, question: str):
    """Run one engine. Returns (answer, seconds, had_error)."""
    start = time.perf_counter()
    try:
        answer = build(st.session_state.retriever)(question)
        return answer, time.perf_counter() - start, False
    except Exception as e:
        return friendly_error(e), time.perf_counter() - start, True


def best_similarity(question: str):
    """Similarity of the best matching chunk (what the LangGraph gate checks)."""
    try:
        scored = st.session_state.retriever.vectorstore.similarity_search_with_relevance_scores(
            question, k=4
        )
        return max((s for _, s in scored), default=0.0)
    except Exception:
        return None


def user_bubble(text: str):
    safe = html.escape(text).replace("\n", "<br>")
    st.markdown(
        f'<div class="pv-user-row"><div class="pv-user">{safe}</div></div>',
        unsafe_allow_html=True,
    )


def render_compare(c: dict):
    """Two answers side by side, with time and similarity info."""
    cols = st.columns(2)
    for col, key, label in ((cols[0], "chain", "LangChain"), (cols[1], "graph", "LangGraph")):
        with col:
            st.markdown(f"**{label}**\n\n{c[key]['answer']}")
            note = f"{c[key]['time']:.1f}s"
            if key == "graph" and c.get("gated"):
                note += " · skipped the model (low similarity)"
            st.caption(note)
    if c.get("score") is not None:
        st.caption(
            f"Best chunk similarity: {c['score']:.2f} · "
            f"LangGraph threshold: {SCORE_THRESHOLD}"
        )


def render_message(msg: dict):
    if msg["role"] == "user":
        user_bubble(msg["content"])
    else:
        with st.chat_message("assistant", avatar="📄"):
            if msg.get("compare"):
                render_compare(msg["compare"])
            else:
                st.markdown(msg["content"])
                if msg.get("meta"):
                    st.caption(msg["meta"])


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    # TEMP DEBUG: confirms the API key reached the app (prints True/False only)
    st.caption(f"API key loaded: {bool(GOOGLE_API_KEY)}")

    st.markdown('<div class="pv-label">01 · Upload documents</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="pv-note">Up to <b>{MAX_PAGES} pages</b> in total. PDF or TXT.</div>',
        unsafe_allow_html=True,
    )
    files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # ---- check the page limit as soon as files are picked ----
    total_pages = 0
    unreadable = []
    for f in files or []:
        try:
            total_pages += count_pages(f)
        except Exception:
            unreadable.append(f.name)

    over_limit = total_pages > MAX_PAGES
    upload_ok = bool(files) and not over_limit and not unreadable

    if files:
        st.progress(min(total_pages / MAX_PAGES, 1.0), text=f"{total_pages} / {MAX_PAGES} pages")
        if unreadable:
            st.error("Couldn't read: " + ", ".join(unreadable)
                     + ". The file may be corrupted or password-protected.")
        elif over_limit:
            st.error(
                f"Too many pages: {total_pages} (limit {MAX_PAGES}). "
                "Remove a file or upload a shorter document."
            )

    st.markdown('<div class="pv-label">02 · Choose engine</div>', unsafe_allow_html=True)
    engine_name = st.radio(
        "Engine", list(ENGINES.keys()), label_visibility="collapsed", horizontal=True
    )
    st.markdown(f'<div class="pv-engine-desc">{ENGINES[engine_name]}</div>', unsafe_allow_html=True)

    process = st.button("Process documents", type="primary", disabled=not upload_ok)

    if process and upload_ok:
        with st.status("Processing documents...", expanded=True) as status:
            bar = st.progress(0.0, text="Reading files...")

            def on_progress(done, total):
                bar.progress(done / total, text=f"Embedding {done}/{total} chunks")

            try:
                retriever, n_pages, n_chunks = build_retriever(
                    files, progress_callback=on_progress
                )
                st.session_state.retriever = retriever
                st.session_state.messages = []
                st.session_state.docs_info = {
                    "files": [f.name for f in files],
                    "pages": n_pages,
                    "chunks": n_chunks,
                }
                status.update(label="Ready to answer questions", state="complete", expanded=False)
            except Exception as e:
                st.session_state.retriever = None
                st.session_state.docs_info = None
                status.update(label="Processing failed", state="error")
                st.error(friendly_error(e))
                st.exception(e)  # TEMP DEBUG: shows the full error. Remove when fixed.

    info = st.session_state.docs_info
    if info and st.session_state.retriever:
        st.markdown(
            f'<div class="pv-stats">'
            f'<div class="pv-stat"><div class="v">{info["pages"]}</div><div class="l">Pages</div></div>'
            f'<div class="pv-stat"><div class="v">{info["chunks"]}</div><div class="l">Chunks</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button("Clear chat", type="secondary"):
            st.session_state.messages = []
            st.rerun()

    st.caption("LangChain · LangGraph · Gemini · ChromaDB")

# ------------------------------------------------------------------
# Top bar
# ------------------------------------------------------------------
info = st.session_state.docs_info
ready = bool(st.session_state.retriever and info)

st.markdown(
    f"""
    <div class="pv-topbar">
        <div class="pv-logo">{APP_NAME_MAIN}<b>{APP_NAME_ACCENT}</b></div>
        <div class="pv-status">
            <span class="pv-dot {'' if ready else 'idle'}"></span>
            {'Library ready' if ready else 'No documents loaded'}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Empty state: hero + how it works (before documents are loaded)
# ------------------------------------------------------------------
if not ready:
    st.markdown(
        f"""
        <div class="pv-hero">
            <span class="pv-badge">Welcome</span>
            <h1>Ask your documents</h1>
            <p>Upload a document (up to {MAX_PAGES} pages), then ask questions in plain language.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="pv-section">How it works</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    steps = [
        ("01", "Upload", f"Add PDF or TXT files from the sidebar (up to {MAX_PAGES} pages in total)."),
        ("02", "Process", "We split your files into chunks and index them for search."),
        ("03", "Ask", "Chat in plain language and get answers from your own content."),
    ]
    for col, (num, title, text) in zip(cols, steps):
        is_upload = num == "01"
        cls = "pv-card pv-card-upload" if is_upload else "pv-card"
        attrs = ' role="button" tabindex="0"' if is_upload else ""
        hint = '<div class="pv-card-hint">Click to choose files →</div>' if is_upload else ""
        col.markdown(
            f"""<div class="{cls}"{attrs}><div class="num">{num}</div>
            <h4>{title}</h4><p>{text}</p>{hint}</div>""",
            unsafe_allow_html=True,
        )

    # Invisible script: clicking the Upload card opens the file picker
    components.html(
        """
        <script>
        const doc = window.parent.document;
        if (!window.parent.__uploadCardBound) {
            window.parent.__uploadCardBound = true;

            function openPicker() {
                // Open the sidebar first if it is collapsed
                const expand = doc.querySelector(
                    '[data-testid="stExpandSidebarButton"], [data-testid="collapsedControl"]'
                );
                if (expand) expand.click();
                setTimeout(function () {
                    const input =
                        doc.querySelector('[data-testid="stFileUploaderDropzone"] input[type="file"]') ||
                        doc.querySelector('input[type="file"]');
                    if (input) input.click();
                }, 250);
            }

            doc.addEventListener("click", function (e) {
                if (e.target.closest(".pv-card-upload")) openPicker();
            });
            doc.addEventListener("keydown", function (e) {
                if ((e.key === "Enter" || e.key === " ") && e.target.closest &&
                    e.target.closest(".pv-card-upload")) {
                    e.preventDefault();
                    openPicker();
                }
            });
        }
        </script>
        """,
        height=0,
    )

    st.markdown(
        '<div class="pv-tip">Tip: text-based PDFs work best. Scanned pages without a text layer can\'t be read.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ------------------------------------------------------------------
# Chat (ready state): file chips, scrolling chat box, input at the bottom
# ------------------------------------------------------------------
names = info["files"]
shown = names[:3]
chips = "".join(
    f'<span class="pv-chip">📎 {html.escape(n if len(n) <= 28 else n[:25] + "...")}</span>'
    for n in shown
)
if len(names) > 3:
    chips += f'<span class="pv-chip">+{len(names) - 3} more</span>'
st.markdown(f'<div class="pv-chips">{chips}</div>', unsafe_allow_html=True)

chat_box = st.container(height=CHAT_HEIGHT)
question = st.chat_input("Ask a question about your documents...")
pending = st.session_state.pending
if pending:
    question = pending
    st.session_state.pending = None

with chat_box:
    if not st.session_state.messages and not question:
        st.markdown(
            """
            <div class="pv-empty">
                <h3>What would you like to know?</h3>
                <div>Pick a suggestion or type your own question below.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        cols = st.columns(len(STARTERS))
        for i, (col, text) in enumerate(zip(cols, STARTERS)):
            col.button(text, key=f"starter_{i}", on_click=set_pending, args=(text,),
                       type="secondary")

    for msg in st.session_state.messages:
        render_message(msg)

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        user_bubble(question)

        with st.chat_message("assistant", avatar="📄"):
            if engine_name == "Compare":
                # ---- run both engines and show the answers side by side ----
                with st.spinner("Running both engines..."):
                    score = best_similarity(question)
                    chain_answer, chain_time, _ = run_engine(build_chain_engine, question)
                    graph_answer, graph_time, _ = run_engine(build_graph_engine, question)
                compare = {
                    "chain": {"answer": chain_answer, "time": chain_time},
                    "graph": {"answer": graph_answer, "time": graph_time},
                    "score": score,
                    "gated": score is not None and score < SCORE_THRESHOLD,
                }
                render_compare(compare)
                st.session_state.messages.append(
                    {"role": "assistant", "content": "", "compare": compare}
                )
            else:
                # ---- single engine, with the typing effect ----
                build = build_graph_engine if engine_name == "LangGraph" else build_chain_engine
                with st.spinner("Searching your documents..."):
                    answer, elapsed, error = run_engine(build, question)

                if error:
                    st.markdown(answer)
                    meta = None
                else:
                    st.write_stream(stream_text(answer))
                    meta = f"{engine_name} · answered in {elapsed:.1f}s"
                    st.caption(meta)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "meta": meta}
                )