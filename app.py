"""
Guesstimate Coach
------------------
A Streamlit app that walks a user through solving a Fermi/guesstimate problem
step by step (Clarify -> Structure -> Estimate -> Calculate -> Sanity-Check),
using an AI coach that nudges rather than answers, and then checks the
user's final guess against real-world numbers pulled from Tavily web search.

The coaching framework is built from three sources:
  1. "Guesstimation" (Weinstein & Adam)      -> order-of-magnitude rounding,
                                                 anchoring on known reference quantities
  2. "Guesstimation 2.0" (Weinstein & Edwards)-> same technique, applied to a
                                                 wider/harder problem set
  3. Hacking the Case Interview's 5-step      -> Clarify, Structure, Estimate,
     consulting guesstimate framework            Calculate, Sanity-Check

Run with:
    streamlit run app.py
"""

import re
import json
import streamlit as st
import requests

# ---------------------------------------------------------------------------
# Framework definition
# ---------------------------------------------------------------------------

STEPS = [
    {
        "key": "clarify",
        "title": "1. Clarify",
        "book": "Hacking the Case Interview",
        "prompt": (
            "Restate the question in your own words. What exactly are you being "
            "asked to estimate? Define any ambiguous terms (geography, time period, "
            "who/what counts) before you touch a single number."
        ),
    },
    {
        "key": "structure",
        "title": "2. Structure",
        "book": "Hacking the Case Interview",
        "prompt": (
            "Break the problem into a logical, non-overlapping (MECE) tree of "
            "components. Decide your approach: TOP-DOWN (start from a big known "
            "number and narrow down), BOTTOM-UP (start from one small unit and "
            "scale up), or for market-sizing questions, SUPPLY-SIDE vs DEMAND-SIDE "
            "(e.g. number of stations x throughput per station, vs. number of "
            "customers x frequency of use -- these two paths should roughly agree). "
            "Write out the pieces you'll multiply or add together."
        ),
    },
    {
        "key": "estimate",
        "title": "3. Estimate",
        "book": "Guesstimation & Guesstimation 2.0",
        "prompt": (
            "For each piece of your structure, assign a plausible value. Round "
            "aggressively to the nearest convenient number or power of ten. "
            "Anchor each guess on a real reference figure you know or that's "
            "provided in the anchors panel (population, household counts, typical "
            "prices, penetration rates) rather than picking numbers at random."
        ),
    },
    {
        "key": "calculate",
        "title": "4. Calculate",
        "book": "Guesstimation & Guesstimation 2.0",
        "prompt": (
            "Combine your estimates with simple arithmetic to arrive at a single "
            "number. Keep the math visible and simple -- round as you go rather "
            "than carrying false precision."
        ),
    },
    {
        "key": "sanity_check",
        "title": "5. Sanity-Check",
        "book": "Hacking the Case Interview",
        "prompt": (
            "Does your number pass the smell test? Compare it to a benchmark you "
            "trust. If it's off by orders of magnitude from something you know, "
            "figure out which assumption is the weak link and revise it."
        ),
    },
]

DEFAULT_QUESTIONS = [
    "How many auto-rickshaws are there in Mumbai?",
    "How many people ride the Mumbai local trains every day?",
    "How many smartphones are sold in India every year?",
    "How many chai/tea stalls are there across India?",
    "How many petrol pumps (fuel stations) are there in India?",
    "How many two-wheelers are sold in India every year?",
    "How many people order food online in Bengaluru every day?",
    "How many wedding events happen in India each year?",
    "What is the daily revenue of a single Chai Point or CCD outlet?",
    "How many engineering college seats are there in India each year?",
]

# Curated order-of-magnitude reference anchors for India, in the spirit of
# "Guesstimation"'s anchoring technique -- approximate, for practice purposes,
# not live/authoritative data. Passed to the AI coach so its nudges can be
# grounded in real numbers instead of generic advice.
INDIA_ANCHORS = {
    "Population": "~1.44 billion",
    "Number of households": "~300 million (avg. household size ~4.4-4.8 people)",
    "Urban population share": "~35% urban, ~65% rural",
    "States & Union Territories": "28 states + 8 union territories, 766+ districts",
    "Internet users": "~900 million+",
    "Smartphone users": "~750-800 million",
    "Two-wheeler ownership": "~2 per 10 people in urban areas; India is the largest two-wheeler market globally",
    "Cities with population > 1 million": "~50",
    "Mumbai metro population": "~21 million",
    "Delhi NCR population": "~32 million",
    "Bengaluru population": "~13-14 million",
    "GDP per capita": "~US $2,500-2,700/year (~₹2.1-2.3 lakh/year)",
    "Average urban household monthly spend": "~₹25,000-35,000 depending on city tier",
    "Working days per year": "~250 (accounting for weekends/holidays)",
}

GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_state():
    defaults = {
        "question": "",
        "current_step": 0,
        "step_answers": {s["key"]: "" for s in STEPS},
        "hints": {s["key"]: "" for s in STEPS},
        "final_guess": "",
        "verification_result": None,
        "started": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_state():
    for s in STEPS:
        st.session_state.step_answers[s["key"]] = ""
        st.session_state.hints[s["key"]] = ""
    st.session_state.current_step = 0
    st.session_state.final_guess = ""
    st.session_state.verification_result = None
    st.session_state.started = False


# ---------------------------------------------------------------------------
# AI coaching calls
# ---------------------------------------------------------------------------

COACH_SYSTEM_PROMPT = """You are an expert guesstimate / Fermi-estimation coach, trained on three sources:

1. "Guesstimation" and "Guesstimation 2.0" (Weinstein et al.) -- breaking a problem into a
chain of simpler estimates, rounding aggressively to convenient numbers or powers of ten,
and anchoring each guess on a known reference quantity.

2. Hacking the Case Interview's 5-step guesstimate framework -- Clarify, Structure
(top-down / bottom-up / supply-side vs demand-side, MECE breakdown), Estimate, Calculate,
Sanity-Check.

You are coaching ONE specific step, on ONE specific question the person actually gave you.
A "Reference anchors" list of real India-specific numbers may be provided -- these are the
same kind of known reference quantities the Guesstimation books use for anchoring.

HARD RULES -- a hint that breaks any of these is a failure:
- It must be SPECIFIC to the exact question asked. Never write a nudge that could be copy-pasted
  onto a different guesstimate question unchanged. Name the actual entities in the question
  (the city, the product, the population segment) explicitly.
- If a Reference anchors list is provided, actively cite the ones relevant to this question by
  name and value (e.g. "Mumbai's metro population is ~21 million" -- not just "use population data").
- For the Structure step specifically: propose at least one concrete breakdown path using real
  variables for this question (not "break it into components" -- say which components).
- For the Estimate step specifically: name the exact anchor figure(s) to start from and what
  adjustment (a rate, a percentage, a frequency) turns that anchor into the quantity needed --
  but do not do the multiplication or state the final numeric answer yourself.
- NEVER give away the final numeric answer or fully solve the arithmetic chain.
- If the person has already written something for this step, react to what they specifically
  wrote: name the gap, the unstated assumption, or the MECE overlap in their own draft.
- 3-5 sentences. Dense and concrete, not a generic pep talk. No filler like "great start!" or
  "consider thinking about...". Get straight to the specific, actionable point.
"""

VERIFY_SYSTEM_PROMPT = """You are grading a Fermi/guesstimate answer. You will be given:
- the original question
- the user's final numeric guess
- raw search result snippets pulled from the web about the real-world figure

Your job:
1. From the snippets, infer the best real-world reference number you can (it's fine if \
it's approximate or a range -- guesstimates are judged on order of magnitude, not \
precision).
2. Compare the user's guess to that reference number.
3. Judge using Fermi-problem standards: a guess within roughly 2-3x of the true value \
(same order of magnitude) is considered a GOOD estimate. Beyond that is OFF.
4. Write feedback that is SPECIFIC to this question and this guess -- name the actual gap \
(e.g. "you likely used India's total population instead of urban households, which would \
inflate the estimate by roughly the urban share") rather than generic advice like "review \
your assumptions." If you can't tell which assumption caused the gap from the information \
given, say that plainly instead of inventing a cause.
5. Respond ONLY with strict JSON, no markdown fences, no preamble, in this exact shape:
{
  "reference_estimate": "<best real-world number or range you found, as a short string, or 'unknown' if snippets are unclear>",
  "reference_source_note": "<one short sentence on where this number came from>",
  "verdict": "GOOD" | "OFF" | "UNKNOWN",
  "ratio_note": "<one short sentence describing how far off the guess was, e.g. 'about 2.5x too high'>",
  "feedback": "<2-3 sentences of SPECIFIC coaching feedback tying back to which step of the process likely caused the gap, if any>"
}
"""


def call_ai(system_prompt: str, user_message: str, api_key: str, model: str, max_tokens: int) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    combined = f"{system_prompt}\n\n---\n\n{user_message}"
    payload = {
        "contents": [{"parts": [{"text": combined}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": 0.4,
            # Gemini 2.5 models reserve part of maxOutputTokens for internal
            # "thinking" by default, which can silently truncate short answers.
            # Disable it so the full budget goes to the visible response.
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    resp = requests.post(url, params={"key": api_key}, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts).strip()


def get_hint(api_key: str, model: str, question: str, step_index: int) -> str:
    step = STEPS[step_index]

    context_lines = [f"Guesstimate question: {question}", ""]
    context_lines.append("Reference anchors (India, approximate, for grounding only):")
    for name, val in INDIA_ANCHORS.items():
        context_lines.append(f"- {name}: {val}")
    context_lines.append("")
    for i, s in enumerate(STEPS):
        if i < step_index:
            context_lines.append(f"[{s['title']} - completed] {st.session_state.step_answers[s['key']] or '(left blank)'}")
    context_lines.append("")
    context_lines.append(f"Current step to coach: {step['title']} (technique source: {step['book']})")
    context_lines.append(f"Step goal: {step['prompt']}")
    current_answer = st.session_state.step_answers[step["key"]]
    if current_answer.strip():
        context_lines.append(f"\nUser's current draft for this step: {current_answer}")
    else:
        context_lines.append("\nUser hasn't written anything for this step yet.")

    user_message = "\n".join(context_lines)
    return call_ai(COACH_SYSTEM_PROMPT, user_message, api_key, model, max_tokens=550)


def verify_guess(api_key: str, model: str, question: str, guess: str, snippets: list) -> dict:
    snippet_text = "\n\n".join(
        f"Source: {s['title']}\nURL: {s['link']}\nSnippet: {s['snippet']}" for s in snippets
    ) or "No search snippets available."

    user_message = (
        f"Question: {question}\n\n"
        f"User's final guess: {guess}\n\n"
        f"Search snippets:\n{snippet_text}"
    )

    raw = call_ai(VERIFY_SYSTEM_PROMPT, user_message, api_key, model, max_tokens=700)
    raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "reference_estimate": "unknown",
            "reference_source_note": "Could not parse model response.",
            "verdict": "UNKNOWN",
            "ratio_note": "",
            "feedback": raw,
        }


# ---------------------------------------------------------------------------
# Tavily Search (AI-oriented web search; open to new signups, generous free tier)
# ---------------------------------------------------------------------------

def tavily_search(query: str, api_key: str, num: int = 5) -> list:
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": num,
        "search_depth": "basic",
    }
    resp = requests.post(url, json=payload, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    items = data.get("results", [])
    return [
        {"title": it.get("title", ""), "link": it.get("url", ""), "snippet": it.get("content", "")}
        for it in items
    ]


# ---------------------------------------------------------------------------
# Theme: "Drafting Table" — blueprint linework, amber pencil accent,
# ruler-tick progress tracker, ink-stamp verdicts.
# ---------------------------------------------------------------------------

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #12283F;
    --panel: #17324C;
    --line: #3D6B8C;
    --line-faint: rgba(61, 107, 140, 0.35);
    --text: #E7EEF2;
    --text-dim: #93AEC2;
    --accent: #E8A33D;
    --good: #4FB286;
    --off: #E0584C;
}

.stApp {
    background-color: var(--bg);
    background-image:
        linear-gradient(var(--line-faint) 1px, transparent 1px),
        linear-gradient(90deg, var(--line-faint) 1px, transparent 1px);
    background-size: 28px 28px;
    color: var(--text);
    font-family: 'IBM Plex Sans', sans-serif;
}

h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }
h1 { font-weight: 700 !important; }

p, li, label, span, div { font-family: 'IBM Plex Sans', sans-serif; }

.eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.72rem;
    color: var(--accent);
    margin-bottom: 0.15rem;
}

/* Sidebar as instrument panel */
section[data-testid="stSidebar"] {
    background-color: #0E2033;
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-family: 'IBM Plex Mono', monospace !important;
    text-transform: uppercase;
    font-size: 0.95rem !important;
    letter-spacing: 0.08em;
    color: var(--text-dim);
}

/* Bordered containers -> blueprint panels */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: var(--panel);
    border: 1px solid var(--line) !important;
    border-radius: 2px !important;
}

/* Inputs */
.stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
    background-color: #0E2033 !important;
    color: var(--text) !important;
    border: 1px solid var(--line) !important;
    border-radius: 2px !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder { color: var(--text-dim) !important; }

/* Buttons: outlined technical style */
.stButton > button {
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.78rem;
    background-color: transparent;
    color: var(--accent);
    border: 1px solid var(--accent);
    border-radius: 2px;
    transition: background-color 0.15s ease, color 0.15s ease;
}
.stButton > button:hover {
    background-color: var(--accent);
    color: #12283F;
}
.stButton > button[kind="primary"] {
    background-color: var(--accent);
    color: #12283F;
    border: 1px solid var(--accent);
}
.stButton > button[kind="primary"]:hover {
    background-color: transparent;
    color: var(--accent);
}

/* Alerts / info / success boxes */
div[data-testid="stAlert"] {
    background-color: var(--panel);
    border: 1px solid var(--line);
    border-radius: 2px;
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Ruler tracker (signature element) */
.ruler-wrap { margin: 0.4rem 0 1.6rem 0; }
.ruler-track {
    position: relative;
    height: 2px;
    background: var(--line);
    margin: 0 6px;
}
.ruler-ticks { display: flex; justify-content: space-between; position: relative; top: -9px; }
.ruler-tick { display: flex; flex-direction: column; align-items: center; width: 100%; }
.ruler-dot {
    width: 16px; height: 16px; border-radius: 50%;
    background: var(--bg); border: 2px solid var(--line);
    display: flex; align-items: center; justify-content: center;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.55rem; color: var(--text-dim);
}
.ruler-dot.done { background: var(--good); border-color: var(--good); color: #0E2033; }
.ruler-dot.current { background: var(--accent); border-color: var(--accent); color: #0E2033; box-shadow: 0 0 0 4px rgba(232,163,61,0.2); }
.ruler-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.62rem; letter-spacing: 0.06em;
    text-transform: uppercase; color: var(--text-dim); margin-top: 6px; text-align: center;
}
.ruler-label.active { color: var(--accent); }

/* Hero + original watermark (no external imagery) */
.hero-wrap { position: relative; padding: 0.2rem 0 1.1rem 0; }
.hero-watermark {
    position: absolute;
    top: -18px;
    right: -12px;
    width: 168px;
    height: auto;
    opacity: 0.16;
    z-index: 0;
    pointer-events: none;
}
.hero-content { position: relative; z-index: 1; }
.hero-content h1 {
    font-size: 2.1rem !important;
    margin-bottom: 0.15rem !important;
}
.hero-caption {
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--text-dim);
    font-size: 0.92rem;
    max-width: 34rem;
}
.hero-signature {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    letter-spacing: 0.03em;
    color: var(--text-dim);
    margin-top: 0.55rem;
}
.hero-signature .cross { color: var(--off); }
.hero-signature .check { color: var(--good); }
.hero-signature .arrow { color: var(--accent); margin: 0 0.35rem; }

/* Stamp verdict badge */
.stamp {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 1.05rem;
    padding: 0.5rem 1.1rem;
    border: 3px double var(--stamp-color, var(--good));
    color: var(--stamp-color, var(--good));
    border-radius: 4px;
    transform: rotate(-4deg);
    margin: 0.4rem 0 1rem 0;
}
</style>
"""


def render_ruler(current_index: int, total_labels: list) -> str:
    ticks = ""
    for i, label in enumerate(total_labels):
        if i < current_index:
            dot_class, num = "done", "\u2713"
        elif i == current_index:
            dot_class, num = "current", f"{i+1:02d}"
        else:
            dot_class, num = "", f"{i+1:02d}"
        label_class = "active" if i == current_index else ""
        ticks += (
            f'<div class="ruler-tick">'
            f'<div class="ruler-dot {dot_class}">{num}</div>'
            f'<div class="ruler-label {label_class}">{label}</div>'
            f'</div>'
        )
    return f'<div class="ruler-wrap"><div class="ruler-track"></div><div class="ruler-ticks">{ticks}</div></div>'


def render_stamp(verdict: str) -> str:
    color = {"GOOD": "var(--good)", "OFF": "var(--off)", "UNKNOWN": "var(--text-dim)"}.get(verdict, "var(--text-dim)")
    label = {"GOOD": "Approved", "OFF": "Revise", "UNKNOWN": "Unclear"}.get(verdict, verdict)
    return f'<div class="stamp" style="--stamp-color:{color}">{label} \u00b7 {verdict}</div>'


RULER_LABELS = [s["title"].split(". ")[1] for s in STEPS] + ["Verify"]

# Original line-art watermark: a drafting compass over a tick-marked ruler.
# Hand-built from primitive shapes -- no external or third-party imagery.
WATERMARK_SVG = """
<svg class="hero-watermark" viewBox="0 0 220 190" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="112" cy="26" r="7" stroke="var(--line)" stroke-width="3"/>
  <line x1="112" y1="33" x2="66" y2="150" stroke="var(--line)" stroke-width="3" stroke-linecap="round"/>
  <line x1="112" y1="33" x2="158" y2="150" stroke="var(--line)" stroke-width="3" stroke-linecap="round"/>
  <line x1="90" y1="95" x2="134" y2="95" stroke="var(--line)" stroke-width="2" stroke-linecap="round"/>
  <circle cx="66" cy="150" r="4.5" fill="var(--line)"/>
  <circle cx="158" cy="150" r="4.5" fill="var(--line)"/>
  <line x1="20" y1="176" x2="200" y2="176" stroke="var(--line)" stroke-width="2" stroke-linecap="round"/>
  <line x1="30" y1="168" x2="30" y2="176" stroke="var(--line)" stroke-width="2"/>
  <line x1="50" y1="171" x2="50" y2="176" stroke="var(--line)" stroke-width="1.5"/>
  <line x1="70" y1="168" x2="70" y2="176" stroke="var(--line)" stroke-width="2"/>
  <line x1="90" y1="171" x2="90" y2="176" stroke="var(--line)" stroke-width="1.5"/>
  <line x1="130" y1="171" x2="130" y2="176" stroke="var(--line)" stroke-width="1.5"/>
  <line x1="150" y1="168" x2="150" y2="176" stroke="var(--line)" stroke-width="2"/>
  <line x1="170" y1="171" x2="170" y2="176" stroke="var(--line)" stroke-width="1.5"/>
  <line x1="190" y1="168" x2="190" y2="176" stroke="var(--line)" stroke-width="2"/>
</svg>
"""

def get_secret(name: str) -> str:
    """Read a key from .streamlit/secrets.toml if it exists; else empty string."""
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Guesstimate Coach", page_icon="\U0001F4D0", layout="centered")
st.markdown(THEME_CSS, unsafe_allow_html=True)
init_state()

st.markdown(
    f"""
    <div class="hero-wrap">
        {WATERMARK_SVG}
        <div class="hero-content">
            <div class="eyebrow">Fermi Estimation &middot; Drafting Table</div>
            <h1>Guesstimate Coach</h1>
            <div class="hero-caption">
                A step-by-step Fermi-estimation coach built on <em>Guesstimation</em>,
                <em>Guesstimation 2.0</em>, and Hacking the Case Interview's 5-step framework.
            </div>
            <div class="hero-signature">
                🙋 I guess&hellip; <span class="cross">&times;</span>
                <span class="arrow">&rarr;</span>
                🤖 AI guess&hellip; <span class="check">&#10003;</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Instrument Panel")

    ai_key = get_secret("GEMINI_API_KEY")
    if ai_key:
        st.success("Gemini key loaded from secrets.toml")
    else:
        ai_key = st.text_input("Gemini API key", type="password")
        st.caption("Free, no credit card: get one at aistudio.google.com/apikey")

    model = st.selectbox("Model", GEMINI_MODELS, index=0)
    st.markdown("---")
    st.subheader("Tavily Verification")

    tavily_key = get_secret("TAVILY_API_KEY")
    if tavily_key:
        st.success("Tavily key loaded from secrets.toml")
    else:
        tavily_key = st.text_input("Tavily API key", type="password")
        st.caption(
            "Needed only for the final 'check my guess' step, via the Tavily "
            "search API (open signup, ~1,000 free credits/month)."
        )
    st.markdown("---")
    if st.button("Start Over"):
        reset_state()
        st.rerun()

# --- Question selection -----------------------------------------------------
if not st.session_state.started:
    with st.container(border=True):
        st.markdown('<div class="eyebrow">Step 00 &middot; Choose your problem</div>', unsafe_allow_html=True)
        st.subheader("Pick or write a guesstimate question")
        choice = st.selectbox("Sample questions", ["-- write my own --"] + DEFAULT_QUESTIONS)
        if choice == "-- write my own --":
            q = st.text_input("Your question")
        else:
            q = choice
            st.text_input("Your question", value=q, disabled=True)

        if st.button("Start coaching", type="primary", disabled=not q):
            st.session_state.question = q
            st.session_state.started = True
            st.rerun()
    st.stop()

st.info(f"**Question:** {st.session_state.question}")

with st.expander("📐 Reference anchors (India) — use these for Structure & Estimate"):
    st.caption("Approximate figures for grounding your estimates. Not live/verified data.")
    for name, val in INDIA_ANCHORS.items():
        st.markdown(f"- **{name}:** {val}")

# --- Step-by-step coaching --------------------------------------------------
step_idx = st.session_state.current_step

st.markdown(render_ruler(step_idx, RULER_LABELS), unsafe_allow_html=True)

if step_idx < len(STEPS):
    step = STEPS[step_idx]
    with st.container(border=True):
        st.markdown(f'<div class="eyebrow">Technique &middot; {step["book"]}</div>', unsafe_allow_html=True)
        st.subheader(step["title"])
        st.write(step["prompt"])

        if st.button("Get a hint", key=f"hint_{step['key']}"):
            if not ai_key:
                st.error("Add your Gemini API key in the sidebar first.")
            else:
                with st.spinner("Thinking..."):
                    hint = get_hint(ai_key, model, st.session_state.question, step_idx)
                    st.session_state.hints[step["key"]] = hint

        if st.session_state.hints[step["key"]]:
            st.success(st.session_state.hints[step["key"]])

        st.session_state.step_answers[step["key"]] = st.text_area(
            "Your work for this step",
            value=st.session_state.step_answers[step["key"]],
            height=140,
            key=f"answer_{step['key']}",
        )

        col1, col2 = st.columns(2)
        with col1:
            if step_idx > 0 and st.button("Back"):
                st.session_state.current_step -= 1
                st.rerun()
        with col2:
            next_disabled = not st.session_state.step_answers[step["key"]].strip()
            if st.button("Next", type="primary", disabled=next_disabled):
                st.session_state.current_step += 1
                st.rerun()

else:
    # --- Final guess + verification -----------------------------------------
    with st.container(border=True):
        st.markdown('<div class="eyebrow">Final stage</div>', unsafe_allow_html=True)
        st.subheader("Final Guess")

        with st.expander("Review your reasoning"):
            for s in STEPS:
                st.markdown(f"**{s['title']}**")
                st.write(st.session_state.step_answers[s["key"]] or "_(blank)_")

        st.session_state.final_guess = st.text_input(
            "What's your final numeric estimate?", value=st.session_state.final_guess
        )

        if st.button("Check my guess", type="primary", disabled=not st.session_state.final_guess.strip()):
            if not ai_key:
                st.error("Add your Gemini API key in the sidebar first.")
            elif not tavily_key:
                st.error("Add your Tavily API key in the sidebar first.")
            else:
                with st.spinner("Searching and grading..."):
                    try:
                        snippets = tavily_search(
                            f"{st.session_state.question} actual number statistic India",
                            tavily_key,
                        )
                    except requests.HTTPError as e:
                        st.error(f"Tavily API error: {e}")
                        snippets = []
                    result = verify_guess(
                        ai_key, model, st.session_state.question,
                        st.session_state.final_guess, snippets,
                    )
                    st.session_state.verification_result = {"result": result, "snippets": snippets}

        if st.session_state.verification_result:
            result = st.session_state.verification_result["result"]
            snippets = st.session_state.verification_result["snippets"]

            verdict = result.get("verdict", "UNKNOWN")
            st.markdown(render_stamp(verdict), unsafe_allow_html=True)
            st.write(f"**Reference estimate found:** {result.get('reference_estimate', 'unknown')}")
            st.caption(result.get("reference_source_note", ""))
            st.write(f"**How far off:** {result.get('ratio_note', '')}")
            st.write(result.get("feedback", ""))

            if snippets:
                with st.expander("Raw search snippets used"):
                    for s in snippets:
                        st.markdown(f"- [{s['title']}]({s['link']}) — {s['snippet']}")

    if st.button("Try a new question"):
        reset_state()
        st.rerun()
