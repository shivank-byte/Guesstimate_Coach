# Guesstimate Coach

An AI-guided Fermi-estimation trainer. Walks you through a guesstimate question
step by step, then checks your final guess against real-world numbers pulled
from a live web search.

## The framework

The coaching prompts are built directly from three sources:

1. **Guesstimation** (Weinstein & Adam) — round aggressively to convenient
   numbers / powers of ten, anchor every guess on something you actually know.
2. **Guesstimation 2.0** (Weinstein & Edwards) — same technique, wider problem set.
3. **Hacking the Case Interview's 5-step framework** — Clarify → Structure
   (top-down/bottom-up, MECE breakdown) → Estimate → Calculate → Sanity-Check.

The app walks you through exactly those five steps. At each step you can ask
for a hint — the AI nudges you (points out gaps, unstated assumptions, MECE
violations) but never hands you the numbers or the answer. At the end, the
app also writes a short conclusion that synthesizes your own reasoning across
all five steps into a single narrative, alongside your raw notes for each step.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

### API keys — two options

**Option A: `secrets.toml` (recommended — enter keys once, not every run)**

1. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
2. Fill in your real keys:
   ```toml
   GEMINI_API_KEY = "AIza..."
   TAVILY_API_KEY = "tvly-..."
   ```
3. `secrets.toml` is already listed in `.gitignore` — it will never be committed
   to GitHub. Only the `.example` file (with no real keys) is meant to be shared.
4. When keys are found in secrets, the sidebar shows "loaded from secrets.toml"
   and skips the manual input fields entirely.

**Option B: paste into the sidebar**

If no `secrets.toml` is present, the app falls back to password-style input
fields in the sidebar. Keys are held only in that session's memory — nothing
is saved to disk.

Either way, you'll need:

- **Gemini API key** — for the coaching hints, the final grading, and the
  closing conclusion. Get one for free (no credit card required) at
  https://aistudio.google.com/app/apikey
- **Tavily API key** — for the final verification step, using the Tavily
  search API. Sign up at https://tavily.com — the free tier includes about
  1,000 search credits/month, more than enough for personal practice.
  (Note: this app originally used Google's Custom Search JSON API, but that
  API is closed to new customers and is being fully retired by January 1,
  2027, so Tavily is used instead — it's open to new signups and built
  specifically for feeding search results to an LLM.)

The Tavily step is optional — you can use the app purely as a coaching tool
without it, you just won't get the automated grading at the end.

## How verification works

When you submit a final guess, the app:
1. Searches the web (via the Tavily search API) for the question + "actual number statistic".
2. Sends the top result snippets + your guess to Gemini.
3. Gemini infers the best real-world reference number from the snippets and
   judges your guess by Fermi-problem standards: within ~2-3x (same order of
   magnitude) counts as a **GOOD** estimate, since guesstimates are about
   structured reasoning, not precision.

## Notes

- API keys are only held in Streamlit session memory for the run — nothing is
  persisted to disk.
- If Tavily search results don't clearly contain a number, the verdict will
  come back UNKNOWN rather than guessing.
- The closing "Your Complete Thought Process" section works even without a
  Tavily key — you'll still get your raw step-by-step notes and, if you click
  "Write my conclusion," a Gemini-written narrative summary of your reasoning.
  If you did run the verification step, that verdict gets woven into the
  summary too.
