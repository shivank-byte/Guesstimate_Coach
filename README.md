# Guesstimate Coach

An AI-guided Fermi-estimation trainer. Walks you through a guesstimate question
step by step, then checks your final guess against real-world numbers pulled
from Google.

## The framework

The coaching prompts are built directly from three sources:

1. **Guesstimation** (Weinstein & Adam) — round aggressively to convenient
   numbers / powers of ten, anchor every guess on something you actually know.
2. **Guesstimation 2.0** (Weinstein & Edwards) — same technique, wider problem set.
3. **Hacking the Case Interview's 5-step framework** — Clarify → Structure
   (top-down/bottom-up, MECE breakdown) → Estimate → Calculate → Sanity-Check.

The app walks you through exactly those five steps. At each step you can ask
for a hint — the AI nudges you (points out gaps, unstated assumptions, MECE
violations) but never hands you the numbers or the answer.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

You'll need, entered in the sidebar at runtime (nothing is hardcoded):

- **Anthropic API key** — for the coaching hints and the final grading.
  Get one at https://console.anthropic.com
- **Google API key** + **Custom Search Engine ID (cx)** — for the final
  verification step, using the Google Custom Search JSON API.
  1. Create a Programmable Search Engine at https://programmablesearchengine.google.com/
     (set it to search the whole web) — this gives you the `cx` ID.
  2. Enable the "Custom Search API" in Google Cloud Console and generate an
     API key: https://console.cloud.google.com/apis/library/customsearch.googleapis.com

The Google step is optional — you can use the app purely as a coaching tool
without it, you just won't get the automated grading at the end.

## How verification works

When you submit a final guess, the app:
1. Searches Google (via the Custom Search API) for the question + "actual number statistic".
2. Sends the top snippets + your guess to Claude.
3. Claude infers the best real-world reference number from the snippets and
   judges your guess by Fermi-problem standards: within ~2-3x (same order of
   magnitude) counts as a **GOOD** estimate, since guesstimates are about
   structured reasoning, not precision.

## Notes

- API keys are only held in Streamlit session memory for the run — nothing is
  persisted to disk.
- If Google search snippets don't clearly contain a number, the verdict will
  come back UNKNOWN rather than guessing.
