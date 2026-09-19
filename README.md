# jevkit

Ten patterns that put [TypeSafe's Jev](https://docs.typesafe.ai/) where it belongs: in the
places an agent has to **decide** something, not write something.

Jev is a System One model. You send it a `state` and a map of typed questions; it returns one
typed answer per question, drawn from options you supplied, with a calibrated probability for
each. It never generates text, so there is nothing to parse, and it cannot name a tool, file, or
selector you did not offer it. What it can do is be wrong — which is why every pattern here gates
its side effects on confidence and fails toward the safe action.

```python
from typesafe_sdk import Choice, Noul, Score

from jevkit import Jev

with Jev() as jev:                                  # reads TYPESAFE_API_KEY
    reply = jev.ask(
        {"ticket": "Help! My payouts have been failing for 3 days."},
        {
            "team": Choice(
                instructions="Which team should handle `ticket`?",
                criteria={"billing": "Payments, invoicing, refunds",
                          "technical": "Bugs, outages, integrations"},
            ),
            "urgent": Noul(instructions="Does `ticket` convey urgency?"),
            "frustration": Score(instructions="How frustrated is the customer?",
                                 criteria=["calm", "frustrated", "very angry"]),
        },
    )

reply.picked("team")        # 'technical' — always one of the keys you passed
reply.confidence("team")    # 0.82 — threshold this before you act on it
reply.noul("urgent")        # 0.92 — a probability, not a magnitude
reply.unit("frustration")   # 0.8 — the Score normalised to 0..1 for weighting
print(jev.ledger.summary()) # 1 requests · 312 input tokens · $0.000013 · p50 ...
```

Three questions, one request, one round trip. That last line is the point of the ledger: every
number this repo reports about speed or cost is measured at runtime, not written into a doc.

## Install

```bash
uv sync                      # or: pip install -e '.[dev]'
cp .env.example .env         # then put a key from console.typesafe.ai/keys in it
```

Python 3.10+. One runtime dependency, the official `typesafe-sdk`.

## The ten patterns

Each lives in its own module under `jevkit/recipes/`, with a doc in `docs/`, a runnable example in
`examples/`, and offline tests in `tests/`.

<!-- RECIPE TABLE -->

## What every recipe looks like

Open any recipe and the first thing under the docstring is the block that matters:

```python
# --- questions and thresholds (review this block) -------------------------
VERDICTS = {"allow": "...", "confirm": "...", "block": "..."}
BLOCK_ABOVE = 0.80          # irreversible and probable enough to stop outright
CONFIRM_ABOVE = 0.35        # ask a human first
# --- end of review block --------------------------------------------------
```

Everything a reviewer should argue about — the wording of each question, every threshold — sits
there. The code below it only routes on what comes back. `scripts/check_review_block.py` enforces
this, and `tests/test_conventions.py` runs it in CI, because the convention is worthless if it
decays.

Three more rules hold throughout, and they are the ones that make these patterns safe rather than
merely fast:

- **One request per decision.** Questions in a request run in parallel against one state, so asking
  a question you might not need costs a few tokens and no latency. Ask the whole decision tree at
  once and let code ignore the irrelevant answers.
- **Confidence gates side effects, scaled to the stakes.** A reversible action and a destructive one
  never share a threshold, and every recipe has an explicit path for "the model is not sure".
- **Fail closed.** An exception, a malformed answer, or a missing one produces the conservative
  outcome — block, escalate, keep, ask a human — never the permissive one.

## Honest claims

The patterns here are motivated by claims about cost and speed. None of those claims are asserted
in this codebase. Instead:

- `jevkit.cost` holds the published price in one table, and computes dollars from the
  `input_tokens` the API actually reported.
- `jevkit.ledger.Ledger` accumulates tokens, dollars, p50/p95 latency, and the sequential decision
  rate for a run. Examples print it.
- Where a pattern's value depends on your data — reranking accuracy, guardrail thresholds, whether
  routing pays for itself at all — the recipe ships a helper that measures it on *your* labelled
  examples and prices, and can tell you the answer is no.

Calibration is a property of groups of predictions, not a promise about any single answer. A
constrained answer removes parse failures and invented identifiers; it does not remove mistakes.

## Checks

```bash
.venv/bin/ruff check .
.venv/bin/python -m pytest -q
.venv/bin/python scripts/check_review_block.py
```

The test suite runs entirely offline: `jevkit.testing` scripts answers over a mock transport
through the real SDK, so tests exercise real request encoding and real answer validation without a
key and without spending anything. **No test in this repo may call a paid API.** You can use the
same harness for your own recipes:

```python
from jevkit.testing import fake_jev

jev, calls = fake_jev({"team": "billing", "urgent": 0.9})
```

## Reading further

- `docs/api-notes.md` — the complete API surface this repo is written against, with the limits and
  prices, verified against the docs and `typesafe-sdk` 0.7.0.
- `AGENTS.md` — the contract a new recipe has to satisfy.
- [TypeSafe docs](https://docs.typesafe.ai/), the [cookbooks](https://docs.typesafe.ai/cookbooks),
  and the [agent skill](https://docs.typesafe.ai/agent-skill) if you want your coding agent to know
  this API too.
- [browser-use/jev-ultrafast](https://github.com/browser-use/jev-ultrafast) — prior art for the
  action-selection pattern.

MIT licensed.
