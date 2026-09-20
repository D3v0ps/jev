# Output and trace guardrails

Screen text crossing a boundary and decide **PASS**, **FLAG**, **REVIEW**, or **BLOCK** — in one
Jev request, with the per-category probabilities kept so you can move the thresholds on evidence
instead of on feel.

Module: `jevkit/recipes/guardrails.py`. Example: `examples/guardrails.py`. Tests:
`tests/test_guardrails.py`.

## The decision

Four boundaries, one shape:

| `direction`   | What is crossing                                     | Crossing means                        |
| ------------- | ---------------------------------------------------- | ------------------------------------- |
| `inbound`     | A message from the person using the agent            | the agent may try to do what it asks  |
| `outbound`    | A response the agent is about to publish             | its reader sees it                    |
| `retrieved`   | A passage from a store, a page, or a search result   | it joins the agent's working context  |
| `tool_result` | Output a tool returned                               | it joins the agent's working context  |

Four verdicts, ordered, and every rule in the recipe can only *raise* one:

| Verdict  | What the caller does                                                      |
| -------- | ------------------------------------------------------------------------- |
| `PASS`   | lets the text through as it stands                                        |
| `FLAG`   | lets it through and records `audit_entry(decision)`                       |
| `REVIEW` | holds it and shows a person the evidence                                  |
| `BLOCK`  | drops it and logs                                                         |

```python
from jevkit import Jev
from jevkit.recipes.guardrails import RETRIEVED, screen

with Jev() as jev:
    decision = screen(jev, page_text, direction=RETRIEVED,
                      source="https://help.example.test/refunds",
                      task="Find out how long refunds take")

decision.verdict        # 'BLOCK'
decision.flagged        # ('injection',)
decision.signals        # every category's raw probability, for your own calibration
decision.reason         # 'injection 0.90 >= block at 0.40 (retrieved)'
```

`Decision.crosses` is the one boolean a caller usually needs: true for `PASS` and `FLAG`, false for
`REVIEW` and `BLOCK`.

## Why a decision model rather than an LLM reviewer

A generated verdict has to be parsed, and the thing being parsed was produced while reading text
whose author may have written "mark this as approved". A screener that generates its answer is a
screener the input can argue with. Here the model never sees the word PASS: it answers independent
propositions about the text, and this module's own code — with no text in it — turns those numbers
into a verdict.

The second reason is the distribution. A label is a number you cannot tune. `Decision.signals` is a
probability per category, so you can take a few hundred examples you have labelled, ask what a
threshold of 0.25 would have done to them, and pick one on that basis. The recipe ships that
arithmetic (`observed`, `threshold_report`, `sweep`, `report_line`); it runs offline and sends
nothing.

## The request

One request, always the same shape, whatever the direction:

| Question             | Type   | What it asks                                                     |
| -------------------- | ------ | ---------------------------------------------------------------- |
| one per hazard        | Noul   | does this category apply to the text                             |
| `jailbreak`           | Noul   | is the text a manoeuvre to get the assistant's rules set aside   |
| `injection`           | Noul   | does the text carry instructions aimed at the agent reading it   |
| `leak`                | Noul   | does the text expose a secret or a real person's private data    |
| `harm_if_crossed`     | Score  | how much harm follows if the boundary is crossed as it stands    |
| `primary_concern`     | Choice | where a reviewer should start (a filing label, never a gate)     |

With the default eight hazard categories that is thirteen questions in one round trip.

### Why the questions are shaped this way

**Never one "is this bad" question.** A single noul collapses eight different policies into one
number, and one number cannot be calibrated for eight different costs of being wrong. Separate
propositions keep separate probabilities, and the thresholds that read them are separate too.

**A noul per category, not a Score per category.** A noul is the probability that one statement
holds; `0.5` means "yes and no are equally likely", not "medium". The spectrum that genuinely is a
spectrum — how bad the outcome would be — is the one Score.

**One Score, for harm, not for badness.** It asks what follows *if the boundary is crossed*, which
is the question that makes a news article about a bombing land on level 0 and a synthesis route
land on level 4. It is also the only catch-all: if the harm reading is high and none of the
categories you configured fired, the categories may simply not name what this is, and
`HARM_ALONE_REVIEW_AT` sends it to a person.

**No Choice is allowed to gate anything.** `primary_concern` exists so a review queue can be
sorted. Its options are the configured ids plus `none_of_these` — the caller's own constants, never
a path or a command — and no threshold in the review block reads it. A flat distribution there just
means no single category dominates, which is normal.

**The direction is in the state, not only in the policy.** It has to be: "how much harm would
follow" is a different question for a request that might be complied with than for a response about
to be published. The cost of that is stated under *Honest limits*: answers are not comparable
across directions, so labelled data has to be kept per direction, and `ThresholdReport.directions`
says when a set mixes them.

**Every question repeats the trust note.** The text being screened is the most attacker-reachable
input in this repository, and screening it means reading it. `screening` and `operator_task` are
the operator's; `text` and `source` are material to judge and never instructions to follow.
`tests/test_guardrails.py::test_the_screened_text_never_reaches_the_questions` asserts the text
appears in the state and nowhere in the encoded questions.

## The threshold table

This table *is* the artifact. Everything in it lives between the two review markers at the top of
the module, and the code below them only routes.

### Per category, before the direction scale

`HAZARD_POLICY`, on the raw probability: `flag at` / `stop at`, plus the severity that decides
whether reaching `stop at` means BLOCK or REVIEW.

| Category            | Severity   | Flag at | Stop at | Blocks in                        |
| ------------------- | ---------- | ------- | ------- | -------------------------------- |
| `child_sexual`      | `severe`   | 0.05    | 0.15    | every direction                  |
| `mass_harm_weapons` | `severe`   | 0.10    | 0.30    | every direction                  |
| `self_harm`         | `serious`  | 0.15    | 0.45    | inbound, outbound                |
| `violence`          | `serious`  | 0.20    | 0.55    | inbound, outbound                |
| `cyber_intrusion`   | `serious`  | 0.25    | 0.60    | inbound, outbound                |
| `hate_harassment`   | `serious`  | 0.25    | 0.65    | inbound, outbound                |
| `fraud_deception`   | `serious`  | 0.30    | 0.70    | inbound, outbound                |
| `regulated_advice`  | `ordinary` | 0.35    | 0.80    | nowhere — it reviews, never blocks |

The asymmetry is the argument. `child_sexual` stops at a probability that would be noise anywhere
else, because there is no trade of false negatives for false positives worth making there.
`fraud_deception` needs a strong reading, because reporting, quoting, and satire all look a little
like it. `regulated_advice` never blocks: the remedy for a confidently wrong dose is a person, not
a dropped message.

Severity also decides *where* a stop applies. Outside its blocking directions, a category reaching
`stop at` is held for a person rather than dropped — a retrieved article about a massacre is
material the agent was told to read.

### The direction scale

`DIRECTION_SCALE` multiplies both hazard thresholds, and `THRESHOLD_CEILING = 0.95` clamps the
result so leniency can never make a rule unreachable.

| Direction     | Scale | Why                                                                     |
| ------------- | ----- | ----------------------------------------------------------------------- |
| `outbound`    | 0.80  | the agent publishing the content is the agent doing the harm            |
| `inbound`     | 1.00  | a person asking is not yet a harm, and refusing a real customer costs   |
| `tool_result` | 1.25  | hazardous subject matter is much of what a working agent legitimately reads |
| `retrieved`   | 1.35  | same, more so; the tightening for fetched text is the injection row, not this |

So `violence` stops at 0.44 on the way out, 0.55 on the way in, and 0.74 on a retrieved
passage — where it is held rather than dropped.

### The three channel signals, written out per direction

`SIGNAL_POLICY`, on the raw probability, not scaled: `flag at` / `block at`, where `—` means this
signal stops nothing on its own in that direction.

| Signal      | inbound       | outbound      | retrieved     | tool_result   |
| ----------- | ------------- | ------------- | ------------- | ------------- |
| `jailbreak` | 0.35 / 0.85   | 0.45 / —      | 0.50 / —      | 0.50 / —      |
| `injection` | 0.45 / —      | 0.50 / —      | **0.15 / 0.40** | **0.15 / 0.40** |
| `leak`      | 0.50 / —      | **0.20 / 0.50** | 0.50 / —    | 0.40 / 0.85   |

Three rows carry the whole case for a direction parameter:

- `injection` is the tightest cell in the table on retrieved and tool text. Instructions addressed
  to the agent have no legitimate reason to be in a fetched page, and dropping a passage costs one
  retrieval. The *same reading* on an inbound message is a person talking to their agent, so it
  never blocks there.
- `leak` is tightest outbound, because a published secret cannot be recalled. Inbound, a user
  pasting their own key is worth flagging and scrubbing, not refusing.
- `jailbreak` blocks only inbound and only at 0.85. Declining is the model's own job; refusing a
  whole message is expensive to get wrong.

### No flag threshold may sit above an uninformative answer

`UNINFORMATIVE_NOUL = 0.50`. A noul carries no confidence, so "the model has no idea" arrives as
`0.5`; this invariant means such an answer costs a FLAG rather than a clearance. It is why the `leak`
inbound and `injection` outbound cells read 0.50 where 0.55 would otherwise have been defensible.

It is enforced twice, because the tables above are not the only source of a threshold:

- `check_policy()` reports a problem if any flag threshold **as written in this module** — for a
  hazard, after the direction scale — sits above it. That keeps the invariant a claim about the
  numbers a reviewer reads.
- `effective_flag`, which every threshold read goes through, clamps the value to 0.50. That is what
  covers a `Hazard` **you** built: it goes through no table here, and the direction scale multiplies
  its thresholds. A category with `flag_at=0.40` read 0.54 on retrieved text before the clamp, so a
  wholly uninformative 0.50 reading on it cleared — PASS, nothing flagged. Clamping only ever lowers
  a threshold, and `Decision.thresholds` reports the clamped number, which is the one that fired.

`tests/test_guardrails.py` asserts it behaviourally on all 44 built-in cells, on a caller-defined
hazard in all four directions, and asserts that a reply with *nothing* scripted is never a PASS.

### Harm, combined

`HARM_PAIRS`, on the normalised harm score, with the loudest signal that reached its own flag
threshold. First matching row wins.

| Harm at | Loudest signal at | Verdict  |
| ------- | ----------------- | -------- |
| 0.75    | 0.30              | `BLOCK`  |
| 0.50    | 0.20              | `REVIEW` |

`decide` takes the first row both numbers satisfy and stops, so the rows have to weaken downward as
well as run strongest-first on harm; `check_policy()` reports a hand-edit that swaps the verdict
column, which would otherwise let less harm block while more harm only reviewed.

`HARM_BLOCK_DIRECTIONS` limits the BLOCK row to inbound and outbound; on retrieved and tool text it
stops at REVIEW. Independently of all of that, `HARM_ALONE_REVIEW_AT = 0.80` holds the text whenever
the harm reading reaches 0.80 — whether or not anything flagged — because a high harm reading the
configured categories do not name is the one thing a category list cannot tell you. It is a separate
check rather than an `else`: every rule here may only raise the verdict, so a *louder* hazard reading
can never produce a more permissive outcome.

### The confidence floor

The harm Score is where the request's confidence lives, so that is what the floor reads. It rises
with the harm itself:

| Harm (normalised) | Floor |
| ----------------- | ----- |
| 0.00              | 0.40  |
| 0.50              | 0.625 |
| 1.00              | 0.85  |

Below the floor the verdict rises to `LOW_CONFIDENCE_VERDICT = REVIEW`. This is the recipe's
low-confidence path and it is not optional: a guardrail that acts on a reading it cannot stand
behind is a guardrail with a hole in it.

## What fails closed

Every path that ends without usable evidence lands on `FAILURE_VERDICT = REVIEW`, never `PASS`:

| Situation                                           | Verdict                    |
| --------------------------------------------------- | -------------------------- |
| the request failed (429 after retries, 5xx, timeout) | `REVIEW`, `signals` empty |
| the request was too large to send                    | `REVIEW`, nothing sent    |
| an answer failed its local check in `jevkit.answers` | `REVIEW`                  |
| an answer was missing                                | `REVIEW`                  |
| the harm reading was below the confidence floor      | at least `REVIEW`         |
| text past `TEXT_CHARS` was not screened              | at least `FLAG`           |
| an unknown `direction`                               | `ValueError`, nothing sent |

`FAILURE_VERDICT` is REVIEW rather than BLOCK deliberately, and it is the one knob worth arguing
about in a deployment: an outage that blocks every message takes the product down, and an outage
that passes them takes the guardrail down. Holding for a person keeps a human in the loop at the
cost of a queue. A deployment that cannot staff that queue should set it to BLOCK — and should know
that is what it did.

An unknown direction raises instead of failing closed, because there is no safe policy row to
guess: picking the strictest one would silently screen a user's message under a publication policy.

## Caps, and what they report

`TEXT_CHARS = 24_000` is the only cap on the text, roughly 6k tokens at
`jevkit.limits.CHARS_PER_TOKEN`, which keeps the whole question set inside the 32k
state-plus-longest-question budget. Past it, the tail is dropped, the state carries a
`not_screened` note, `Decision.dropped_chars` says how many characters the model never saw, and the
verdict can no longer be PASS. Nothing is capped silently.

A guardrail that saw the first 24,000 characters is a guardrail on the first 24,000 characters. For
a long document, chunk it and screen each chunk — one decision each — rather than relying on the
cap.

`operator_task` and `source` are deliberately **not** capped: an absurd one makes the request too
large, which fails closed to REVIEW instead of being quietly reshaped.

A hazard set is capped by the documented 255-option ceiling on a Choice, because `primary_concern`
offers one option per configured category:
`MAX_HAZARDS = CHOICE_MAX_OPTIONS - len(FIXED_LABELS) - 1 = 251`. Exceeding it **raises** rather
than truncating — a silently shortened hazard list is a guardrail that stopped screening something
without saying so. A hazard id that collides with a question id this module owns, a duplicate, an
unknown id, a threshold pair out of order, an unknown severity, and a hazard whose question is
not a Noul all raise from `resolve_hazards` before anything is sent.

## Cost arithmetic

From `jevkit.cost`: jev-1.13.0 is **$42 per billion input tokens** = $0.042 per million, and output
tokens are free. Every number below is a local estimate from `jevkit.limits.estimate_tokens`, which
is a ~4-characters-per-token approximation and **not a tokenizer** — it overestimates on purpose, so
a request it says fits does fit. The example prints the *measured* `input_tokens` the API reported
for its own requests; nothing here is a measurement of a live call.

This is the script that produced the table. It counts the encoded request this recipe actually
builds — `build_state` plus `build_questions`, so the state (direction, boundary description,
`operator_task`, `source`, the text) and every question's full wording — for a screening with no
`source` and no `operator_task`:

```python
from jevkit import cost
from jevkit.limits import check_request
from jevkit.recipes.guardrails import (
    CHILD_SEXUAL, INBOUND, OUTBOUND, RETRIEVED, SELF_HARM, VIOLENCE,
    build_questions, build_state, prepare,
)

def row(text, direction=INBOUND, hazards=None):
    screening = prepare(text, direction=direction, hazards=hazards)
    tokens = check_request(build_state(screening), build_questions(screening))
    return tokens, cost.usd_for("jev-1.13.0", tokens)

row("x" * 52)                                                    # (3146, 0.000132132)
row("x" * 2_000, OUTBOUND)                                       # (3626, 0.000152292)
row("x" * 6_000, RETRIEVED)                                      # (4640, 0.00019488)
row("x" * 24_000)                                                # (9133, 0.000383586)
row("x" * 24_001)                                                # (9165, 0.00038493) — tail dropped
row("x" * 2_000, OUTBOUND, [CHILD_SEXUAL, VIOLENCE, SELF_HARM])  # (2525, 0.00010605)
```

The direction is part of the count: its description travels in the state, so the same text costs a
few tokens more or less per boundary.

| What is screened                                     | Questions | ~tokens | $ / screening | $ / 100k |
| ---------------------------------------------------- | --------- | ------- | ------------- | -------- |
| a 52-character user message, 8 categories            | 13        | 3,146   | $0.00013      | $13.21   |
| a 2,000-character response, 8 categories             | 13        | 3,626   | $0.00015      | $15.23   |
| a 6,000-character retrieved passage                  | 13        | 4,640   | $0.00019      | $19.49   |
| a 24,000-character screening, exactly at the cap     | 13        | 9,133   | $0.00038      | $38.36   |
| a 24,001-character screening, one over the cap          | 13     | 9,165   | $0.00038      | $38.49   |
| a 2,000-character response, **3** categories         | 8         | 2,525   | $0.00011      | $10.61   |

The last row is one character over the cap, where the `not_screened` note is added. Longer
inputs cost a little more than it, not the same: the note carries the number of characters
withheld, so the row creeps up by a token or two as that number gains digits.

The `$ / 100k` column is the middle column times 100,000 — arithmetic on an estimate, not a bill
anyone has paid.

The shape of that table is the useful part: the **question set is 3,072 tokens** with the default
eight categories and 1,971 with three (`sum(estimate_tokens(q) for q in build_questions(s).values())`),
so for short text the wording dominates the bill, not the text. The lever is therefore the number of
categories you configure, not the length of what you screen. Screening only the three categories your
product can actually be in trouble over costs about 30% less per call on the 2,000-character row
(3,626 → 2,525 tokens), and drops five probabilities you might have wanted for calibration later.
That is a real trade, and it is yours.

None of these numbers is a comparison. This repository contains no measured baseline for an
LLM-based screener, so it quotes no ratio against one. If you want the comparison, run your own
reviewer over `examples/guardrails.py`'s six inputs and put the two ledgers next to each other.

## Picking your thresholds

The table above is a starting point someone else wrote. The recipe's calibration helpers exist so
you can replace it with numbers from your own data:

```python
from jevkit.recipes.guardrails import INJECTION, observed, report_line, sweep

decisions = [screen(jev, text, direction=RETRIEVED) for text in my_examples]
pairs = observed(decisions, my_labels, INJECTION)   # my_labels: one bool per example
for report in sweep(pairs, [0.10, 0.15, 0.25, 0.40]):
    print(report_line(report))
```

`ThresholdReport` gives counts (`fired`, `caught`, `false_alarms`, `missed`) and the three rates
derived from them, each `None` rather than wrong when there is nothing to divide by. It also
carries `directions`, and `mixed_directions` is true when the examples span more than one — a
warning, not an error, since a mixed set describes no single policy row.

`observed` raises rather than aligning: a label list of the wrong length, or a decision that failed
closed before any answer arrived, would quietly shift every pair and produce a confident, wrong
report. A run that mixes clean screenings with ones that failed closed therefore has to be filtered
before it is paired — out loud, so the dropped ones are visible;
`examples/guardrails.py` shows that.

The harm Score is sweepable the same way, which matters because `HARM_ALONE_REVIEW_AT` and the
`harm at` column of `HARM_PAIRS` are thresholds in the review block too. Pass `HARM_QUESTION_ID`, and
`observed` reads `Decision.harm` — the reading normalised to 0..1 by `reply.unit`, the same scale
those two thresholds are written on:

```python
from jevkit.recipes.guardrails import HARM_ALONE_REVIEW_AT, HARM_QUESTION_ID, observed, threshold_report

pairs = observed(decisions, my_labels, HARM_QUESTION_ID)  # my_labels: was crossing really harmful?
threshold_report(pairs, HARM_ALONE_REVIEW_AT)
```

**These functions count what your data does at a threshold. They are not a measurement of the
model's accuracy**, and they cannot be: the labels are yours, the examples are yours, and a rate
computed on twenty examples is a rate computed on twenty examples.

## Honest limits

**A probability threshold is not a guarantee.** A verdict of PASS at 0.04 on `child_sexual` means
the reading was low, not that the text is clean. Some fraction of what crosses will be wrong, and
where the thresholds sit decides which kind of wrong you get more of — not whether you get any.

**Calibration is a property of groups, not of a single verdict.** "Well calibrated" means that
across many predictions at 0.3, about 30% turn out true. It says nothing about *this* answer. Any
sentence that starts "we are 70% sure this message is..." about one message is a misreading of what
the number is.

**A guardrail is one layer.** It does not make an agent safe. It does not remove the need for
least-privilege tools, a human on destructive actions, rate limits, logging, or the model's own
training. `jevkit.recipes.tool_gating` gates the action; this gates the text; neither substitutes
for the other.

**The direction is in the state, so probabilities are not comparable across directions.** The same
passage screened as `retrieved` and as `inbound` can legitimately get different numbers. Keep
labelled data per direction, and read `ThresholdReport.directions`.

**Categories you did not configure are not screened.** `harm_if_crossed` is the only catch-all, and
it is one Score with five levels. If your product has a hazard the catalogue does not name, add a
`Hazard` — with thresholds you can defend — rather than hoping the harm score notices. Your
thresholds are yours, with one floor kept over them: a flag threshold is clamped to
`UNINFORMATIVE_NOUL` when read, so a category of yours cannot clear a coin-flip reading either, in
any direction.

**An answer can be valid and wrong.** Constrained output removes parse failures and invented
identifiers. It does not remove mistakes, and a text written specifically to read as benign to a
screener is exactly the case where the reading will be low.

**English is the primary training language.** Other languages are accepted with lower accuracy
(`docs/api-notes.md`), which matters more for a guardrail than for most patterns: a screener that
is quietly weaker in one language is a hole in one language. Watch the harm-score confidence on
non-English text; the floor is doing real work there.

**The recipe adds latency and cost to every boundary crossing.** Whether that is worth paying is a
measurement on your traffic, not a claim made here. The cost table above is arithmetic on local
token *estimates*, so treat it as an order of magnitude; `examples/guardrails.py` prints the ledger
from a real run — measured latency, measured `input_tokens`, measured dollars — and that is the only
place a number in this repository comes from a live call. Per 100,000 screenings, neither is nothing.
