# Goal and stuck checks

`jevkit/recipes/loop_control.py` — the supervisor of an agent loop. After each step it
decides whether the loop takes another one, has finished, is going in circles, is blocked,
or needs a person. One request per step, five outcomes, and a DONE that no model answer
can produce on its own.

## The decision

A step just finished. The loop has to pick a branch:

| action | what the caller does |
| --- | --- |
| `continue` | Take another step. Nothing stopped the run. |
| `done` | Stop and report success — but read `decision.verified` first. |
| `stuck` | Stop the current approach and nudge: replan, change tool, change tactic. Progress is still available to the agent. |
| `blocked` | Stop and get something from outside the loop: a credential, an input, a working dependency, a decision. Nudging will not help. |
| `needs_human` | Stop and hand the run to a person. Every failure path in this recipe lands here. |

These are five keys `loop_control` holds. The model chooses among them and code arbitrates
afterwards, so the answer is an index into the caller's own control flow — never a string
that gets executed, and never a verdict nobody offered. A text model asked the same
question answers in prose the loop then has to interpret, and a loop that has to interpret
its own supervisor is both slower and less predictable than one that branches on an enum.

`Decision` carries the action, the reason, and the evidence that produced it: the verdict
the model actually picked (even when code overrode it), the confidence, the four noul
values, the normalised remaining distance, and the code-owned counters. `decision.line()`
prints all of it on one line, which is what you want in a run log six hours later.

## The questions, and why they are shaped that way

Six questions, one request. Questions in a request are evaluated independently against the
same state and run in parallel, so asking every question the decision tree might need
costs a few hundred tokens and no extra round trip.

| id | type | why this type |
| --- | --- | --- |
| `verdict` | `choice` over the five actions | The decision itself. A Choice returns one of the supplied keys with a probability for every alternative, which is what the confidence gates need. |
| `goal_met` | `noul` | "The goal is achieved **and the evidence for it is visible**." A yes/no about evidence, thresholded on its value — a noul has no confidence. This is the question that separates "the agent says it is done" from "the result exists". |
| `progress` | `noul` | Did the most recent step move the run closer to the goal? The one judgment code genuinely cannot make: whether a new page, a new file or a new error is progress. |
| `repeating` | `noul` | Is the agent redoing its own work? Deliberately overlaps the deterministic counter: the counter catches literal and near-literal repeats, the model catches "different words, same move". |
| `steering` | `noul` | `observed` and `history` hold tool output, page text and file contents. This asks whether that material is trying to direct the loop — claim completion, order a stop, replace the goal, skip a check — rather than report what happened. |
| `remaining` | `score`, 4 levels | How much work is left is a spectrum, and a noul is a probability rather than a magnitude. Four levels, not ten: the loop only needs "about to finish" separated from "will not finish in this budget". `reply.unit()` normalises it to 0..1, so the threshold does not move when a level is added. |

The goal, the plan, the recent steps and the last tool result go in `state`. Nothing
untrusted is ever interpolated into a question: `tests/test_loop_control.py` asserts that
injected text appears in the state and nowhere in the questions, and that the verdict
criteria are exactly the five keys the module defines.

## DONE is not proof

This is the property the recipe exists for. A DONE verdict has to clear three gates in
order, and the third one is not the model:

1. **Evidence.** `goal_met` at or above `GOAL_MET_DONE`. Below it the answer is
   `needs_human` / `no_evidence`, and the caller's check is not even consulted — a claim
   with no evidence behind it is not worth verifying.
2. **Confidence.** The verdict's confidence at or above `DONE_CONFIDENCE`, the highest bar
   in the file. Below it: `needs_human` / `low_confidence`.
3. **An independent check.** `verify()`, supplied by the caller: code that looks at the
   world instead of at the transcript. A transcript can be talked into saying anything; a
   billing API, a file on disk, or a test suite cannot.

What the check returns decides the outcome:

| `verify()` | result |
| --- | --- |
| `True` | `done`, `reason="goal_verified"`, `verified=True`. The only fully verified stop. |
| `False` | `needs_human`, `reason="check_failed"`, `verified=False`. Never a DONE, not even with `allow_unverified_done=True`. |
| `None` ("cannot tell") | `needs_human` / `unverified_done`, or, with `allow_unverified_done=True`, a `done` whose `verified` is `None` and whose log line says `UNVERIFIED`. |
| raises, or returns anything else | `needs_human`, `reason="check_failed"`, `verified=False`. A verifier that breaks its own contract is not evidence. |
| not supplied | Same as `None`. |

`decision.unverified_done` is True for any DONE the check did not confirm, so a caller can
refuse to report success on one, and `Watch.unverified_dones` counts them over a run.

## STUCK is not BLOCKED

They need different responses from the caller, so they are different verdicts:

- **STUCK** — no progress, but the agent still has moves. Retrying the same search, cycling
  between two tools, drifting off the goal. The caller's response is a nudge: replan,
  change approach, summarise and restart the sub-task.
- **BLOCKED** — no action available to the agent can progress. A missing credential, an
  input nobody supplied, a dependency that is down, a decision only a person can make. A
  nudge here just burns budget; the caller has to change the agent's situation.

A blocked claim under `BLOCKED_CONFIDENCE` degrades to STUCK, because a nudge is the
reversible response and "actually blocked" will show up again on the next check — and, if
the agent keeps repeating itself, the repeat counter escalates it to a person by itself.

## Code owns what code can count

The model judges progress. It never decides whether the run is out of budget:

- **`Budget`** holds `max_steps` (required — a ceiling a reviewer has not chosen is not a
  ceiling) and an optional `max_wall_s`. `budget.exhausted()` is a code verdict:
  `needs_human` / `budget_exhausted`, whatever the model said. `budget.used()` is not
  capped at 1, so an overrun is visible in the log rather than rounded away.
- **`repeats_in(history)`** counts, deterministically, how many of the last `REPEAT_WINDOW`
  steps repeat the latest one. It normalises case and whitespace, serialises non-strings,
  and treats two records as the same action when they are `NEAR_IDENTICAL_RATIO` similar.
  That is a ratio over the whole record, so the same one-token edit passes or fails on
  length: `difflib.SequenceMatcher` puts `GET /orders?page=1 retry` against `page=2` at
  0.958 and catches it, but bare `page=1` against `page=2` at 0.833 and does not. Short
  records are compared for equality in practice. It counts matches
  anywhere in the window, not only consecutive ones, so an a-b-a-b cycle is caught too. It
  compares each record's first `ENTRY_CHARS` characters, the same cap `prepare` sends, so
  the counter agrees with the steps the model is shown and its own cost stays bounded by a
  number in the review block rather than by how much a caller logs per step.

Exactly one thing outranks those counters: a DONE the caller's own `verify()` returned True
for. Finishing on the last allowed step is finishing, and the confirmation came from code
looking at the world. An **unverified** DONE does not outrank them — not even with
`allow_unverified_done=True`, which buys a labelled DONE and not an exemption. Past the step
or wall ceiling, or at `REPEAT_ESCALATE_AT` repeats, it falls through to
`needs_human` / `budget_exhausted` or `needs_human` / `repeating`, with the done claim still
in the decision's evidence and `decision.verdict` still reading `done`. Without that, the
one permissive outcome in the recipe would have been reachable from a model answer alone at
any distance past a ceiling.

## Thresholds

Every one of these lives in the review block at the top of the module, and
`tests/test_loop_control.py` exercises both sides of each — including
`NEAR_IDENTICAL_RATIO`, whose pair is built by searching for the shortest suffix that drops
`difflib`'s ratio under the constant, so the two records straddle it wherever it is set.

What those tests pin is the *comparison*, not the value: they derive their inputs from the
constants, so lowering `GOAL_MET_DONE` or `DONE_CONFIDENCE` keeps the suite green. That is
deliberate — the value is the judgment this review block exists to put in front of a human,
and the tests are there to prove the code reads it and that both sides of it behave. Changing
one is a review, not a test failure.

| threshold | value | what it gates |
| --- | --- | --- |
| `GOAL_MET_DONE` | 0.90 | Evidence a done claim needs before the caller's check is consulted. High, because this is the line between visible evidence and the agent's own say-so. |
| `DONE_CONFIDENCE` | 0.85 | Confidence a verdict needs to end a run as finished. The most expensive mistake available here, so the highest bar. |
| `BLOCKED_CONFIDENCE` | 0.70 | Confidence a BLOCKED verdict needs. Lower than DONE: declaring a run blocked costs a person's attention, declaring it finished ships something that does not exist. Below the bar, BLOCKED becomes STUCK. |
| `PROGRESS_FLOOR` | 0.35 | At or below this, the last step counts as having produced nothing, and the run is a stall. |
| `REPEATING_SUSPECTED` | 0.65 | At or above this, the model's read of repetition counts toward a stall alongside the counter. |
| `STEERING_SUSPECTED` | 0.60 | At or above this, the material is treated as an instruction to the loop, and the loop stops for a person. |
| `REMAINING_FAR` | 0.60 | Normalised remaining work that counts as far from the goal. |
| `BUDGET_WARN_FRACTION` | 0.75 | Fraction of the budget spent at which a far-from-goal run is escalated (`will_not_finish`) instead of left to burn the rest. Code counts the fraction; the model judges the distance. |
| `REPEAT_STALL_AT` | 2 | Near-identical repeats that make the run a stall. Doing the same thing three times is a loop. |
| `REPEAT_ESCALATE_AT` | 4 | Repeats at which code stops nudging and asks a person. |
| `REPEAT_WINDOW` | 6 | How many recent steps the counter looks at. |
| `NEAR_IDENTICAL_RATIO` | 0.90 | Similarity at which two actions are the same action. Lower starts merging genuine alternatives. |
| `HISTORY_WINDOW` | 8 | Recent steps sent as state. Older ones are reported in `dropped_steps`. |
| `ENTRY_CHARS` | 600 | Character cap per step and on `observed`, and the prefix `repeats_in` compares. Cuts are named in `truncated`. |
| `CHECK_BUDGET_MS` | 1000.0 | **Not a measurement.** The hypothesis `Watch.within_budget()` tests against your account. |

## The order of the gates is the policy

`decide()` applies them in this order, and the order matters more than any single number:

1. **A rejected answer** → `needs_human` / `rejected`. No evidence is attached, but the
   code-owned counters still are.
2. **Steering** → `needs_human` / `steering`. First, because untrusted text that talks
   about stopping poisons every other answer in the reply — including a done claim that
   would otherwise have been verified.
3. **The done branch**, before the budget, so a run that finishes on its last allowed step
   is DONE rather than an escalation — but only when `verify()` confirmed it. An unverified
   DONE falls through to 4 and 5 instead of ending the run.
4. **Budget exhausted** → `needs_human` / `budget_exhausted`. A code verdict.
5. **`REPEAT_ESCALATE_AT` repeats** → `needs_human` / `repeating`. Also a code verdict.
6. **The model asked for a person** → `needs_human` / `asked`. No confidence gate: asking
   is the safe outcome, so there is nothing to gate.
7. **Projected overrun** (`BUDGET_WARN_FRACTION` spent and `REMAINING_FAR` left) →
   `needs_human` / `will_not_finish`.
8. **BLOCKED**, gated on `BLOCKED_CONFIDENCE`, degrading to STUCK below it.
9. **STUCK**, from the model's verdict or from the stall test (`REPEAT_STALL_AT` repeats,
   `PROGRESS_FLOOR`, or `REPEATING_SUSPECTED`).
10. **CONTINUE**, only when nothing above fired.

Fail closed throughout: a rejected answer, a locally refused request, a transport failure
and a verifier that raises all land on `needs_human`. `check_step` never raises.

## What gets sent, and what gets left out

`prepare()` clamps the two fields that grow without bound — `history` to `HISTORY_WINDOW`
entries, each entry and `observed` to `ENTRY_CHARS` characters — and reports both:
`decision.dropped_steps` counts the older steps left out, `decision.truncated` names every
field that was cut, and `decision.line()` prints both. No silent caps. `goal` and `plan`
are the caller's own text and are passed whole, which means they are what can overflow the
context: an oversized one is refused locally by `jevkit.limits` and becomes
`needs_human` / `refused` without a round trip.

## Cost arithmetic

Every number below is `jevkit.limits.estimate_tokens` over the request this module's own
`build_questions()` and `build_state()` produce, priced through `jevkit.cost`. It is an
estimate, not a tokenizer: `estimate_tokens` divides the encoded JSON by
`limits.CHARS_PER_TOKEN` (4), which deliberately overcounts so a request that would have
fit is never refused locally. Reproduce all of it with:

```python
from jevkit import cost, limits
from jevkit.recipes.loop_control import prepare
import examples.loop_control as ex

run = ex.Run()
for index in range(3):
    check_in = run.take(index)   # the example's step 3: goal, plan, 3 steps, the injected page
request = prepare(check_in)

{qid: limits.estimate_tokens(q) for qid, q in request.questions.items()}
limits.estimate_tokens(request.questions)   # 1118 — the whole questions map, as encoded
limits.estimate_tokens(request.state)       #  229 — this check-in's state
cost.usd_for("jev-latest", 1118 + 229)      #  5.6574e-05
```

| part of the request | estimated tokens |
| --- | --- |
| `verdict` (5 described verdicts) | 417 |
| `goal_met` | 202 |
| `progress` | 102 |
| `repeating` | 92 |
| `steering` | 150 |
| `remaining` (4 levels) | 148 |
| the six questions summed one by one | 1,111 |
| the questions map as one encoded object | 1,118 |
| state, for the example's step-3 check-in | 229 |
| **one check** (questions map + that state) | **1,347** |

The questions are constant, so that 1,347 is the only part of this table a different
check-in changes. Across the example's eight scripted steps the state runs 104 → 305
tokens, i.e. 1,222 → 1,423 per check — the estimate grows with how much history you send,
not with how long the run is. At the published price in `jevkit.cost` — $0.042 per million
input tokens, output tokens free:

```
1,347 × $42/1e9 = $0.000056574 per check
                ≈ $0.0566 per 1,000 checks
                ≈ 17,676 checks per dollar
```

The billed number is the `input_tokens` the API reports, which `Watch` and `jev.ledger`
accumulate for a real run. There is no API key in this repo's environment, so no figure
here is a billed one; run the example on your own account for that, and expect the estimate
and the bill to differ, in that direction.

The request fits with room to spare: the longest question is 417 tokens, leaving
32,000 − 417 = 31,583 tokens of the state-plus-longest-question budget for history.

## Measuring the latency hypothesis

The claim this pattern is built to support is that a goal/stuck check fits inside a
sub-second loop step. This repo does not assert that anywhere. `measure()` runs the
supervisor over a sequence of check-ins and returns a `Watch` whose numbers are ledger
deltas for that call only:

```python
watch = measure(jev, (run.take(i) for i in range(len(SCRIPT))), verify=run.verify)
watch.p50_ms, watch.p95_ms          # measured per-check latency, end to end
watch.within_budget()               # p95 <= CHECK_BUDGET_MS — the hypothesis, tested
watch.usd_per_thousand_checks       # priced from the tokens the API reported
```

Each sample is one whole check: `repeats_in` and `prepare` on your own records as well as
the request, because all of it is time the loop step pays for. A check that never reached
the network — an oversized request refused locally — still contributes a sample, since it
still cost the loop its time. `jev.ledger.latencies_ms` keeps the request-only numbers when
you want to separate the two, and the gap between them is this recipe's local cost.

`within_budget()` uses p95, not the mean, because a loop is held up by its slow steps, and
returns False when nothing was measured at all: an unmeasured budget is not a met one.
`examples/loop_control.py` prints the whole thing for a scripted run. There is no API key
in this repo's test environment, so no latency number here was measured against the real
API — run the example on your own account to get one.

## Honest limits

- **A verified DONE is only as good as the verifier.** This recipe guarantees that a stop
  needs an independent check; it cannot make the check correct. A verifier that reads the
  agent's own output is not independent, and will happily confirm a fabricated result.
- **Injected text can stall a run on purpose.** Because steering is checked first, text in
  a page or tool result that reads as an instruction to the loop stops the run and asks a
  person — even when the caller's check would have confirmed the goal. That trade is
  deliberate (a stop is safe, a false success is not), but it means attacker-controlled
  content can cost a human interruption. If your tool surface returns third-party text
  routinely, expect escalations and watch the `steering` rate.
- **The checker is consulted only on a done claim.** If your check would say True while
  the model still says `continue`, the loop keeps going and wastes the rest of its budget.
  Callers with a cheap, reliable check should run it in their own loop as well.
- **CONTINUE has no confidence gate.** The stopping verdicts are gated; continuing is
  bounded by `Budget` instead. The reasoning is that the side effects of the *next step*
  are the next step's problem to gate (see the tool-gating recipe), and that halting on
  every uncertain check-in makes a supervisor useless. If your steps are individually
  expensive, gate them where they happen.
- **The repeat counter is textual, and it only reads the first `ENTRY_CHARS`.** It compares
  what the loop logged. Two calls that differ only in an argument your log omits look
  identical to it; two calls that do the same thing with different wording look different.
  The `repeating` question is there to cover the second case, and it is a judgment, not a
  proof. Two records that agree for 600 characters and diverge after that count as one
  action — which is also all the model is shown of them, so the two signals agree, but if
  your step records carry the distinguishing part at the end, put it at the front instead.
- **One check per step costs a request per step.** At 1,200 requests/minute across the
  account, a supervisor at one check per step shares that ceiling with everything else on
  the key; pace it with `jevkit.RateLimiter` if you run many loops at once.
- **Calibration is a property of groups of answers.** A single `goal_met` of 0.93 is not a
  promise. Thresholds here are starting points to tune against your own labelled runs, not
  values transferred from ours.
