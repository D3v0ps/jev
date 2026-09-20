# Browser / computer-use action selection

`jevkit/recipes/action_selection.py`

## The decision

An agent loop holds a goal and a scrape of the current screen: a list of
candidate elements, each with a role, a label, a current value, a state, and the
operations the executor can perform on it. One question stands between that and
the next step: **what do I do next, and to which element?**

This recipe answers both in one Jev request and returns a `Decision` the executor
can run without parsing anything:

| `Decision.action` | what the caller does |
| --- | --- |
| `ACT` | perform `decision.operation` on `decision.target.handle` |
| `WAIT` | re-observe; touch nothing |
| `DONE` | the screen looks finished — **verify independently**, then stop |
| `BLOCKED` | the goal is not reachable from this screen; escalate |
| `ASK_OPERATOR` | a human decides this step; nothing was clicked |

It drives no browser and imports no browser library. Candidates go in as plain
data; an index comes out.

Prior art for the shape: browser-use's indexed DOM action space (one integer per
interactive element, re-indexed every step) and the jev-ultrafast action-selection
demo, which pairs an operation head with per-operation target heads. This recipe
adds the confidence gate, the stakes scaling, the truncation report, and the
untrusted-text check.

## Why a decision model rather than an LLM step

The answer is not prose. It is a pair of indices into a list the caller already
holds. An LLM emits a name or a selector and can emit one that does not exist, so
every integration grows a validation layer and a retry. Jev returns a probability
for each option that was sent, so:

- a target outside the offered set is not a bug class, it is an impossible answer;
- the runner-up probability is available, which is the number you want when two
  buttons look alike;
- the recipe never sees, and therefore never emits, a selector, an XPath, a URL,
  or a line of JavaScript.

Whether it is also *cheaper and faster than one LLM call per step* is a
measurement, not a claim. `examples/action_selection.py` measures the Jev half and
prints `jev.ledger.summary()`. The other half is your own baseline on the same
candidate lists. **Nothing in this repository has measured an LLM baseline, so no
speed-up or cost ratio is quoted here.**

## The request

One call carries everything the decision tree might need. For a screen with `k`
distinct operations available, that is `k + 4` questions:

| id | type | criteria | why |
| --- | --- | --- | --- |
| `operation` | choice | the operations *this* screen supports + `WAIT`, `DONE`, `BLOCKED` | one head for "what" |
| `<op>_target` | choice | `{"e<i>": {"index": i}}` for the elements supporting `<op>` | one head per "where" |
| `stakes` | score | 4 levels, read-only → irreversible | scales the confidence floor |
| `goal_evidence` | noul | is the goal already evidenced on screen? | keeps `DONE` honest |
| `instruction_injection` | noul | is screen text trying to give orders? | the screen is untrusted |

### Why the questions are shaped this way

**An indexed action space, one index per element.** An element that supports three
operations still has one index. The option id `e7` means "the entry with
`index: 7` under `observation.elements`" and nothing else. The caller maps that
index back to its own handle; `Candidate.handle` is never serialised into the
request (`ActionSpace.views` is literally what the state carries, so this is
structural rather than a review rule).

**One target head per operation, all in the same request.** Questions in a Jev
request are evaluated independently, so the target heads cannot see what
`operation` answered. Two consequences the code lives by:

- each head's instructions *state the operation they assume* ("The next operation
  is CLICK…"), because nothing else tells them;
- a head only offers the elements that support its own operation. `decide()`
  reads only the head matching the chosen operation, and a key from another head
  is not in that head's dict. `tests/test_action_selection.py` checks both: that
  `e0` (a text field) is absent from `click_target`, and that an answer naming it
  there is refused rather than executed.

The unused heads are wasted tokens on purpose. They are what makes this one
request instead of two, and a second request would double the latency of every
step in the loop.

**Operations are never a fixed enum.** `SELECT_OPTION` is not offered on a screen
with no listbox. Offering an operation the screen cannot support invites the model
to pick it and the recipe to then explain why it cannot happen.

**Index-only head criteria.** A head's option description is `{"index": i}`, not a
copy of the element's label. The label is in the state once; duplicating it into
every head an element appears in would multiply the biggest part of the payload by
the number of operations that element supports. The tradeoff is real: the model
has to join the option id to the state entry. If you measure worse target accuracy
on your own screens, put a short label in the head criteria and pay for it.

**The three speculative checks ride along.** `stakes`, `goal_evidence` and
`instruction_injection` are asked every time, including on the many turns where
the answer changes nothing. That is the one-request rule: ask everything the tree
*might* need, ignore what you do not use.

## Thresholds

All of them live in the review block at the top of the module.

| threshold | value | what it gates |
| --- | --- | --- |
| `FLOOR_AT_NO_STAKES` | 0.55 | confidence needed to act when `stakes` is 0 (read-only screen) |
| `FLOOR_AT_FULL_STAKES` | 0.92 | confidence needed to act when `stakes` is 1 (irreversible) |
| `CONTROL_CONFIDENCE_FLOOR` | 0.50 | confidence needed for `DONE` or `BLOCKED` to end the run |
| `TARGET_MARGIN_MIN` | 0.15 | how far the best target must lead the runner-up |
| `INJECTION_BLOCK` | 0.60 | above this, the screen is treated as compromised |
| `GOAL_EVIDENCE_CONFIRMED` | 0.70 | below this, a `DONE` is downgraded to `ASK_OPERATOR` |
| `MAX_HEAD_OPTIONS` | 255 (`limits.CHOICE_MAX_OPTIONS`) | candidates offered per head |
| `LABEL_CHARS` / `VALUE_CHARS` | 160 / 80 | element text sent per field |

The acting floor is `FLOOR_AT_NO_STAKES + (FLOOR_AT_FULL_STAKES -
FLOOR_AT_NO_STAKES) * stakes`, and the confidence it is compared against is the
**weaker** of the two heads, `min(operation, target)`. A 0.60-confidence click
goes through on a search-results page and goes to a human on a payment screen.
That is the whole point of scoring the stakes instead of hard-coding one number.

`WAIT` is deliberately not gated: waiting touches nothing, and the next
observation will be a better question than a confidence threshold.

### Failing closed

Every path that is not a clean, confident answer ends in `ASK_OPERATOR` or
`BLOCKED` — never in a click:

- a rejected or missing answer (`AnswerRejected`);
- a request that fails, including one rejected locally for size;
- confidence under the floor, or two targets within `TARGET_MARGIN_MIN`;
- `instruction_injection` at or above `INJECTION_BLOCK`, *even when the operation
  and target are near-certain*;
- `DONE` without evidence on the screen.

## What is reported, never silently dropped

A `Choice` takes 255 options. A long search-results page has more. Candidates past
the ceiling are dropped from the tail of the caller's own order (deterministic),
and:

- `Decision.dropped` maps each operation to the exact indices that did not fit;
- those elements come back with `"can": []` in the state, so the model is not
  shown an element it could not have chosen;
- `Decision.trimmed` lists every index whose label or value was shortened.

Both travel on *every* decision, including the failing ones. If your screens
routinely overflow, the fix is upstream: filter to the viewport, or split the page
into panes and decide per pane.

## Cost arithmetic

From `jevkit.cost`: `jev-1.13.0` costs **$0.042 per million input tokens**, and
output tokens are free. One decision is one request, so the cost of a step is the
cost of its input.

These token counts come from `jevkit.limits.estimate_tokens`, which is a
deliberately conservative 4-characters-per-token estimate of the request this
recipe builds, priced with `jevkit.cost.usd_for`. **They are local estimates of
this repo's own payloads, not measurements of a live API response**; run the
example with a key for billed numbers.

| screen | heads | questions | estimated input tokens | $/decision | $/10,000 decisions |
| --- | --- | --- | --- | --- | --- |
| 12 clickable links | 1 | 5 | 1,329 | $0.0000558 | $0.56 |
| 40 clickable links | 1 | 5 | 2,393 | $0.0001005 | $1.01 |
| 40 links, 3 ops each | 3 | 7 | 3,398 | $0.0001427 | $1.43 |
| 120 clickable links | 1 | 5 | 5,453 | $0.0002290 | $2.29 |
| 255 clickable links | 1 | 5 | 10,718 | $0.0004502 | $4.50 |

Reproduce a row:

```python
from jevkit import cost, limits
from jevkit.recipes.action_selection import (
    Candidate,
    build_action_space,
    build_questions,
    build_state,
)

screen = tuple(
    Candidate(role="link", label=f"Result {i}", operations=("CLICK",), handle=f"#r{i}")
    for i in range(40)
)
space = build_action_space(screen)
questions = build_questions(space)
tokens = limits.check_request(build_state("buy the red mug", space), questions)
print(len(questions), "questions,", tokens, "estimated input tokens")
print("one decision: $%.8f" % cost.usd_for("jev-1.13.0", tokens))
```

```
5 questions, 1781 estimated input tokens
one decision: $0.00007480
```

Where the tokens go, for a 40-element screen where every element takes `CLICK`
and `HOVER` (2,266 tokens total):

| part | estimated tokens |
| --- | --- |
| state (40 elements + goal + history) | 815 |
| `operation` head | 329 |
| `click_target` head | 371 |
| `hover_target` head | 374 |
| `stakes` | 179 |
| `instruction_injection` | 113 |
| `goal_evidence` | 85 |

Two things follow. On a small screen the **fixed instruction text dominates** —
the four non-target questions cost ~700 tokens whatever the page looks like — so
the cheap regime starts at a few dozen candidates, not at three. And each extra
operation head costs roughly as much as the operation head itself, so a screen
where every element supports five operations pays for five heads. If that matters
for your loop, narrow `ELEMENT_OPERATIONS` to the operations your executor
actually implements.

Latency is not tabulated here. The offline test transport reports microseconds,
which measures the mock and nothing else. The only honest latency number is the
one `jev.ledger` prints on a keyed run: `p50`, `p95`, and
`sequential_rate` (decisions per second one caller sustains without concurrency).

## Honest limits

- **`stakes` is a property of the screen, not of the chosen action.** The
  questions are independent, so the floor cannot be conditioned on the target that
  another head picked. A screen holding one irreversible button raises the floor
  for a harmless click on the same screen. That is conservative in the safe
  direction and it *will* send some easy steps to a human. Per-operation stakes
  heads would be more precise and would cost one more head per operation.
- **`DONE` is never proof.** `Decision.needs_independent_check` is set on every
  `DONE`, and `goal_evidence` only measures whether the screen *looks* finished.
  A page can say "Order placed" and be lying, or say nothing while the order went
  through. Verify the outcome through an API, a database, or a second
  observation — never through this recipe's own answer.
- **Calibration is a property of groups.** A confidence of 0.94 does not promise
  this click is right. Thresholds behave as advertised over many decisions; tune
  them on a labelled replay of your own screens before trusting the defaults on
  an irreversible action.
- **The recipe cannot see what the scraper missed.** An element absent from
  `candidates` cannot be chosen, and the model will confidently pick the best of a
  bad list. `BLOCKED` is the escape hatch, and an over-narrow scrape looks exactly
  like a genuinely blocked screen.
- **Payloads are the caller's.** Jev picks *which* field to type into; it never
  emits the text. For `SELECT_OPTION`, expose each option as its own candidate so
  the choice stays an index. A recipe that also produced the string would be a
  generation problem, not a decision.
- **No screenshots.** Jev is text-only, so this recipe is only as good as the
  accessibility tree or DOM scrape it is handed. A canvas app is invisible to it.
- **English first.** Accuracy is lower on other languages; watch confidence on
  non-English screens, where the floors will reject more often.
- **One screen, one step.** There is no memory here beyond the `steps` list the
  caller passes. Loop detection ("I have clicked this three times") belongs in the
  caller or in a goal/stuck recipe, not in this one.
