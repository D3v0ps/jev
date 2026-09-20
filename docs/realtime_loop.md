# Real-time control loop

`jevkit.recipes.realtime_loop` — pick the next move in a loop that ticks many times
per second, with a deadline, a staleness check, and a safe default.

## The decision

Each tick the caller has a world observation and a set of moves that are legal *right
now* — a dynamic subset of a fixed move enum. The decision is: **which of those moves
does the agent make in this tick?**

The answer is one of the caller's own move ids plus a distribution over the
alternatives. Nothing is parsed, nothing is named: `offer(catalogue, legal)` intersects
the caller's fixed move enum with the legal set, so an illegal move is not an option in
the request and cannot come back from it. The reply is an index into constants the
caller already holds.

Set `Tick.catalogue` to that fixed enum and `next_move` does the intersection itself,
which is where the guarantee belongs: descriptions and option order then come from the
catalogue rather than from `legal`, an id in `legal` that the catalogue does not hold
holds the tick (`reason="failed"`, no request sent) instead of being offered, and a
`safe_default` outside the catalogue raises — a hold that executes an unknown move is
not a hold. Leaving `catalogue` unset keeps the older behaviour, where `legal` is trusted
because the caller already called `offer` itself.

Three things a loop needs that a text model does not give you:

1. **A deadline.** `next_move` wraps the request in `asyncio.wait_for(budget_ms)`. A
   late answer is cancelled in flight, the tick takes the caller's safe default, and
   the miss is counted. The loop never blocks past its deadline and never acts on an
   answer that arrived after it.
2. **A staleness check.** The world moves while the request is in flight. The caller
   passes a `fingerprint` in, and a `fingerprint_now` callable that is read again when
   the answer lands. If it changed, the decision is dropped. This is the bug naive
   versions of this loop ship: a confident, correct decision about a world that no
   longer exists. It is a tested contract here, not a nicety — and `fingerprint_now`
   has **no default**, so it cannot be lost by forgetting a keyword argument. A caller
   who has genuinely frozen the world for the tick passes `fingerprint_now=None`; that
   opt-out is visible at the call site, every decision made under it carries
   `staleness_checked=False`, `LoopReport.unchecked_staleness` counts them, and
   `sustains()` refuses such a run, because `stale_drops == 0` means nothing when the
   check never ran.
3. **Fail closed.** Rejected answer, refused request (too large), transport failure,
   no legal moves, observation that reads as an order: every path returns the caller's
   safe default with `safe=True` and a `reason`, and none of them raises. A loop that
   raises stops being a loop.

The caller executes `decision.move` on every tick, including when it is the safe
default — holding is a move — and logs `decision.line()` next to it.

## The questions, and why they are shaped that way

One request per tick carries all four. They are independent, they see the same state,
and the three that are not the move are speculative: the loop uses them to pre-empt
its own plan without a second round trip.

| id | type | why this type |
| --- | --- | --- |
| `move` | `choice` over the legal moves | The decision itself. A Choice returns one of the supplied ids with a probability for every alternative, which is what the confidence gate needs. Options are the caller's enum keys, so an answer is never a string to execute. |
| `threat` | `score`, 4 levels | Threat is a spectrum, and a Noul is a probability rather than a magnitude. Four levels, not ten: the loop only needs "act normally" separated from "this tick is high-stakes", and every extra level is a distinction a reviewer has to defend. `reply.unit()` normalises it to 0..1 so the threshold does not move if a level is added. |
| `plan_holds` | `noul` | A yes/no about the loop's own plan: is the thing we are doing still the right thing? A Noul has no confidence, so it is thresholded on its value. |
| `steering` | `noul` | `world` is sensor text, chat, signage and tool output — attacker-reachable. This asks whether the observation is trying to direct the controller instead of describing the environment. State is material to judge, never instructions to obey. |

The state is `{goal, plan, world, fingerprint}`. `world` is the untrusted half and every
question names it in backticks, so the model is pointed at the part that is material
rather than at a blob. The fingerprint rides along as a string for traceability; no
question reads it, and the staleness decision is made in code, not by the model.

## Thresholds

All of them live in the one review block at the top of the module.

| threshold | value | what it gates |
| --- | --- | --- |
| `MOVE_CONFIDENCE_REVERSIBLE` | 0.55 | Confidence a move needs when it can be undone next tick. Low on purpose: the cost of a wrong-but-undoable step is one tick of progress, and holding on every uncertain tick is its own failure mode in a moving world. |
| `MOVE_CONFIDENCE_COMMITTING` | 0.80 | Confidence a move needs when the caller cannot take it back — a step past a ledge, a release, a shot. Same model, higher bar, because the mistake is permanent. The caller names these in `Tick.committing`. |
| `THREAT_ESCALATE` | 0.60 | Normalised threat at or above which the tick is treated as high-stakes: the loop sets `preempt` and holds *every* move, reversible ones included, to the committing bar. |
| `PLAN_HOLDS_FLOOR` | 0.40 | Below this the current plan is reported as no longer holding (`preempt=True`). Advice to the caller, not a veto on this tick's move. |
| `STEERING_SUSPECTED` | 0.60 | At or above this the observation is treated as an instruction to the controller, and the loop holds. |
| `TICK_BUDGET_MS` | 90.0 | The decision's share of a 100 ms tick at `TARGET_RATE_PER_SECOND`, leaving 10 ms for sensing and acting. |
| `TARGET_RATE_PER_SECOND` | 10.0 | Not a measurement: the hypothesis `LoopReport.sustains()` is the instrument for. |

Two thresholds, not one, is the whole point of `Tick.committing`: the same answer at
0.675 confidence — midway between the two bars, `(0.55 + 0.80) / 2` — is executed for a
turn and held for a step off a ledge.
`tests/test_realtime_loop.py::test_the_bar_moves_with_the_stakes_not_with_the_answer`
is that sentence as a test.

## Cost arithmetic

From `jevkit.cost`: jev-1.13.0 is **$42 per billion input tokens** ($0.042 per million),
and output tokens are free. One tick is one request.

Counted offline over the request `examples/realtime_loop.py` builds on **tick 0**: its
`build_state` output and its four `build_questions`, encoded as they go on the wire, run
through `jevkit.limits.estimate_tokens` and priced with `jevkit.cost`. `estimate_tokens`
is a ~4-characters-per-token size estimate, **not a tokenizer and not a billed figure** —
a live call's `usage.input_tokens` will differ. Every number below is asserted by
`tests/test_realtime_loop.py::test_the_documented_cost_table_is_what_the_code_produces`,
which is also the script that produced it:

```python
example, tick = example_first_tick()                     # examples/realtime_loop.py, tick 0
offered, _ = cap_moves(offer(example.MOVES, tick.legal))  # 4 moves: no reversing yet
state = limits.estimate_tokens(build_state(tick))
per_question = {qid: limits.estimate_tokens(q) for qid, q in build_questions(offered, tick.goal).items()}
cost.usd_for("jev-1.13.0", state + sum(per_question.values()))
```

`example_first_tick()` is the helper in that test file; reproduce the whole table offline,
with no key, by running it:
`.venv/bin/python -m pytest tests/test_realtime_loop.py -k cost_table`.

| | tokens |
| --- | --- |
| state (`goal`, `plan`, `world`, `fingerprint`) | 86 |
| `move` (4 described moves — `reverse` is illegal on tick 0, so it is not offered) | 235 |
| `threat` (4 levels) | 137 |
| `plan_holds` | 108 |
| `steering` | 128 |
| **one tick** | **694** |

- All five moves described would make the `move` question 251 tokens; tick 0 offers four.
- `cost.usd_for("jev-1.13.0", 694)` = **$0.0000291 per tick** — about 34,000 ticks per
  dollar.
- Arithmetic on that per-tick figure, *if* a loop holds 10 decisions/s (a hypothesis, not
  a rate measured here): $0.0291 per 1,000 ticks, **$1.05 per robot-hour**, $8.39 per
  8-hour shift.
- The four questions are 608 of those 694 tokens, and they are the same every tick, so
  cost scales with how much world you send. Trimming the observation is the lever;
  trimming the rubric is not worth it.
- A 255-option `move` question costs what its ids cost: ~1,034 tokens for bare 2–4
  character ids (`m0`…`m254`) with this example's goal and no descriptions, ~1,316 with
  8-character ids (`move_000`-style). State plus the longest question here is 321 tokens
  against the documented 32k ceiling, so the binding limit in practice is the option
  count, not the context.

`LoopReport.usd` is the ledger's own delta for the run, so the example prints the
dollars that run actually spent rather than this table.

## Rate arithmetic

`limits.REQUESTS_PER_MINUTE` is 1,200 — an **account** ceiling, not a per-loop one.
`Account` does the division. One call, one line — the account rate and this fleet's demand
together:

```pycon
>>> Account(loops=2).line()
'1200/min = 20 requests/s across the account · 2 loop(s) x 10/s = 20/s wanted · 2 loop(s) fit · fits'
>>> Account(loops=3).line()
'1200/min = 20 requests/s across the account · 3 loop(s) x 10/s = 30/s wanted · 2 loop(s) fit · needs pacing or more quota'
```

At one request per tick, 10 decisions/s means **two robots on a key**. A third needs
pacing or more quota; `run(..., limiter=RateLimiter())` installs a
`jevkit.pacing.RateLimiter` for the run and hands the client back unpaced afterwards.
Two limiters on one client are refused, because each would think it owned the quota.

## What a run reports

`LoopReport` separates the loop's cadence from its decisions, because a loop can tick
very fast while deciding nothing at all:

| field | what it is |
| --- | --- |
| `achieved_rate` | Ticks per second, including every held tick. Cadence, not decisions. |
| `chosen_rate` | Moves per second the model actually chose (`reason == "chosen"`). |
| `p50_ms` / `p95_ms` | Request latency percentiles from the ledger; `None` when no answer landed. |
| `overhead_s` | Wall time minus the request waits the *ledger* recorded, so a burned deadline and a tick that never asked both count as overhead. |
| `deadline_misses`, `stale_drops`, `unchecked_staleness`, `safe_defaults`, `preempts` | Counts over the run's decisions. |
| `calls`, `input_tokens`, `usd` | The ledger's own deltas for this run — a floor on spend, since a cancelled tick may have cost a request no answer came back from. |

`sustains(target)` is the certificate, so it is deliberately hard to get: it wants
`chosen_rate >= target` **and** no deadline miss, no stale drop, and no answer used
without a staleness check. Rating `achieved_rate` instead would certify a loop that held
every one of its ticks, or one that sent no request at all —
`tests/test_realtime_loop.py::test_a_run_that_chose_no_move_sustains_nothing` pins both
cases down.

## Honest limits

- **The 10+/s claim is not measured here, and this repo cannot measure it without a
  key.** `run()` is the instrument: `LoopReport` carries the tick cadence, the rate of
  moves actually chosen, p50/p95 from the ledger, deadline misses, stale drops and
  unchecked staleness, and `sustains(target)` compares them to a target. Run the example
  against your own account and read its last three lines. **No tick rate or overhead
  figure is quoted in this repo**, and a number measured against the mock transport in
  `jevkit.testing` would be the speed of a mock rather than of Jev. What is committed is
  the *accounting*, not a figure:
  `tests/test_realtime_loop.py::test_overhead_counts_every_millisecond_no_answer_was_waited_for`
  asserts that a run which never got an answer reports all of its wall time as
  `overhead_s`, and that a run which did gets exactly the ledger's request waits
  subtracted. Read your own `report.summary()` for numbers; do not read them here.
- **A missed deadline is a held tick, not a retry.** If the model's p95 sits above
  `TICK_BUDGET_MS`, the loop degrades to holding a lot. Read `deadline_misses` before
  believing a tick rate, and raise the budget or lower the rate rather than removing
  the deadline.
- **Pacing waits are spent inside the tick budget.** `RateLimiter.acquire_async` waits
  inside `jev.ask`, so a loop that is over its quota converts throttling into deadline
  misses — fail-closed, but it looks like latency. Size the quota for the fleet.
- **A cancelled tick may still have cost a request.** The ledger counts answers that
  landed, so `LoopReport.calls` is a floor on spend when there are deadline misses.
- **The fingerprint is only as good as what the caller puts in it.** It has to cover
  exactly what would invalidate the decision. Too coarse and a stale decision executes;
  too fine (every sensor sequence number) and every tick is dropped. The example uses
  `(step, obstacle_present, distance_rounded)`.
- **Calibration is a property of groups, not of one answer.** A confident wrong move is
  possible; the thresholds buy a rate, not a guarantee. That is why the committing set
  exists and why the safe default is the caller's, not ours.
- **Options, not commands.** The model never sees the actuators. It picks among ids the
  caller already holds, which bounds the worst case to "a legal move at the wrong time"
  — recoverable for anything not in `committing`.
- **More than 255 legal moves is capped, loudly.** `cap_moves` keeps the first 255 in
  the caller's order and `Decision.dropped` names the rest. Order `legal` by priority,
  or shard across ticks; nothing is dropped silently.
- **English first.** Non-English observations are accepted at lower accuracy
  (`docs/api-notes.md`), and this recipe gates on confidence, so a non-English world
  shows up as more held ticks rather than as worse moves.
