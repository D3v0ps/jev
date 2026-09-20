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

Three things a loop needs that a text model does not give you:

1. **A deadline.** `next_move` wraps the request in `asyncio.wait_for(budget_ms)`. A
   late answer is cancelled in flight, the tick takes the caller's safe default, and
   the miss is counted. The loop never blocks past its deadline and never acts on an
   answer that arrived after it.
2. **A staleness check.** The world moves while the request is in flight. The caller
   passes a `fingerprint` in, and a `fingerprint_now` callable that is read again when
   the answer lands. If it changed, the decision is dropped. This is the bug naive
   versions of this loop ship: a confident, correct decision about a world that no
   longer exists. It is a tested contract here, not a nicety.
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
0.68 confidence is executed for a turn and held for a step off a ledge.
`tests/test_realtime_loop.py::test_the_bar_moves_with_the_stakes_not_with_the_answer`
is that sentence as a test.

## Cost arithmetic

From `jevkit.cost`: jev-1.13.0 is **$42 per billion input tokens** ($0.042 per million),
and output tokens are free. One tick is one request.

Measured offline for the request `examples/realtime_loop.py` builds on its first tick,
with `jevkit.limits.estimate_tokens` over the real encoded body and `jevkit.cost` for
the price. These are size estimates at 4 chars/token, not a billed figure from a live
call:

| | tokens |
| --- | --- |
| state (`goal`, `plan`, `world`, `fingerprint`) | 86 |
| `move` (5 described moves) | 235 |
| `threat` (4 levels) | 137 |
| `plan_holds` | 108 |
| `steering` | 128 |
| **one tick** | **694** |

- `cost.usd_for("jev-1.13.0", 694)` = **$0.0000291 per tick** — about 34,000 ticks per
  dollar.
- At 10 decisions/s: $0.0291 per 1,000 ticks, **$1.05 per robot-hour**, $8.39 per
  8-hour shift.
- The four questions are 608 of those 694 tokens, and they are the same every tick, so
  cost scales with how much world you send. Trimming the observation is the lever;
  trimming the rubric is not worth it.
- A 255-move Choice costs ~1,012 tokens for that one question. State plus the longest
  question here is 321 tokens against the documented 32k ceiling, so the binding limit
  in practice is the option count, not the context.

`LoopReport.usd` is the ledger's own delta for the run, so the example prints the
dollars that run actually spent rather than this table.

## Rate arithmetic

`limits.REQUESTS_PER_MINUTE` is 1,200 — an **account** ceiling, not a per-loop one.
`Account` does the division:

```
1200/min = 20 requests/s across the account
2 loop(s) x 10/s = 20/s wanted · 2 loop(s) fit · fits
3 loop(s) x 10/s = 30/s wanted · 2 loop(s) fit · needs pacing or more quota
```

At one request per tick, 10 decisions/s means **two robots on a key**. A third needs
pacing or more quota; `run(..., limiter=RateLimiter())` installs a
`jevkit.pacing.RateLimiter` for the run and hands the client back unpaced afterwards.
Two limiters on one client are refused, because each would think it owned the quota.

## Honest limits

- **The 10+/s claim is not measured here, and this repo cannot measure it without a
  key.** `run()` is the instrument: `LoopReport` carries the achieved rate, p50/p95 from
  the ledger, deadline misses and stale drops, and `sustains(target)` compares them to a
  target. Run the example against your own account and read its last three lines. The
  only rate this repo can produce offline is the recipe's own overhead: driving
  `examples/realtime_loop.py` for its 12 ticks against the mock transport in
  `jevkit.testing` reports 0.4 ms per tick of non-request time (`LoopReport.overhead_s`,
  three runs, same figure). That says the harness is not the bottleneck. It says nothing
  whatsoever about model latency, and the ~1,000 ticks/s such a run prints is the speed
  of a mock, not of Jev.
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
