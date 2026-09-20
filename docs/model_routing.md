# Model routing

`jevkit.recipes.model_routing` — send each request to the cheapest handler on the
caller's ladder that can actually do it, escalate on doubt, and be honest about whether
the routing pays for itself.

## The decision

The caller holds an ordered registry of handlers — deterministic code, a small model, a
frontier model, a human queue — and one request that exactly one of them has to answer.
The decision is: **which rung does this request go to?**

The answer is one of the caller's own handler ids plus a distribution over the rest.
Nothing is parsed and nothing is named: `Registry` holds the ids, `offer()` turns them
into the Choice's options, and `decide()` checks the id that comes back against the
registry again before returning it. A handler that is not in the registry cannot be
chosen, and `decision.handler` is always a key the caller already had.

The caller sends the request to `decision.handler` and logs `decision.line()` next to it.
`decision.runner_up` and `decision.runner_up_probability` are the second choice and its
probability, so a share of traffic can be shadow-routed there — which is the only way to
measure the misroute rate the cost arithmetic below needs.

Why a decision model rather than a text model: the router sits on the critical path of
every single request. Asking a frontier model which model to use makes the router the
most expensive component in the system, and it answers in a form that has to be mapped
back onto a handler id. Jev returns the id, with a number to gate it on.

## The questions, and why they are shaped that way

One request carries all six. They are evaluated independently against the same state, so
the five gate questions are free speculation: most requests need none of them, and a
second request to find out whether this one needs a tool would cost more than the route
saves.

| id | type | why this type |
| --- | --- | --- |
| `handler` | `choice` over the registry | The decision itself. A Choice returns one of the supplied ids with a probability for every alternative, which is what the confidence gate and the shadow-route need. Options are the caller's own keys in cheapest-first order. |
| `depth` | `score`, 5 levels | Reasoning depth is a spectrum, and a Noul is a probability rather than a magnitude. Five levels, not ten: the router only separates "a template can do this" from "this needs a model" from "this needs the strongest thing we have". `reply.unit()` normalises to 0..1 so the two depth thresholds survive an edit to the rubric. |
| `needs_tools` | `noul` | A yes/no about the request, thresholded on its value because a Noul carries no confidence. It removes handlers the caller did not mark `tools` from the eligible set. |
| `needs_fresh_data` | `noul` | Same shape, for information that changes over time. A stale answer to "is the train running" is a wrong answer, not a cheap one. |
| `safety_sensitive` | `noul` | Does both jobs: it removes handlers the caller has not approved for sensitive work, and it raises the confidence floor. This is where the threshold scales with the stakes. |
| `steering` | `noul` | The request is attacker-reachable text. This asks whether it is addressing the *router* — naming a tier, claiming authority, arriving dressed as configuration — rather than describing work. A request that tries to pick its own handler does not get to. |

The state is `{request, context, channel, truncated}`. `request` and `context` are the
untrusted half and every question names them in backticks, so the model is pointed at
the material rather than at a blob. `channel` is the caller's own label and is trusted.
`truncated` tells the model it is looking at the head of a longer request. No question
reads `channel` or `truncated` as an instruction; they are context.

## Thresholds

All of them live in the one review block at the top of the module. Every row is exercised
on both sides in `tests/test_model_routing.py` — for the two budget rows that means a
request that fits (`test_a_request_at_the_budget_is_sent_whole`), one that is clipped
(`test_a_long_request_is_clipped_to_the_state_budget_and_reports_what_was_cut`), and one
sent beside a 255-handler question that still has to reach the network
(`test_a_wide_described_registry_leaves_the_request_room_it_can_actually_use`).

| threshold | value | what it gates |
| --- | --- | --- |
| `FLOOR_ROUTE_DOWN` | 0.65 | Confidence the Choice needs before a handler weaker than the strongest eligible one is accepted. Below it, the route moves up one rung. |
| `FLOOR_SENSITIVE` | 0.85 | The same floor once `safety_sensitive` has fired. The stakes changed, so the number did. |
| `SAFETY_SENSITIVE_TRUE` | 0.40 | Probability at which a request counts as safety-sensitive. Deliberately the lowest bar here: treating a harmless request as sensitive costs one more expensive call, and the reverse costs the harm. |
| `NEEDS_TOOLS_TRUE` | 0.60 | Probability at which handlers without `tools` leave the eligible set. |
| `NEEDS_FRESH_DATA_TRUE` | 0.60 | Probability at which handlers without `fresh_data` leave the eligible set. |
| `STEERING_SUSPECTED` | 0.60 | Probability at which the request is treated as addressing the router. The route then goes to the strongest handler the gates allow, and never to the pick. |
| `DEPTH_NEEDS_A_MODEL` | 0.25 | Normalised depth at which the cheapest rung stops being eligible: anything past a lookup needs something that can reason. Skipped on a ladder whose rungs all share one tier, where it would only empty the set. |
| `DEPTH_NEEDS_THE_TOP` | 0.70 | Normalised depth at which only the top of the ladder is eligible, whatever the Choice preferred. Sits between the "Moderate" and "Deep" levels. |
| `MIX_SUM_TOLERANCE` | 0.01 | How far a measured traffic mix may miss 1.0 before `expected_cost` refuses it. |
| `STATE_TOKEN_RESERVE` | 4,000 tokens | Held back from the 32k state-plus-longest-question budget for the questions, so `STATE_CHARS_BUDGET` = 112,000 characters is the *most* request text ever sent. It is a ceiling, not the clip: see below. |
| `STATE_OVERHEAD_RESERVE` | 1,000 tokens | Held back for the rest of the state — `context`, `channel`, `truncated`, and the JSON around them — when `request_chars_budget()` derives the clip from the questions actually built. |

## Escalate on doubt

The two mistakes are not symmetric, and the code moves in one direction — with a single
exception, forced by the gates, that labels itself rather than passing as an escalation:

- **A missed floor moves the route up one rung** inside the eligible set
  (`_one_step_up`). It never moves it down. If the pick is already the strongest eligible
  handler there is nowhere to go, so the handler stands — but the reason is
  `low_confidence` and not `routed`, because a route taken without the confidence it
  wanted is not the same event as one taken with it, and counting reasons has to separate
  them. `floor_met=False` records it on the decision either way.
- **Capability gates are applied before the floor**, and a pick the gates rule out is
  escalated to the cheapest eligible handler that is not weaker than it (`_at_or_above`).
  When the gates leave nothing as strong as the pick, the route is the strongest *eligible*
  handler, which is the one place this recipe moves a request *down*: `downgraded=True`
  says so, `escalated` is then False, and the detail reads "downgraded to …" rather than
  pretending otherwise.
- **No fallback sidesteps the gates.** Every path that cannot use the pick — steering,
  a ruled-out pick, an empty eligible set — lands on the strongest handler that satisfies
  the gates, not on `registry.strongest` regardless of them. Reaching for the top rung
  there is how a safety-sensitive request ends up on the one handler the caller marked
  `sensitive_ok=False`, and the steering path is attacker-triggerable, so it is the last
  place to relax a constraint. If the gates rule out *everything*, `eligible` is empty,
  `reason="not_eligible"`, and the route is the strongest rung the caller stands behind.
- **Every failure path lands on the strongest rung the caller stands behind**
  (`to_the_top`): a malformed answer, an answer naming a handler this registry does not
  hold, a request too large to send, a transport failure. `route()` never raises, because
  a gateway that raises stops being a gateway, and the expensive handler is the safe
  answer here, not the cheap one. "Up" stops at the caller's own safety claim: with no
  answers the request might be anything, so if the top rung is not `sensitive_ok` the
  fallback is the strongest rung that is, and the detail names the swap. Mark your top
  rung `sensitive_ok=True` if you want it to catch everything.
- **A request that tries to steer the router** gets the strongest eligible handler too,
  before its own preferred route is even considered.

`decision.reason` is one of `routed`, `low_confidence`, `not_eligible`, `steering`,
`rejected`, `refused`, `failed`. Only `routed` is the pick honoured on confidence.
Counting those reasons over a day is the operational signal: a spike in `failed` means Jev
is unreachable and every request is being paid for at the fallback rung's price, and a
spike in `not_eligible` with an empty `eligible` means this registry cannot serve the
traffic it is being given.

## The cost arithmetic

Jev costs **$42 per billion input tokens** — `jevkit.cost.per_million("jev-1.13.0")` is
`0.042` dollars per Mtok, and output tokens are free. That fee lands on *every* request,
including the ones the router would have sent to the baseline anyway.

Counted offline, with no key and no live call, over the request `prepare()` builds for the
ladder and the six requests in `examples/model_routing.py`: `jevkit.limits.estimate_tokens`
on the state, plus `estimate_tokens` on each of the six questions, priced through
`jevkit.cost`. `estimate_tokens` is a ~4-characters-per-token estimate and *not* a
tokenizer — it overestimates on purpose, so these are upper bounds and a live reply's own
`usage.input_tokens` is normally lower. `test_the_fee_per_decision_is_the_repos_own_estimate`
asserts every figure in this table, and `test_the_estimated_fee_is_what_a_decision_records_in_the_ledger`
ties it to what one decision books in the ledger.

| | tokens |
| --- | --- |
| the six questions | 1,016 (`handler` is the largest at 344) |
| state for the six example requests | 29–77 |
| **one routing decision** | **1,045–1,093 → $0.0000439–$0.0000459 → $43.89–$45.91 per million requests** |

The fee is dominated by the fixed question block, so it barely moves with request size
until the request itself gets long — and it grows with the *registry*, since every handler
description is in the handler question. `router_usd_from(jev.ledger)` is how you get your
own number from a real run; it refuses an empty ledger and one holding replies from an
unpriced model rather than averaging something wrong.

`expected_cost()` puts that fee where it belongs:

```
routed_per_request = fee + Σ mix[h] · price[h] + misroute_rate · price[rerun]
saving             = price[baseline] − routed_per_request
break_even_rate    = (price[baseline] − fee − Σ mix[h] · price[h]) / price[rerun]
```

`mix` comes from `mix_from(decisions)` — the routes actually taken. `price[...]` is the
caller's own dollars per request. `rerun` defaults to the baseline, since that is where a
misrouted request would have gone in the first place.

Two verdicts from the same traffic, over an **illustrative** mix of a third each to
`template`, `frontier` and `human` — a shape, not a measurement: a real mix comes from
`mix_from(decisions)` after a real run, and there is no run in this repo. The other inputs
are the baseline `frontier` at $0.012 a request, the assumed 5% misroute rate, and the
$0.0000459 fee from the table above (the largest of the six, so the router's own cost is
not understated). `test_the_doc_cost_table_is_expected_cost_over_the_stated_mix` asserts
both rows, so the arithmetic below is this repo's and not a recollection of a run:

| price list | handlers | routed | vs baseline | break-even misroute rate | verdict |
| --- | --- | --- | --- | --- | --- |
| human rung free | $0.004000 | $0.004646 | 0.39x | 66.3% | routing pays |
| human rung at $2.00 a ticket | $0.670667 | $0.671313 | 55.94x | −5489.3% | routing does not pay |

Run `examples/model_routing.py` with a key to replace the mix with your own; the verdict is
a property of your prices and your traffic, not of this pattern.

Read `break_even_misroute_rate` against the rate you measured. Negative means routing
loses even if it never misroutes: the fee plus the handlers it chose already cost more
than the baseline. Above 1 means no misroute rate can make it lose. `None` means re-runs
are free, so misroutes cannot tip it.

There is no savings claim in this module, and `pays=False` is a real answer:
`test_routing_does_not_pay_when_a_human_rung_takes_one_request_in_a_hundred` and
`test_a_small_fee_against_a_cheap_baseline_also_loses` exist to keep it that way.

## The honest limits

- **The fee is on every request; the saving is only on the ones that move down.** A
  router whose measured mix is still mostly the baseline is pure cost. Check `mix_from`
  before believing anything else here.
- **The floor and the saving trade against each other.** Escalating on doubt is a
  deliberate decision to spend money to avoid wrong answers. Raise `FLOOR_ROUTE_DOWN`
  enough and the mix collapses back onto the baseline; the arithmetic will show it, but
  it cannot pick the trade-off for you.
- **The misroute rate is not observable from inside the router.** Nothing in this module
  measures it — it is an input. Shadow-route a share of traffic to `decision.runner_up`,
  compare the outcomes, and feed the result back in. Until then `expected_cost` is
  faithfully reporting the consequences of a guess.
- **Calibration is a property of groups of predictions, not of any single answer.** A
  confidence of 0.7 on one request does not mean this request is 70% likely to be routed
  correctly. The floor earns its keep across a day of traffic, not on one ticket.
- **English is the primary training language.** Non-English requests are answered with
  lower accuracy and the floor does not know that. Count escalations per language before
  assuming the router behaves the same across them.
- **The capability flags are the caller's claims, not facts.** If `frontier.tools` is
  wrong, the router routes wrongly and confidently. The flags are hard constraints
  precisely so they are reviewed in one place.
- **A router adds a round trip to the critical path of every request.** Watch
  `jev.ledger.p50_ms`. If it is comparable to the small model's own latency, the router
  can cost more time than the cheap rung saves, and the arithmetic above says nothing
  about that.
- **The ladder is treated as an ordinal.** Handlers that are *different* rather than
  *stronger* — a code model beside a writing model — are not what `tier` models. Give
  them the same tier and distinguish them with capability flags and descriptions; lateral
  moves are then possible, but the recipe will never prefer one for its own sake.
- **A long request is judged on its head.** `request_chars_budget()` works out how much
  request text fits beside the questions it actually built, capped at
  `STATE_CHARS_BUDGET` = 112,000 characters and smaller when the handler question is
  large: 255 described handlers can cost thousands of tokens and take that room away from
  the request. `truncated_chars` says how much was cut, and routing on the opening of a
  long document is a guess about the rest of it. Split it upstream if that matters.
- **A registry can be too wide to route with at all.** The handler question has to fit
  inside the 32k state-plus-longest-question budget on its own, and clipping the request
  cannot buy that back. 255 handlers described in 480 characters each already leave the
  request nothing (the question estimates at 31,533 tokens); at 500 characters each the
  question is 32,808 and `route()` returns `reason="refused"` without a call — correct,
  and useless. Those two figures are asserted in
  `test_a_registry_too_wide_to_ask_about_is_refused_before_the_network`. Shorten the
  descriptions, or shard the ladder and route in two passes; nothing here does it for you.
- **The no-evidence fallback can be weaker than your top rung.** If the strongest handler
  is not marked `sensitive_ok`, a request nobody could check goes to the strongest handler
  that is, because the alternative is handing possibly-harmful content to a rung you
  declared unfit for it. That rung may lack tools or fresh data. Mark the top of the
  ladder `sensitive_ok=True` if you want it to catch everything, and read the decision's
  detail, which names the swap.
- **Escalating on steering is a denial-of-wallet path.** An attacker who wants expensive
  handling only has to sound like they are addressing the router. That is still the right
  trade — the alternative is letting requests pick their own handler — but rate-limit per
  sender, and count `reason == "steering"` so the bill is not the thing that tells you.
- **A Jev outage routes everything to the top of the ladder.** Safe, and expensive. The
  `refused` and `failed` reasons are there to be alerted on.
