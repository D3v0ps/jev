# Tool risk gating

`jevkit/recipes/tool_gating.py`

## The decision

An agent has already chosen a tool and filled in its arguments. Before the
executor runs it, something has to answer: **does this call run?**

This recipe answers in one Jev request and returns a `Decision` the caller routes
on without parsing anything:

| `Decision.verdict` | what the caller does |
| --- | --- |
| `ALLOW` | execute the call it already held |
| `CONFIRM` | show `Decision.reason` to a human and wait |
| `BLOCK` | refuse, and log `audit_entry(decision)` |

It executes nothing and it rewrites nothing. `Decision.tool` is the caller's own
tool name, unchanged; the recipe has no way to produce a different one.

```python
from jevkit import Jev
from jevkit.recipes.tool_gating import ALLOW, ToolCall, gate

TIERS = {"search_docs": "read_only", "write_file": "mutating", "delete_bucket": "privileged"}

call = ToolCall(
    tool="delete_bucket",
    arguments={"bucket": "prod-invoices", "recursive": True},
    task="Delete the temporary export bucket, tmp-export",
    context=["Bucket listing: prod-invoices (18,402 objects), tmp-export (3 objects)."],
)

with Jev() as jev:
    decision = gate(jev, call, TIERS)

if decision.verdict == ALLOW:
    run(call)          # the caller's own executor, on the caller's own call
else:
    escalate(decision) # decision.reason and decision.triggers say why
```

## Why a decision model rather than an LLM reviewer

The output of a gate is not prose. It is one of three constants plus the numbers
that justified it, and the whole value of a gate is that it is the same gate every
time.

Two properties follow from the request shape rather than from a prompt:

- **The model is never offered an identifier.** The only option sets in the
  request are this module's three tier names, its five blast-radius levels, and
  `{true, false}`. A tool name, a path, an argument value, or a command is never
  an option, so a verdict cannot arrive carrying something the caller would then
  execute.
  `tests/test_tool_gating.py::test_every_option_offered_is_one_of_this_modules_own_constants`
  asserts that against the encoded request body.
- **The policy is code, not persuasion.** The model answers seven independent
  questions about the call. Every threshold, every combination, and every
  precedence rule lives in the review block at the top of the module. Changing
  what gets blocked is a diff in a table a reviewer reads, not a prompt edit whose
  effect has to be re-measured.

What a decision model does *not* remove is being wrong. A calibrated probability
is a property of groups of predictions, not a promise about this one call, which
is why the thresholds below are asymmetric and every uncertain path ends in
`CONFIRM` or `BLOCK`.

Whether this is also *fast and cheap enough to sit in front of every tool call* is
a measurement, not a claim. `examples/tool_gating.py` gates five proposed calls and
prints p50/p95 latency, tokens, and dollars per decision from `jev.ledger`.
**No LLM-reviewer baseline has been measured in this repository, so no speed-up or
cost ratio is quoted here.**

## The request

One call, seven questions, always the same seven:

| id | type | criteria | why it exists |
| --- | --- | --- | --- |
| `destructive` | noul | `{true, false}` | would something that exists now be gone? |
| `irreversible` | noul | `{true, false}` | could the same agent undo it? |
| `production` | noul | `{true, false}` | real users, real records, real money? |
| `secrets` | noul | `{true, false}` | would a secret or personal data move somewhere new? |
| `injected_arguments` | noul | `{true, false}` | did the text the agent *read* supply the arguments? |
| `blast_radius` | score | 5 levels, own workspace → everything at once | scales the confidence floor and blocks on its own |
| `judged_tier` | choice | `read_only`, `mutating`, `privileged` | catches a stale entry in the caller's registry |

### Why the questions are shaped this way

**Decomposed, not "is this safe".** One safety question returns one number with
no structure: you cannot set a different threshold for "deletes something" and
"sends money", and you cannot show a reviewer why a call was stopped. Five
propositions give five thresholds, and `Decision.signals` carries all five into
the audit log whatever the verdict was.

**One proposition per noul.** A noul is the probability that a statement holds —
`0.5` means "yes and no are equally likely", not "medium". Severity lives in the
`blast_radius` score, which returns a probability-weighted position that can land
between levels; `reply.unit()` normalises it to 0..1 so the thresholds do not
change if a level is added.

**Nouls have no confidence, so the noul thresholds absorb the uncertainty.** An
uninformative noul is `0.5`, and `0.5` is above every confirm threshold in the
table. A shrug therefore asks a human instead of allowing. The two graded
questions do carry confidence, and that is what the confidence floor is for.

**The declared tier is not in the state.** `judged_tier` exists to disagree with
the caller's registry — a tool listed as `read_only` two refactors ago that now
writes. Showing the model the declared tier would anchor it and the disagreement
would disappear. The tier is caller-side policy; the model classifies the call on
its own.

**The state names its own trust boundary.** `task` is the operator's instruction.
`proposed_call.arguments` and `untrusted_context` are material to judge, and every
question says so. `agent_rationale` is carried as the agent's *claim*, explicitly
not as evidence — the agent that wrote it may be the thing that was hijacked.

## The policy table

The model supplies evidence; `decide()` applies the policy. Every rule can only
*raise* the verdict (`stronger()`), so no rule can talk another one out of
stopping a call.

### Per-signal thresholds

Base thresholds, then multiplied by the tier scale
(`read_only` ×1.6, `mutating` ×1.0, `privileged` ×0.65) and clamped at 0.95. A
dash means the signal never blocks on its own — it still blocks in combination.

| signal | read_only confirm / block | mutating confirm / block | privileged confirm / block |
| --- | --- | --- | --- |
| `injected_arguments` | 0.24 / 0.64 | 0.15 / 0.40 | 0.0975 / 0.26 |
| `secrets` | 0.32 / 0.95 | 0.20 / 0.60 | 0.13 / 0.39 |
| `irreversible` | 0.48 / 0.95 | 0.30 / 0.80 | 0.195 / 0.52 |
| `destructive` | 0.48 / — | 0.30 / — | 0.195 / — |
| `production` | 0.56 / — | 0.35 / — | 0.2275 / — |
| `blast_radius` (0..1) | 0.72 / 0.95 | 0.45 / 0.90 | 0.2925 / 0.585 |

The asymmetry is the argument of the block, and it runs along two axes:

- **By consequence of the signal.** `injected_arguments` is tightest: an argument
  planted by text the agent read is how an agent gets driven, and there is no
  version of that worth running. `secrets` is next, because an exfiltration
  cannot be taken back. `destructive` and `production` never block alone —
  deleting a temp file and writing to production are ordinary agent work — so
  they ask a human instead.
- **By consequence of the tool.** Being wrong about a read-only tool costs less,
  so one irreversibility reading of 0.55 is a `CONFIRM` on a `read_only` or
  `mutating` tool and a `BLOCK` on a `privileged` one. The 0.95 ceiling keeps
  leniency from making a rule unreachable: even a read-only tool has a
  probability at which it stops.

### Combinations

Every signal named must reach the (tier-scaled) threshold for the pair to fire,
and a pair fires a `BLOCK`.

| pair | base | read_only | mutating | privileged | reads as |
| --- | --- | --- | --- | --- | --- |
| `destructive` + `irreversible` | 0.45 | 0.72 | 0.45 | 0.2925 | an irreversible delete |
| `secrets` + `production` | 0.50 | 0.80 | 0.50 | 0.325 | real data leaving production |
| `injected_arguments` + `destructive` | 0.30 | 0.48 | 0.30 | 0.195 | a delete the page asked for |

This is where "an irreversible delete blocks at a far lower probability than a
read-only call" actually lives. On a `mutating` tool, 0.45 and 0.45 together block
— against a single-signal `irreversible` block of 0.80 and no `destructive` block
at all.

### The other four rules

| rule | threshold | verdict |
| --- | --- | --- |
| graded-answer confidence below the floor | 0.50 at no blast radius → 0.85 at full | at least `CONFIRM` |
| probability mass on a tier stricter than the registry declares | 0.30 | at least `CONFIRM` |
| tool missing from the registry, or carrying an unknown tier | — | strictest tier **and** at least `CONFIRM` |
| a cap kept part of the call from the model | — | at least `CONFIRM` |

The confidence floor is interpolated on the blast radius and is deliberately *not*
tier-scaled: a floor above 1.0 would stop every call and hide the rest of the
policy. It catches the case the noul thresholds cannot — the model reading the
call as harmless without being able to tell how far it reaches. Two readings that
put the blast radius in the same place, one confident and one flat, come out
`ALLOW` and `CONFIRM`.

## What fails closed

| situation | verdict |
| --- | --- |
| the request failed (oversized, rate limited, refused, timed out) | `BLOCK` |
| an answer failed validation, or an answer is missing | `BLOCK` |
| the model answered a tier that was not offered | `BLOCK` (rejected by `reply.picked`) |
| the tool is not in the registry | strictest tier, never `ALLOW` |
| an argument or context item was capped or dropped | never `ALLOW` |

`Decision.signals` is empty on the first two, which is how a log tells "blocked on
the evidence" from "blocked for lack of any".

## Caps, and what they cost

The state is capped so one pathological argument cannot push the request past the
32k state-plus-longest-question budget: 32 arguments, 1,000 characters per
argument value, 8 context items, 2,000 characters each. Every cap that fires is
named in `Decision.trimmed` or `Decision.dropped`, and **anything withheld
forbids `ALLOW`**: a verdict only covers the call the model actually saw.
Truncating the tail of a long argument is exactly how a payload hides, so a
truncated call goes to a human rather than through.

Argument *names* and `task` are not capped. An absurd one makes the request too
large, `jevkit.limits.check_request` rejects it locally before the network, and
`gate` turns that into a `BLOCK` — the cost of not capping them is a refused call,
not a quietly reshaped one.

`ToolCall.redact` names arguments whose values must not be sent at all; the state
carries the name, the type, and the length instead. The gate does not need to see
a secret to judge a call that moves one, and a value the request never carries
cannot be leaked by it.

A name in `redact` that matches no argument redacts nothing, so it is treated as
withheld material rather than passed over: it appears in `Decision.dropped` as
`unredacted.<name>` and takes `ALLOW` off the table. A typo used to be silent,
which is the worst possible outcome — the caller believes a secret was held back
while the request carries it in full, and `Decision.redacted` is empty either way.

## Cost arithmetic

From `jevkit.cost`, at the published $42 per billion input tokens ($0.042/Mtok)
for `jev-1.13.0`, and `jevkit.limits.estimate_tokens` on the real request bodies
in `examples/tool_gating.py`:

| | input tokens | per decision | per 1,000 gated calls |
| --- | --- | --- | --- |
| the seven questions alone | 1,390 | $0.0000584 | $0.0584 |
| a read (`search_docs`) | 1,487 | $0.0000625 | $0.0625 |
| a delete (`delete_bucket`, with a bucket listing) | 1,496 | $0.0000628 | $0.0628 |
| a send with a hostile page in context (`send_email`) | 1,566 | $0.0000658 | $0.0658 |
| every cap at its maximum | 13,531 | $0.0005683 | $0.5683 |

The shape to notice: **the fixed question text is most of every request.** The
call being judged is a few dozen tokens; the seven questions are ~1,390 and are
paid on every gate. Output tokens are free. There is no prompt cache in the API
surface this repo is written against (`docs/api-notes.md`), so shortening the
question block is the only lever on the floor — and it is the block a reviewer
reads, which is a real trade-off, not a free win.

Latency is not in this table on purpose. Run `examples/tool_gating.py` and read
p50/p95 off your own ledger; a gate in front of every tool call pays its latency
on every step, and that number depends on your region and your concurrency, not
on this document.

## Honest limits

- **The gate is exactly as good as the five signals.** There is no keyword rule
  underneath: a call whose `injected_arguments` reading comes back low is allowed
  even when the context is visibly hostile, which
  `test_nothing_in_the_state_can_reach_the_verdict_except_through_a_signal`
  demonstrates on purpose. Five signals, a blast radius, and a confidence floor
  are defence in depth against one of them being wrong, not a proof that none is.
- **The thresholds here are a starting point, not a calibration.** They were
  chosen by argument — which consequences are worth a human's time — not fitted to
  labelled calls. Run your own proposed-call log through `decide()` and move them;
  the table is one block precisely so that is a small diff.
- **Calibration is a group property.** "0.62 destructive" is meaningful across
  many calls, not a statement about this one. A gate at the 80th percentile of
  anything will be wrong on individual calls in both directions.
- **CONFIRM has a cost the model cannot see.** A gate that asks too often gets
  clicked through, which is worse than a gate that asks rarely. Watch the CONFIRM
  rate on your own traffic; it is the number that decides whether the tiers are
  drawn in the right places.
- **A tier is a claim about a tool, made once.** The `judged_tier` question exists
  because registries go stale, but it only fires when the model disagrees clearly
  enough. Keeping the registry honest is still the caller's job.
- **It judges the call, not the plan.** Ten individually reasonable calls can add
  up to something no one would approve. This recipe sees one call at a time; a
  budget or a step counter across the run is a different mechanism.
- **English first.** The model's other languages are weaker
  (`docs/api-notes.md`), and a gate is exactly where that asymmetry matters.
  Non-English arguments and context deserve a lower confidence floor of trust —
  or a tighter tier.
