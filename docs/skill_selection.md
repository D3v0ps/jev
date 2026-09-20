# Skill and tool selection

`jevkit/recipes/skill_selection.py`

## The decision

An agent holds a catalogue of skills — procedures, tool bundles, plugins — far
larger than its prompt. Before it answers a turn, something has to answer: **which
of these, if any, does this turn need?**

This recipe answers in at most two Jev requests and returns a `Decision` the
loader acts on without parsing anything:

| `Decision.action` | what the caller does |
| --- | --- |
| `SELECT` | load `decision.handles` — its own paths — and nothing else |
| `SELECT_NONE` | load nothing and answer the turn unaided |
| `ASK_USER` | ask whether they meant `decision.suggested`; load nothing yet |

`SELECT_NONE` is the expected outcome for most turns, and it is the cheapest one.

```python
from jevkit import Jev
from jevkit.recipes.skill_selection import PRIVILEGED, Skill, Turn, SELECT, select_skills

catalogue = [
    Skill(name="spreadsheets", summary="Read, clean and write .xlsx workbooks",
          description="Use for any task whose input or output is a spreadsheet...",
          handle="skills/spreadsheets/SKILL.md"),
    Skill(name="deploy", summary="Release the service to production",
          description="Runs the production release pipeline...",
          power=PRIVILEGED, handle="skills/deploy/SKILL.md"),
    # ... 294 more
]

turn = Turn(
    request="The Q3 sales workbook has headers halfway down and no totals. Sort it out.",
    context=["attached: q3-sales.xlsx (14 sheets, 22k rows)"],   # untrusted
    loaded=("incident-review",),
)

with Jev() as jev:
    decision = select_skills(jev, turn, catalogue, limit=2)

if decision.action == SELECT:
    for path in decision.handles:        # the caller's own paths, never the model's
        load(path)
```

## Why a decision model rather than an LLM

The answer is a set of positions in a list the caller already holds, and the
common answer is the empty set. Two properties follow from the request shape
rather than from a prompt:

- **The model is never offered an identifier.** The option ids are `s0`, `s1`,
  `s2` … — positions in the caller's own catalogue — plus `__none__`. A skill
  name, a path, or a loader key is never an option and never appears in a
  request, so an activation the loader cannot resolve is impossible rather than a
  case to guard against.
  `tests/test_skill_selection.py::test_no_option_is_ever_a_name_a_path_or_a_handle`
  asserts that against the encoded bodies.
- **"Load nothing" is a first-class option, not a reluctant one.** `__none__` is
  offered in every Choice in both rounds. It gives the model somewhere to put its
  mass when nothing fits, and it gives the code a reference point to compare
  against — see [the tournament](#the-255-option-ceiling-is-a-tournament).

What a decision model does *not* remove is being wrong. A calibrated probability
is a property of groups of predictions, not a promise about one turn, which is
why two independent signals have to agree before anything is loaded and why the
floors scale with what the skill can do.

Whether this is also cheaper than the alternative is a *measurement*, not a claim.
The arithmetic is [below](#cost-arithmetic), and `examples/skill_selection.py`
prints it for its own catalogue before it runs. **No LLM baseline has been
measured in this repository, so no speed-up or cost ratio against one is quoted
here.**

## Two rounds, and why the second one exists

| round | requests | sees | answers |
| --- | --- | --- | --- |
| one: nominate | one per shard | the turn, and one-line summaries of the whole catalogue | a shortlist, plus whether a skill is needed at all |
| two: confirm | one | the turn, and the **full descriptions** of the shortlist only | which to load, or none of them |

The second request is a real second round trip, and the reason is size first and
sequencing second:

- **The full text does not fit.** In the example's catalogue, 296 entries, round
  one costs 7,483 tokens in total; the full descriptions of the same entries are
  18,834 — about 64 tokens each, so a catalogue of 500 passes the 32k
  state-plus-longest-question budget on its own, and the 255-option ceiling would
  make you pay it per shard anyway. Round one ranks what is cheap; round two spends
  real tokens on the few that survived.
- **Round two's option set does not exist until round one has answered.** The
  shortlist *is* round one's answer.

That is the whole justification, and it is stated in the code at both call sites.
Everything else in the decision travels in one request: the "does this turn need a
skill at all" Noul and the injection check ride along with the first shard's
Choice, because they are speculative questions about the turn and asking them
separately would cost another round trip to learn that no round trip was needed.

### Round one

| id | type | criteria | why it exists |
| --- | --- | --- | --- |
| `nominate` | choice | up to 254 one-line summaries + `__none__` | rank the catalogue cheaply |
| `needs_skill` | noul | `{true, false}` | does this turn need *any* capability? |
| `turn_injection` | noul | `{true, false}` | is the text the agent read trying to pick the skill? |

`needs_skill` is asked as a Noul rather than folded into the Choice because it is
about the turn, not about the catalogue: a turn can need a skill this catalogue
does not contain, and that is a different answer from "the third entry fits".
Below `NEEDS_SKILL_MIN` the decision ends at one request.

### Round two

| id | type | criteria | why it exists |
| --- | --- | --- | --- |
| `pick` | choice | the shortlist + `__none__` | which candidate wins, against the others *and* against nothing |
| `fits_<id>` | noul | `{true, false}` | does this one candidate cover the turn, on its own terms? |
| `description_injection` | noul | `{true, false}` | is a description written to get itself loaded? |

**Two signals have to agree.** The ranking alone cannot admit more than one skill
and cannot say "the winner is also a bad fit"; the per-candidate Nouls alone
cannot say which of two overlapping skills to prefer. Requiring both is what makes
`limit=2` meaningful and what makes rejecting everything easy.

**The full descriptions live in the state, not in `criteria`.** They have to,
because `description_injection` is a Noul and a Noul can only read the state. A
catalogue entry is attacker-reachable text in any marketplace, so the round that
reads entries in full also screens them.

## The 255-option ceiling is a tournament

A Choice takes at most 255 options, so a catalogue over 254 entries (one slot is
reserved for `__none__`) is sharded. The subtlety is the point:

> **Probabilities are normalised within a request.** A nominee at 0.61 in shard 3
> and a nominee at 0.44 in shard 1 are not on the same scale. Shard 3 may have
> been a shard of near-misses and shard 1 a shard where one entry was obviously
> right. Sorting nominees from different shards by probability is **wrong**, and
> the fact that it usually looks reasonable is what makes it worth saying out loud.

Two rules follow, and the recipe has no others:

1. **Nomination is a within-request comparison.** An entry is eligible when its
   probability reaches `NOMINEE_OVER_NONE` times *that same request's* `__none__`
   mass. Because `__none__` is offered in every shard, this asks "did this entry
   beat 'nothing here fits' in its own shard?", which is a question one
   distribution can answer. An absolute floor could not: 1/255 and 1/42 are
   different numbers for the same shrug.
2. **The shortlist is filled round-robin by rank, never by probability.** Every
   shard's best goes in before any shard's second, in shard order, up to
   `SHORTLIST_MAX`. `test_the_shortlist_is_filled_round_robin_not_by_probability`
   puts shard 0's best at 0.30 and shard 1's at 0.80 and asserts both are
   shortlisted in rank order.

The only legitimate comparison between entries from different shards happens in
round two, where they are options in **one** request. That is the final round of
the tournament, and it is the round that actually decides.

Entries a cap kept out are reported, never silently unselectable:

| field | what it holds |
| --- | --- |
| `Decision.dropped` | entries that beat their shard's `__none__` but did not fit `NOMINEES_PER_SHARD` or `SHORTLIST_MAX` |
| `Decision.unjudged` | entries past `MAX_SHARDS` shards — never offered to the model at all |
| `Decision.trimmed` | entries whose summary or description was truncated |
| `Decision.clipped` | parts of the turn that were truncated or left out |

`MAX_SHARDS × SHARD_OPTIONS` = 2,032 entries is the largest catalogue one decision
judges. Past that, the tail is reported rather than quietly unreachable; a
catalogue that size wants a retrieval step in front of this one, not a ninth shard.

## The thresholds

Everything in this table is in the review block at the top of the module.

### Round one

| threshold | value | effect |
| --- | --- | --- |
| `NEEDS_SKILL_MIN` | 0.45 | below it: `SELECT_NONE` at one request, round two never sent |
| `INJECTION_BLOCK` (turn) | 0.60 | at or above: `SELECT_NONE`, round two never sent |
| `NOMINEE_OVER_NONE` | 0.5 | an entry is eligible at half its shard's `__none__` mass |
| `NOMINEES_PER_SHARD` | 3 | per shard, by rank |
| `SHORTLIST_MAX` | 8 | candidates round two reads in full |

Nominating at *half* the `__none__` mass is deliberately generous: a shortlist is
cheap and round two is the strict round. Confirming at *equal* mass is the strict
half of the same pair.

### Round two

| threshold | ADVISORY | ACTING | PRIVILEGED |
| --- | --- | --- | --- |
| `FITS_MIN` — the candidate's own Noul | 0.55 | 0.72 | 0.85 |
| `CONFIDENCE_FLOOR` — the ranking's confidence | 0.35 | 0.55 | 0.75 |
| below the confidence floor | not loaded | not loaded | **`ASK_USER`** |
| description was truncated | may still load | may still load | **`ASK_USER`** |

| rule | value | effect |
| --- | --- | --- |
| `SELECT_OVER_NONE` | 1.0 | every selected candidate must be at least as probable as `__none__` |
| `pick` lands on `__none__` | — | `SELECT_NONE`: the shortlist was rejected as a whole |
| `INJECTION_BLOCK` (descriptions) | 0.60 | `SELECT_NONE` |
| `SELECTION_LIMIT` | 2 | at most this many skills per turn; the caller may lower it |

The power tier is **declared by the caller's registry, not judged by the model**.
This is the opposite of `tool_gating.py`, where a `judged_tier` question exists
precisely to disagree with a stale registry. The difference: there the model reads
the actual call, here it would be reading the skill's own marketing copy, which is
the least reliable witness to what the skill can do. The tools inside a skill are
gated per call by `tool_gating.py`; this recipe decides only what gets loaded.

A fit reading of 0.60 therefore loads a checklist and does not load a deploy
pipeline — `test_the_fit_floor_scales_with_what_the_skill_can_do` asserts both
sides of all three floors.

### The low-confidence paths

A flat ranking means "some candidate here, but which one is a coin flip". What
happens next depends on the stakes, not on the model:

- **Advisory**: load nothing. The agent answers unaided, which is what it would
  have done anyway. Interrupting a person over a checklist costs more than it saves.
- **Privileged**: `ASK_USER`, with `Decision.suggested` naming the skill. "Did you
  mean the deploy skill?" is a cheap question; auto-loading a release pipeline on
  a coin flip is not.

## What fails closed

| situation | outcome |
| --- | --- |
| a request failed (oversized, refused, rate limited, timed out) | `SELECT_NONE` |
| an answer failed validation, or an answer is missing | `SELECT_NONE` |
| the model answered an option that was not offered | `SELECT_NONE` (`reply.picked` refuses it) |
| round one returned fewer replies than shards | `SELECT_NONE`, nothing nominated |
| the catalogue is empty | `SELECT_NONE`, no request at all |
| the turn's context is trying to pick the skill | `SELECT_NONE` at one request |
| an entry declares an unknown power | `ValueError` — a registry bug, not a decision |

Loading nothing is always safe: the agent proceeds with its defaults. That is why
every failure path lands there and why `ASK_USER` is reserved for the one case
where a person's answer is worth more than silence.

## Cost arithmetic

From `jevkit.cost` at the published $42 per billion input tokens ($0.042/Mtok) for
`jev-1.13.0`, and `jevkit.limits.estimate_tokens` over the real request bodies for
the 296-entry catalogue in `examples/skill_selection.py`. Run it without a key and
it prints this table for itself.

| | input tokens | dollars |
| --- | --- | --- |
| round one, shard 0 (254 entries) | 6,166 | $0.000259 |
| round one, shard 1 (42 entries) | 1,317 | $0.000055 |
| **round one, both shards** | **7,483** | **$0.000314** |
| round two, a 6-candidate shortlist | 2,493 | $0.000105 |
| **one two-round decision** | **9,976** | **$0.000419** |
| a turn that needs nothing (round one only) | 7,483 | $0.000314 |
| 1,000 two-round decisions | 9,976,000 | $0.42 |

Where it goes:

- **The summaries are round one.** The `nominate` instructions are 298 tokens; the
  254 summaries around them are ~5,400. Shortening summaries is the only real lever
  on round one's floor.
- **The per-candidate Nouls are round two.** Each `fits_<id>` question is 242
  tokens, so a shortlist of 8 costs ~1,900 tokens of question text against 546 for
  the fixed `pick` and `description_injection` pair. `SHORTLIST_MAX` is a price,
  not a free parameter.
- Output tokens are free, and there is no prompt cache in the API surface this
  repo is written against (`docs/api-notes.md`), so nothing here amortises.

### The prompt bloat this is measured against

The hypothesis behind the pattern is "precise activation without prompt bloat".
The honest measurement is **how much skill text ends up in the agent's own prompt**:

| | tokens in the agent's prompt, per turn |
| --- | --- |
| every description pasted in (296 entries) | 18,834 |
| this pattern, one skill selected | 88 |
| this pattern, two skills selected | 158 |
| this pattern, nothing selected (the common case) | 0 |

Those numbers are `activation_tokens()` over the same catalogue. They are counts of
text, not a claim about the agent's accuracy or its bill — the caller's model's
price is not in this repo, so no dollar comparison between the two columns is
offered.

**And the honest other direction:** the Jev side is not free. For this catalogue's
description lengths the two rounds cost more tokens than the whole catalogue does
until about **54 entries** (`catalogue_tokens` 3,489 against `decision_tokens`
3,457 at that size). Below that, pasting everything into the prompt is simply
cheaper and this recipe is the wrong tool. The pattern pays when the catalogue is
large, when its text would otherwise be paid on every turn, and when the prompt
space matters for its own sake.

## Honest limits

- **It is two round trips, and that is latency in front of every turn.** A turn
  that needs nothing costs one; a turn that loads something costs two, serialised.
  If the catalogue fits one Choice and the descriptions are short, one round with
  the full text in `criteria` would be a better recipe than this one.
- **Precision over recall, on purpose.** The ranking and the per-candidate Noul
  both have to agree, `__none__` winning round two vetoes everything, and the floors
  are set high for skills that act. A skill that would have helped will sometimes
  not be loaded. That is the trade this pattern makes; if a missed activation costs
  you more than a wrong one, lower `FITS_MIN` and raise `SELECTION_LIMIT` — they are
  one block at the top of the file.
- **The thresholds were chosen by argument, not fitted.** They come from which
  mistakes are worth what, not from labelled turns. Run your own turn log through
  `decide()` and move them.
- **A shard is still an arbitrary cut.** Round-robin nomination keeps the
  *comparison* honest, but a shard with four genuinely good candidates sends three
  and reports the fourth in `dropped`. If your catalogue clusters, order it so that
  related entries do not all land in one shard.
- **`__none__` mass is a reference, not a calibration.** "Beat your shard's
  `__none__`" is a comparison the normalisation supports; it is not a claim that
  `__none__` means the same thing in two different shards.
- **Summaries carry round one, and they are the caller's to write.** A skill whose
  one-liner does not describe it will not be nominated, and nothing in round two can
  rescue it. This recipe makes the catalogue's own metadata load-bearing.
- **It judges the turn, not the conversation.** Ten turns of drift can end
  somewhere no single turn would have gone. Re-deciding each turn is cheap; carrying
  a skill forever because turn 3 needed it is the caller's decision, not this one's.
- **English first.** The model's other languages are weaker (`docs/api-notes.md`),
  and a one-line summary is very little text to judge. Watch the confidence on
  non-English turns, or raise the floors for them.
