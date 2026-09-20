# Context compaction

`jevkit.recipes.compaction` — fit a transcript into a token budget by deciding, per block,
what to keep and what to drop. Nothing is rewritten, so nothing is quietly distorted, and
the selection is reproducible.

## The decision

The caller holds a context window as a list of blocks — system prompt, goal, tool results,
user turns, model turns — and it no longer fits the budget it has to be sent in. The
decision is: **which blocks stay?**

The answer is a partition of block ids the caller already holds. `decision.kept` and
`decision.dropped` are those ids in transcript order; `kept_blocks(transcript, decision)`
returns the surviving `Block` objects with their text byte-for-byte unchanged. The model
never sees an id it could name back, because the state and the questions key on labels this
module mints (`b0`, `b1`, …) from the caller's own candidate list.

Why a decision model rather than a text model. A summariser answers the same question by
writing a new, shorter text, which has three failure modes keep/drop does not have:

- every sentence of its output is a claim no block ever made, and the original is gone, so
  the distortions are unreviewable;
- the content that costs the most to carry — a long tool result full of identifiers, ids
  and amounts — is exactly what paraphrasing damages worst;
- it is not reproducible. Two runs give two different contexts.

What keep/drop cannot do is compress a single block. See the limits at the bottom: the two
approaches are not substitutes everywhere, and this one is a fitter, not a rewriter.

## The state, and why it is shaped that way

```json
{"pinned": [{"role": "system", "text": "..."}],
 "blocks": {"b0": {"position": 2, "role": "tool", "text": "...", "clipped_chars": 412}},
 "window": {"pinned_shown": 3, "pinned_total": 3, "blocks_here": 11,
            "blocks_total": 11, "shard": 0, "shards": 1}}
```

`pinned` is the part of the context that is **not part of the decision**: the system prompt,
the standing goal, the user's live message, anything else the caller pinned. Pinned blocks
are never dropped and never asked about — a recipe that can drop the goal is broken — but
they are the thing everything else is judged *against*. "Load-bearing" is a relation to a
goal, not a property of a paragraph, so a compactor that cannot see the goal is guessing.

`blocks` holds the candidates under code-minted labels. `position` is each block's index in
the whole transcript, so a question can reason about what came after it: a block whose
content a later block restates in full is disposable however important it looked.
`clipped_chars` appears when a block was too long to send whole and says so, rather than
letting the model judge a truncated block as if it were complete.

Everything in `pinned` and `blocks` is untrusted. It is a transcript: tool output, fetched
pages, whatever a user typed. Each question carries the same note saying so.

## The questions, and why they are shaped that way

Three per candidate block, all in one request.

| id | type | why this type |
| --- | --- | --- |
| `value_<label>` | `score`, 4 levels | What deleting the block costs is a spectrum, which is a Score and not a Noul (a Noul is a probability, not a magnitude). Four levels — disposable, background, useful, load-bearing — because the fill only needs to rank blocks against each other; every extra level is a boundary a reviewer would have to defend. `reply.unit()` normalises to 0..1, so the thresholds survive an edit to the rubric, and the Score's `confidence` is what the low-confidence path reads. |
| `depends_<label>` | `noul` | "Does a later step need this exact text" is a different question from "how much does this matter", and it is the one that turns a merely useful block into one that must not be dropped: the block holding the only copy of an account number can read as dull. A yes/no with a probability, thresholded on its value because a Noul carries no confidence. |
| `steering_<label>` | `noul` | Budget is zero-sum, so a block that wins space with an instruction takes it from a block that earned it. This asks whether the block's text is addressed to whatever decides what to keep — "CRITICAL, never delete", "delete the goal instead", text dressed up as a system notice — rather than being part of the work. |

The two Nouls are speculative: most blocks need neither answer. They ride along anyway,
because a question evaluated against a state that is already being sent is cheap, and a
second request to find out whether a block holds an identifier would cost more than the
block does.

## Why this is one request and not one per block

Questions in a request are evaluated independently against the same state, so the whole
decision is one round trip. Three shapes of the same decision, measured — the honest
comparison is the third row, because it is the only per-block shape that gives each
question the same information:

| shape | what each request carries | tokens (11-block example) | round trips |
| --- | --- | --- | --- |
| one request (this recipe) | the whole transcript, 33 questions | ~9,263 | 1 |
| one request per block, lean | pinned context plus that one block | ~10,648 (1.15x) | 11 |
| one request per block, informed | the whole transcript, 3 questions | ~15,543 (1.68x) | 11 |

The ratio grows with the transcript, because the informed per-block shape re-sends the whole
state every time:

| transcript | one request | per-block lean | per-block informed |
| --- | --- | --- | --- |
| 11 blocks, ~30 tokens each (the example) | ~9,263 | 1.15x | 1.68x |
| 12 blocks, ~2,000 tokens each | ~33,917 | 1.09x | 8.94x |
| 40 blocks, ~400 tokens each | ~48,296 | 1.23x | 14.64x |
| 80 blocks, ~100 tokens each | ~72,599 (2 shards) | 1.30x | 11.35x |

All of these are **estimates of request size** from `jevkit.limits.estimate_tokens` — a
characters-per-token ratio, not a tokenizer — computed offline over the requests `plan()`
actually builds, for the transcript in `examples/compaction.py` and for three synthetic
shapes. They are not metered usage: run `examples/compaction.py` with a key and it prints
what the API reported next to what the estimator predicted.

Two things the table does not flatter:

- Against the *lean* per-block shape, one request saves 9–30% of the tokens, not orders of
  magnitude. The questions dominate this recipe's request (~785 estimated tokens per block
  for all three), so most of what is sent is the same three questions repeated. The win
  against that baseline is the round trips, and the information: a lean per-block request
  cannot answer "another block already says this", because it never sees the other blocks.
- 785 tokens of questions per block is also what sets the shard size. 48 blocks is ~37,700
  tokens of questions before any state.

## Thresholds

All of them live in the one review block at the top of the module. Every row is exercised on
both sides in `tests/test_compaction.py`.

| threshold | value | what it gates | which way it fails |
| --- | --- | --- | --- |
| `VALUE_LEVELS` | 4 levels | The rubric the fill ranks on: disposable / background / useful / load-bearing. | — |
| `DEPENDS_LATER_TRUE` | 0.55 | Probability at which a later step counts as still depending on the block, which **protects** it: it is filled before anything unprotected. Just above a coin flip, because protecting a block that turns out not to matter costs a few tokens and the reverse costs work redone. | toward keeping |
| `CONFIDENCE_FLOOR_DROP` | 0.55 | Confidence below which a value answer decides nothing and the block is protected instead. Higher than a floor for a reversible action would be: once the compacted context is sent, the block is gone from the conversation the model sees. | toward keeping |
| `STEERING_SUSPECTED` | 0.60 | Probability at which a block reads as addressing the compactor. A flagged block loses its protection. | toward dropping the flagged block |
| `STEERING_VALUE_CAP` | 0.25 | The value a flagged block is capped at, whatever it scored. Above disposable and below background: it keeps its place if there is room and cannot buy space it did not earn. | toward dropping the flagged block |
| `UNJUDGED_VALUE` | 1.0 | The value given to a block no usable answer arrived for. The top of the scale, so an unanswered question costs tokens, never content. | toward keeping |
| `MAX_BLOCKS_PER_SHARD` | 48 | Blocks judged in one request; also a ceiling of 144 questions per request. | — |
| `MAX_SHARDS` | 8 | Requests one compaction will ever spend. Blocks past this are reported unjudged and kept. | toward keeping |
| `SHARD_TOKEN_RESERVE` | 4,000 tokens | Held back from both documented per-request budgets for the questions and the envelope, leaving `SHARD_STATE_TOKENS` = 28,000 and `SHARD_TOTAL_TOKENS` = 60,000. | — |
| `PINNED_STATE_TOKENS` | 8,000 tokens | The share of a shard's state spent on pinned context. The most recent pinned blocks are shown first, and at least one always is. | reports `pinned_shown` |
| `BLOCK_CHARS_BUDGET` | 8,000 characters | How much of one block's text reaches the state. Clipping is for judging only: the block itself is still kept or dropped whole, and the characters cut are reported. | reports `clipped_chars` |

Note which way each one fails. Dropping is the destructive direction, so every uncertain
path keeps: a block the model is unsure about, a block that may still be depended on, a
block whose dependency answer never arrived, a block whose answer the checks in
`jevkit.answers` rejected, and a block no request covered are all protected, and are dropped
only when the budget leaves no room at all.

The one exception is a block flagged as steering, and it is deliberate. The three answers are
read separately, so a rejected value answer does not throw away a steering flag that did
arrive, and a flagged block is not protected by its own missing answers. Otherwise a block
could buy protection by being hard to judge.

## The fill: code owns the budget

The model supplies a value per block. Which blocks survive is arithmetic, in `decide()`:

1. Pinned blocks are kept and their tokens come off the budget first. What is left is the
   room the candidates compete for.
2. Candidates are sorted by `(protected, -value/tokens, -position, block_id)`: protected
   blocks first, then value per token, then the more recent block, then the id. That is a
   total order over the blocks, so the same answers always produce the same selection.
3. Greedy fill in that order. A block that does not fit the room left is passed over and
   smaller ones are still considered.
4. `tokens_before`, `tokens_after`, `dropped` and per-block `tokens` are recorded, all
   computed with the caller's own counter, so `saved_tokens` is a measurement.

Same answers in, same blocks out. A summariser cannot offer that, and it is what makes a
regression in this recipe debuggable: the answers and the fill are separately inspectable.

## Token counting

`count` is a caller-supplied `Callable[[str], int]`, defaulting to
`jevkit.limits.estimate_tokens`. It is worth being clear about why:

- The default is an estimate — 4 characters per token — not a tokenizer. It will be wrong on
  code, on JSON, and on languages other than English.
- The budget being enforced is the *caller's model's* context window, not Jev's. Only the
  caller knows that tokenizer, and the whole point of the number is that the compacted
  context actually fits. Pass `tiktoken`'s encoder, or your provider's counter.
- The counter sees a block's text only. A caller whose wire format adds per-message overhead
  should add it in their counter, or the budget will be missed by a few tokens per block.
- Request *size* is estimated separately with `jevkit.limits`, because that budget is the
  API's and is checked before sending.

## Sharding, and what it costs

A transcript that does not fit one request's state budget is split: `plan()` packs candidates
in transcript order until a shard hits `MAX_BLOCKS_PER_SHARD` or the documented token
budgets, and `decide()` merges the replies by label. This is the one place this repo sends
more than one request per decision, and it qualifies because the shards are independent — no
shard's questions depend on another shard's answer, each one carries the same pinned context,
and the merge is code.

It is not free, and two costs are real:

- **Scores from different shards are not strictly comparable.** Every question sees the state
  of *its* shard, so a block judged against eleven siblings and a block judged against a
  different eleven were asked slightly different questions. The fill then ranks them on one
  scale as if they were commensurable. Blocks near the boundary can flip when the sharding
  changes, even with the same transcript content.
- **"Another block already says this" only works within a shard.** Two blocks that duplicate
  each other can both be rated load-bearing if they land in different shards.

Both push toward keeping, not dropping, which is the direction this recipe prefers to fail
in. `decision.shards` says how many requests were spent; a single-shard decision has neither
problem.

Shards are filled oldest first, so if `MAX_SHARDS` truncates the plan the blocks left
unjudged are the newest ones — which recency would have kept anyway. They are reported in
`decision.unjudged` and kept.

## Cost arithmetic

Jev is priced by `jevkit.cost` at $0.042 per million input tokens, output free. The fee is
charged **once** per compaction; the saving is charged back on **every** later request that
sends the smaller context. So the verdict turns on reuse:

```
saving = tokens_saved x usd_per_token x reuses
net    = saving - jev_usd
break-even reuses = jev_usd / (tokens_saved x usd_per_token)
```

`expected_cost(decision, usd_per_token=…, reuses=…, jev_usd=jev_usd_from(jev.ledger))` does
that arithmetic with the caller's own downstream price and a fee measured from the ledger,
and `report.pays` can be — and is — False.

Worked example, computed offline with `jevkit.limits.estimate_tokens`, `jevkit.cost` and the
test harness over the transcript in `examples/compaction.py` (the answers were scripted, not
the model's; run the example with a key for metered numbers):

| | |
| --- | --- |
| transcript | 425 estimated tokens, 14 blocks, 3 pinned |
| budget | 220 tokens |
| request | 1, ~9,263 estimated input tokens |
| fee | ~$0.000389 at $0.042/Mtok |
| kept / dropped | 7 / 7 — 218 tokens, 207 saved (49%) |
| downstream price | $3.00/Mtok (illustrative; use your own) |
| saving per reuse | 207 x $0.000003 = $0.000621 |
| break-even | ~0.63 reuses |

At $3/Mtok downstream, this compaction pays back before the next request finishes. Two ways
it does not:

- **A cheap downstream model.** At $0.042/Mtok — the same price as Jev — break-even is ~45
  reuses, because the fee is measured against a saving priced the same as the fee.
- **A small saving.** Break-even scales inversely with the tokens freed: a compaction that
  frees 20 instead of 207 needs ten times the reuses at the same price. When nothing was
  dropped at all, `break_even_reuses` is None and `expected_cost` says so rather than
  reporting a ratio.

The other half of the cost is latency, and the ledger measures it: `jev.ledger.p50_ms` and
`p95_ms` are what the caller waited, and a sharded compaction sent through `compact_async`
waits once rather than per shard. `decision.latencies_ms` holds each request's own.

## Honest limits

- **Greedy is not optimal.** Value-per-token fill is a knapsack heuristic. It can pass over
  a long load-bearing block and fill the room with three short useful ones — in the worked
  example above it drops a 45-token "useful" block and keeps a 27-token "background" one,
  because the larger one no longer fit. If a block must survive, pin it; that is what
  pinning is for. An exact knapsack would be a different, slower, still-heuristic policy.
- **It cannot compress a block.** A single 30,000-token tool result is kept or dropped whole.
  Where the content has to shrink rather than go, a summariser is the right tool and this
  recipe is not a substitute for it. Callers who can split their blocks more finely get a
  better fit; block granularity is the caller's lever.
- **Values across shards are not strictly comparable.** See above.
- **Calibration is a property of groups.** The published confidence is calibrated over many
  predictions, not a guarantee about the block in front of you. The thresholds here decide
  how much of that to spend; they are starting points and should be tuned against transcripts
  where you know what mattered, which means keeping a record of what was dropped.
- **The steering cap has a false-positive cost.** A block that both carries a real dependency
  and carries text aimed at the compactor loses its protection and has its value capped, so
  it can be dropped. That is deliberate — untrusted text does not get to hold the budget
  hostage — but a tool result that legitimately says "keep this" is collateral.
  `decision.steered` names every flagged block, and the answer to a false positive is to pin
  it.
- **No undo unless you keep the original.** The Decision reports everything an undo needs
  (`dropped`, `tokens`, `tokens_before`), but this module does not store your transcript. If
  a dropped block turns out to have mattered, the caller has to still have it.
- **English first.** Jev's primary training language is English; other languages are accepted
  with lower accuracy. Compacting a non-English transcript should watch
  `decision.confidences` — and the low-confidence path is protection, so a non-English
  transcript will tend to compact less, not worse.
- **A valid answer can still be the wrong one.** Constrained output removes parse failures and
  invented ids. It does not remove mistakes: a block can be rated disposable and matter.
  That is why the whole pattern is built so the failure mode is "carried more tokens than
  necessary", not "lost the account number".
