# RAG reranking

`jevkit.recipes.rerank` — reorder retrieved passages against a query, drop the ones that
should never reach the answering model, and hand the caller the evidence to score the
ranking on its own labels.

## The decision

A retriever returned a list of candidate passages, cheaply and by similarity. For each
candidate, three things have to be decided before any of them enters an answering model's
context:

1. **how useful it is** for answering this query, as a position on a rubric;
2. **whether it is safe to show** — a retrieved passage is the one input in a RAG pipeline
   that an outsider can write, and a passage that addresses the reader is not material;
3. **whether it contradicts what the query assumes**, which is a property worth surfacing
   and never a reason to hide the passage.

The answer is a permutation of a list the caller already holds, plus a set of drops.
`RerankDecision.order` is the caller's own candidate ids, best first; `top(k)` is what goes
to the answering model; `dropped` never does. `judged[id]` carries the relevance score, its
0..1 normalisation, the confidence, both noul values, the retriever's original rank and
every drop or flag reason, so the ranking can be scored afterwards against the caller's own
labels rather than believed.

Why a decision model rather than an LLM: a reranker sits on the critical path of every
retrieval and its output is a number per candidate — the shape a text model is worst at.
Asking one to "return the ids in order" invites it to invent an id, drop one, or reorder on
a whim, so the caller needs a parser and a reconciliation step against the list it sent.
Here each candidate gets its own Score against its own question, the questions are evaluated
independently against one state, and the ordering is a `sorted()` call in this module that a
reviewer can read. Nothing in the answer is a string the caller dereferences: candidates are
keyed by labels (`c0`, `c1`, …) this module mints from the retriever's order, and the
caller's ids and payloads are never serialised into the request at all.

## The two modes

Both are implemented; the caller picks, and the trade-off is real.

| | `MODE_BATCHED` | `MODE_PER_PAIR` |
| --- | --- | --- |
| requests | one for the whole set (or one per shard) | one per query-candidate pair |
| state per candidate | shares 32k tokens with every other candidate | the whole state to itself |
| judged in sight of rivals | yes | no |
| bounded by | `MAX_CANDIDATES_PER_BATCH` and the 64k/32k token budgets | `MAX_REQUESTS` |
| latency, via `rerank_async` | one round trip | one round trip, if `FAN_OUT_CONCURRENCY` covers the fan-out |
| latency, via `rerank` | one round trip per shard, serial | **one round trip per candidate, serial** |
| rate limits | trivial | 1,200 requests/minute is reachable; pass a `RateLimiter` |

**Batched is right** for the ordinary case: ten to thirty short passages, a normal query,
one round trip. It is also the only mode where a candidate is scored while its rivals are
visible in the same state.

**Per-pair is right** when the candidates are long enough that sharing one state degrades
them — a set of 2,000-token passages where the batched state would be mostly other people's
passages — or when a shard boundary would otherwise split the set in a way you cannot
defend. It costs one request per candidate and repeats the query in every one of them, which
is exactly the cost that shows up in the arithmetic below when the query is long.

Per-pair is not a second round trip for one decision. Each request carries its own state and
its own candidate, no request's questions depend on another's answer, and nothing in them has
to be ordered; `decide()` merges the replies afterwards. The questions themselves are
byte-identical in the two modes (`test_the_questions_are_identical_in_both_modes`), so the
only variable between them is how much material shares a state.

**Which entry point you call decides whether the fan-out is concurrent.** `rerank_async` runs
the plan through `AsyncJev.map` with `FAN_OUT_CONCURRENCY` requests in flight. The token and
dollar numbers below are the same either way — the requests are identical, and only the wall
clock differs. The synchronous `rerank` accepts per-pair mode too
and sends the identical requests **one at a time, waiting for each**: a candidate per
request, in candidate order, so 32 candidates is 32 serial round trips and at `MAX_REQUESTS`
it is 64 (`test_the_sync_entry_point_sends_a_per_pair_fan_out_serially` for the shape,
`test_a_per_pair_run_sends_one_request_per_candidate_in_order` at 32). That is a supported way to
run it — the reason to choose per-pair is accuracy, not latency — but if the fan-out needs to
overlap, it has to be `rerank_async`. Both produce the same decision from the same answers.

**A batched set that does not fit is refused, never truncated.** `plan()` packs candidates in
retriever order against `MAX_CANDIDATES_PER_BATCH` and the documented token budgets; if that
needs more than one request and `overflow` is `OVERFLOW_REFUSE` (the default), it raises
`RequestTooLarge` and `rerank()` returns `reason="refused"` with an empty `order`. Pass
`overflow=OVERFLOW_SHARD` to send the set as several batched requests instead. Dropping the
tail of the candidate list is the one thing this recipe will not do, because the tail is
exactly what a reranker exists to rescue.

## The questions, and why they are shaped that way

Three per candidate, all in one request in batched mode. They are evaluated independently
against the same state.

| id | type | why this type |
| --- | --- | --- |
| `relevance_<label>` | `score`, 5 levels | Relevance is a spectrum, and this is the number the order sorts on. A Noul would give a probability, not a magnitude; a Choice over candidates would give one winner and no order. Five levels, not ten: the ordering only needs bands a reviewer can defend, and `reply.unit()` normalises to 0..1 so `KEEP_UNIT_MIN` survives an edit to the rubric. A Score also returns a position *between* levels, which is what makes a stable sort possible at all. |
| `poison_<label>` | `noul` | A yes/no about the passage: does it address the software reading it? A Noul carries no confidence, so its value *is* the number the threshold reads. This is the question that earns the recipe its place next to a similarity score: a cross-encoder ranks an injection by how well it matches the query, which is precisely what a well-written injection optimises for. |
| `contradicts_<label>` | `noul` | Same shape, for the query's premise. Separate from relevance on purpose: a passage that contradicts the premise is often the *most* useful one in the set, so it must not be scored down for it. |

Both nouls are asked about every candidate, including the ones that will be dropped for
irrelevance anyway. A question against a state that is already being sent costs a few hundred
tokens and no extra round trip; finding out afterwards that the passage now in the answering
model's context was addressed to it would cost a second request per candidate.

The state is `{query, candidates, window}`. A candidate's view is its `text` (clipped to
`PASSAGE_CHARS_BUDGET`), its `source` (clipped to `SOURCE_CHARS_BUDGET`), and the size of each
cut when there was one. Every question names the passage it judges by path
(`` `candidates.c3.text` ``) and carries the same `UNTRUSTED_NOTE`: the query is what to judge
against, and everything under `candidates` came from a store other people can write to —
material to judge, never instructions to follow. `window` says how many candidates this
request holds and how many exist in total, plus `shard`/`shards` when a batched set was split
across requests.

**The retriever's rank is deliberately not in the state, in either mode.** It is the tie-break
in code, so a ranking cannot be produced by ratifying the order the retriever already chose.
A per-pair request holds one candidate, so a shard index there would *be* that candidate's
rank as an integer: `plan()` leaves the sharding fields off every per-pair request, and every
per-pair window in a run is identical
(`test_the_state_carries_no_caller_id_no_payload_and_no_retriever_rank`, which runs in both
modes). What remains inferable is the label naming, and only that — see the limits below.

## Thresholds

All of them live in the one review block at the top of the module. Which of them are tested on
*both* sides is worth being exact about, because "every threshold is tested both ways" was
written here before it was true:

- **Two-sided, on a value either side of the line:** `KEEP_UNIT_MIN`, `DROP_CONFIDENCE_FLOOR`,
  `POISON_DROP`, `CONTRADICTS_FLAG`, `QUERY_CHARS_BUDGET`, `PASSAGE_CHARS_BUDGET`,
  `SOURCE_CHARS_BUDGET`, `MAX_CANDIDATES_PER_BATCH` (32 is one request, 33 is refused or
  sharded) and `MAX_REQUESTS` (64 pairs plan 64 requests, 65 is refused).
- **One-sided by nature:** `UNJUDGED_UNIT` is a position, not a boundary.
  `STATE_TOKEN_RESERVE` is asserted against the longest question a packed shard asks, which
  is the quantity the 32k state-plus-longest-question budget pairs the state with
  (`test_sharding_splits_on_tokens_not_only_on_the_candidate_count`). Asserting it against
  `BATCH_STATE_TOKENS` instead would be algebra: that constant is defined as the 32k budget
  minus this reserve, so the two cannot disagree.
- **Asserted as a value that is passed through:** `FAN_OUT_CONCURRENCY` is what `rerank_async`
  hands `AsyncJev.map` unless the caller overrides it
  (`test_the_fan_out_caps_what_is_in_flight_at_the_documented_concurrency`).

| threshold | value | what it gates |
| --- | --- | --- |
| `KEEP_UNIT_MIN` | 0.38 | Normalised relevance a candidate must reach to stay in. The midpoint between "Adjacent" (0.25) and "Partial" (0.50): a passage the model places nearer Partial survives, nearer Adjacent does not. |
| `DROP_CONFIDENCE_FLOOR` | 0.60 | Confidence a *low* relevance score needs before it removes a candidate. Below it the passage is kept and flagged `low_confidence`. There is no threshold in the other direction, because keeping a candidate is not a side effect. |
| `POISON_DROP` | 0.55 | Probability at which a passage counts as carrying instructions aimed at its reader and is **dropped**, whatever it scored. Just above a coin flip, and deliberately not lower: product documentation legitimately tells a human what to do, and the question is worded to separate the two. |
| `CONTRADICTS_FLAG` | 0.60 | Probability at which a passage is **flagged** as contradicting the query's premise. Never a drop. |
| `UNJUDGED_UNIT` | 0.5 | Where a candidate with no usable relevance answer is ranked: the middle. Never dropped, never promoted, always named in `unjudged`. |
| `MAX_CANDIDATES_PER_BATCH` | 32 | Candidates one batched request judges, so 96 questions. A ceiling a reviewer can see rather than one that emerges from how long the passages happened to be. |
| `MAX_REQUESTS` | 64 | Requests one rerank will ever send, in either mode. A longer per-pair fan-out is refused. |
| `FAN_OUT_CONCURRENCY` | 8 | Requests in flight during a per-pair fan-out. |
| `PASSAGE_CHARS_BUDGET` | 8,000 chars | Characters of one candidate's `text` that reach the state. A longer passage is clipped **in the state only** — it is still kept or dropped whole — and the clip is reported as `clipped_chars` plus a `clipped` flag, which also forces `screened` to False. |
| `SOURCE_CHARS_BUDGET` | 400 chars | Characters of one candidate's `source` that reach the state. `source` is caller metadata — a path, a URL, a collection name — and 400 characters is generous for one; without a bound, a single candidate's `source` could take an unbounded share of a batched request and push the others into another shard. Clipped rather than refused, because whoever can put a document into the store can often name it too, and an over-long name must not be able to refuse a whole rerank. Reported as `source_clipped_chars` plus a `source_clipped` flag; it does **not** touch `screened`, because no question judges `source` either way. |
| `QUERY_CHARS_BUDGET` | 8,000 chars | A longer query is refused, not clipped: ranking against the head of a query ranks against a different query. |
| `STATE_TOKEN_RESERVE` | 4,000 tokens | Held back from the 64k/32k budgets for the questions and the JSON envelope. |

## Deterministic order

`decide()` sorts the survivors by `(-ranking_unit, rank)`: normalised relevance first,
the retriever's original rank as the tie-break. Same answers, same order, every time — a set
where every candidate scores identically comes back in exactly the retriever's order, in
whichever order it was handed in (`test_ties_keep_the_retrievers_order_in_either_direction`).

`max_kept` is the caller's own top-k. It is applied *after* the sort, and every candidate it
cuts is reported in `dropped` with the reason `beyond_max_kept`. Nothing is ever cut
silently: a clip is reported, a shard boundary is reported, and a set that does not fit is
refused rather than shortened.

## Failing closed, in two directions

The two mistakes are not symmetric, and the code says so:

- **The poison check gates showing the passage.** No usable answer — missing, or rejected by
  `jevkit.answers` — drops the candidate with reason `unscreened`. Nothing certified it safe
  to show, so it is not shown.
- **The relevance score gates dropping the passage.** A missing or rejected relevance answer
  keeps the candidate at `UNJUDGED_UNIT` and flags it `unjudged`. A low score without
  `DROP_CONFIDENCE_FLOOR` behind it keeps it and flags it `low_confidence`. Dropping is the
  side effect here; keeping is the default.
- **A contradiction is never a drop.** Flagged in `contradicting`, left in the order.
- **An unanswered premise question is not a clean pass.** A missing or rejected
  `contradicts_<label>` answer keeps the candidate — a contradiction is never a drop — but it
  is named in `contradiction_unscreened` and it takes `screened` down with it. Without that,
  an empty `contradicting` would mean both "nothing contradicts the premise" and "nobody
  looked", which is the more dangerous of the two readings to lose
  (`test_an_unusable_contradiction_answer_is_named_and_costs_the_screened_claim`).
- **A failed request ranks nothing.** `reason="failed"`, `order=()`, and the retriever's own
  order in `unscreened`. The caller may still use it, but this module will not hand it back
  as a ranking it checked. `AsyncJev.map` fails the whole fan-out if any request fails, for
  the same reason: a partial rerank silently drops the candidates whose request never came
  back.
- **`screened` is the one-line check.** It is True only when the reason is `reranked` and all
  three of `unjudged`, `partly_screened` and `contradiction_unscreened` are empty. A clipped
  passage was screened on its head, so `partly_screened` names it and `screened` goes False:
  certifying the head of a passage is not certifying the passage.
- **A failure still reports what it cost.** `requests` on a `failed` decision counts requests
  *attempted*, not requests that came back: the failed one was sent, may have been retried by
  the SDK, and the rest of a fan-out was dispatched and billed. Read spend off `jev.ledger`,
  but the Decision will not hide it either.

A malformed *call* — unknown mode, blank query, duplicate candidate ids — still raises.
That is a bug, not a runtime condition.

## The cost arithmetic

Jev costs **$42 per billion input tokens**: `jevkit.cost.per_million("jev-1.13.0")` is
`0.042` dollars per Mtok, and output tokens are free. `estimate_cost()` prices a candidate
set before anything is sent, and `measured_cost(jev.ledger, …)` reports what a real run
actually spent (it refuses an empty ledger, and one holding replies from an unpriced model,
rather than averaging something wrong).

Every number below is `estimate_cost` output, regenerated by
`tests/test_rerank.py::test_the_documented_cost_table_is_what_estimate_cost_produces` and
`::test_the_documented_per_question_tokens_are_what_build_questions_produces` — so if a
question or the state shape changes, the tests fail rather than the doc going quietly stale.
Tokens come from `jevkit.limits.estimate_tokens`, which is `ceil(len(json) / 4)`: a
**four-characters-per-token estimate, not a tokenizer**. A live call reports its own count,
normally lower, and `measured_cost(jev.ledger, …)` is the only thing here that reports a real
one. Nothing on this page is a latency or a live-run figure; there is no key in this repo and
no such number was produced.

These are **costs, not savings**: this repo has nothing to compare them against, because the
thing they would be compared against is your existing reranker.

Per candidate, one rerank asks three questions:

| question | tokens |
| --- | --- |
| `relevance_<label>` | 374 |
| `poison_<label>` | 306 |
| `contradicts_<label>` | 258 |
| **one candidate's questions** | **938** |

The passage itself is small beside that: the eight candidates in `examples/rerank.py` cost
31–89 tokens each in the state. So the fee is dominated by the fixed question block, once per
candidate, in **both** modes. The two sets priced below are the eight candidates in
`examples/rerank.py`, and a synthetic set defined in `tests/test_rerank.py` — 32 passages of
exactly 900 characters, priced once against the example's 65-character query and once against
a 5,760-character one:

| set | mode | requests | input tokens | one rerank | per million reranks |
| --- | --- | --- | --- | --- | --- |
| the example's 8 short passages | batched | 1 | 7,955 | $0.000334 | $334.11 |
| the example's 8 short passages | per-pair | 8 | 8,257 | $0.000347 | $346.79 |
| 32 passages of 900 chars (`PRICED_SET`) | batched | 1 | 37,639 | $0.001581 | $1,580.84 |
| 32 passages of 900 chars (`PRICED_SET`) | per-pair | 32 | 38,988 | $0.001637 | $1,637.50 |
| the same 32, with a 5,760-char query | batched | 1 | 39,063 | $0.001641 | $1,640.65 |
| the same 32, with a 5,760-char query | per-pair | 32 | 84,534 | $0.003550 | $3,550.43 |

The dollar column is the token column at $42 per billion, and the last is that times a
million; to price your own set, call `estimate_cost(query, candidates, mode=…)` and read
`line()`.

Two things worth reading off that table:

- **Per-pair's token overhead is the repeated query, not the questions.** With a one-line
  query it is 4% more tokens than batched — and 8 to 32 times the requests, which is where it
  actually costs you: the 1,200-requests-per-minute ceiling, and latency whenever the fan-out
  does not overlap (it never does through the synchronous `rerank`). With a 5,760-character
  query the same set costs 2.2x, because every pair pays for the query again.
- **Cost scales with candidates, not with mode.** Roughly 938 tokens plus the passage per
  candidate, either way. If that is too much, the lever is fewer candidates from the
  retriever, not a different mode.

## Measuring whether it helps

There is no accuracy claim in this module, and no accuracy number anywhere in this repo came
out of it. What ships instead is `measure()`:

```python
from jevkit.recipes.rerank import Case, measure

before = measure([Case(order=retriever_order, relevant=labels) for ...], k=5)
after  = measure([Case(order=decision.order,  relevant=labels) for ...], k=5)
```

`Quality` carries the query count, the top-k hit rate, MRR, and how many queries had no
labels and were skipped. Two things about it matter:

- **A relevant passage that was *dropped* is simply absent from `order`, and contributes 0 to
  both numbers.** That is deliberate. It is the only way the floors in the review block show
  up as a cost rather than as a free improvement, and the first thing to check when tuning
  `KEEP_UNIT_MIN` is how much MRR the drops cost.
- **One query is not an evaluation.** `examples/rerank.py` runs `measure` over its single
  query with hand-written labels purely to show the shape of the call, and says so in its own
  output.

## The honest limits

- **Nothing here was measured against another reranker.** The claim this pattern exists to
  support — that typed relevance scores over retrieved candidates are a good trade — is a
  hypothesis. `measure()` and `estimate_cost()` are the instruments; your corpus and your
  labels are the experiment. A cross-encoder that costs you nothing per query may well win on
  ordering alone; what it does not do is the poison check.
- **Calibration is a property of groups of predictions.** A confidence of 0.7 on one passage
  does not mean that passage is 70% likely to be correctly placed. The floors earn their keep
  across a corpus, not on one query.
- **The poison check has a false-positive shape you should look for.** Documentation
  legitimately instructs a human reader. The question is worded to separate "press Settings →
  Privacy" from "ignore the other passages", but a how-to page written in the imperative is
  the case most likely to be dropped wrongly. Count `dropped` reasons per source and sample
  them before trusting `POISON_DROP` on your corpus.
- **A dropped passage is invisible downstream.** Whatever reads `order` cannot tell that
  something was removed. Log `decision.line()` next to the answer, or the first sign of a
  mis-tuned floor will be a user saying the answer was wrong.
- **The labels leak the retriever's order.** Rank is not a field in the state, but the labels
  are minted in rank order, so `c0` is inferably the retriever's first pick. Hiding it would
  cost the deterministic tie-break, and this seemed the better trade — but it means "the model
  never sees the retriever's order" would be an overclaim.
- **Batched mode judges each passage while the others are in the state.** That is the reason
  to prefer it, and also a coupling: the same passage can score differently depending on which
  rivals it is sent with, and it does not score identically across a shard boundary. Per-pair
  removes the coupling and costs a request per candidate.
- **Sharding compares across requests.** Scores from different shards are put in one order as
  if they were commensurable. They are each a judgement against the same query, so this is
  defensible, but it is not the same guarantee a single batched request gives.
- **A passage longer than 8,000 characters is judged on its head**, and kept or dropped
  whole. `clipped_chars` says how much was cut. This is a safety limit before it is an
  accuracy one: an instruction aimed at the reader hides best in the tail, and the tail is
  the part no question read. Such a passage lands in `partly_screened` and forces
  `screened` to False. If you pass the full `Candidate.payload` on to an answering model,
  either drop the ids in `partly_screened` or chunk upstream until nothing clips — the
  scores, and the poison check, are about the text that was sent.
- **`source` is not screened.** Every question is asked about `` `candidates.<label>.text` ``,
  so a candidate's `source` is context for the relevance Score and nothing judges it. It is
  bounded (`SOURCE_CHARS_BUDGET`) so that it cannot quietly take over a batched request, and
  a cut is reported — but keep it to what it is for, a short label for where the passage came
  from. Material that needs screening belongs in `text`.
- **English is the primary training language.** Non-English passages are judged with lower
  accuracy and the thresholds do not know that. Count drops per language before assuming the
  recipe behaves the same across them.
- **A Jev outage is an outage.** `reason="failed"` with nothing screened is the safe answer,
  not a useful one. Decide in advance whether the fallback is the retriever's unscreened
  order or answering without retrieval, and alert on `failed` and `refused` either way.
- **`MAX_REQUESTS` bounds a fan-out, it does not pace it.** A per-pair rerank of 64
  candidates is 64 requests as fast as concurrency allows — and through the synchronous
  `rerank`, 64 one after another; pass a `jevkit.pacing.RateLimiter` to the client if several
  reranks can be in flight at once.
