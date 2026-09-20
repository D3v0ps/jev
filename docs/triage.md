# Ticket and email triage

`jevkit.recipes.triage` — classify an inbound ticket, put it in a queue the caller already
owns, and say what automation is allowed to do with it. Built for volume: one request per
ticket, a batch path that keeps failures visible, and cost arithmetic you run on your own
numbers.

## The decision

A message arrives. Before anyone reads it, a support system has to settle several things
at once:

- **what it is about** — one of the caller's categories
- **who works it** — one of the caller's queues
- **how fast** — a rung of a written urgency rubric
- **how the sender sounds** — a rung of a frustration rubric
- **does it ask for money back**, **does it carry reproduction steps**, **does a person
  have to read it first** — three probabilities
- and, because the body is a place strangers can write, **is it trying to triage itself**

Every answer is an index into a table the caller already holds. `Desk` holds the
categories and queues; `build_questions` turns them into the two Choices' options; and
`decide` checks the ids that come back against the same `Desk` before returning them. A
queue that was retired last quarter cannot come back from the request, and
`decision.queue` is always a key the caller had before it asked.

The caller sends the ticket to `decision.queue` and logs `decision.line()` next to it.
`decision.category` indexes the caller's own canned-reply table; `decision.auto_reply` says
whether that reply may go out with nobody reading the ticket. `decision.flag_refund` and
`decision.flag_repro` set labels. Nothing in this module sends, labels or closes anything.

Why a decision model rather than a text model: triage is a classification with a known
answer set, run on every message, where the input is written by whoever felt like writing
it. An LLM asked to "reply with the queue name" gives you a string to map back onto a
queue, a prompt that has to defend itself against the ticket body, and a bill on the whole
inbox. Jev returns the id and a calibrated number to gate it on, and cannot return an id
that was not offered.

## The questions, and why they are shaped that way

One request carries all nine. They are evaluated independently against the same state, so
seven of them are speculation that most tickets do not need — and the code says so out
loud: `flag_repro` is only computed when the chosen category is one the caller marked as
bugs, and on a billing ticket the reproduction answer is simply dropped. What that
speculation costs is measured below rather than waved at.

| id | type | why this type |
| --- | --- | --- |
| `category` | `choice` over the desk's categories | The classification itself. A Choice returns one of the supplied ids with a probability for every alternative, which is what the auto-reply and refund gates need. |
| `queue` | `choice` over the desk's queues | Asked separately from the category on purpose. A desk with a fixed category → queue map does not need this question and can drop it; a desk where a billing ticket sometimes belongs to on-call does. |
| `urgency` | `score`, 5 levels | Urgency is a spectrum and a Noul is a probability, not a magnitude. The rubric rates the consequence of waiting, not the volume of the writing, so a polite outage report outranks an angry question about a receipt. `reply.unit()` normalises it so the two priority thresholds survive an edit to the rubric. |
| `frustration` | `score`, 5 levels | A separate spectrum, because tone and urgency come apart in both directions. It gates one thing only: whether a canned answer is allowed to be the first thing this person hears back. |
| `refund_requested` | `noul` | A yes/no about this message, thresholded on its value because a Noul carries no confidence. Its criteria separate "wants money back" from "mentions a charge". |
| `has_repro` | `noul` | Whether a defect report says enough for someone else to make it happen again. This is the answer that routes a bug to engineering instead of back to the customer for details. |
| `needs_human` | `noul` | The brake. Distressing, ambiguous, or a reply saying the last answer did not help — anything a template would get wrong. It only ever switches automation off. |
| `english` | `noul` | English is the model's strongest language; other languages are accepted with lower accuracy. This does not route anything — it raises the confidence a placement needs. |
| `steering` | `noul` | The ticket is attacker-reachable text. This asks whether some of it addresses *triage* — naming a queue, ordering an auto-reply, dressed up as a system notice — rather than describing the sender's problem. A hit sends the ticket to a person with automation off. |

The state is
`{subject, body, quoted_history, sender, channel, account, truncated}`. `subject`, `body`,
`quoted_history` and `sender` are the untrusted half; every question names them in
backticks and repeats, in its own `untrusted` field, that instructions found there are
part of the material. `channel` and `account` are the caller's own labels and are trusted.
`truncated` tells the model it is looking at the head of a longer message.

`quoted_history` is why a forwarded thread works. `split_quoted` cuts the body at the first
quote marker in `QUOTE_MARKERS`, so `body` is the newest message and the older
correspondence is background the questions are told to treat as background. Without that
split, a refund asked for three replies ago reads as this message's request.

## Thresholds

All of them live in the one review block at the top of the module. Every threshold that
gates a behaviour is tested on both sides in `tests/test_triage.py`; the budgets and the
option cap are tested at their limits.

| threshold | value | what it gates |
| --- | --- | --- |
| `FLOOR_QUEUE` | 0.55 | Confidence the queue Choice needs before the ticket is placed on the strength of it. Below it the ticket goes to the desk's fallback queue. Moderate, because the mistake it guards is one click for whoever reads that queue. |
| `FLOOR_QUEUE_NON_ENGLISH` | 0.75 | The same floor when `english` did not fire. The same confidence buys less accuracy in a weaker language, so it has to be higher. |
| `FLOOR_AUTO_REPLY` | 0.85 | Confidence **both** the category and the queue need before an automated reply may go out unread. The customer sees this mistake. |
| `FLOOR_REFUND_FLAG` | 0.80 | Confidence the category needs before the refund flag is set. It feeds a money workflow. |
| `REFUND_REQUESTED_TRUE` | 0.60 | Probability at which the ticket counts as asking for money back. It also blocks an auto-reply outright. |
| `HAS_REPRO_TRUE` | 0.60 | Probability at which a bug report counts as reproducible — on a category the caller marked `bug`, and nowhere else. |
| `NEEDS_HUMAN_TRUE` | 0.40 | Probability at which a person must read the ticket first. The lowest bar of the three: a needless human read costs a minute, the reverse costs a canned reply to somebody in trouble. |
| `ENGLISH_TRUE` | 0.50 | Probability at or above which the ticket reads as English. Below it, the queue floor rises. |
| `STEERING_SUSPECTED` | 0.60 | Probability at which the ticket is treated as addressing triage. Fallback queue, automation off, nothing flagged. |
| `PRIORITY_URGENT_AT` | 0.70 | Normalised urgency at which the priority label becomes `urgent`. |
| `PRIORITY_HIGH_AT` | 0.45 | Normalised urgency at which it becomes `high`. Below, `normal`. |
| `NO_AUTO_REPLY_ABOVE_FRUSTRATION` | 0.40 | Normalised frustration at which no canned answer goes out first. Below the midpoint: "clearly annoyed" is already too late for one. |
| `EVIDENCE_TOP_N` | 2 | Options carried on the Decision as evidence, enough to shadow-route to the runner-up. |
| `STATE_TOKEN_RESERVE` | 6,000 tokens | Held back from the 32k state-plus-longest-question budget. Leaves `BODY_CHARS_BUDGET` = 104,000 characters of newest message and `QUOTED_CHARS_BUDGET` = 26,000 of history. |
| `BATCH_CONCURRENCY` | 16 | Requests in flight by default in `triage_batch`. |

The two floors that gate side effects sit above the one that gates a placement, which is
the whole shape of the policy: **a ticket may be filed on less confidence than it may be
answered on.**

## Fail closed

Every path that could not read the ticket ends in `to_a_human`: the fallback queue,
`needs_human=True`, `auto_reply=False`, no flags, `priority="unknown"`, and a `detail`
saying what happened.

| reason | when | where it goes |
| --- | --- | --- |
| `routed` | the queue cleared its floor | the chosen queue |
| `low_confidence` | it did not | fallback queue |
| `steering` | the ticket addressed triage | fallback queue, nothing flagged |
| `empty` | no subject, body or history — **no request is sent at all** | fallback queue |
| `rejected` | an answer failed `jevkit.answers`' checks, or named something off the desk | fallback queue |
| `refused` | `jevkit.limits` refused the request locally, before the network | fallback queue |
| `failed` | anything else: transport, timeout, a ticket that could not even be prepared | fallback queue |

`priority` is `unknown` rather than `urgent` on those paths on purpose: a stream of
unreadable tickets must not be able to flood the urgent lane. Neither `triage`,
`triage_async` nor `triage_batch` raises — a worker draining an inbox that raises stops
draining the inbox.

## Volume

```python
jev = AsyncJev(limiter=RateLimiter())
result = await triage_batch(jev, inbox, desk, concurrency=16)
for index, decision in result.failures:
    retry(inbox[index])
```

`triage_batch` fans out over `AsyncJev.map`, one request per ticket, at most `concurrency`
in flight. A `limiter` is attached to the client for the duration and restored afterwards;
pass one when several workers share an account, because the answer to a 429 is not to send
it. `result.decisions[i]` is about `tickets[i]`, always — an empty ticket is a Decision
with reason `empty`, a broken one is a Decision with a failing reason, neither is a gap.

**On failure.** `AsyncJev.map` fails the whole fan-out if any one request fails, which
says nothing about which ticket broke. By default `triage_batch` then asks the remaining
tickets again one at a time, so the failure lands on the ticket that caused it and a queue
can retry that ticket rather than the batch. That costs a second request for tickets that
had already succeeded, on the failure path only, and the result says so:
`BatchResult.retried` is True and the cost helpers will read high for that batch. Pass
`isolate_failures=False` and a failed fan-out instead marks every ticket in it failed —
cheaper, and it blames everyone.

## The cost arithmetic

Nothing here is a claim. `jevkit.cost` holds the one published price ($42 per billion input
tokens, output free); `Ledger` accumulates the `input_tokens` the API itself reported; and
these three helpers do arithmetic on those two things:

- `usd_per_ticket(ledger)` and `usd_per_1000_tickets(ledger)` — what the run spent, per
  request. Both refuse an empty ledger and a ledger holding replies from a model with no
  price on file, rather than average a number they cannot defend.
- `compare_to_llm(ledger, usd_per_million_input=...)` — the same tokens at **your** price.
  Optional `prompt_tokens_per_ticket`, `output_tokens_per_ticket` and
  `usd_per_million_output` let you price the LLM fairly; leave them out and the LLM side
  counts only the tokens Jev actually read, which makes `ratio` a floor, not an estimate.
- `measure_speculative_overhead(jev, ticket, desk)` — two requests on purpose, the whole
  tree against `CORE_QUESTIONS`, so the price of never needing a second round trip is a
  number from the usage report.

### Numbers, and where they came from

The figures below were produced **offline**, by the harness in `jevkit.testing` over the
ten-ticket inbox in `examples/triage.py`. The token counts are therefore
`jevkit.limits.estimate_tokens`' estimate (4 characters per token), not the model's own
tokenizer, and the dollars are that estimate priced through `jevkit.cost`. There is no API
key in this repo's test environment, so nothing here was measured against the live API.
Run `examples/triage.py` with a key and the same helpers print the real ones.

| measured | value |
| --- | --- |
| tickets in the inbox | 10 (one empty, so 9 requests) |
| input tokens per request | ~1,704 |
| Jev, per 1,000 tickets | **$0.0716** |
| the same 1,000 at $3.00/Mtok input | $5.11 → **71.4x** |
| the same 1,000 at Jev's own $0.042/Mtok | $0.0716 → 1.0x (the arithmetic's sanity check) |

The 71.4x is not a property of this pattern. It is `3.00 / 0.042` — the ratio of two
prices — and it holds only while the LLM reads the same tokens and emits nothing. Give the
LLM classifier an 800-token instruction prompt and 60 tokens of output at $15/Mtok and the
ratio moves; `compare_to_llm` takes both and will tell you. What this repo can say is the
denominator: **$0.07 per thousand tickets, measured.** The numerator is yours.

### What the speculative questions actually cost

Measured on the bug report in the example inbox, the same way:

| | input tokens |
| --- | --- |
| all nine questions, one request | 1,703 |
| `CORE_QUESTIONS` only (category + queue) | 691 |
| the seven speculative questions | **1,012 (59% of the request)** |
| asking them as a second request instead | 1,784 total (+81, the state sent twice) |

Two honest readings of that table:

1. The speculative questions are **most of the request** on a short ticket. "Almost
   nothing" is true in money — 1,012 tokens is $0.043 per thousand tickets — and false as
   a share. They are a fixed cost, so the share falls as bodies grow: on a 16,000-character
   ticket the same 1,012 tokens are 18% of the request.
2. Splitting them out does not save it. Two requests cost *more* tokens than one (the state
   is sent twice) and add a whole round trip. The question text is the lever a reviewer
   holds here, not the question count: these nine carry long criteria on purpose, and
   trimming them trims every request.

## The awkward inputs, and what happens to them

| input | behaviour |
| --- | --- |
| empty body and subject | reason `empty`, fallback queue, **zero requests** |
| subject only, no body | triaged normally; one request |
| forwarded thread | `split_quoted` puts the newest message in `body`, the rest in `quoted_history`; the questions judge `body` |
| a body longer than the budget | clipped to `BODY_CHARS_BUDGET`, `truncated: true` in the state, `decision.body_chars_cut` says how much was lost |
| non-English | sent as written — no translation step — and the queue floor rises to `FLOOR_QUEUE_NON_ENGLISH` |
| a body instructing the triage system | the `steering` question covers it; fallback queue, automation off, nothing flagged |
| more categories or queues than a Choice takes | capped at 255 with the fallback queue always kept, and every dropped id is on `decision.dropped` |

## The honest limits

- **Calibration is a property of groups.** `confidence` over a thousand tickets means
  something; on one ticket it is a number, and a confident answer can be wrong. The floors
  buy you a rate, not a guarantee, and they need re-tuning against your own labelled
  tickets — the values here are defensible starting points, not measurements of your inbox.
- **The non-English floor is a guess in the right direction.** The documented caveat says
  accuracy is lower outside English; it does not say how much lower, and 0.75 is not
  derived from anything. If non-English tickets matter to you, label a hundred of them and
  move that number.
- **`english` is itself a Jev answer.** A ticket the model misreads as English is judged on
  the low floor. The failure is quiet.
- **`needs_human` is the brake, not a category.** It switches automation off; it does not
  move the ticket. A desk that wants "everything uncertain in one place" should point
  `fallback` at that place and accept that a confident placement is not affected.
- **The refund flag is a label, not a refund.** Nothing in this module can spend money, and
  `flag_refund` at 0.80 category confidence will still be wrong sometimes. Keep the human
  step between the flag and the credit.
- **A quote marker that does not fire is a silent failure.** If your mail client's quoting
  style is not in `QUOTE_MARKERS`, the whole thread lands in `body` and an old request can
  be read as the current one. Add your marker and test it; this is the part of the recipe
  most likely to break on a real inbox.
- **The batch's failure path costs money.** Isolating one failure re-asks every other
  ticket in the fan-out. On a 500-ticket batch with a single 429, that is 499 extra
  requests. Smaller batches, or `isolate_failures=False` plus a retry queue, are the way
  out.
- **A cheap decision is not a free one.** At $0.07 per thousand tickets the interesting
  question is not the fee, it is what the misroutes cost. That number is not in this repo
  because it depends on your queues; shadow-route a share of traffic to
  `decision.top_queues[1]` and measure it.
