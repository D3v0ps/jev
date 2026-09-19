# The API this repo is written against

Verified against `https://docs.typesafe.ai/api` and `typesafe-sdk` 0.7.0 on
2026-09-19. Everything below is the whole surface: there is no fourth question
type and no other answer field. Do not invent one.

## Request

`POST https://api.typesafe.ai/v1/systemone`, `Authorization: Bearer <key>`.

```json
{
  "state": "string | object | array",
  "model": "jev-latest",
  "questions": {"<your id>": {"type": "...", "instructions": ..., "criteria": ...}}
}
```

`state` is the material to judge. Every question in a request sees the same state,
is evaluated independently, and returns under the id you chose. Question ids are
not sent to the model, so the full question goes in `instructions`. `instructions`
and every `criteria` value accept a string, object, array, or null — structure is
understood, so pass JSON rather than a serialised template. To point a question at
part of a structured state, name the path in backticks:
``Does `ticket.messages[0].text` request a refund?``

## The three question types

| Type     | `criteria`                               | Answer fields                                    |
| -------- | ---------------------------------------- | ------------------------------------------------ |
| `noul`   | optional `{true: ..., false: ...}`       | `noul` (0–1). **No confidence.**                 |
| `choice` | map of option id → description or null   | `choice`, `probabilities`, `confidence`          |
| `score`  | ordered array of 2–10 level descriptions | `score`, `legend`, `probabilities`, `confidence` |

- A Choice takes at most **255** options. `probabilities` covers every option and
  sums to 1; `choice` is the highest-probability one.
- A Score returns a probability-weighted position that can land *between* levels,
  in `0 .. len(levels) - 1`. Normalise with `reply.unit(qid)` before weighting.
- A Noul is a probability, not a magnitude. `0.5` means "yes and no are equally
  likely", not "medium". Use a Score for a spectrum.
- `confidence` is derived from the shape of `probabilities`; flat means uncertain.
  It is a convenience statistic, not a separate model output — `probabilities` is
  there when you want your own measure.

## Response

```json
{"model": "jev-1.13.0",
 "answers": {"<your id>": {"type": "choice", "choice": "...", "probabilities": {...}, "confidence": 0.82}},
 "usage": {"input_tokens": 312, "output_tokens": 48}}
```

## Limits and prices (jev-1.13.0)

| | |
| --- | --- |
| Price | $42 per billion input tokens ($0.042/Mtok). **Output tokens are free.** |
| Rate | 250,000 tokens/second, 1,200 requests/minute |
| Context | 64k tokens per request; 32k for state plus the longest question |
| Input | Text only — string, JSON object, or array. No images, audio, or video. |

Errors: `401` bad key, `422` malformed request, `429` rate limited, `529`
overloaded. The SDK retries 429/529 with backoff by default.

Aliases `jev-latest` and `jev-preview` both resolve to `jev-1.13.0` today. The
response reports the versioned id that answered; pin it once thresholds are tuned.

## Caveats worth designing around

- English is the primary training language; other languages are accepted with
  lower accuracy. Watch confidence when routing non-English content.
- Calibration is a property of groups of predictions, not a guarantee about any
  single answer.
- A valid answer can still be the wrong one. Constrained output removes parsing
  failures and invented identifiers, not mistakes.
