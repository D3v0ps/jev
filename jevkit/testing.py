"""An offline Jev: scripted answers over a mock transport, so tests cost nothing.

The SDK accepts a `transport`, so a test can exercise the real client, the real
request encoding, and the real answer parsing without a network call or an API
key. Script the answers by value and the harness builds distributions that
satisfy the same checks jevkit.answers applies to live ones.

    jev, calls = fake_jev({"risk": 0.95, "operation": "CLICK"})
    reply = jev.ask({"page": "..."}, questions)
    assert calls[0].questions["operation"]["type"] == "choice"

A plan may be one mapping (every call answers the same way), a sequence (one per
call, in order), or a callable receiving (call_index, body).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx2

from .client import AsyncJev, Jev
from .limits import estimate_tokens

#: The model a fake reply claims to come from. Priced in jevkit.cost, so tests see real dollars.
FAKE_MODEL = "jev-1.13.0"
#: Probability mass a scripted Choice puts on the option the test named.
DEFAULT_PEAK = 0.9

Values = Mapping[str, Any]
Plan = Values | Sequence[Values] | Callable[[int, dict], Values] | None


@dataclass(frozen=True)
class Call:
    """One request the fake received, decoded."""

    body: dict
    index: int

    @property
    def state(self) -> Any:
        return self.body.get("state")

    @property
    def questions(self) -> dict:
        return self.body.get("questions", {})

    @property
    def model(self) -> str | None:
        return self.body.get("model")

    def ids(self) -> list[str]:
        """Question ids in this request, in order."""
        return list(self.questions)


@dataclass(frozen=True)
class Fail:
    """A scripted HTTP failure, to exercise retry and error paths."""

    status: int
    detail: str = "scripted failure"
    headers: Mapping[str, str] = field(default_factory=dict)


def _normalise(weights: Mapping[Any, float]) -> dict[str, float]:
    items = {str(key): float(value) for key, value in weights.items()}
    if any(value < 0 or not math.isfinite(value) for value in items.values()):
        raise ValueError(f"weights must be finite and non-negative: {weights}")
    total = sum(items.values())
    if total <= 0:
        raise ValueError(f"weights must not sum to zero: {weights}")
    return {key: value / total for key, value in items.items()}


def _argmax(probabilities: Mapping[str, float]) -> str:
    return max(sorted(probabilities), key=lambda key: probabilities[key])


def _confidence(probabilities: Mapping[str, float]) -> float:
    """A stand-in for the model's own statistic: the top mass, clamped to [0, 1].

    Real confidence is derived from the whole distribution and is not this
    formula. Tests that care about a confidence value should script it.
    """
    return min(1.0, max(0.0, max(probabilities.values())))


def choice_answer(options: Sequence[str], value: Any = None, *, peak: float = DEFAULT_PEAK) -> dict:
    """A Choice answer over `options`. `value` is an option id, a weight map, or None for uniform."""
    if not options:
        raise ValueError("a Choice answer needs at least one option")
    if value is None:
        probabilities = _normalise({option: 1.0 for option in options})
    elif isinstance(value, Mapping):
        probabilities = _normalise(value)
        if set(probabilities) != set(options):
            raise ValueError(f"weight map {sorted(probabilities)} does not match options {sorted(options)}")
    else:
        picked = str(value)
        if picked not in options:
            raise ValueError(f"{picked!r} is not one of the offered options {list(options)}")
        if len(options) == 1:
            probabilities = {picked: 1.0}
        else:
            rest = (1.0 - peak) / (len(options) - 1)
            probabilities = {option: (peak if option == picked else rest) for option in options}
    return {
        "type": "choice",
        "choice": _argmax(probabilities),
        "probabilities": probabilities,
        "confidence": _confidence(probabilities),
    }


def score_answer(levels: Sequence[Any], value: Any = None) -> dict:
    """A Score answer over `levels`.

    An int puts all mass on that level, so `score` equals it exactly. A float
    splits mass across the two neighbouring levels, so `score` equals the float.
    A weight map is used as given and `score` is its weighted mean.
    """
    count = len(levels)
    if count < 2:
        raise ValueError("a Score answer needs at least two levels")
    legend = {str(index): level for index, level in enumerate(levels)}
    if value is None:
        value = (count - 1) / 2
    if isinstance(value, Mapping):
        probabilities = _normalise(value)
        if set(probabilities) != set(legend):
            raise ValueError(f"weight map {sorted(probabilities)} does not match levels {sorted(legend)}")
    elif float(value) != int(value):
        target = float(value)
        if not 0 <= target <= count - 1:
            raise ValueError(f"score {target} falls outside 0..{count - 1}")
        low = math.floor(target)
        probabilities = {str(index): 0.0 for index in range(count)}
        probabilities[str(low)] = 1 - (target - low)
        probabilities[str(low + 1)] = target - low
    else:
        index = int(value)
        if not 0 <= index <= count - 1:
            raise ValueError(f"level {index} falls outside 0..{count - 1}")
        probabilities = {str(position): float(position == index) for position in range(count)}
    score = sum(float(level) * probability for level, probability in probabilities.items())
    return {
        "type": "score",
        "score": score,
        "legend": {key: value for key, value in legend.items()},
        "probabilities": probabilities,
        "confidence": _confidence(probabilities),
    }


def noul_answer(value: Any = 0.5) -> dict:
    """A Noul answer. `value` is the probability that the statement is true."""
    probability = float(value)
    if not 0 <= probability <= 1:
        raise ValueError(f"a noul is a probability in [0, 1], got {probability}")
    return {"type": "noul", "noul": probability}


def answers_for(questions: Mapping[str, Any], values: Values) -> dict:
    """Build one answer per question, using `values[qid]` where present.

    Unscripted questions get a deliberately uninformative answer — uniform for a
    Choice, the middle level for a Score, 0.5 for a Noul — so a test only scripts
    the questions its assertion depends on. Scripting an id that was not asked is
    an error, because it usually means the question was renamed.
    """
    unknown = set(values) - set(questions)
    if unknown:
        raise AssertionError(f"scripted answers for questions that were not asked: {sorted(unknown)}")
    built = {}
    for qid, question in questions.items():
        kind = question.get("type")
        value = values.get(qid)
        if kind == "noul":
            built[qid] = noul_answer(0.5 if value is None else value)
        elif kind == "choice":
            built[qid] = choice_answer(list(question.get("criteria") or {}), value)
        elif kind == "score":
            built[qid] = score_answer(list(question.get("criteria") or []), value)
        else:
            raise AssertionError(f"{qid}: unsupported question type {kind!r}")
    return built


def response_body(body: dict, values: Values) -> dict:
    """A complete System One response for a decoded request body."""
    questions = body.get("questions", {})
    input_tokens = estimate_tokens(body.get("state")) + sum(estimate_tokens(q) for q in questions.values())
    return {
        "model": FAKE_MODEL,
        "answers": answers_for(questions, values),
        "usage": {"input_tokens": input_tokens, "output_tokens": 4 * len(questions)},
    }


def _resolve(plan: Plan, index: int, body: dict) -> Values | Fail:
    if plan is None:
        return {}
    if callable(plan):
        return plan(index, body)
    if isinstance(plan, Mapping):
        return plan
    if index >= len(plan):
        raise AssertionError(f"the fake was called {index + 1} times but the plan scripts {len(plan)}")
    return plan[index]


def make_transport(plan: Plan, calls: list[Call]) -> httpx2.MockTransport:
    """A mock transport that answers /v1/systemone from `plan` and appends to `calls`."""

    def handle(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        call = Call(body=body, index=len(calls))
        calls.append(call)
        step = _resolve(plan, call.index, body)
        if isinstance(step, Fail):
            return httpx2.Response(
                step.status,
                json={"error": {"message": step.detail}},
                headers=dict(step.headers),
            )
        return httpx2.Response(200, json=response_body(body, step))

    return httpx2.MockTransport(handle)


def fake_jev(plan: Plan = None, **kwargs: Any) -> tuple[Jev, list[Call]]:
    """A `Jev` that answers from `plan`, plus the list its calls are recorded in."""
    calls: list[Call] = []
    kwargs.setdefault("api_key", "test-key-not-a-secret")
    return Jev(transport=make_transport(plan, calls), **kwargs), calls


def fake_async_jev(plan: Plan = None, **kwargs: Any) -> tuple[AsyncJev, list[Call]]:
    """An `AsyncJev` that answers from `plan`, plus the list its calls are recorded in."""
    calls: list[Call] = []
    kwargs.setdefault("api_key", "test-key-not-a-secret")
    return AsyncJev(transport=make_transport(plan, calls), **kwargs), calls
