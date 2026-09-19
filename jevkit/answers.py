"""Reading answers: typed accessors, local sanity checks, and confidence bands.

Jev returns a distribution over the options you supplied, so the value your code
acts on is always one of your own constants. These helpers keep that guarantee
explicit: `Reply` exposes the answer fields, the measurements (latency, tokens,
dollars), and nothing that would require parsing prose.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from typesafe_sdk import ChoiceAnswer, NoulAnswer, ScoreAnswer, SystemOneResponse

from . import cost
from .errors import AnswerRejected

#: Distributions are floats that sum to 1; allow for rounding on the wire.
SUM_TOLERANCE = 0.02

Band = Literal["low", "medium", "high"]


def offered(question: Any) -> Any:
    """The criteria of a question, whether it is an SDK object or a raw dict."""
    if isinstance(question, Mapping):
        return question.get("criteria")
    return getattr(question, "criteria", None)


def _finite_unit(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and 0 <= value <= 1


def _check_distribution(probabilities: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(probabilities) != expected:
        raise AnswerRejected(
            f"{label}: probabilities cover {sorted(probabilities)}, expected {sorted(expected)}"
        )
    if not all(_finite_unit(p) for p in probabilities.values()):
        raise AnswerRejected(f"{label}: probabilities are not all finite values in [0, 1]")
    if abs(sum(probabilities.values()) - 1) > SUM_TOLERANCE:
        raise AnswerRejected(f"{label}: probabilities sum to {sum(probabilities.values()):.4f}, not 1")


def check_choice(answer: ChoiceAnswer, options: Mapping[str, Any], label: str = "choice") -> ChoiceAnswer:
    """Reject a Choice answer that is not a clean pick from `options`."""
    if answer.choice not in options:
        raise AnswerRejected(f"{label}: chose {answer.choice!r}, which was not offered")
    _check_distribution(answer.probabilities, set(options), label)
    top = max(answer.probabilities.values())
    if answer.probabilities[answer.choice] < top - 1e-6:
        raise AnswerRejected(f"{label}: {answer.choice!r} is not the highest-probability option")
    if not _finite_unit(answer.confidence):
        raise AnswerRejected(f"{label}: confidence {answer.confidence!r} is not in [0, 1]")
    return answer


def check_score(answer: ScoreAnswer, levels: int, label: str = "score") -> ScoreAnswer:
    """Reject a Score answer whose value or distribution does not match `levels`."""
    if len(answer.legend) != levels:
        raise AnswerRejected(f"{label}: legend has {len(answer.legend)} levels, expected {levels}")
    _check_distribution(answer.probabilities, set(answer.legend), label)
    in_range = (
        isinstance(answer.score, (int, float))
        and math.isfinite(answer.score)
        and 0 <= answer.score <= levels - 1
    )
    if not in_range:
        raise AnswerRejected(f"{label}: score {answer.score!r} falls outside 0..{levels - 1}")
    if not _finite_unit(answer.confidence):
        raise AnswerRejected(f"{label}: confidence {answer.confidence!r} is not in [0, 1]")
    return answer


def check_noul(answer: NoulAnswer, label: str = "noul") -> NoulAnswer:
    """Reject a Noul answer outside [0, 1]."""
    if not _finite_unit(answer.noul):
        raise AnswerRejected(f"{label}: noul {answer.noul!r} is not in [0, 1]")
    return answer


@dataclass(frozen=True)
class Reply:
    """One evaluated request: the answers, plus what it cost to get them.

    `usd` is None when the responding model carries no price in jevkit.cost.
    """

    response: SystemOneResponse
    latency_ms: float
    questions: Mapping[str, Any]

    @property
    def model(self) -> str:
        return self.response.model

    @property
    def input_tokens(self) -> int:
        return self.response.usage.input_tokens

    @property
    def usd(self) -> float | None:
        return cost.usd_for(self.response.model, self.response.usage.input_tokens)

    def _answer(self, qid: str) -> Any:
        try:
            return self.response.answers[qid]
        except KeyError:
            raise AnswerRejected(f"no answer returned for question {qid!r}") from None

    def choice(self, qid: str) -> ChoiceAnswer:
        """The validated Choice answer for `qid`."""
        answer = self._answer(qid)
        if not isinstance(answer, ChoiceAnswer):
            raise AnswerRejected(f"{qid}: expected a choice answer, got {type(answer).__name__}")
        return check_choice(answer, offered(self.questions[qid]) or {}, qid)

    def score(self, qid: str) -> ScoreAnswer:
        """The validated Score answer for `qid`."""
        answer = self._answer(qid)
        if not isinstance(answer, ScoreAnswer):
            raise AnswerRejected(f"{qid}: expected a score answer, got {type(answer).__name__}")
        return check_score(answer, len(offered(self.questions[qid]) or []), qid)

    def noul(self, qid: str) -> float:
        """The probability that `qid` is true, in [0, 1]."""
        answer = self._answer(qid)
        if not isinstance(answer, NoulAnswer):
            raise AnswerRejected(f"{qid}: expected a noul answer, got {type(answer).__name__}")
        return check_noul(answer, qid).noul

    def picked(self, qid: str) -> str:
        """The chosen option id of a Choice question."""
        return self.choice(qid).choice

    def unit(self, qid: str) -> float:
        """A Score normalised to 0..1, so weights in code stay independent of level count.

        A single-level Score cannot exist (the API requires two), so the divisor is never zero.
        """
        answer = self.score(qid)
        return answer.score / (len(answer.legend) - 1)

    def confidence(self, qid: str) -> float:
        """Confidence for a Choice or Score answer. Noul answers carry none."""
        answer = self._answer(qid)
        if isinstance(answer, ChoiceAnswer):
            return self.choice(qid).confidence
        if isinstance(answer, ScoreAnswer):
            return self.score(qid).confidence
        raise AnswerRejected(f"{qid}: a noul answer has no confidence; threshold its value instead")

    def probabilities(self, qid: str) -> Mapping[str, float]:
        """The full distribution behind an answer, for code that wants the shape itself."""
        answer = self._answer(qid)
        if isinstance(answer, NoulAnswer):
            raise AnswerRejected(f"{qid}: a noul answer has no distribution; its value is the probability")
        return dict(answer.probabilities)

    def band(self, qid: str, *, low: float, high: float) -> Band:
        """Split confidence into act / proceed-with-care / do-not-act.

        `low` and `high` belong to the caller: the stakes of the action decide them.
        """
        if not 0 <= low <= high <= 1:
            raise ValueError(f"expected 0 <= low <= high <= 1, got low={low}, high={high}")
        value = self.confidence(qid)
        if value >= high:
            return "high"
        if value >= low:
            return "medium"
        return "low"

    def top(self, qid: str, n: int = 3) -> list[tuple[str, float]]:
        """The n most probable options, highest first. Ties break on option id."""
        items = self.probabilities(qid).items()
        return sorted(items, key=lambda kv: (-kv[1], kv[0]))[:n]
