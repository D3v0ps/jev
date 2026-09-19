"""Documented model limits, checked locally so an oversized request fails free.

Every number here comes from https://docs.typesafe.ai/models and the primitives
pages. A request that breaks one of these would be rejected by the API with 422
after a round trip; checking first turns that into an immediate, local error.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, TypeVar

from .errors import QuestionShapeError, RequestTooLarge

#: A Choice may offer at most this many options in one question.
CHOICE_MAX_OPTIONS = 255
#: A Score needs at least two levels and takes at most ten.
SCORE_MIN_LEVELS = 2
SCORE_MAX_LEVELS = 10
#: State plus every question in one request.
CONTEXT_TOKENS = 64_000
#: State plus the single longest question.
STATE_PLUS_LONGEST_QUESTION_TOKENS = 32_000
#: Account-level ceilings, used by jevkit.pacing.
TOKENS_PER_SECOND = 250_000
REQUESTS_PER_MINUTE = 1_200

#: Conservative characters-per-token ratio. The docs put 32k tokens at roughly
#: 150k characters of English (~4.7 chars/token); 4 overestimates the token
#: count, so estimates err toward rejecting a request that would have fit.
CHARS_PER_TOKEN = 4

T = TypeVar("T")


def payload(value: Any) -> Any:
    """A JSON-serialisable view of a value, unwrapping SDK question models."""
    dump = getattr(value, "model_dump", None)
    return dump(exclude_none=True) if callable(dump) else value


def estimate_tokens(value: Any) -> int:
    """Rough token count for any JSON-serialisable value. An estimate, not a tokenizer."""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(payload(value), ensure_ascii=False, default=str)
    return -(-len(text) // CHARS_PER_TOKEN)  # ceil


def check_choice(options: Mapping[str, Any], *, name: str = "choice") -> None:
    """Reject a Choice whose option set cannot be sent as one question."""
    if not options:
        raise QuestionShapeError(f"{name}: a Choice needs at least one option")
    if len(options) > CHOICE_MAX_OPTIONS:
        raise QuestionShapeError(
            f"{name}: {len(options)} options exceeds the {CHOICE_MAX_OPTIONS} limit; "
            "shard the options with jevkit.limits.shard and merge the answers in code"
        )


def check_score(levels: Sequence[Any], *, name: str = "score") -> None:
    """Reject a Score whose level count falls outside the documented range."""
    if not SCORE_MIN_LEVELS <= len(levels) <= SCORE_MAX_LEVELS:
        raise QuestionShapeError(
            f"{name}: a Score takes {SCORE_MIN_LEVELS}-{SCORE_MAX_LEVELS} levels, got {len(levels)}"
        )


def check_request(state: Any, questions: Mapping[str, Any]) -> int:
    """Estimate a request's size and reject it if it cannot fit. Returns the estimate."""
    state_tokens = estimate_tokens(state)
    question_tokens = {key: estimate_tokens(question) for key, question in questions.items()}
    total = state_tokens + sum(question_tokens.values())
    if total > CONTEXT_TOKENS:
        raise RequestTooLarge(
            f"~{total} tokens for state plus {len(questions)} question(s) exceeds the "
            f"{CONTEXT_TOKENS} token request budget"
        )
    longest = max(question_tokens.values(), default=0)
    if state_tokens + longest > STATE_PLUS_LONGEST_QUESTION_TOKENS:
        raise RequestTooLarge(
            f"~{state_tokens + longest} tokens for state plus the longest question exceeds the "
            f"{STATE_PLUS_LONGEST_QUESTION_TOKENS} token limit"
        )
    return total


def shard(items: Sequence[T], size: int = CHOICE_MAX_OPTIONS) -> list[Sequence[T]]:
    """Split a candidate list into runs that each fit one Choice question."""
    if size < 1:
        raise ValueError("shard size must be positive")
    return [items[i : i + size] for i in range(0, len(items), size)]


def fits(state: Any, questions: Iterable[Any]) -> bool:
    """True when this state and these questions would pass check_request."""
    try:
        check_request(state, {str(i): q for i, q in enumerate(questions)})
    except RequestTooLarge:
        return False
    return True
