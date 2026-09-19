"""jevkit: ten production patterns for TypeSafe's Jev.

Jev returns a typed decision drawn from options you supply, with a calibrated
probability for each. This package holds the plumbing every pattern needs
(client, measurement, limits, pacing, answer reading) and one module per pattern
under `jevkit.recipes`.

Each recipe module opens with a single block of questions and thresholds. That
block is the part worth reviewing; the code under it only routes on what comes
back.
"""

from .answers import Reply, check_choice, check_noul, check_score
from .client import AsyncJev, Jev
from .cost import per_million, usd_for
from .errors import AnswerRejected, JevkitError, QuestionShapeError, RequestTooLarge
from .ledger import Ledger
from .limits import CHOICE_MAX_OPTIONS, SCORE_MAX_LEVELS, SCORE_MIN_LEVELS, estimate_tokens, shard
from .pacing import RateLimiter

__all__ = [
    "CHOICE_MAX_OPTIONS",
    "SCORE_MAX_LEVELS",
    "SCORE_MIN_LEVELS",
    "AnswerRejected",
    "AsyncJev",
    "Jev",
    "JevkitError",
    "Ledger",
    "QuestionShapeError",
    "RateLimiter",
    "Reply",
    "RequestTooLarge",
    "check_choice",
    "check_noul",
    "check_score",
    "estimate_tokens",
    "per_million",
    "shard",
    "usd_for",
]
