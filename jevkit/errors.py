"""Failures jevkit raises on its own, before or after the SDK's own exceptions."""


class JevkitError(Exception):
    """Base class for every error raised by jevkit itself."""


class RequestTooLarge(JevkitError):
    """A request was rejected locally because it cannot fit the model's context budget."""


class QuestionShapeError(JevkitError):
    """A question violates a documented limit (option count, level count, empty criteria)."""


class AnswerRejected(JevkitError):
    """An answer failed a local sanity check, so no caller-visible action was taken.

    Jev constrains answers to the options supplied, so this should not fire in
    normal operation. It exists because the alternative to checking is acting on
    a malformed answer, and these patterns gate side effects.
    """
