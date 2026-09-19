"""Contracts for the plumbing: limits, cost, answer checks, pacing, measurement."""

from __future__ import annotations

import math

import pytest
from typesafe_sdk import Choice, ChoiceAnswer, Noul, Score

from jevkit import cost, limits
from jevkit.answers import check_choice
from jevkit.errors import AnswerRejected, QuestionShapeError, RequestTooLarge
from jevkit.ledger import Ledger, percentile
from jevkit.pacing import RateLimiter
from jevkit.testing import Fail, choice_answer, score_answer

QUESTIONS = {
    "route": Choice(instructions="Where does this go?", criteria={"billing": None, "technical": None}),
    "urgent": Noul(instructions="Is this urgent?"),
    "severity": Score(instructions="How severe?", criteria=["none", "minor", "major"]),
}


def test_one_request_carries_every_question(jev):
    client, calls = jev({"route": "technical", "urgent": 0.92, "severity": 2})
    reply = client.ask({"ticket": "payouts failing for 3 days"}, QUESTIONS)
    assert len(calls) == 1, "speculative fan-out must not become one call per question"
    assert calls[0].ids() == ["route", "urgent", "severity"]
    assert reply.picked("route") == "technical"
    assert reply.noul("urgent") == pytest.approx(0.92)
    assert reply.score("severity").score == pytest.approx(2.0)
    assert reply.unit("severity") == pytest.approx(1.0)


def test_unscripted_questions_answer_without_information(jev):
    client, _ = jev({"route": "billing"})
    reply = client.ask("state", QUESTIONS)
    assert reply.noul("urgent") == 0.5
    assert reply.unit("severity") == pytest.approx(0.5)


def test_scripting_a_question_that_was_not_asked_is_an_error(jev):
    client, _ = jev({"typo": "billing"})
    with pytest.raises(AssertionError, match="not asked"):
        client.ask("state", QUESTIONS)


def test_measurements_are_recorded(jev):
    client, _ = jev({"route": "billing"})
    reply = client.ask({"ticket": "hello"}, QUESTIONS)
    assert reply.latency_ms >= 0
    assert reply.input_tokens > 0
    assert reply.model == "jev-1.13.0"
    assert reply.usd == pytest.approx(reply.input_tokens * 42 / 1e9)
    assert client.ledger.calls == 1
    assert client.ledger.usd == pytest.approx(reply.usd)
    assert "requests" in client.ledger.summary()


def test_ask_without_questions_is_refused(jev):
    client, calls = jev()
    with pytest.raises(ValueError, match="at least one question"):
        client.ask("state", {})
    assert calls == []


def test_http_failure_surfaces_after_retries(jev):
    client, calls = jev([Fail(429), Fail(429), Fail(429), Fail(429), Fail(429), Fail(429)])
    with pytest.raises(Exception) as caught:
        client.ask("state", {"urgent": Noul(instructions="Is this urgent?")})
    assert "429" in str(caught.value) or "rate" in str(caught.value).lower()
    assert calls, "the request must have been attempted"


async def test_async_fan_out_preserves_order(async_jev):
    client, calls = async_jev(lambda index, body: {"route": ["billing", "technical"][index % 2]})
    replies = await client.map(
        [(f"ticket {index}", {"route": QUESTIONS["route"]}) for index in range(6)],
        concurrency=3,
    )
    assert [reply.picked("route") for reply in replies] == ["billing", "technical"] * 3
    assert len(calls) == 6
    assert client.ledger.calls == 6
    await client.aclose()


# --- limits ---------------------------------------------------------------


def test_oversized_request_fails_locally(jev):
    client, calls = jev({"urgent": 0.5})
    huge = "x" * (limits.CONTEXT_TOKENS * limits.CHARS_PER_TOKEN + 1)
    with pytest.raises(RequestTooLarge, match="token request budget"):
        client.ask(huge, {"urgent": Noul(instructions="Is this urgent?")})
    assert calls == [], "an oversized request must not reach the network"


def test_state_plus_longest_question_is_checked():
    state = "x" * (limits.STATE_PLUS_LONGEST_QUESTION_TOKENS * limits.CHARS_PER_TOKEN - 400)
    question = Noul(instructions="y" * 2000)
    with pytest.raises(RequestTooLarge, match="longest question"):
        limits.check_request(state, {"q": question})


def test_choice_option_ceiling():
    limits.check_choice({str(i): None for i in range(limits.CHOICE_MAX_OPTIONS)})
    with pytest.raises(QuestionShapeError, match="shard"):
        limits.check_choice({str(i): None for i in range(limits.CHOICE_MAX_OPTIONS + 1)})
    with pytest.raises(QuestionShapeError, match="at least one option"):
        limits.check_choice({})


@pytest.mark.parametrize("count,ok", [(1, False), (2, True), (10, True), (11, False)])
def test_score_level_range(count, ok):
    levels = ["level"] * count
    if ok:
        limits.check_score(levels)
    else:
        with pytest.raises(QuestionShapeError):
            limits.check_score(levels)


def test_shard_splits_candidates_to_fit_one_choice():
    items = list(range(600))
    parts = limits.shard(items)
    assert [len(part) for part in parts] == [255, 255, 90]
    assert [item for part in parts for item in part] == items


def test_token_estimate_unwraps_question_models():
    question = Choice(instructions="pick", criteria={"a": None})
    assert limits.estimate_tokens(question) > 0
    assert limits.payload(question)["type"] == "choice"


# --- cost -----------------------------------------------------------------


def test_prices_resolve_through_aliases():
    assert cost.price_of("jev-latest") == cost.price_of("jev-1.13.0")
    assert cost.per_million("jev-1.13.0") == pytest.approx(0.042)
    assert cost.usd_for("jev-1.13.0", 1_000_000) == pytest.approx(0.042)
    assert cost.usd_for("some-unreleased-model", 1000) is None


# --- answer checks --------------------------------------------------------


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("unknown_choice", "not offered"),
        ("missing_option", "probabilities cover"),
        ("nan", "finite"),
        ("bad_sum", "sum to"),
        ("not_argmax", "highest-probability"),
        ("bad_confidence", "confidence"),
    ],
)
def test_malformed_choice_answers_are_rejected(mutation, message):
    """Jev constrains answers to the options sent; the check is what keeps that a fact."""
    options = {"a": None, "b": None}
    fields = choice_answer(["a", "b"], "a")
    if mutation == "unknown_choice":
        fields["choice"] = "invented"
    elif mutation == "missing_option":
        del fields["probabilities"]["b"]
    elif mutation == "nan":
        fields["probabilities"]["b"] = math.nan
    elif mutation == "bad_sum":
        fields["probabilities"]["b"] = 0.9
    elif mutation == "not_argmax":
        fields["choice"] = "b"
    else:
        fields["confidence"] = 1.5
    answer = ChoiceAnswer.model_construct(**fields)
    with pytest.raises(AnswerRejected, match=message):
        check_choice(answer, options, "q")


def test_score_answer_value_matches_its_distribution():
    answer = score_answer(["none", "minor", "major"], 1.25)
    assert answer["score"] == pytest.approx(1.25)
    assert sum(answer["probabilities"].values()) == pytest.approx(1.0)


def test_confidence_bands_split_on_caller_thresholds(jev):
    client, _ = jev({"route": {"billing": 0.55, "technical": 0.45}})
    reply = client.ask("state", {"route": QUESTIONS["route"]})
    assert reply.band("route", low=0.3, high=0.9) == "medium"
    assert reply.band("route", low=0.9, high=0.95) == "low"
    assert reply.band("route", low=0.1, high=0.5) == "high"
    with pytest.raises(ValueError, match="low <= high"):
        reply.band("route", low=0.9, high=0.1)


def test_noul_has_no_confidence(jev):
    client, _ = jev({"urgent": 0.7})
    reply = client.ask("state", {"urgent": QUESTIONS["urgent"]})
    with pytest.raises(AnswerRejected, match="no confidence"):
        reply.confidence("urgent")


def test_top_orders_options_by_probability(jev):
    client, _ = jev({"route": {"billing": 0.7, "technical": 0.3}})
    reply = client.ask("state", {"route": QUESTIONS["route"]})
    assert reply.top("route", 1) == [("billing", pytest.approx(0.7))]


# --- ledger and pacing ----------------------------------------------------


def test_percentiles_and_rate():
    ledger = Ledger(calls=3, latencies_ms=[40.0, 50.0, 300.0])
    assert ledger.p50_ms == 50.0
    assert ledger.p95_ms == 300.0
    assert ledger.sequential_rate == pytest.approx(20.0)
    assert percentile([1.0], 0.5) == 1.0
    with pytest.raises(ValueError):
        percentile([], 0.5)


def test_limiter_spreads_requests_over_the_minute():
    clock = [0.0]
    limiter = RateLimiter(requests_per_minute=60, tokens_per_second=1_000, monotonic=lambda: clock[0])
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        clock[0] += seconds

    assert limiter.acquire(10, sleep=sleep) == 0.0
    waited = limiter.acquire(10, sleep=sleep)
    assert waited == pytest.approx(1.0), "60 rpm means one request per second"
    assert slept


def test_limiter_waits_for_the_token_bucket():
    clock = [0.0]
    limiter = RateLimiter(requests_per_minute=100_000, tokens_per_second=1_000, monotonic=lambda: clock[0])
    limiter.consume(1_000)
    assert limiter.delay_for(500) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        limiter.delay_for(-1)
