"""What a request actually cost, in dollars, from the usage the API reports.

Prices are per input token; Jev charges nothing for output tokens. Published at
https://docs.typesafe.ai/models as $42 per billion input tokens for jev-1.13.0.
Prices change: this table is the one place to update, and an unpriced model
yields None rather than a wrong number.
"""

from __future__ import annotations

USD_PER_INPUT_TOKEN: dict[str, float] = {
    "jev-1.13.0": 42 / 1e9,
}

ALIASES: dict[str, str] = {
    "jev-latest": "jev-1.13.0",
    "jev-preview": "jev-1.13.0",
}


def price_of(model: str) -> float | None:
    """USD per input token for a model id or alias, or None when unpriced here."""
    return USD_PER_INPUT_TOKEN.get(ALIASES.get(model, model))


def usd_for(model: str, input_tokens: int) -> float | None:
    """Cost of one request, or None when the model carries no price in this table."""
    price = price_of(model)
    return None if price is None else input_tokens * price


def per_million(model: str) -> float | None:
    """USD per million input tokens, the unit vendors usually quote."""
    price = price_of(model)
    return None if price is None else price * 1e6
