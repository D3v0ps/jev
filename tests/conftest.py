"""Shared fixtures. No test in this repo may call a paid API."""

from __future__ import annotations

import pytest

from jevkit.testing import fake_async_jev, fake_jev


@pytest.fixture
def jev():
    """Factory: `jev(plan)` returns (Jev, calls). See jevkit.testing for plan shapes."""
    made = []

    def build(plan=None, **kwargs):
        client, calls = fake_jev(plan, **kwargs)
        made.append(client)
        return client, calls

    yield build
    for client in made:
        client.close()


@pytest.fixture
def async_jev():
    """Factory: `async_jev(plan)` returns (AsyncJev, calls)."""
    made = []

    def build(plan=None, **kwargs):
        client, calls = fake_async_jev(plan, **kwargs)
        made.append(client)
        return client, calls

    yield build


@pytest.fixture(autouse=True)
def no_live_api(monkeypatch):
    """Fail loudly rather than reach the real endpoint if a test forgets the fake."""
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("TYPESAFE_BASE_URL", "http://tests.invalid")
