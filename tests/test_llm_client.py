"""Behavioral tests for the single Anthropic transport boundary."""

from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

import pytest


class FakeRateLimit(Exception):
    pass


class FakeBillingError(Exception):
    pass


fake_anthropic = types.SimpleNamespace(
    RateLimitError=FakeRateLimit,
    APIConnectionError=type("APIConnectionError", (Exception,), {}),
    InternalServerError=type("InternalServerError", (Exception,), {}),
    AuthenticationError=type("AuthenticationError", (Exception,), {}),
    PermissionDeniedError=type("PermissionDeniedError", (Exception,), {}),
    BadRequestError=type("BadRequestError", (Exception,), {}),
)

from src.llm.client import (  # noqa: E402
    AnthropicModelClient,
    ModelCallError,
    ModelPricing,
    TokenUsage,
    estimate_cost,
)


@pytest.fixture(autouse=True)
def fake_anthropic_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)


def pricing() -> ModelPricing:
    return ModelPricing(5.0, 25.0, 6.25, 0.50)


def response(*, payload: dict, input_tokens: int, output_tokens: int):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=json.dumps(payload))],
        stop_reason="end_turn",
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )


class SequenceTransport:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.requests = []
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        self.requests.append(kwargs)
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class RecordingTransport:
    def __init__(self):
        self.retrieved = []
        self.models = SimpleNamespace(retrieve=self.retrieve)

    def retrieve(self, *, model_id):
        self.retrieved.append(model_id)


class SequenceClock:
    def __init__(self, values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


def test_retries_rate_limit_then_returns_usage():
    """Removing retry behavior must surface the first transient failure."""
    transport = SequenceTransport([
        FakeRateLimit(),
        response(payload={"ok": True}, input_tokens=100, output_tokens=20),
    ])
    sleeps = []
    client = AnthropicModelClient(
        transport, pricing(), max_retries=3, sleeper=sleeps.append,
        clock=SequenceClock([0.0, 0.4]), jitter=lambda _lo, _hi: 0.0,
    )

    got = client.call_json(
        model="m", system="s", prompt="p", schema={"type": "object"}, max_tokens=50,
    )

    assert got.attempts == 2
    assert got.usage == TokenUsage(100, 20, 0, 0)
    assert got.latency_seconds == pytest.approx(0.4)
    assert sleeps == [1.0]
    assert transport.requests[0]["output_config"] == {
        "format": {"type": "json_schema", "schema": {"type": "object"}},
    }
    assert transport.requests[0]["system"] == [{
        "type": "text", "text": "s", "cache_control": {"type": "ephemeral"},
    }]


def test_billing_error_is_terminal_without_retry():
    """Treating exhausted credit as retryable would waste paid-run time."""
    client = AnthropicModelClient(
        SequenceTransport([FakeBillingError("credit balance depleted")]), pricing(),
        max_retries=3, sleeper=lambda _: None,
    )

    with pytest.raises(ModelCallError) as caught:
        client.call_json(
            model="m", system="s", prompt="p",
            schema={"type": "object"}, max_tokens=50,
        )

    assert caught.value.category == "billing"
    assert caught.value.attempts == 1
    assert caught.value.retryable is False


def test_cost_uses_all_anthropic_token_categories():
    """Omitting cache tokens would understate the recorded model cost."""
    usage = TokenUsage(1_000_000, 1_000_000, 1_000_000, 1_000_000)

    assert estimate_cost(usage, pricing()) == pytest.approx(36.75)


def test_preflight_retrieves_the_configured_model():
    """Skipping retrieval would allow a bad model name into a labeling run."""
    transport = RecordingTransport()

    AnthropicModelClient(transport, pricing()).preflight("claude-opus-5")

    assert transport.retrieved == ["claude-opus-5"]
