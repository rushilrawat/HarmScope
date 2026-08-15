"""The single reliable transport boundary for Anthropic model calls."""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.config import CONFIG


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass(frozen=True)
class ModelPricing:
    input_usd_per_million: float
    output_usd_per_million: float
    cache_write_usd_per_million: float
    cache_read_usd_per_million: float


@dataclass(frozen=True)
class ModelCallResult:
    payload: dict[str, Any]
    model: str
    stop_reason: str
    usage: TokenUsage
    attempts: int
    latency_seconds: float
    estimated_cost_usd: float


class ModelCallError(RuntimeError):
    """A failed call, including paid accounting when a response arrived."""

    def __init__(
        self,
        category: str,
        attempts: int,
        retryable: bool,
        *,
        usage: TokenUsage | None = None,
        latency_seconds: float = 0.0,
        estimated_cost_usd: float = 0.0,
        response_received: bool = False,
    ) -> None:
        super().__init__(f"{category} after {attempts} attempt(s)")
        self.category = category
        self.attempts = attempts
        self.retryable = retryable
        self.usage = usage or TokenUsage()
        self.latency_seconds = latency_seconds
        self.estimated_cost_usd = estimated_cost_usd
        self.response_received = response_received


def estimate_cost(usage: TokenUsage, price: ModelPricing) -> float:
    return (
        usage.input_tokens * price.input_usd_per_million
        + usage.output_tokens * price.output_usd_per_million
        + usage.cache_creation_input_tokens * price.cache_write_usd_per_million
        + usage.cache_read_input_tokens * price.cache_read_usd_per_million
    ) / 1_000_000


def _token_count(value: object, field: str, *, optional: bool = False) -> int:
    if optional and value is None:
        return 0
    if type(value) is not int or value < 0:
        raise ValueError(f"response usage {field} must be a non-negative integer")
    return value


def _response_usage(response: object) -> TokenUsage:
    usage = response.usage
    return TokenUsage(
        input_tokens=_token_count(usage.input_tokens, "input_tokens"),
        output_tokens=_token_count(usage.output_tokens, "output_tokens"),
        cache_read_input_tokens=_token_count(
            getattr(usage, "cache_read_input_tokens", None),
            "cache_read_input_tokens",
            optional=True,
        ),
        cache_creation_input_tokens=_token_count(
            getattr(usage, "cache_creation_input_tokens", None),
            "cache_creation_input_tokens",
            optional=True,
        ),
    )


def _best_effort_response_usage(response: object) -> TokenUsage:
    try:
        usage = response.usage
    except Exception:
        return TokenUsage()

    def safe(field: str, *, optional: bool = False) -> int:
        try:
            return _token_count(getattr(usage, field, None), field, optional=optional)
        except Exception:
            return 0

    return TokenUsage(
        input_tokens=safe("input_tokens"),
        output_tokens=safe("output_tokens"),
        cache_read_input_tokens=safe("cache_read_input_tokens", optional=True),
        cache_creation_input_tokens=safe("cache_creation_input_tokens", optional=True),
    )


def classify_error(exc: Exception) -> tuple[str, bool]:
    import anthropic

    message = str(exc).lower()
    if "credit balance" in message or "billing" in message:
        return "billing", False
    if isinstance(exc, anthropic.RateLimitError):
        return "rate_limit", True
    if isinstance(exc, (anthropic.APIConnectionError, anthropic.InternalServerError)):
        return "transient", True
    if isinstance(exc, anthropic.AuthenticationError):
        return "authentication", False
    if isinstance(exc, anthropic.PermissionDeniedError):
        return "permission", False
    if isinstance(exc, anthropic.BadRequestError):
        return "invalid_request", False
    return "unexpected", False


def _configured_pricing() -> ModelPricing:
    config = CONFIG.llm
    return ModelPricing(
        input_usd_per_million=config.input_usd_per_million,
        output_usd_per_million=config.output_usd_per_million,
        cache_write_usd_per_million=config.cache_write_usd_per_million,
        cache_read_usd_per_million=config.cache_read_usd_per_million,
    )


class AnthropicModelClient:
    """Make structured Anthropic calls with explicit, observable retries."""

    def __init__(
        self,
        transport: Any | None = None,
        pricing: ModelPricing | None = None,
        *,
        max_retries: int | None = None,
        retry_base_seconds: float | None = None,
        retry_max_seconds: float | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        jitter: Callable[[float, float], float] = random.uniform,
    ) -> None:
        config = CONFIG.llm
        if transport is None:
            import anthropic

            transport = anthropic.Anthropic(max_retries=0)
        self.transport = transport
        self.pricing = pricing or _configured_pricing()
        self.max_retries = config.max_retries if max_retries is None else max_retries
        self.retry_base_seconds = (
            config.retry_base_seconds if retry_base_seconds is None else retry_base_seconds
        )
        self.retry_max_seconds = (
            config.retry_max_seconds if retry_max_seconds is None else retry_max_seconds
        )
        self.sleeper = sleeper
        self.clock = clock
        self.jitter = jitter

    def preflight(self, model: str) -> None:
        try:
            self.transport.models.retrieve(model_id=model)
        except Exception as exc:
            category, retryable = classify_error(exc)
            raise ModelCallError(category, 1, retryable) from exc

    def call_json(
        self, model: str, system: str, prompt: str, schema: dict, max_tokens: int
    ) -> ModelCallResult:
        started_at = self.clock()
        attempts = 0
        while True:
            attempts += 1
            try:
                response = self.transport.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=[
                        {
                            "type": "text",
                            "text": system,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    output_config={
                        "format": {"type": "json_schema", "schema": schema},
                    },
                    messages=[{"role": "user", "content": prompt}],
                )
            except Exception as exc:
                category, retryable = classify_error(exc)
                if not retryable or attempts > self.max_retries:
                    raise ModelCallError(
                        category,
                        attempts,
                        retryable,
                        latency_seconds=self.clock() - started_at,
                    ) from exc
                base = min(
                    self.retry_base_seconds * 2 ** (attempts - 1),
                    self.retry_max_seconds,
                )
                self.sleeper(base + self.jitter(0, min(base * 0.25, 1.0)))
                continue

            # A response is paid usage even when its content is malformed.
            # Keep extraction outside the transport exception block: parsing
            # failures must never be misclassified as retryable provider calls.
            latency_seconds = self.clock() - started_at
            try:
                usage = _response_usage(response)
                estimated_cost_usd = estimate_cost(usage, self.pricing)
            except Exception as exc:
                usage = _best_effort_response_usage(response)
                try:
                    estimated_cost_usd = estimate_cost(usage, self.pricing)
                except Exception:
                    estimated_cost_usd = 0.0
                raise ModelCallError(
                    "usage_extraction",
                    attempts,
                    False,
                    usage=usage,
                    latency_seconds=latency_seconds,
                    estimated_cost_usd=estimated_cost_usd,
                    response_received=True,
                ) from exc
            try:
                stop_reason = response.stop_reason
                if type(stop_reason) is not str:
                    raise TypeError("response stop_reason must be a string")
            except Exception as exc:
                raise ModelCallError(
                    "malformed_response",
                    attempts,
                    False,
                    usage=usage,
                    latency_seconds=latency_seconds,
                    estimated_cost_usd=estimated_cost_usd,
                    response_received=True,
                ) from exc
            if stop_reason == "refusal":
                payload = {"refused": True, "stop_reason": "refusal"}
            else:
                category = (
                    "truncated_response" if stop_reason == "max_tokens" else "malformed_response"
                )
                try:
                    text = next(block.text for block in response.content if block.type == "text")
                    payload = json.loads(text)
                except Exception as exc:
                    raise ModelCallError(
                        category,
                        attempts,
                        False,
                        usage=usage,
                        latency_seconds=latency_seconds,
                        estimated_cost_usd=estimated_cost_usd,
                        response_received=True,
                    ) from exc
                if stop_reason == "max_tokens":
                    raise ModelCallError(
                        category,
                        attempts,
                        False,
                        usage=usage,
                        latency_seconds=latency_seconds,
                        estimated_cost_usd=estimated_cost_usd,
                        response_received=True,
                    )
            return ModelCallResult(
                payload=payload,
                model=model,
                stop_reason=stop_reason,
                usage=usage,
                attempts=attempts,
                latency_seconds=latency_seconds,
                estimated_cost_usd=estimated_cost_usd,
            )
