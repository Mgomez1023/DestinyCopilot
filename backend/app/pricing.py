"""Estimated cost of OpenAI usage. Prices are configuration, never constants in this code.

Prices are USD per 1M tokens (web search per 1K calls), read from the same settings the
evaluation tooling uses: EVAL_PRICE_INPUT_PER_1M, EVAL_PRICE_CACHED_INPUT_PER_1M,
EVAL_PRICE_OUTPUT_PER_1M, EVAL_PRICE_WEB_SEARCH_PER_1K (OPENAI_PRICE_* work too). The figure is
an estimate from the usage the API reported, not a bill.
"""

from typing import Any

from app.config import Settings


def prices_from_settings(settings: Settings) -> dict[str, float]:
    """The configured prices; a price that is not set is absent."""

    configured = {
        "input_per_1m": settings.price_input_per_1m,
        "cached_input_per_1m": settings.price_cached_input_per_1m,
        "output_per_1m": settings.price_output_per_1m,
        "web_search_per_1k": settings.price_web_search_per_1k,
    }
    return {name: value for name, value in configured.items() if value is not None}


def estimate_cost(totals: dict[str, int], prices: dict[str, Any]) -> float | None:
    """USD estimate, or None when input and output prices are not configured."""

    if "input_per_1m" not in prices or "output_per_1m" not in prices:
        return None
    cached = min(totals.get("cached_input_tokens", 0), totals.get("input_tokens", 0))
    uncached = totals.get("input_tokens", 0) - cached
    cached_price = prices.get("cached_input_per_1m", prices["input_per_1m"])
    cost = (
        uncached * prices["input_per_1m"]
        + cached * cached_price
        + totals.get("output_tokens", 0) * prices["output_per_1m"]
    ) / 1_000_000
    cost += totals.get("web_search_calls", 0) * prices.get("web_search_per_1k", 0.0) / 1000
    return round(cost, 6)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def response_usage(response: Any) -> dict[str, int]:
    """Token and web-search totals of one model response (zeros when it reports none)."""

    usage = _field(response, "usage")
    details = _field(usage, "input_tokens_details")
    searches = sum(
        1
        for item in _field(response, "output", []) or []
        if _field(item, "type") == "web_search_call"
    )
    return {
        "input_tokens": int(_field(usage, "input_tokens", 0) or 0),
        "cached_input_tokens": int(_field(details, "cached_tokens", 0) or 0),
        "output_tokens": int(_field(usage, "output_tokens", 0) or 0),
        "web_search_calls": searches,
    }
