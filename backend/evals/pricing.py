"""Cost estimate for an eval run. Prices come from the environment, never from this code.

Set (USD per 1M tokens; web search per 1K calls) in the environment or the root .env:

    EVAL_PRICE_INPUT_PER_1M, EVAL_PRICE_CACHED_INPUT_PER_1M, EVAL_PRICE_OUTPUT_PER_1M,
    EVAL_PRICE_WEB_SEARCH_PER_1K   (optional, defaults to 0 when unset)

Unset input/output prices mean no estimate is shown. The figure is an estimate from the
usage the API reported, not a bill.
"""

import os
from pathlib import Path
from typing import Any

from app.pricing import estimate_cost  # noqa: F401  (shared with the app's spend cap)

ROOT_ENV = Path(__file__).resolve().parents[2] / ".env"
PRICE_KEYS = {
    "input_per_1m": "EVAL_PRICE_INPUT_PER_1M",
    "cached_input_per_1m": "EVAL_PRICE_CACHED_INPUT_PER_1M",
    "output_per_1m": "EVAL_PRICE_OUTPUT_PER_1M",
    "web_search_per_1k": "EVAL_PRICE_WEB_SEARCH_PER_1K",
}


def _dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        key, separator, value = line.partition("=")
        if separator and key.strip().startswith("EVAL_PRICE_"):
            values[key.strip()] = value.strip().strip("\"'")
    return values


def load_prices(
    env: dict[str, str] | None = None, dotenv: Path | None = ROOT_ENV
) -> dict[str, Any]:
    """Prices from `env` (default: the process environment), then the .env file. {} if unset."""

    source = {**(_dotenv(dotenv) if dotenv else {}), **(os.environ if env is None else env)}
    prices: dict[str, Any] = {}
    for name, key in PRICE_KEYS.items():
        raw = source.get(key)
        try:
            if raw not in (None, ""):
                prices[name] = float(raw)
        except ValueError:
            continue
    return prices
