"""Beta guardrails (quota, spend cap), labels left empty by trimming, and Subclass 3.0."""

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from app.ai import RecommendationService
from app.config import Settings
from app.local_repairs import clean_layout, trim_excess
from app.models import ChatRequest
from app.pricing import estimate_cost, prices_from_settings, response_usage
from app.usage_limits import RESTING_MESSAGE, UsageLimits, quota_message
from tests.test_guardian_tools import (
    FakeResponses,
    guardian_context,
)

NOON = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)
PRICES = {
    "price_input_per_1m": 2.0,
    "price_cached_input_per_1m": 0.5,
    "price_output_per_1m": 8.0,
    "price_web_search_per_1k": 10.0,
}


def settings(**overrides: Any) -> Settings:
    return Settings(openai_api_key="test-key", **{**PRICES, **overrides})


def usage(input_tokens: int = 0, cached: int = 0, output_tokens: int = 0, searches: int = 0) -> Any:
    return SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            input_tokens_details=SimpleNamespace(cached_tokens=cached),
            output_tokens=output_tokens,
        ),
        output=[SimpleNamespace(type="web_search_call")] * searches,
    )


class Clock:
    def __init__(self) -> None:
        self.now = NOON

    def __call__(self) -> datetime:
        return self.now


# --- Pricing ---------------------------------------------------------------------------------


def test_prices_come_from_settings_and_the_estimate_matches_the_eval_one() -> None:
    prices = prices_from_settings(settings())
    assert prices == {
        "input_per_1m": 2.0,
        "cached_input_per_1m": 0.5,
        "output_per_1m": 8.0,
        "web_search_per_1k": 10.0,
    }
    totals = response_usage(usage(1_000_000, 400_000, 100_000, 20))
    # 600k * 2 + 400k * 0.5 + 100k * 8 per 1M, plus 20 searches at 10 per 1K
    assert estimate_cost(totals, prices) == pytest.approx(1.2 + 0.2 + 0.8 + 0.2)
    from evals.pricing import estimate_cost as eval_estimate

    assert eval_estimate is estimate_cost
    assert estimate_cost(totals, {}) is None


def test_prices_can_be_given_by_environment_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_PRICE_INPUT_PER_1M", "1.5")
    monkeypatch.setenv("OPENAI_PRICE_OUTPUT_PER_1M", "6")
    configured = Settings(openai_api_key="k")
    assert configured.price_input_per_1m == 1.5 and configured.price_output_per_1m == 6.0


# --- Daily message quota ---------------------------------------------------------------------


def test_the_quota_counts_per_membership_and_refuses_with_a_friendly_message() -> None:
    limits = UsageLimits(settings(daily_message_quota=2), Clock())

    assert limits.admit("a") is None and limits.admit("a") is None
    refusal = limits.admit("a")
    assert refusal == quota_message(2)
    assert "2 Copilot messages" in refusal and "00:00 UTC" in refusal
    # Another user is unaffected; the refused message was not counted for anyone else.
    assert limits.admit("b") is None


def test_the_quota_resets_at_the_utc_day_boundary() -> None:
    clock = Clock()
    limits = UsageLimits(settings(daily_message_quota=1), clock)
    assert limits.admit("a") is None and limits.admit("a") is not None

    clock.now = NOON.replace(hour=23, minute=59)
    assert limits.admit("a") is not None
    clock.now = NOON + timedelta(hours=12)  # 00:00 UTC the next day
    assert limits.admit("a") is None


def test_a_quota_of_zero_turns_the_limit_off() -> None:
    limits = UsageLimits(settings(daily_message_quota=0), Clock())
    assert all(limits.admit("a") is None for _ in range(500))


def test_default_limits_are_a_beta_friendly_50_messages_and_3_dollars() -> None:
    defaults = Settings(openai_api_key="k")
    assert defaults.daily_message_quota == 50 and defaults.daily_spend_cap_usd == 3.0


# --- Daily spend cap -------------------------------------------------------------------------


def test_spend_accumulates_warns_at_80_percent_and_rests_at_the_cap(
    caplog: pytest.LogCaptureFixture,
) -> None:
    limits = UsageLimits(settings(daily_spend_cap_usd=1.0, daily_message_quota=0), Clock())
    with caplog.at_level(logging.WARNING, logger="app.usage_limits"):
        limits.record_response(usage(input_tokens=100_000))  # 0.2 USD
        assert not caplog.records
        limits.record_response(usage(output_tokens=75_000))  # +0.6 = 0.8
        assert len(caplog.records) == 1 and "80%" in caplog.records[0].getMessage()
        limits.record_response(usage(output_tokens=1_000))  # still one warning
        assert len(caplog.records) == 1
        assert limits.admit("a") is None
        limits.record_response(usage(output_tokens=30_000))  # over the cap
        assert "cap reached" in caplog.records[-1].getMessage()
    assert limits.admit("a") == RESTING_MESSAGE
    assert limits.spend_today == pytest.approx(0.2 + 0.6 + 0.008 + 0.24)


def test_the_spend_resets_each_utc_day_and_a_refused_message_is_not_counted() -> None:
    clock = Clock()
    limits = UsageLimits(settings(daily_spend_cap_usd=0.1, daily_message_quota=1), clock)
    limits.record_response(usage(input_tokens=1_000_000))
    assert limits.admit("a") == RESTING_MESSAGE

    clock.now = NOON + timedelta(days=1)
    assert limits.spend_today == 0
    assert limits.admit("a") is None  # the resting refusal above did not consume the quota


def test_without_prices_the_cap_cannot_be_enforced_and_says_so(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="app.usage_limits"):
        limits = UsageLimits(
            Settings(openai_api_key="k", daily_spend_cap_usd=1.0, daily_message_quota=0), Clock()
        )
    assert "not enforced" in caplog.text
    limits.record_response(usage(input_tokens=10**9))
    assert limits.admit("a") is None and limits.spend_today == 0


# --- Through the service ---------------------------------------------------------------------


class UsageResponses(FakeResponses):
    """A client whose responses report usage, to drive the spend cap through real calls."""

    def __init__(self, output_tokens: int) -> None:
        super().__init__(request_tool=False)
        self.output_tokens = output_tokens

    async def create(self, **kwargs: Any) -> Any:
        response = await super().create(**kwargs)
        response.usage = SimpleNamespace(
            input_tokens=0,
            input_tokens_details=SimpleNamespace(cached_tokens=0),
            output_tokens=self.output_tokens,
        )
        return response


def chat(service: RecommendationService, message: str = "What should I do next?") -> Any:
    return asyncio.run(service.chat(ChatRequest(message=message), guardian_context()))


def test_a_user_over_quota_gets_a_message_and_no_model_call() -> None:
    responses = UsageResponses(output_tokens=1)
    service = RecommendationService(settings(daily_message_quota=1))
    service.client = SimpleNamespace(responses=responses)

    first = chat(service)
    second = chat(service)

    assert first.source == "openai" and second.source == "local"
    assert second.message == quota_message(1)
    assert len(responses.requests) == 1


def test_when_the_cap_is_hit_the_service_rests_without_calling_the_model() -> None:
    responses = UsageResponses(output_tokens=1_000_000)  # 8 USD against a 1 USD cap
    service = RecommendationService(settings(daily_spend_cap_usd=1.0, daily_message_quota=0))
    service.client = SimpleNamespace(responses=responses)

    first = chat(service)
    calls_after_first = len(responses.requests)
    second = chat(service)

    assert first.source == "openai"
    assert second.source == "local" and second.message == RESTING_MESSAGE
    assert len(responses.requests) == calls_after_first
    assert service.limits.spend_today == pytest.approx(8.0 * calls_after_first)


def test_evaluation_settings_have_no_limits() -> None:
    from evals.runtime import eval_settings

    configured = eval_settings()
    assert configured.daily_message_quota == 0 and configured.daily_spend_cap_usd == 0


# --- Labels are never left empty by trimming -----------------------------------------------


def test_trimming_never_strands_a_label_at_the_end_or_between_sections() -> None:
    lines = ["Run Kepler Uplink for the quest."]
    lines += ["", "Plan"] + [f"- Step {i}: " + "word " * 12 for i in range(6)]
    lines += [
        "",
        "Backup (if you want something different)",
        "- Run Strikes for the milestone today.",
    ]
    lines += ["", "Why:", "- It is quick and repeatable for the week.", "- It fits your window."]
    answer = "\n".join(lines)

    fixed = trim_excess(answer, 40)

    assert fixed is not None and fixed.startswith("Run Kepler Uplink")
    shown = fixed.splitlines()
    for index, line in enumerate(shown):
        if line.strip() in {"Plan", "Why:"} or line.startswith("Backup ("):
            after = next((value for value in shown[index + 1 :] if value.strip()), "")
            assert after.startswith("-"), (line, after)
    assert not shown[-1].startswith(("Backup", "Plan")) and not shown[-1].endswith(":")


def test_clean_layout_drops_labels_without_content_but_keeps_real_lines() -> None:
    assert clean_layout("Run it now.\n\nBackup (if you want a change)") == "Run it now."
    assert (
        clean_layout("Run it now.\n\nPlan\n\nWhy:\n- It is quick.")
        == "Run it now.\n\nWhy:\n- It is quick."
    )
    assert clean_layout("Run it now.\n\nPlan\n- Launch it.") == "Run it now.\n\nPlan\n- Launch it."
    # A short closing line is not a label.
    assert clean_layout("Run it now.\n\nHave fun") == "Run it now.\n\nHave fun"
    # Markdown headings with text under them (after a blank line) stay.
    assert clean_layout("## Plan\n\nLaunch it today.") == "## Plan\n\nLaunch it today."
    # Never empties an answer.
    assert clean_layout("Plan") == "Plan"


# --- Subclass 3.0 in core knowledge -----------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "What does the top subclass tree do for my Titan?",
        "Which attunement should I pick for Sunbreaker?",
        "How do aspects and fragments work?",
    ],
)
def test_pre_subclass_3_terms_surface_the_current_structure(message: str) -> None:
    from app.config import BUNDLED_DATA_DIR
    from app.core_knowledge import DestinyKnowledgeEngine

    engine = DestinyKnowledgeEngine.from_file(BUNDLED_DATA_DIR / "core_knowledge.json")
    bundle = engine.bundle(message)

    assert bundle is not None
    assert "Subclass customization (Subclass 3.0)" in [
        value["name"] for value in bundle.payload["entities"]
    ]
    assert "no skill trees or Attunement paths" in bundle.text
