"""Beta guardrails: a per-user daily message quota and a global daily OpenAI spend cap.

Both are kept in process memory and reset at 00:00 UTC. That is correct for a single instance
only; with more than one instance each keeps its own count, so they move to Redis together with
the session store. The checks run before any model call, so a refused message costs nothing.
"""

import logging
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any

from app.config import Settings
from app.pricing import estimate_cost, prices_from_settings, response_usage

logger = logging.getLogger(__name__)

WARN_FRACTION = 0.8


def quota_message(limit: int) -> str:
    return (
        f"You've used today's {limit} Copilot messages (the beta daily limit). "
        "It resets at 00:00 UTC, so you can ask again then."
    )


RESTING_MESSAGE = "Copilot is resting until reset (00:00 UTC). Please check back then."


class UsageLimits:
    """Counts messages per Bungie membership id and estimated spend per UTC day."""

    def __init__(
        self,
        settings: Settings,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.quota = settings.daily_message_quota
        self.spend_cap = settings.daily_spend_cap_usd
        self.prices = prices_from_settings(settings)
        self.clock = clock or (lambda: datetime.now(UTC))
        self._day: date | None = None
        self._messages: dict[str, int] = {}
        self._spend = 0.0
        self._warned_80 = False
        self._warned_cap = False
        if (
            self.spend_cap > 0
            and estimate_cost({"input_tokens": 1, "output_tokens": 1}, self.prices) is None
        ):
            logger.warning(
                "Daily spend cap is set (%.2f USD) but OpenAI prices are not configured "
                "(EVAL_PRICE_INPUT_PER_1M / EVAL_PRICE_OUTPUT_PER_1M); spend is not tracked "
                "and the cap is not enforced.",
                self.spend_cap,
            )

    def _roll(self) -> None:
        today = self.clock().astimezone(UTC).date()
        if today != self._day:
            self._day = today
            self._messages = {}
            self._spend = 0.0
            self._warned_80 = self._warned_cap = False

    @property
    def spend_today(self) -> float:
        self._roll()
        return self._spend

    def spend_cap_reached(self) -> bool:
        self._roll()
        return self.spend_cap > 0 and self._spend >= self.spend_cap

    def admit(self, membership_id: str) -> str | None:
        """None when the message may proceed (and it is counted), else the message to show."""

        self._roll()
        if self.spend_cap_reached():
            return RESTING_MESSAGE
        if self.quota > 0:
            used = self._messages.get(membership_id, 0)
            if used >= self.quota:
                return quota_message(self.quota)
            self._messages[membership_id] = used + 1
        return None

    def record_response(self, response: Any) -> None:
        """Add one model response's estimated cost to today's spend."""

        self._roll()
        cost = estimate_cost(response_usage(response), self.prices)
        if cost is None:
            return
        self._spend += cost
        if self.spend_cap <= 0:
            return
        if self._spend >= self.spend_cap and not self._warned_cap:
            self._warned_cap = True
            logger.warning(
                "Daily OpenAI spend cap reached: %.2f of %.2f USD. Copilot answers with the "
                "resting message until 00:00 UTC.",
                self._spend,
                self.spend_cap,
            )
        elif self._spend >= WARN_FRACTION * self.spend_cap and not self._warned_80:
            self._warned_80 = True
            logger.warning(
                "Daily OpenAI spend at %.0f%% of the cap: %.2f of %.2f USD.",
                100 * self._spend / self.spend_cap,
                self._spend,
                self.spend_cap,
            )
