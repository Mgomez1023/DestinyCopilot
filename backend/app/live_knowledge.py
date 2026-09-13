import re
import unicodedata
from typing import Any

LIVE_PROVIDER_CATEGORY = "live"

_DIRECT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("featured_dungeon", re.compile(r"\bfeatured\s+dungeon\b|\bdungeon\b.*\bfeatured\b")),
    ("featured_raid", re.compile(r"\bfeatured\s+raid\b|\braid\b.*\bfeatured\b")),
    (
        "weekly_nightfall",
        re.compile(
            r"\bnightfall\b.*\b(this|current)\s+week\b|"
            r"\bcurrent\s+nightfall\b|"
            r"^(what|which|whats)\s+(is|s)\s+(the\s+)?nightfall\b"
        ),
    ),
    ("weekly_rotation", re.compile(r"\bweekly\s+(?:loot\s+)?rotation\b")),
    ("current_loot_rotation", re.compile(r"\bcurrent\s+loot\s+rotation\b")),
    ("current_modifiers", re.compile(r"\bcurrent\s+modifiers?\b")),
    (
        "current_vendor_inventory",
        re.compile(r"\bcurrent\s+vendor\s+inventory\b|\bvendor\b.*\binventory\b"),
    ),
    ("current_meta", re.compile(r"\bcurrent\s+meta\b")),
    (
        "current_exotic_mission",
        re.compile(
            r"\bexotic\s+mission\b.*\b(this|current)\s+week\b|"
            r"\bcurrent\s+(?:weekly\s+)?exotic\s+mission\b"
        ),
    ),
    (
        "reset_changes",
        re.compile(r"\b(what|whats|what s)\b.*\b(reset|changed at reset)\b"),
    ),
    (
        "weekly_farming",
        re.compile(
            r"\bfarm\b.*\b(this|current)\s+week\b|"
            r"\bworth\b.*\b(this|current)\s+week\b|"
            r"\bwhat\b.*\bdo\b.*\bthis\s+week\b"
        ),
    ),
    ("daily_availability", re.compile(r"\bavailable\b.*\btoday\b")),
    ("drop_rate", re.compile(r"\bdrop\s+rates?\b")),
    ("double_loot", re.compile(r"\bdouble\s+loot\b")),
    (
        "xur_location",
        re.compile(r"\bxur\b.*\b(where|location)\b|\b(where|location)\b.*\bxur\b"),
    ),
    (
        "xur_inventory",
        re.compile(
            r"\bxur\b.*\b(inventory|selling|sell|stock|have)\b|"
            r"\b(inventory|selling|sell|stock)\b.*\bxur\b"
        ),
    ),
    ("xur_status", re.compile(r"\bcurrent\s+xur\b")),
)

_TIME_WORDS = re.compile(r"\b(this\s+week|today|right\s+now|now|currently|current)\b")
_VOLATILE_SUBJECTS = re.compile(
    r"\b(featured|rotation|nightfall|xur|vendor|inventory|modifier|farm|available|"
    r"loot|dungeon|raid|meta|exotic mission|reset|activity|worth)\b"
)


def _normalize(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return " ".join(re.findall(r"[a-z0-9]+", ascii_value.casefold()))


def volatile_topic(query: str) -> str | None:
    """Return a normalized live-data topic only for time-sensitive current-state queries."""

    normalized = _normalize(query)
    for topic, pattern in _DIRECT_PATTERNS:
        if pattern.search(normalized):
            return topic
    if _TIME_WORDS.search(normalized) and _VOLATILE_SUBJECTS.search(normalized):
        return "current_destiny_state"
    return None


def unavailable_live_data(query: str, topic: str | None = None) -> dict[str, Any]:
    resolved_topic = topic or volatile_topic(query) or "current_destiny_state"
    return {
        "query": query,
        "topic": resolved_topic,
        "freshness": "volatile",
        "status": "unavailable_live_data",
        "requires_live_provider": True,
        "required_provider_category": LIVE_PROVIDER_CATEGORY,
        "live_data_available": False,
        "disallowed_as_current_sources": [
            "guardian_account",
            "bungie_manifest",
            "guide_provider",
            "model_memory",
        ],
        "warning": (
            "No LiveDestinyProvider is connected. Static, account, and guide sources cannot "
            "establish the current rotation or other live state."
        ),
    }


def unavailable_live_message(topic: str, *, provider_connected: bool = False) -> str:
    if topic in {"featured_dungeon", "featured_raid", "weekly_nightfall"}:
        subject = {
            "featured_dungeon": "this week's featured dungeon",
            "featured_raid": "this week's featured raid",
            "weekly_nightfall": "this week's Nightfall",
        }[topic]
        if provider_connected:
            return (
                "I can identify and explain the activity, but the connected live source does "
                f"not reliably expose {subject}, so I can't give you a current answer."
            )
        return (
            "I can identify and explain the activity, but I don't have a live weekly-rotation "
            f"source connected yet, so I can't reliably tell you {subject}."
        )
    if provider_connected:
        return (
            "The connected live Destiny sources don't reliably expose this current or rotating "
            "state yet. I can still explain a named activity, item, or mechanic using Manifest "
            "and guide knowledge."
        )
    return (
        "I don't have a live Destiny source connected yet, so I can't reliably answer this "
        "current or rotating-state question. I can still explain a named activity, item, or "
        "mechanic using Manifest and guide knowledge."
    )
