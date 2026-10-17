"""Scenario and expectation types shared by the bank, properties, and runner."""

from dataclasses import dataclass, field
from typing import Literal

Style = Literal["fact", "planning", "build", "explain"]

AXES: dict[str, tuple[str, ...]] = {
    "intent": (
        "what_next",
        "short_session",
        "long_session",
        "build_critique",
        "build_creation",
        "owned_gear",
        "perk_explanation",
        "item_explanation",
        "activity_explanation",
        "location",
        "quest_help",
        "progression_help",
        "returning_player",
        "live_state",
        "ambiguity",
        "rules",
    ),
    "character": ("unspecified", "Titan", "Hunter", "Warlock", "switch"),
    "time": ("none", "15m", "30m", "1h", "long"),
    "social": ("unspecified", "solo", "fireteam", "no_lfg"),
    "ownership": ("none", "only_owned", "equipped_only", "not_owned"),
    "knowledge": (
        "none",
        "stable",
        "live",
        "ambiguous",
        "unknown",
        "shorthand",
        "unregistered_shorthand",
        "typo",
    ),
    "account": (
        "general",
        "near_complete",
        "no_near_complete",
        "large_vault",
        "sparse",
        "class_locked_exotics",
        "no_useful_exotics",
        "active_campaign",
        "no_matching_quest",
        "stale_data",
        "missing_class",
    ),
}


@dataclass(frozen=True)
class Expectations:
    """What a correct answer must (and must not) do; consumed by properties.py."""

    style: Style = "fact"
    character: str | None = None
    only_owned: bool = False
    equipped_only: bool = False
    not_owned: tuple[str, ...] = ()
    ambiguity: str | None = None
    grounded_entities: tuple[str, ...] = ()
    unknown_fields: tuple[tuple[str, str], ...] = ()
    live_sensitive: bool = False
    inactive_quests: tuple[str, ...] = ()
    mentions_any: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    tools_any: tuple[str, ...] = ()
    question_allowed: bool = False
    missing_class: str | None = None
    # Conversation-level checks: the character each turn should be about, and strings that
    # belong to another character and must not appear in answers from `leak_from_turn` on.
    turn_characters: tuple[str | None, ...] = ()
    leak_terms: tuple[str, ...] = ()
    leak_from_turn: int = 1


@dataclass(frozen=True)
class Scenario:
    id: str
    profile: str
    turns: tuple[str, ...]
    expect: Expectations = field(default_factory=Expectations)
    tags: dict[str, str] = field(default_factory=dict)
    golden: bool = False
    representative: bool = False
    note: str = ""

    @property
    def prompt(self) -> str:
        return self.turns[-1]
