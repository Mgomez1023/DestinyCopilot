"""The manual "never regress" golden set, drawn from observed failures and product goals.

Golden prompts are regression tests, not special cases: no production code may branch on
them. Multi-turn entries replay earlier turns live, so the model's own prior answers form
the conversation history.
"""

from evals.model import Expectations as E
from evals.model import Scenario

BUILD_TOOLS = ("analyze_current_build", "find_build_alternatives", "search_inventory")
PLANNING_TOOLS = ("get_active_quests", "get_available_activities", "get_character_summary")

GOLDEN: tuple[Scenario, ...] = (
    Scenario(
        "golden.what_next",
        "titan_veteran",
        ("What should I do next?",),
        E(style="planning", inactive_quests=("A Single Thread",), tools_any=PLANNING_TOOLS),
        {"intent": "what_next", "account": "active_campaign"},
        golden=True,
        note="Failure A: previously recommended an inactive quest (A Single Thread).",
    ),
    Scenario(
        "golden.why_altars",
        "titan_veteran",
        ("Why should I finish Altars of Sorrow?",),
        E(
            style="explain",
            grounded_entities=("Altars of Sorrow",),
            unknown_fields=(("Altars of Sorrow", "matchmaking"), ("Altars of Sorrow", "rewards")),
            mentions_any=("46", "Waves", "four", "4 more"),
        ),
        {"intent": "progression_help", "account": "near_complete", "knowledge": "stable"},
        golden=True,
    ),
    Scenario(
        "golden.where_altars",
        "titan_veteran",
        ("Where is Altars of Sorrow?",),
        E(grounded_entities=("Altars of Sorrow",), mentions_any=("Moon",)),
        {"intent": "location", "knowledge": "stable"},
        golden=True,
        note="Failure E: location and distinction from Haunted Altars.",
    ),
    Scenario(
        "golden.titan_loadout_owned",
        "titan_veteran",
        ("Look at what I own and recommend a Titan loadout",),
        E(style="build", character="Titan", only_owned=True, tools_any=BUILD_TOOLS),
        {
            "intent": "build_creation",
            "character": "Titan",
            "ownership": "only_owned",
            "account": "class_locked_exotics",
        },
        golden=True,
    ),
    Scenario(
        "golden.only_owned_followup",
        "titan_veteran",
        ("Look at what I own and recommend a Titan loadout", "Only recommend things I own"),
        E(style="build", character="Titan", only_owned=True, tools_any=BUILD_TOOLS),
        {"intent": "owned_gear", "character": "Titan", "ownership": "only_owned"},
        golden=True,
    ),
    Scenario(
        "golden.rampage",
        "titan_veteran",
        ("What does the Rampage perk do?",),
        E(style="explain", mentions_any=("damage", "kill")),
        {"intent": "perk_explanation", "knowledge": "stable"},
        golden=True,
    ),
    Scenario(
        "golden.weakest_build",
        "titan_veteran",
        ("What's the weakest part of my current build?",),
        E(style="build", character="Titan", tools_any=("analyze_current_build",)),
        {"intent": "build_critique", "character": "Titan"},
        golden=True,
        note="Failure B: previously returned a grounded-refusal fallback.",
    ),
    Scenario(
        "golden.thirty_minutes",
        "titan_veteran",
        ("I only have 30 minutes. Pick one thing for me.",),
        E(style="planning", inactive_quests=("A Single Thread",), tools_any=PLANNING_TOOLS),
        {"intent": "short_session", "time": "30m"},
        golden=True,
    ),
    Scenario(
        "golden.hunter_switch",
        "titan_veteran",
        (
            "I only have 30 minutes. Pick one thing for me.",
            "Actually what if I wanted to play on my Hunter?",
        ),
        E(style="planning", character="Hunter", tools_any=PLANNING_TOOLS),
        {"intent": "short_session", "character": "switch", "time": "30m"},
        golden=True,
        note="Failure C: switched to Hunter but gave a generic plan.",
    ),
    Scenario(
        "golden.altars_ambiguity",
        "titan_veteran",
        ('What does "Altars" refer to?',),
        E(ambiguity="altars", question_allowed=True),
        {"intent": "ambiguity", "knowledge": "ambiguous"},
        golden=True,
    ),
    Scenario(
        "golden.altars_matchmade",
        "titan_veteran",
        ("Is Altars of Sorrow matchmade?",),
        E(
            grounded_entities=("Altars of Sorrow",),
            unknown_fields=(("Altars of Sorrow", "matchmaking"),),
        ),
        {"intent": "activity_explanation", "knowledge": "unknown"},
        golden=True,
    ),
    Scenario(
        "golden.two_exotic_armor",
        "titan_veteran",
        ("Can I equip two Exotic armor pieces?",),
        E(mentions_any=("one", "1", "only"), forbidden=(r"\byes\b[^.]*\btwo\b",)),
        {"intent": "rules", "knowledge": "stable"},
        golden=True,
    ),
    Scenario(
        "golden.exotic_weapon_and_armor",
        "titan_veteran",
        ("Can I equip an Exotic weapon and Exotic armor together?",),
        E(mentions_any=("yes",), forbidden=(r"^\s*no\b",)),
        {"intent": "rules", "knowledge": "stable"},
        golden=True,
    ),
    Scenario(
        "golden.use_hc",
        "titan_veteran",
        ("What's the weakest part of my current build?", "Use a HC for this build."),
        E(style="build", character="Titan", mentions_any=("Rose", "Fatebringer", "Ace of Spades")),
        {"intent": "build_creation", "character": "Titan", "knowledge": "shorthand"},
        golden=True,
    ),
    Scenario(
        "golden.what_is_dungeon",
        "titan_veteran",
        ("What is a dungeon?",),
        E(mentions_any=("three", "3")),
        {"intent": "activity_explanation", "knowledge": "stable"},
        golden=True,
    ),
    Scenario(
        "golden.nightfall_this_week",
        "titan_veteran",
        ("What is the Nightfall this week?",),
        E(live_sensitive=True),
        {"intent": "live_state", "knowledge": "live"},
        golden=True,
    ),
    Scenario(
        "golden.what_is_nightfall",
        "titan_veteran",
        ("What is a Nightfall?",),
        E(mentions_any=("strike", "Strike")),
        {"intent": "activity_explanation", "knowledge": "stable"},
        golden=True,
    ),
)
