"""Held-out evaluation scenarios: unseen prompts that were never used to repair the product.

None of these appear in the main bank, the playtest subset, or the saved-draft regression
fixtures, and no fix may be written against them: a failure points at a behavior class to triage,
not at a prompt to special-case. They are kept out of `BANK` so the main bank's size and
coverage rules stay unchanged.

Expectations name evidence a tool can really provide. Where the data cannot answer the question
(win streaks, K/D), the scenario expects a plain statement of that limit and no invented figures
instead of rewarding a particular tool call.

Run once with: `python -m evals.runner --subset heldout --no-judge`
"""

from typing import Any

from evals.golden import PLANNING_TOOLS
from evals.model import Expectations, Scenario

UNAVAILABLE = ("can't", "cannot", "don't have", "not available", "isn't available", "no data")
LOADOUT_TOOLS = ("get_equipped_loadout", "get_build_details", "analyze_current_build")

# Defects already seen in live runs; recorded for later triage, not asserted here.
KNOWN_ISSUES: tuple[str, ...] = (
    "Prismatic first-turn false refusal.",
    "Unverified equipment improvements despite empty alternative-search results.",
    "Planning fallback and unsupported Haunted Altars reward/solo claims.",
    "Sunshot Exotic replacement check is likely a false positive.",
    "Current-source answers should expose the sources they cite.",
)


def held(
    scenario_id: str,
    profile: str,
    turns: str | tuple[str, ...],
    tags: dict[str, str],
    **expect: Any,
) -> Scenario:
    return Scenario(
        f"heldout.{scenario_id}",
        profile,
        (turns,) if isinstance(turns, str) else turns,
        Expectations(**expect),
        tags,
        note="held-out: do not fix against this prompt",
    )


HELDOUT: tuple[Scenario, ...] = (
    # --- Single turn ---------------------------------------------------------------------
    held(
        "vault_shotgun",
        "hunter_veteran",
        "got any shotguns stashed away? my hunter could use one",
        {"intent": "owned_gear", "ownership": "only_owned", "character": "Hunter"},
        character="Hunter",
        only_owned=True,
        mentions_any=("Heritage",),
        # The vault listing is the evidence; a recommendation tool also reads it.
        tools_any=("search_inventory", "find_build_alternatives"),
    ),
    held(
        "no_stat_for_trials",
        "titan_veteran",
        "whats my win streak in trials rn",
        {"intent": "progression_help", "knowledge": "unknown"},
        # No tool reads Trials streaks: the answer must say so, not invent a number.
        mentions_any=UNAVAILABLE,
        forbidden=("win streak of", "you're on a"),
    ),
    held(
        "firefly_on_owned_gun",
        "titan_veteran",
        "what does the firefly perk on my fatebringer actually do",
        {"intent": "perk_explanation", "ownership": "only_owned", "knowledge": "stable"},
        mentions_any=("Firefly",),
        tools_any=("get_item_details", "search_destiny_entities", "search_inventory"),
    ),
    held(
        "ember_fragments_casual",
        "titan_veteran",
        "what are those two ember things on my titan subclass even doing for me",
        {"intent": "build_critique", "character": "Titan"},
        style="build",
        character="Titan",
        mentions_any=("Ember of Ashes", "Ember of Empyrean"),
        tools_any=LOADOUT_TOOLS,
    ),
    held(
        "frenzy_somewhere",
        "titan_veteran",
        "do any of my guns across all my guys have frenzy",
        {"intent": "perk_explanation", "ownership": "only_owned"},
        # Funnelweb (Subsistence, Frenzy) is equipped on the Hunter.
        mentions_any=("Funnelweb",),
        only_owned=True,
    ),
    held(
        "witherhoard_warlock",
        "warlock_veteran",
        "is witherhoard any good for my warlock or is it trash",
        {"intent": "item_explanation", "character": "Warlock", "ownership": "only_owned"},
        character="Warlock",
        mentions_any=("Witherhoard",),
    ),
    held(
        "three_friends_tonight",
        "titan_veteran",
        "me and two buddies are on in an hour, what should we run together",
        {"intent": "long_session", "social": "fireteam"},
        style="planning",
        tools_any=PLANNING_TOOLS,
    ),
    held(
        "lost_my_place_in_story",
        "returning_player",
        "wait i stopped in the middle of some story thing ages ago, what was it",
        {"intent": "quest_help", "account": "active_campaign"},
        mentions_any=("Transgression", "Final Shape"),
        tools_any=("get_active_quests", "get_quest_details", "get_progression"),
    ),
    held(
        "armor_stats_casual",
        "titan_veteran",
        "what stats should i even care about on armor",
        {"intent": "rules", "knowledge": "stable"},
        # Armor 3.0 stats are Weapons/Health/Class/Grenade/Melee/Super; the old six are retired.
        mentions_any=("Weapons", "Health", "Class", "Grenade", "Melee", "Super"),
        forbidden=("Resilience", "Recovery", "Mobility"),
    ),
    # --- Multi turn ----------------------------------------------------------------------
    held(
        "hunter_dungeon_leftovers",
        "hunter_veteran",
        (
            "i've been grinding gambit all week lol, anything else worth my time",
            "does my hunter have anything going toward a dungeon?",
        ),
        {"intent": "quest_help", "character": "Hunter"},
        character="Hunter",
        mentions_any=("Rite of the Nine", "Duality"),
    ),
    held(
        "exotic_armor_stacking",
        "titan_veteran",
        (
            "can my titan wear heart of inmost light",
            "and keep synthoceps on at the same time?",
        ),
        {"intent": "rules", "character": "Titan"},
        character="Titan",
        # One Exotic armor piece at a time; Synthoceps is what the Titan wears now.
        mentions_any=("one Exotic", "only one", "one at a time", "can't", "cannot"),
    ),
    held(
        "twenty_minutes_then_an_hour",
        "near_complete",
        (
            "ugh only got like 20 mins before work, anything quick?",
            "actually what if i stretch it to a full hour",
        ),
        {"intent": "short_session", "time": "1h", "account": "near_complete"},
        style="planning",
        mentions_any=("Duality", "Altars", "Crucible", "Vanguard"),
        tools_any=PLANNING_TOOLS,
    ),
    held(
        "sniper_then_fusion",
        "large_vault",
        (
            "thinking about a new sniper for my hunter, do i already have one",
            "ok what about a linear fusion instead",
        ),
        {"intent": "owned_gear", "ownership": "only_owned", "character": "Hunter"},
        character="Hunter",
        only_owned=True,
        # The vault holds Taipan-4fr and no sniper rifle.
        mentions_any=("Taipan-4fr",),
    ),
    held(
        "unavailable_then_solo",
        "titan_veteran",
        (
            "whats my k/d in crucible this season",
            "ok whatever, something fun i can do solo then?",
        ),
        {"intent": "what_next", "social": "solo", "character": "Titan"},
        style="planning",
        character="Titan",
        forbidden=("K/D of", "your K/D is"),
        tools_any=PLANNING_TOOLS,
    ),
    held(
        "pvp_then_dungeon_build",
        "hunter_veteran",
        (
            "how does my hunter hold up in pvp right now",
            "fine. and for the duality dungeon?",
        ),
        {"intent": "build_critique", "character": "Hunter"},
        style="build",
        character="Hunter",
        mentions_any=("Star-Eater Scales", "Sunshot", "Rose"),
        tools_any=LOADOUT_TOOLS,
    ),
)
