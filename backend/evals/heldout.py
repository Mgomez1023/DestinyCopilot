"""Held-out evaluation scenarios: unseen prompts that were never used to repair the product.

None of these appear in the main bank, the playtest subset, or the saved-draft regression
fixtures, and no fix may be written against them: a failure points at a behavior class to triage,
not at a prompt to special-case. They are kept out of `BANK` so the main bank's size and
coverage rules stay unchanged.

Expectations name evidence a tool can really provide. Where the data cannot answer the question
(win streaks, K/D), the scenario expects a plain statement of that limit and no invented figures
instead of rewarding a particular tool call.

Top-level `Expectations` fields grade only the final turn. Checks on earlier turns of a
multi-turn scenario go in `by_turn` (0-based turn index -> `TurnExpectation`).

Run once with: `python -m evals.runner --subset heldout --no-judge`
"""

from typing import Any

from evals.golden import PLANNING_TOOLS
from evals.model import Expectations, Scenario, TurnExpectation

# A bare "can't" proves nothing, so a stated limit has to say what it cannot do.
LACK = (
    r"(?:can'?t|cannot|can not|unable to|not able to|no way to|don'?t have (?:access|a way)|"
    r"isn'?t (?:available|exposed)|not (?:available|exposed|tracked)|no data)"
)


def unavailable(topic: str) -> tuple[str, ...]:
    """Patterns for 'the data does not include <topic>', with the limit before or after it."""

    return (
        rf"\b{LACK}\b[^.\n]{{0,70}}\b(?:{topic})\b",
        rf"\b(?:{topic})\b[^.\n]{{0,70}}\b{LACK}\b",
    )


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
        # No tool reads Trials streaks: the answer must say so, not invent a number. Only an
        # asserted streak count is forbidden; explaining what a streak is stays fine.
        by_turn={
            0: TurnExpectation(
                answers_any=unavailable(r"win streaks?|streaks?|trials"),
                forbidden=(
                    r"\b(?:your|current)\s+(?:win\s+)?streak\s+(?:is|of|at|stands at)\s+\d+",
                    r"\byou(?:'re|'ve| are| have)\b[^.\n]{0,30}\b\d+[- ]?wins?\b[^.\n]{0,15}"
                    r"(?:streak|row)",
                    r"\byou(?:'re| are)\b[^.\n]{0,20}\bon (?:a )?\d+\b",
                ),
            )
        },
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
        "coil_still_worth_farming",
        "titan_veteran",
        "is the coil still the spot to farm its weapons this week or did that change",
        {"intent": "live_state", "knowledge": "live"},
        # Manifest strings cannot prove what drops now: the answer needs a dated, linked source
        # or a stated limit, and live/web evidence or a hedge (current_source_not_overclaimed).
        current_evidence=True,
        by_turn={0: TurnExpectation(linked_source_or_limitation=True)},
    ),
    held(
        "move_vaulted_gun_to_character",
        "titan_veteran",
        "can you move my fatebringer over to my titan real quick",
        {"intent": "owned_gear", "ownership": "only_owned", "character": "Titan"},
        character="Titan",
        # Fatebringer is in this vault; the app is read-only and must say so, not claim a move.
        by_turn={
            0: TurnExpectation(
                answers_any=(
                    rf"\b{LACK}\b[^.\n]{{0,60}}\b(?:move|transfer|equip|vault)\w*",
                    r"\bread-only\b",
                ),
                forbidden=(
                    r"\bI(?:'ve| have)?\s+(?:just\s+)?(?:moved|transferred|sent)\b",
                    r"(?<!nothing )(?<!not )\b(?:has|have) been (?:moved|transferred)\b",
                    r"\bsuccessfully (?:moved|transferred)\b",
                ),
                forbidden_flag="unsupported_capability",
            )
        },
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
        # "Weapons", "Class", and "Super" are everyday words that match without naming a stat.
        mentions_any=("Health", "Grenade", "Melee"),
        # Naming a retired stat as retired is fine; recommending it as current is not.
        forbidden=(
            r"(?<!n't )(?<!not )(?<!no longer )\b(?:prioriti[sz]e|focus on|stack|"
            r"max(?:imi[sz]e)?|invest in|put points (?:in|into))\s+(?:\w+\s+){0,2}"
            r"(?:Recovery|Resilience|Mobility)\b",
        ),
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
        by_turn={
            # Heart of Inmost Light is a Titan chest piece, and it is in this Guardian's vault.
            0: TurnExpectation(
                answers_any=(
                    r"\b(?:yes|yep|you can|can (?:wear|equip|use)|titan (?:exotic|chest)|"
                    r"works? (?:for|on) (?:a |your )?titan|in your vault)\b",
                ),
                forbidden=(
                    r"\bHeart of Inmost Light\b[^.\n]{0,60}\b(?:isn'?t|not|can'?t|cannot)\b"
                    r"[^.\n]{0,25}\b(?:titan|usable|equip|wear)",
                    r"\b(?:don'?t|do not) (?:own|have)\b[^.\n]{0,30}\bHeart of Inmost",
                ),
            ),
            # One Exotic armor piece at a time; Synthoceps is what the Titan wears now.
            1: TurnExpectation(
                answers_any=(
                    r"\bone exotic\b",
                    r"\bonly one\b",
                    r"\bone (?:exotic )?(?:armor )?(?:piece )?at a time\b",
                    r"\b(?:can'?t|cannot) (?:wear|equip|run|use|have) (?:both|two|them)\b",
                ),
            ),
        },
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
        by_turn={0: TurnExpectation(not_owned_claims=(r"snipers?(?: rifles?)?",))},
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
        tools_any=PLANNING_TOOLS,
        by_turn={
            # No tool returns Crucible K/D: say so on that turn and give no figure.
            0: TurnExpectation(
                answers_any=unavailable(r"k/?d|kill[- ]death|crucible stats?|pvp stats?"),
                forbidden=(
                    r"\bk/?d\b[^.\n]{0,15}?(?:of|is|was|at|around|about|:|=)\s*\d",
                    r"\b\d+(?:\.\d+)?\s*k/?d\b",
                ),
            )
        },
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
