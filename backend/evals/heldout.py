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
from evals.model import LACK, Expectations, Scenario, TurnExpectation


def unavailable(topic: str) -> tuple[str, ...]:
    """Patterns for 'the data does not include <topic>', with the limit before or after it."""

    return (
        rf"\b{LACK}\b[^.\n]{{0,70}}\b(?:{topic})\b",
        rf"\b(?:{topic})\b[^.\n]{{0,70}}\b{LACK}\b",
    )


# Wrong answers to "can I wear/run both?": a yes, or a per-slot rule that does not exist. Each
# pattern skips negations ("can't wear both", "not one Exotic per slot") and "one ..., not both".
EXOTIC_WRONG_YES: tuple[str, ...] = (
    r"\b(?:can|could|may)\s+(?:equip|wear|run|use|have)\b(?:(?!\bnot\b|n't|\bonly\b|\bone\b)[^.\n])"
    r"{0,25}\b(?:both|two|them both|all of them)\b",
    r"(?<!not )(?<!n't )\bone exotic (?:armor |weapon )?(?:piece |item )?"
    r"(?:per|for each|in each|in every) (?:\w+ )?slot\b",
    r"^\W*(?:yes|yep|yeah|sure)\b[\W]*(?:you can|both|they (?:stack|work)|that works|it works)",
    r"\beach (?:armor |weapon )?slot (?:can|may|gets?) (?:have |hold )?(?:its|an|one) "
    r"(?:own )?exotic",
)


def exotic_limit(kind: str) -> tuple[str, ...]:
    """Patterns stating that only one Exotic `kind` can be equipped. A bare "one Exotic" is not
    enough: it also appears in the wrong answer "one Exotic per slot"."""

    return (
        rf"\b(?:only|just|at most)\s+(?:(?:wear|equip|run|use|have)\w*\s+)?(?:one|1)\s+"
        rf"(?:exotic\s+)?{kind}\b",
        rf"\b(?:one|1)\s+(?:exotic\s+)?{kind}\b[^.\n]{{0,30}}\b(?:at a time|at once|total|"
        r"overall|max(?:imum)?)\b",
        rf"\b(?:one|1)\s+(?:exotic\s+)?{kind}\b[^.\n]{{0,30}}\bnot (?:both|two|2)\b",
        r"\b(?:can'?t|cannot|can not|not able to|unable to|aren'?t able to)\s+"
        r"(?:wear|equip|run|use|have)\b[^.\n]{0,30}\b(?:both|two|2|them both|together|at once|"
        r"a second)\b",
    )


# Curated guides cover other activities (the Dreaming City arena, Wish-Ender, Hunter's
# Remembrance, Prismatic Facets). An answer about an activity with no guide must not borrow them.
OTHER_ACTIVITY_GUIDES: tuple[str, ...] = (
    r"\b(?:Vorgeth|Dul Incaru|Shattered Throne|Dreaming City|Wish-Ender|Hunter'?s Remembrance|"
    r"Micah-10|Light Vestiges?|Darkness Vestiges?)\b",
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
    # --- Replacements for the four scenarios retired after run 2026-10-22T04:52:27+00:00 ---
    held(
        "envious_assassin_elsewhere",
        "titan_veteran",
        "does my titan have anything running envious assassin?",
        {"intent": "perk_explanation", "ownership": "only_owned", "character": "Titan"},
        # Envious Assassin is only on Edge Transit, which the Hunter carries; the Titan's own
        # weapons (Rose, Calus Mini-Tool, Apex Predator) do not have it. A search that ignores
        # perks or other characters would wrongly say nobody has it.
        only_owned=True,
        mentions_any=("Edge Transit",),
        tools_any=("search_inventory",),
        by_turn={
            0: TurnExpectation(
                forbidden=(
                    r"\bEdge Transit\b[^.\n]{0,40}\b(?:on|for|carried by|equipped (?:on|by)) "
                    r"your Titan\b",
                ),
            )
        },
    ),
    held(
        "xur_stock_now",
        "hunter_veteran",
        "is xur around right now and what's he selling",
        {"intent": "live_state", "knowledge": "live"},
        # A vendor's presence and stock cannot come from the Manifest or the account: the
        # answer needs a returned source or a plainly stated limit, not a canned refusal.
        current_evidence=True,
        by_turn={0: TurnExpectation(linked_source_or_limitation=True)},
    ),
    held(
        "two_exotic_weapons",
        "hunter_veteran",
        (
            "can i run ace of spades along with my sunshot?",
            "what if gjallarhorn goes in the heavy slot instead, does that count",
            "ok and star-eater scales on top of sunshot is fine though right",
        ),
        {"intent": "rules", "character": "Hunter"},
        # Ace of Spades, Gjallarhorn and Le Monarque are in this vault; Sunshot and
        # Star-Eater Scales are equipped on the Hunter. The rule is one Exotic weapon plus one
        # Exotic armor piece at a time, so turns 1-2 are "no" and turn 3 is "yes".
        character="Hunter",
        mentions_any=("Star-Eater Scales",),
        by_turn={
            0: TurnExpectation(
                answers_any=exotic_limit(r"(?:weapon|gun)s?"),
                forbidden=(
                    *EXOTIC_WRONG_YES,
                    r"\b(?:don'?t|do not) (?:own|have)\b[^.\n]{0,30}\bAce of Spades",
                ),
            ),
            1: TurnExpectation(
                answers_any=exotic_limit(r"(?:weapon|gun)s?"),
                forbidden=(
                    *EXOTIC_WRONG_YES,
                    # Heavy weapons are weapons: the slot does not exempt an Exotic.
                    r"\b(?:heavy|power)\b[^.\n]{0,40}\b(?:doesn'?t|does not|don'?t|do not|"
                    r"isn'?t|is not)\s+count\b",
                    r"\b(?:don'?t|do not) (?:own|have)\b[^.\n]{0,30}\bGjallarhorn",
                ),
            ),
            2: TurnExpectation(
                # One Exotic weapon plus one Exotic armor piece can be worn together.
                answers_any=(
                    r"\b(?:yes|yep|yeah|that works|that'?s fine|no problem|allowed|compatible)\b",
                    r"\b(?:one|1) exotic weapon (?:and|plus|with|alongside) (?:one|1) exotic "
                    r"armor",
                    r"\b(?:armor|star-eater)[^.\n]{0,40}\b(?:separate|different|doesn'?t "
                    r"(?:count|conflict))",
                ),
                forbidden=(
                    r"^\W*no\b[\W]*(?:you can'?t|you cannot|those|they|that)",
                    r"\b(?:only|just) one exotic (?:item|thing|total|in total|overall|across)",
                ),
            ),
        },
    ),
    held(
        "onslaught_build_then_waves",
        "warlock_veteran",
        (
            "what should my warlock run for onslaught legend, the later waves are wrecking me",
            "and how do the last waves actually work, anything i need to know",
        ),
        {"intent": "build_critique", "character": "Warlock", "knowledge": "unknown"},
        # No curated guide covers Onslaught. Build advice must come from the Warlock's equipped
        # loadout (Sunbracers, Sunshot, Rose, Falling Guillotine), and the mechanics turn needs
        # a returned source or a stated limit, never another activity's guide.
        style="build",
        character="Warlock",
        mentions_any=("Sunbracers", "Sunshot", "Falling Guillotine", "Rose"),
        tools_any=LOADOUT_TOOLS,
        by_turn={
            0: TurnExpectation(forbidden=OTHER_ACTIVITY_GUIDES),
            1: TurnExpectation(forbidden=OTHER_ACTIVITY_GUIDES, linked_source_or_limitation=True),
        },
    ),
)
