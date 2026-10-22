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
        rf"\b(?:only|just|at most)\s+(?:\w+\s+){{0,2}}?(?:one|1)\s+(?:exotic\s+)?{kind}\b",
        rf"\b(?:one|1)\s+(?:exotic\s+)?{kind}\b[^.\n]{{0,60}}\b(?:at a time|at once|"
        r"at any (?:given )?time|total|overall|max(?:imum)?)\b",
        rf"\b(?:at a time|at once|at any (?:given )?time)\b[^.\n]{{0,40}}\b(?:one|1)\s+"
        rf"(?:exotic\s+)?{kind}\b",
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
        tools_any=(
            "get_active_quests",
            "get_quest_details",
            "get_progression",
            "get_content_progression",
        ),
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
    # --- Replacements for scenarios retired after runs 2026-10-22T04:52:27 and 06:01:24 ---
    held(
        "whirlwind_blade_elsewhere",
        "warlock_veteran",
        "is whirlwind blade on anything my hunter is carrying?",
        {"intent": "perk_explanation", "ownership": "only_owned", "character": "Hunter"},
        # Whirlwind Blade is only on Falling Guillotine, which the Warlock has equipped; the
        # Hunter's Rose, Riskrunner and Apex Predator do not have it. The answer must say the
        # Hunter has none, and name the item with the character that holds it.
        character="Hunter",
        only_owned=True,
        mentions_any=("Falling Guillotine",),
        tools_any=("search_inventory",),
        by_turn={
            0: TurnExpectation(
                answers_any=(
                    r"\bFalling Guillotine\b[^.\n]{0,80}\bWarlock\b",
                    r"\bWarlock\b[^.\n]{0,80}\bFalling Guillotine\b",
                ),
                forbidden=(
                    # "...on your Warlock, not your Hunter" is fine; a plain "on your Hunter" is not
                    r"\bFalling Guillotine\b(?:(?!\bnot\b|n't|\bnever\b)[^.\n]){0,40}"
                    r"\b(?:on|for|carried by|equipped (?:on|by)) your Hunter\b",
                ),
            )
        },
    ),
    held(
        "class_premise_on_vaulted_armor",
        "warlock_veteran",
        (
            "what exotic armor have i got sitting in my vault",
            "nice, ill put the contraverse hold on my titan, those are titan gauntlets right?",
        ),
        {"intent": "owned_gear", "ownership": "only_owned", "character": "Titan"},
        # Contraverse Hold (Warlock gauntlets) and Phoenix Protocol (Warlock chest armor) are in
        # this vault. The player's premise on turn 2 is wrong: the item is Warlock-only, so the
        # answer must correct it, not agree, and must not say a Titan can wear it.
        character="Titan",
        tools_any=("get_item_details", "search_inventory", "search_destiny_entities"),
        by_turn={
            0: TurnExpectation(
                answers_any=(r"\bContraverse Hold\b", r"\bPhoenix Protocol\b"),
            ),
            1: TurnExpectation(
                answers_any=(
                    r"\bContraverse Hold\b[^.\n]{0,80}\bWarlock",
                    r"\bWarlock[^.\n]{0,80}\bContraverse Hold\b",
                ),
                forbidden=(
                    r"^\W*(?:yes|yep|yeah|correct|exactly|that'?s right|you(?:'re| are) right)\b",
                    r"\bTitans? can (?:wear|equip|use|run)\b[^.\n]{0,30}\bContraverse Hold\b",
                ),
            ),
        },
    ),
    held(
        "headshots_with_rose",
        "titan_veteran",
        "how many headshots have i landed with rose this season",
        {"intent": "progression_help", "knowledge": "unknown"},
        # No tool returns weapon kill or precision counts. The answer must say so; stating any
        # count, including zero or "none", reports an absence as a value.
        by_turn={
            0: TurnExpectation(
                answers_any=unavailable(
                    r"headshots?|precision(?: kills?| hits?)?|kill counts?|weapon (?:kill )?stats?"
                ),
                forbidden=(
                    r"\byou(?:'ve| have| landed| got| hit)\b[^.\n]{0,40}"
                    r"\b(?:\d[\d,]*|zero)\s+(?:headshots?|precision (?:kills?|hits?))\b",
                    r"\byour\b[^.\n]{0,30}\b(?:headshots?|precision (?:kills?|hits?))\b[^.\n]{0,20}"
                    r"\b(?:is|are|total|count|stands at|of|at|:)\s*(?:\d|zero|none|nothing)",
                    r"\b(?:no|zero|0)\s+(?:headshots?|precision (?:kills?|hits?))\b"
                    r"(?!\s+(?:data|stats?|counts?|info|tracking|numbers?))",
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
        # The loadout evidence belongs to the build turn; the mechanics turn is graded on its
        # source or stated limit, not on naming gear.
        by_turn={
            0: TurnExpectation(
                answers_any=(r"\b(?:Sunbracers|Sunshot|Falling Guillotine|Rose)\b",),
                forbidden=OTHER_ACTIVITY_GUIDES,
                tools_any=LOADOUT_TOOLS,
            ),
            1: TurnExpectation(forbidden=OTHER_ACTIVITY_GUIDES, linked_source_or_limitation=True),
        },
    ),
)
