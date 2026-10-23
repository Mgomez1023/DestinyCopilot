"""Held-out scenarios that real-model runs exposed, now retired.

Two batches: four from run `started_at=2026-10-22T04:52:27+00:00` and three from run
`started_at=2026-10-22T06:01:24+00:00`. Their prompts have been seen and used to repair the
product, so they no longer measure generalization and are not counted as held-out. They stay
runnable as regression scenarios (`python -m evals.runner --subset heldout_retired --no-judge`)
under their original ids so results remain comparable with those runs. Each was replaced in
`HELDOUT` by a new scenario probing the same behavior class with different wording, items and
activities.
"""

from dataclasses import replace

from evals.heldout import EXOTIC_WRONG_YES, LOADOUT_TOOLS, exotic_limit, held, unavailable
from evals.model import Scenario, TurnExpectation

FIRST_RUN = "2026-10-22T04:52:27+00:00"
SECOND_RUN = "2026-10-22T06:01:24+00:00"


_NUMBER = r"(?:\d+|zero|none|nothing)"
# Any asserted streak value is wrong, zero included: "current Trials win streak is 0".
STREAK_ASSERTED: tuple[str, ...] = (
    rf"\b(?:your|current)\s+(?:\w+\s+){{0,3}}?streaks?\s+(?:is|are|of|at|stands at|sits at|"
    rf"is currently|currently)\s+(?:currently\s+)?{_NUMBER}\b",
    rf"\bstreak\s+(?:is|stands at|sits at)\s+(?:currently\s+)?{_NUMBER}\b",
    r"\byou(?:'re|'ve| are| have)?\s+(?:currently\s+)?(?:have\s+)?(?:no|zero|0)\s+"
    r"(?:\w+\s+){0,2}?streaks?\b(?!\s+(?:data|info|stats?|records?|tracking))",
)


def retired(scenario: Scenario, run: str) -> Scenario:
    note = f"retired held-out (seen in run {run}): regression only"
    return replace(scenario, note=note)


FIRST_BATCH: tuple[Scenario, ...] = tuple(
    retired(value, FIRST_RUN)
    for value in (
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
            # Manifest strings cannot prove what drops now: the answer needs a dated, linked
            # source or a stated limit, and live/web evidence or a hedge.
            current_evidence=True,
            by_turn={0: TurnExpectation(linked_source_or_limitation=True)},
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
                # Heart of Inmost Light is a Titan chest piece, and it is in this vault.
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
                    answers_any=exotic_limit(r"(?:armor|piece)"),
                    forbidden=EXOTIC_WRONG_YES,
                ),
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
)

SECOND_BATCH: tuple[Scenario, ...] = tuple(
    retired(value, SECOND_RUN)
    for value in (
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
                        *STREAK_ASSERTED,
                        r"\byou(?:'re|'ve| are| have)\b[^.\n]{0,30}\b\d+[- ]?wins?\b[^.\n]{0,15}"
                        r"(?:streak|row)",
                        r"\byou(?:'re| are)\b[^.\n]{0,20}\bon (?:a )?\d+\b",
                    ),
                )
            },
        ),
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
                        r"\b(?:yes|yep|yeah|that works|that'?s fine|no problem|allowed|"
                        r"compatible)\b",
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
    )
)

HELDOUT_RETIRED: tuple[Scenario, ...] = FIRST_BATCH + SECOND_BATCH
