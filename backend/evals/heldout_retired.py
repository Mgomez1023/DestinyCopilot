"""Held-out scenarios that run `started_at=2026-10-22T04:52:27+00:00` exposed, now retired.

Their prompts have been seen and used to repair the product, so they no longer measure
generalization and are not counted as held-out. They stay runnable as regression scenarios
(`python -m evals.runner --subset heldout_retired --no-judge`) under their original ids so results
remain comparable with that run. Each was replaced in `HELDOUT` by a new scenario probing the same
behavior class with different wording, items and activities.
"""

from dataclasses import replace

from evals.heldout import EXOTIC_WRONG_YES, LOADOUT_TOOLS, exotic_limit, held
from evals.model import Scenario, TurnExpectation

RETIRED_NOTE = "retired held-out (seen in run 2026-10-22T04:52:27+00:00): regression only"


def retired(scenario: Scenario) -> Scenario:
    return replace(scenario, note=RETIRED_NOTE)


HELDOUT_RETIRED: tuple[Scenario, ...] = tuple(
    retired(value)
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
