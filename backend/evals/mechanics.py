"""Mechanics grounding baseline: scenarios, evidence requirements, and trace aggregation.

Knowledge Engine v0.2, checkpoint M0. These scenarios exist to measure how often answers make
mechanics claims that the turn never retrieved evidence for. Today's tools do not return all
the evidence a perfect answer would need (socket options, mod legality), so some scenarios
are expected to show gaps; that gap is the baseline later checkpoints must close.

Run with `python -m evals.runner --subset mechanics` (live model, so keep it small) or read the
deterministic detector tests in `tests/test_mechanics_claims.py`. Scores here use the existing
rubric and properties; the mechanics-claim trace is informational and never changes a score.
"""

from dataclasses import dataclass
from typing import Any

from evals.model import Expectations as E
from evals.model import Scenario

GUARDIAN = ("get_equipped_loadout", "get_build_details", "analyze_current_build")
KNOWLEDGE = ("get_item_details", "search_destiny_entities")


@dataclass(frozen=True)
class MechanicsSpec:
    """What a grounded answer needs, and what counts as a serious failure."""

    category: str
    evidence_required: tuple[str, ...]
    serious_failure: str


def _scenario(
    scenario_id: str,
    profile: str,
    prompt: str,
    spec: MechanicsSpec,
    intent: str,
    **expect: Any,
) -> tuple[Scenario, MechanicsSpec]:
    scenario = Scenario(
        scenario_id,
        profile,
        (prompt,),
        E(**expect),
        {"intent": intent, "mechanics": spec.category},
    )
    return scenario, spec


_CHANGE_CLAIM = (
    r"\b(?:yes|you can|you could|simply)\b[^.?!\n]{0,40}\b(?:swap|change|switch|reroll|choose|"
    r"pick)\b[^.?!\n]{0,40}\b(?:barrel|grip|magazine|stock|perk|trait)\b"
)

_ENTRIES: tuple[tuple[Scenario, MechanicsSpec], ...] = (
    _scenario(
        "mech.exotic_weapon_effect",
        "titan_veteran",
        "What does Sunshot do?",
        MechanicsSpec(
            "exotic_effect",
            ("manifest: intrinsic and trait plug text (Sunburn, Sun Blast)",),
            "An effect not present in the retrieved plug text.",
        ),
        "item_explanation",
        style="explain",
        mentions_any=("explo", "Solar", "highlight", "scorch"),
        tools_any=KNOWLEDGE,
    ),
    _scenario(
        "mech.exotic_armor_effect_and_slot",
        "titan_veteran",
        "What does Synthoceps do, and which armor slot is it?",
        MechanicsSpec(
            "armor_slot",
            ("manifest: definition slot (Gauntlets) and class (Titan)", "plug text"),
            "Wrong slot or class stated as fact.",
        ),
        "item_explanation",
        style="explain",
        mentions_any=("Gauntlets", "Arms"),
        forbidden=(r"\bsynthoceps\b[^.\n]{0,40}\b(?:helmet|chest|leg armor|class item)\b",),
        tools_any=KNOWLEDGE,
    ),
    _scenario(
        "mech.exotic_armor_class_restriction",
        "hunter_veteran",
        "What does Celestial Nighthawk do, and which class can wear it?",
        MechanicsSpec(
            "armor_class",
            ("manifest: class restriction (Hunter), slot (Helmet), plug text",),
            "A class other than Hunter, or an invented Super interaction.",
        ),
        "item_explanation",
        style="explain",
        mentions_any=("Hunter",),
        forbidden=(r"\b(?:titan|warlock)[- ](?:only|exclusive)\b",),
        tools_any=KNOWLEDGE,
    ),
    _scenario(
        "mech.exotic_conditional_effect",
        "warlock_veteran",
        "What does Verity's Brow do?",
        MechanicsSpec(
            "exotic_conditional",
            ("manifest: full conditional plug text ('damage type matching your grenade')",),
            "Dropping or altering the condition so the effect reads as unconditional.",
        ),
        "item_explanation",
        style="explain",
        mentions_any=("grenade", "Death Throes"),
        tools_any=KNOWLEDGE,
    ),
    _scenario(
        "mech.perk_rampage",
        "titan_veteran",
        "What does the Rampage perk do?",
        MechanicsSpec(
            "perk_explanation",
            ("manifest: Rampage plug text; Trait and Enhanced Trait variants differ",),
            "An effect that matches neither variant, or one variant silently chosen.",
        ),
        "perk_explanation",
        style="explain",
        mentions_any=("damage", "stack"),
        tools_any=KNOWLEDGE,
    ),
    _scenario(
        "mech.perk_outlaw",
        "titan_veteran",
        "What does the Outlaw perk do?",
        MechanicsSpec(
            "perk_explanation",
            ("manifest: Outlaw plug text",),
            "Wrong trigger condition (precision kill versus any kill).",
        ),
        "perk_explanation",
        style="explain",
        mentions_any=("reload", "precision"),
        tools_any=KNOWLEDGE,
    ),
    _scenario(
        "mech.fixed_exotic_sockets",
        "titan_veteran",
        "Can I change Sunshot's barrel or grip?",
        MechanicsSpec(
            "configurable_vs_fixed",
            ("manifest: socket plug sets (single-option, display-only) for Sunshot",),
            "Telling the user a fixed Exotic socket can be changed.",
        ),
        "item_explanation",
        style="explain",
        forbidden=(_CHANGE_CLAIM,),
        tools_any=KNOWLEDGE,
    ),
    _scenario(
        "mech.owned_roll_changes",
        "titan_veteran",
        "Can I change the perks on my Rose?",
        MechanicsSpec(
            "configurable_vs_fixed",
            ("guardian: equipped perks", "instance reusable options (not retrieved today)"),
            "Naming alternative perks as available without option evidence.",
        ),
        "perk_explanation",
        style="explain",
        mentions_any=("Rampage", "Opening Shot"),
        forbidden=(r"\byou can (?:swap|change|switch) (?:it |them )?to \w+",),
        tools_any=GUARDIAN,
    ),
    _scenario(
        "mech.equipped_perks",
        "titan_veteran",
        "Which perks are equipped on my Rose?",
        MechanicsSpec(
            "equipped_configuration",
            ("guardian: socketed plugs on the equipped Rose",),
            "Listing perks the socket data does not show.",
        ),
        "perk_explanation",
        style="explain",
        mentions_any=("Rampage", "Opening Shot"),
        tools_any=GUARDIAN,
    ),
    _scenario(
        "mech.class_restriction_cross_class",
        "warlock_veteran",
        "Can my Warlock equip Synthoceps?",
        MechanicsSpec(
            "armor_class",
            ("manifest: class restriction (Titan)",),
            "Saying a Warlock can equip a Titan Exotic.",
        ),
        "item_explanation",
        style="explain",
        character="Warlock",
        forbidden=(r"^\s*yes\b", r"\byou can equip (?:it|synthoceps)\b"),
        tools_any=KNOWLEDGE,
    ),
    _scenario(
        "mech.slot_for_three_exotics",
        "titan_veteran",
        "Which armor slot do Synthoceps, Celestial Nighthawk and Verity's Brow take?",
        MechanicsSpec(
            "armor_slot",
            ("manifest: slot of each Exotic (Gauntlets, Helmet, Helmet)",),
            "Any slot stated wrongly, or the engram variant's missing slot taken as unknown.",
        ),
        "item_explanation",
        style="explain",
        mentions_any=("Gauntlets",),
        tools_any=KNOWLEDGE,
    ),
    _scenario(
        "mech.armor_mod_slot_compatibility",
        "titan_veteran",
        "Can I put Grenade Kickstart on my helmet?",
        MechanicsSpec(
            "mod_compatibility",
            ("manifest: mod plug category (arms) versus socket whitelist",),
            "Saying yes for a mod whose category does not fit the helmet.",
        ),
        "item_explanation",
        style="explain",
        forbidden=(r"^\s*yes\b", r"\byou can (?:put|slot|use) it on (?:your )?helmet\b"),
        tools_any=KNOWLEDGE,
    ),
    _scenario(
        "mech.armor_mod_effect",
        "titan_veteran",
        "What does the Grenade Kickstart mod do?",
        MechanicsSpec(
            "mod_effect",
            ("manifest: sandbox perk text (the item description may be empty)",),
            "An effect not in the sandbox text, or an energy cost stated from memory.",
        ),
        "perk_explanation",
        style="explain",
        mentions_any=("Armor Charge", "grenade energy"),
        tools_any=KNOWLEDGE,
    ),
    _scenario(
        "mech.fragment_effect",
        "warlock_veteran",
        "What does the Spark of Shock fragment do?",
        MechanicsSpec(
            "fragment_effect",
            ("manifest: sandbox perk text and stat penalty",),
            "Inventing numbers or interactions beyond the retrieved text.",
        ),
        "perk_explanation",
        style="explain",
        mentions_any=("jolt", "Arc grenade"),
        tools_any=KNOWLEDGE,
    ),
    _scenario(
        "mech.objective_damage_source",
        "titan_veteran",
        "How do I make progress on Arc Scourge?",
        MechanicsSpec(
            "damage_source",
            ("manifest: objective text ('Arc ability final blows')",),
            "Telling the user Arc weapon kills count toward an ability requirement.",
        ),
        "quest_help",
        style="explain",
        mentions_any=("ability",),
        forbidden=(r"\barc (?:weapon|primary|special|heavy)s?\b[^.\n]{0,30}\b(?:count|progress)",),
        tools_any=KNOWLEDGE,
    ),
    _scenario(
        "mech.weapon_kills_versus_ability_requirement",
        "titan_veteran",
        "Do Arc weapon kills count for Arc Scourge?",
        MechanicsSpec(
            "damage_source",
            ("manifest: objective text distinguishing ability from weapon final blows",),
            "Answering yes.",
        ),
        "quest_help",
        style="explain",
        mentions_any=("ability",),
        forbidden=(r"^\s*yes\b",),
        tools_any=KNOWLEDGE,
    ),
    _scenario(
        "mech.owned_only_hypothetical_change",
        "titan_veteran",
        "Using only what I own, which Sunshot barrel should I swap to?",
        MechanicsSpec(
            "hypothetical_configuration",
            ("manifest: whether Sunshot's barrel is configurable", "guardian: equipped plugs"),
            "Recommending a named barrel swap without legality evidence.",
        ),
        "owned_gear",
        style="build",
        only_owned=True,
        forbidden=(r"\b(?:swap|switch|change) (?:to|it to)\b[^.\n]{0,30}\bbarrel\b",),
        tools_any=GUARDIAN,
    ),
    _scenario(
        "mech.multi_item_synergy",
        "titan_veteran",
        "Do Sunshot and Synthoceps work well together?",
        MechanicsSpec(
            "multi_item_synergy",
            ("manifest: both items' effect text", "no documented interaction exists"),
            "Asserting an undocumented synergy as established.",
        ),
        "build_critique",
        style="build",
        forbidden=(r"\b(?:guaranteed|always|perfect synergy|stack together)\b",),
        tools_any=KNOWLEDGE + GUARDIAN,
    ),
    _scenario(
        "mech.unknown_perk",
        "titan_veteran",
        "What does the perk Quantum Overdrive do?",
        MechanicsSpec(
            "unknown_mechanic",
            ("manifest search that returns no match",),
            "Describing an effect for a name no retrieval found.",
        ),
        "perk_explanation",
        style="explain",
        forbidden=(
            r"quantum overdrive[^.\n]{0,40}\b(?:grants?|increases?|reduces?|boosts?|gives?)\b",
        ),
        tools_any=KNOWLEDGE,
    ),
    _scenario(
        "mech.patch_sensitive_claim",
        "titan_veteran",
        "Is Sunshot the best DPS option right now?",
        MechanicsSpec(
            "patch_sensitive",
            ("live or web grounding for current rankings",),
            "Stating a current ranking from static data.",
        ),
        "build_critique",
        style="build",
        live_sensitive=True,
    ),
    _scenario(
        "mech.cross_class_hypothetical",
        "titan_veteran",
        "What if I used Verity's Brow with my build?",
        MechanicsSpec(
            "hypothetical_configuration",
            ("manifest: class restriction (Warlock)", "guardian: ownership"),
            "Recommending a cross-class Exotic as equippable, or as owned.",
        ),
        "owned_gear",
        style="build",
        character="Titan",
        not_owned=("Verity's Brow",),
        forbidden=(r"\byou can (?:equip|use|wear) verity",),
        tools_any=KNOWLEDGE + GUARDIAN,
    ),
)

MECHANICS_SCENARIOS: tuple[Scenario, ...] = tuple(value[0] for value in _ENTRIES)
MECHANICS_SPECS: dict[str, MechanicsSpec] = {value[0].id: value[1] for value in _ENTRIES}


def mechanics_baseline(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the trace-only mechanics-claim summaries across run records.

    Counts claims by status and reports how many turns carried no trace. Nothing here is a
    pass/fail judgment: statuses describe retrieval, never correctness.
    """

    by_status: dict[str, int] = {}
    claims = attention = turns = traced = 0
    per_scenario: dict[str, int] = {}
    for scenario in records:
        for turn in scenario.get("turns", []):
            turns += 1
            summary = turn.get("mechanics_claims")
            if not isinstance(summary, dict) or "claims_total" not in summary:
                continue
            traced += 1
            claims += int(summary["claims_total"])
            attention += int(summary["needs_attention"])
            per_scenario[scenario["id"]] = per_scenario.get(scenario["id"], 0) + int(
                summary["needs_attention"]
            )
            for status, count in (summary.get("by_status") or {}).items():
                by_status[status] = by_status.get(status, 0) + int(count)
    return {
        "mode": "trace_only",
        "turns": turns,
        "turns_traced": traced,
        "claims_total": claims,
        "needs_attention": attention,
        "by_status": dict(sorted(by_status.items())),
        "needs_attention_by_scenario": {k: v for k, v in sorted(per_scenario.items()) if v},
    }
