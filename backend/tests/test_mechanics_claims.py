"""M0: trace-only mechanics-claim detection.

Tool results below mirror the shapes the real tools return. The text values are verbatim
Manifest strings copied from the local Manifest during the P1-3A audit.
"""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from app import ai
from app.ai import RecommendationService
from app.config import Settings
from app.guardian_tools import GuardianToolService
from app.mechanics_claims import (
    KNOWN_LIMITATIONS,
    UNSUPPORTED_STATUSES,
    MechanicsEvidenceLedger,
    detect_mechanics_claims,
)
from app.models import ChatRequest
from evals.profiles import PROFILES
from tests.test_build_analysis import ANALYZE_CALL, ScriptedBuildResponses, _answer, build_context

SUNSHOT = {
    "found": True,
    "item": {
        "name": "Sunshot",
        "description": None,
        "item_type": "Weapon",
        "subtype": "Hand Cannon",
        "class_restriction": {"code": 3, "name": "Any class"},
        "intrinsic_perks": [{"name": None, "description": ""}],
        "default_socket_plugs": [
            {
                "name": "Sunburn",
                "description": "This weapon fires explosive rounds and highlights targets that "
                "take damage from Sunshot.",
            },
            {
                "name": "Sun Blast",
                "description": "Targets defeated with Sunshot explode in Solar energy, "
                "scorching other nearby targets.",
            },
            {"name": "Chambered Compensator", "description": "Stable barrel attachment."},
        ],
    },
}
SYNTHOCEPS = {
    "found": True,
    "item": {
        "name": "Synthoceps",
        "description": None,
        "item_type": "Armor",
        "subtype": "Gauntlets",
        "class_restriction": {"code": 0, "name": "Titan"},
        "default_socket_plugs": [
            {
                "name": "Biotic Enhancements",
                "description": "When you're surrounded, gain increased melee and Super damage, "
                "as well as improved weapon handling and reload speed.",
            }
        ],
    },
}
CELESTIAL = {
    "found": True,
    "item": {
        "name": "Celestial Nighthawk",
        "item_type": "Armor",
        "subtype": "Helmet",
        "class_restriction": {"code": 1, "name": "Hunter"},
        "default_socket_plugs": [
            {
                "name": "Hawkeye Hack",
                "description": "Precision final blows reduce the cooldown of your Super. "
                "Modifies Golden Gun to fire a single, high-damage shot.",
            }
        ],
    },
}
KICKSTART = {
    "found": True,
    "item": {
        "name": "Grenade Kickstart",
        "item_type": "Mod",
        "subtype": "Arms Armor Mod",
        "intrinsic_perks": [
            {
                "name": "Grenade Kickstart",
                "description": "When your grenade energy is fully expended, your Armor Charge "
                "is consumed and you gain grenade energy for each Armor Charge used.",
            }
        ],
    },
}
ARC_SCOURGE = {
    "found": True,
    "item": {
        "name": "Arc Scourge",
        "item_type": "Bounty",
        "subtype": "Duality Order",
        "description": "Defeat combatants or Guardians with Arc ability final blows. Combatants "
        "affected by Arc debuffs grant greater progress.",
    },
}
NOT_FOUND = {"found": False, "query": "Shatterdive Protocol", "entity_type": "item"}


def ledger_with(*results: dict[str, Any]) -> MechanicsEvidenceLedger:
    ledger = MechanicsEvidenceLedger()
    for result in results:
        ledger.record("get_item_details", "manifest", result)
    return ledger


def titan_loadout_ledger() -> MechanicsEvidenceLedger:
    service = GuardianToolService(PROFILES["titan_veteran"]())
    ledger = MechanicsEvidenceLedger()
    ledger.record(
        "get_equipped_loadout",
        "guardian",
        service.execute("get_equipped_loadout", {"character_id": None, "character_class": "Titan"}),
    )
    return ledger


def statuses(answer: str, ledger: MechanicsEvidenceLedger, **kwargs: Any) -> dict[str, set[str]]:
    report = detect_mechanics_claims(answer, ledger, **kwargs)
    found: dict[str, set[str]] = {}
    for claim in report.claims:
        found.setdefault(claim.entity, set()).add(claim.status)
    return found


# --- Supported mechanics are not reported as unsupported ------------------------------------


@pytest.mark.parametrize(
    ("answer", "entity"),
    [
        (
            "Sunshot fires explosive rounds that highlight targets and scorch nearby enemies.",
            "Sunshot",
        ),
        (
            "Targets defeated with Sunshot explode in Solar energy and scorch nearby targets.",
            "Sunshot",
        ),
        (
            "Synthoceps boosts your melee and Super damage when you're surrounded.",
            "Synthoceps",
        ),
        (
            "Grenade Kickstart consumes your Armor Charge and gives grenade energy when grenade "
            "energy is expended.",
            "Grenade Kickstart",
        ),
        ("Hawkeye Hack reduces your Super cooldown on precision final blows.", "Hawkeye Hack"),
    ],
)
def test_effects_matching_retrieved_text_are_supported(answer: str, entity: str) -> None:
    ledger = ledger_with(SUNSHOT, SYNTHOCEPS, KICKSTART, CELESTIAL)

    found = statuses(answer, ledger)

    assert found[entity] == {"supported"}
    report = detect_mechanics_claims(answer, ledger)
    assert report.needs_attention == 0


def test_an_item_inherits_the_effect_text_of_the_plugs_it_carries() -> None:
    ledger = ledger_with(SUNSHOT)

    assert ledger.entity("Sunshot") is not None
    assert ledger.entity("Sunshot").has_text is True  # type: ignore[union-attr]


# --- Missing retrieval is classified separately from contradiction -----------------------------


def test_names_only_evidence_is_mentioned_without_evidence_not_a_contradiction() -> None:
    ledger = titan_loadout_ledger()
    assert ledger.entity("Rampage") is not None
    assert ledger.entity("Rampage").has_text is False  # type: ignore[union-attr]

    found = statuses("Rampage increases your damage after each kill.", ledger)

    assert found["Rampage"] == {"mentioned_without_evidence"}


def test_a_missing_retrieval_never_reads_as_the_mechanic_being_absent() -> None:
    report = detect_mechanics_claims(
        "Rampage increases your damage after each kill.", titan_loadout_ledger()
    )

    claim = report.claims[0]
    assert claim.status == "mentioned_without_evidence"
    assert claim.suspected_missing == ("effect_text",)
    assert report.to_trace()["basis"] == "retrieved_evidence_only"
    assert any("never proof" in text for text in KNOWN_LIMITATIONS)


def test_retrieved_text_with_no_shared_content_is_labelled_uncertain_not_contradicted() -> None:
    ledger = ledger_with(KICKSTART)

    found = statuses("Grenade Kickstart makes your sword hit harder against bosses.", ledger)

    assert found["Grenade Kickstart"] == {"evidence_retrieved_no_overlap"}
    assert "evidence_retrieved_no_overlap" not in UNSUPPORTED_STATUSES


def test_structured_slot_and_class_disagreement_is_a_contradiction() -> None:
    ledger = ledger_with(SYNTHOCEPS, CELESTIAL)

    slot = statuses("Synthoceps are a Titan helmet that boosts melee damage.", ledger)
    klass = statuses("Celestial Nighthawk is a Warlock helmet.", ledger)

    assert "contradicted_by_evidence" in slot["Synthoceps"]
    assert "contradicted_by_evidence" in klass["Celestial Nighthawk"]


def test_matching_slot_and_class_are_supported() -> None:
    ledger = ledger_with(SYNTHOCEPS, CELESTIAL)

    assert statuses("Synthoceps are Titan gauntlets.", ledger)["Synthoceps"] == {"supported"}
    assert statuses("Celestial Nighthawk is a Hunter helmet.", ledger)["Celestial Nighthawk"] == {
        "supported"
    }


def test_slot_without_retrieved_slot_data_is_a_gap_not_a_contradiction() -> None:
    ledger = titan_loadout_ledger()  # names-only for armor slot? loadout supplies bucket
    ledger.entities.pop("synthoceps", None)
    ledger.record("get_item_details", "manifest", {"found": True, "item": {"name": "Synthoceps"}})

    found = statuses("Synthoceps are Titan gauntlets.", ledger)

    assert found["Synthoceps"] == {"mentioned_without_evidence"}


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("Arc weapon kills advance Arc Scourge quickly.", "contradicted_by_evidence"),
        ("Arc grenade and melee final blows advance Arc Scourge.", "supported"),
    ],
)
def test_ability_versus_weapon_damage_source_is_checked_against_objective_text(
    answer: str, expected: str
) -> None:
    ledger = ledger_with(ARC_SCOURGE)

    assert expected in statuses(answer, ledger)["Arc Scourge"]


# --- Hypothetical changes versus equipped configurations ----------------------------------------


def test_a_proposed_socket_change_has_no_legality_evidence_by_default() -> None:
    ledger = ledger_with(SUNSHOT)

    report = detect_mechanics_claims("Swap Sunshot's barrel for an Extended Barrel.", ledger)

    claim = next(value for value in report.claims if value.category == "configuration")
    assert claim.status == "config_change_without_legality"
    assert claim.suspected_missing == ("socket_options",)
    assert report.needs_attention == 1


def test_legality_evidence_changes_the_classification() -> None:
    ledger = ledger_with(SUNSHOT)
    ledger.add_legality("Sunshot")

    report = detect_mechanics_claims("Swap Sunshot's barrel for an Extended Barrel.", ledger)

    claim = next(value for value in report.claims if value.category == "configuration")
    assert claim.status == "config_change_with_legality"
    assert claim.suspected_missing == ()
    assert report.needs_attention == 0


def test_an_equipped_perk_stated_as_kept_is_an_equipped_configuration() -> None:
    ledger = titan_loadout_ledger()
    assert "rampage" in ledger.equipped_plugs

    report = detect_mechanics_claims("Keep Rampage, which is already on your Rose.", ledger)

    assert [value.status for value in report.claims] == ["equipped_configuration"]
    assert report.needs_attention == 0


def test_proposing_an_equipped_perk_is_still_a_proposal() -> None:
    report = detect_mechanics_claims(
        "Swap your perk to Rampage on the second trait column.", titan_loadout_ledger()
    )

    assert {value.status for value in report.claims} == {"config_change_without_legality"}


def test_equipped_claim_backed_only_by_knowledge_is_flagged() -> None:
    ledger = ledger_with(SUNSHOT)

    report = detect_mechanics_claims("You are currently running Sun Blast as your trait.", ledger)

    assert [value.status for value in report.claims] == ["equipped_claim_without_evidence"]


# --- Unknown entities stay unknown ---------------------------------------------------------------


def test_an_entity_a_retrieval_could_not_find_is_unresolved() -> None:
    ledger = MechanicsEvidenceLedger()
    ledger.record("get_item_details", "manifest", NOT_FOUND)
    ledger.entities["shatterdive protocol"] = ledger.entities.get(  # type: ignore[assignment]
        "shatterdive protocol"
    ) or __import__("app.mechanics_claims", fromlist=["EntityEvidence"]).EntityEvidence(
        name="Shatterdive Protocol"
    )

    found = statuses("Shatterdive Protocol grants extra damage to your grenades.", ledger)

    assert found["Shatterdive Protocol"] == {"unresolved_entity"}


def test_an_unretrieved_actor_without_a_resolver_is_not_called_unresolved() -> None:
    found = statuses(
        "Mystic Reflex grants extra damage to your grenades.", MechanicsEvidenceLedger()
    )

    assert found["Mystic Reflex"] == {"mentioned_without_evidence"}


def test_resolver_decides_between_unknown_and_unretrieved() -> None:
    answer = "Mystic Reflex grants extra damage to your grenades."

    unknown = statuses(answer, MechanicsEvidenceLedger(), entity_resolver=lambda name: False)
    known = statuses(answer, MechanicsEvidenceLedger(), entity_resolver=lambda name: True)

    assert unknown["Mystic Reflex"] == {"unresolved_entity"}
    assert known["Mystic Reflex"] == {"mentioned_without_evidence"}


def test_hedged_sentences_are_not_claims() -> None:
    report = detect_mechanics_claims(
        "I can't confirm what Mystic Reflex grants to your grenades.", MechanicsEvidenceLedger()
    )

    assert report.claims == ()
    assert report.hedged_sentences == 1


# --- Trace shape and independence from the answer ------------------------------------------------


def test_trace_is_bounded_and_carries_entities_categories_sources_and_missing_evidence() -> None:
    ledger = ledger_with(SUNSHOT, SYNTHOCEPS)
    trace = detect_mechanics_claims(
        "Sunshot makes defeated targets explode and scorch nearby enemies. "
        "Synthoceps are Titan gauntlets. "
        "Swap Sunshot's barrel for an Extended Barrel.",
        ledger,
    ).to_trace()

    assert trace["mode"] == "trace_only"
    assert trace["claims_total"] == len(trace["claims"]) == 3
    claim = next(value for value in trace["claims"] if value["category"] == "configuration")
    assert set(claim) == {
        "category",
        "entity",
        "status",
        "evidence_sources",
        "suspected_missing",
        "sentence",
    }
    assert claim["evidence_sources"] == ["manifest:get_item_details"]
    assert trace["retrieval"]["tools"] == ["get_item_details"]
    assert "Swap Sunshot" not in str(trace)  # no answer text in the trace


def test_detection_does_not_modify_its_inputs() -> None:
    ledger = ledger_with(SUNSHOT)
    before = {key: list(value.texts) for key, value in ledger.entities.items()}
    answer = "Sunshot makes defeated targets explode and scorch nearby enemies."

    detect_mechanics_claims(answer, ledger)

    assert answer == "Sunshot makes defeated targets explode and scorch nearby enemies."
    assert {key: list(value.texts) for key, value in ledger.entities.items()} == before


def test_recording_a_result_does_not_mutate_it() -> None:
    result = {"found": True, "item": dict(SUNSHOT["item"])}
    snapshot = repr(result)
    MechanicsEvidenceLedger().record("get_item_details", "manifest", result)

    assert repr(result) == snapshot


async def _service_answer(*, break_detector: bool) -> tuple[str, dict[str, Any]]:
    responses = ScriptedBuildResponses(
        [
            [ANALYZE_CALL],
            [_answer("Rampage increases damage after kills, so keep Sunshot as your Exotic.")],
        ]
    )
    service = RecommendationService(Settings(openai_api_key="test-key", enable_debug_tools=True))
    service.client = SimpleNamespace(responses=responses)
    original = ai.detect_mechanics_claims
    if break_detector:

        def explode(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("detector bug")

        ai.detect_mechanics_claims = explode  # type: ignore[assignment]
    try:
        reply = await service.chat(ChatRequest(message="Is my Titan build good?"), build_context())
    finally:
        ai.detect_mechanics_claims = original  # type: ignore[assignment]
    return reply.message, service.latest_trace() or {}


def test_service_records_the_trace_without_changing_the_answer() -> None:
    normal_answer, normal_trace = asyncio.run(_service_answer(break_detector=False))
    broken_answer, broken_trace = asyncio.run(_service_answer(break_detector=True))

    assert normal_answer == broken_answer
    claims = normal_trace["mechanics_claims"]
    assert claims["mode"] == "trace_only"
    assert "analyze_current_build" in claims["retrieval"]["tools"]
    assert broken_trace["mechanics_claims"] == {"mode": "trace_only", "error": "detector_failed"}


def test_service_trace_defaults_to_none_when_no_answer_is_generated() -> None:
    service = RecommendationService(Settings(enable_debug_tools=True))
    service.client = None
    asyncio.run(service.chat(ChatRequest(message="hello"), build_context()))

    assert (service.latest_trace() or {})["mechanics_claims"] is None


# --- Detector false positives and misses, measured on scripted examples -----------------------

# (answer, statuses that need attention). Every item is something a user could see.
BENIGN_ANSWERS = [
    "Finish the Edge of Fate mission first; it advances your active campaign quest.",
    "Run Vanguard Ops for the Weekly Vanguard milestone, then try a Nightfall.",
    "Sunshot is a good fit here and you own it.",
    "Your Rose has Rampage and Opening Shot equipped.",
    "I'd keep your current Exotic and add a second weapon for add clear.",
    "With 20 minutes available, pick one short activity.",
    "That depends on your goal: boss damage or add clear?",
    "Crucible is a PvP mode, so skip it tonight.",
    "I can't confirm what Mystic Reflex does with your grenades.",
    "You could try the Duality dungeon if you have an hour.",
    "Your Titan has good recovery and decent resilience.",
    "Use your Rose for the first encounter and switch afterwards.",
    "Equip Synthoceps for the melee-focused plan.",
    "Pick Sunshot as the weapon to build around.",
    "The Weekly Vanguard milestone is 33% complete.",
    "Haunted Altars of Sorrow is on the Moon.",
    "You have 2 hours, so a dungeon fits.",
    "Start with the quest, then spend the rest of your time on bounties.",
    "Your armor stats show 12 Resilience and 8 Recovery.",
    "Keep Rampage on Rose; it is already equipped.",
]
# Sentences the detector is expected to label as needing attention, with the expected status.
FLAGGED_ANSWERS = [
    ("Rampage increases your damage after each kill.", "mentioned_without_evidence"),
    ("Swap Sunshot's barrel for an Extended Barrel.", "config_change_without_legality"),
    ("Reroll the magazine perk on Rose for Alloy Magazine.", "config_change_without_legality"),
    ("Synthoceps are a Hunter helmet.", "contradicted_by_evidence"),
    ("Arc weapon kills advance Arc Scourge quickly.", "contradicted_by_evidence"),
    ("Mystic Reflex grants extra damage to your grenades.", "mentioned_without_evidence"),
    ("Put Alloy Magazine on it for better reloads.", "config_change_without_legality"),
]
# Known blind spots, asserted so the limitation list stays honest.
KNOWN_MISSES = [
    "Sunshot's rounds burst into flame when targets fall.",
    "This pairing lets your grenades chain between enemies.",
    "Alloy Magazine is worth rolling for better reloads.",
]


def corpus_ledger() -> MechanicsEvidenceLedger:
    ledger = titan_loadout_ledger()
    for result in (SUNSHOT, SYNTHOCEPS, KICKSTART, ARC_SCOURGE):
        ledger.record("get_item_details", "manifest", result)
    return ledger


def test_benign_answers_produce_no_attention_claims() -> None:
    ledger = corpus_ledger()
    false_positives = [
        text for text in BENIGN_ANSWERS if detect_mechanics_claims(text, ledger).needs_attention > 0
    ]

    assert false_positives == []


def test_flagged_answers_are_detected_with_the_expected_status() -> None:
    ledger = corpus_ledger()
    misses = []
    for text, expected in FLAGGED_ANSWERS:
        report = detect_mechanics_claims(text, ledger)
        if expected not in {value.status for value in report.claims}:
            misses.append((text, expected, report.counts()))

    assert misses == []


def test_documented_blind_spots_remain_blind_spots() -> None:
    ledger = corpus_ledger()
    for text in KNOWN_MISSES:
        assert detect_mechanics_claims(text, ledger).needs_attention == 0, text


def test_measured_false_positive_rate_on_the_scripted_corpus() -> None:
    ledger = corpus_ledger()
    flagged_benign = sum(
        detect_mechanics_claims(text, ledger).needs_attention > 0 for text in BENIGN_ANSWERS
    )
    detected = sum(
        any(value.status == expected for value in detect_mechanics_claims(text, ledger).claims)
        for text, expected in FLAGGED_ANSWERS
    )

    assert flagged_benign / len(BENIGN_ANSWERS) == 0.0
    assert detected / len(FLAGGED_ANSWERS) == 1.0
