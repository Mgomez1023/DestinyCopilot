"""Possession and equipment evidence for build answers: what counts as proof, what does not,
how removable unsupported wording is repaired, and that genuine failures still fail closed."""

import asyncio
import json
from types import SimpleNamespace
from typing import Any, get_args

import pytest

from app.ai import RecommendationService
from app.config import Settings
from app.guardian_tools import GuardianToolService
from app.models import ChatRequest
from app.ownership_evidence import (
    ClaimType,
    OwnershipEvidence,
    evidence_summary,
    find_ownership_claims,
    record_ownership_evidence,
    repair_unsupported_ownership,
)
from app.response_quality import (
    BuildResponseValidationContext,
    ReadOnlyActionValidationContext,
    ResponseQualityValidator,
    _has_current_claim,
    build_fallback_kind,
    response_mode_context,
)
from evals.profiles import PROFILES
from tests.test_build_analysis import ScriptedBuildResponses, _answer, _call
from tests.test_current_character import ANALYZE, NULLS

MESSAGE = "What's the weakest part of my current build?"
VALIDATOR = ResponseQualityValidator()
SEARCH = {
    **NULLS,
    "query": None,
    "item_type": None,
    "subtype": None,
    "bucket": None,
    "equipped_only": False,
    "limit": 100,
}


def tool_result(profile: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return GuardianToolService(PROFILES[profile]()).execute(name, arguments)


def context_with(
    profile: str, *calls: tuple[str, dict[str, Any]]
) -> BuildResponseValidationContext:
    context = BuildResponseValidationContext(
        is_build_request=True, requires_current_build_analysis=True
    )
    for name, arguments in calls:
        RecommendationService._update_build_validation_context(  # noqa: SLF001
            context,
            {"build_analysis": {"owned_candidates_returned": 0}},
            name,
            tool_result(profile, name, arguments),
        )
    return context


def codes(answer: str, context: BuildResponseValidationContext, message: str = MESSAGE) -> set[str]:
    mode = response_mode_context(
        message, session_planning=False, continuation=False, build_request=True, focused_build=False
    )
    violations = VALIDATOR.validate(
        answer, mode, message, None, context, ReadOnlyActionValidationContext()
    )
    return {value.code for value in violations}


# --- What counts as possession / equipment evidence ----------------------------------------


def test_current_build_analysis_proves_possession_and_equipment() -> None:
    context = context_with("titan_veteran", ("analyze_current_build", ANALYZE))

    for answer in (
        "You have Synthoceps equipped.",
        "Synthoceps is equipped on your Titan.",
        "You have Synthoceps.",
        "Your Rose is equipped, and you have Calus Mini-Tool too.",
    ):
        claims = find_ownership_claims(answer, context.ownership)
        assert claims and all(value.supported for value in claims), answer
        assert "unsupported_ownership_claim" not in codes(answer, context), answer
    claim = find_ownership_claims("You have Synthoceps equipped.", context.ownership)[0]
    assert (claim.item, claim.claim_type, claim.source) == (
        "Synthoceps",
        "equipped",
        "current_build_analysis",
    )


def test_equipped_loadout_alone_is_sufficient_evidence() -> None:
    context = context_with("hunter_veteran", ("get_equipped_loadout", NULLS))

    answer = "Your Sunshot is equipped and you own Star-Eater Scales."
    claims = find_ownership_claims(answer, context.ownership)

    assert [value.supported for value in claims] == [True, True]
    assert {value.source for value in claims} == {"equipped_loadout"}
    assert not context.inventory_ownership_checked
    assert "unsupported_ownership_claim" not in codes(answer, context)


def test_inventory_search_still_proves_possession_but_not_equipment() -> None:
    context = context_with(
        "titan_veteran",
        ("search_inventory", {**SEARCH, "query": "Heart"}),
        ("find_build_alternatives", _find_arguments()),
    )

    owned = find_ownership_claims("You own Heart of Inmost Light.", context.ownership)
    candidate = find_ownership_claims("You have Fatebringer.", context.ownership)
    equipped = find_ownership_claims("Heart of Inmost Light is equipped.", context.ownership)

    assert owned[0].supported and owned[0].source == "inventory_search"
    assert candidate[0].supported and candidate[0].source == "inventory_search"
    assert not equipped[0].supported and equipped[0].claim_type == "equipped"


def _find_arguments() -> dict[str, Any]:
    return {
        **NULLS,
        "slot": None,
        "item_type": "Weapon",
        "subtype": "Hand Cannon",
        "damage_type": None,
        "rarity": None,
        "required_perks": None,
        "exotic": None,
        "equipped": None,
        "locations": None,
        "preserve_exotics": True,
        "locked_items": None,
        "limit": 3,
    }


def test_an_item_with_no_evidence_is_not_supported() -> None:
    context = context_with("titan_veteran", ("analyze_current_build", ANALYZE))

    for answer in (
        "You own Gjallarhorn.",
        "You already have Witherhoard in your vault.",
        "Heart of Inmost Light is equipped.",
    ):
        claims = find_ownership_claims(answer, context.ownership)
        assert claims and not any(value.supported for value in claims), answer
        assert "unsupported_ownership_claim" in codes(answer, context), answer


def test_a_wrong_class_vault_item_is_owned_but_never_equipped() -> None:
    context = context_with("wrong_class_vault", ("search_inventory", SEARCH))

    owned = find_ownership_claims("You have Celestial Nighthawk.", context.ownership)
    equipped = find_ownership_claims("Celestial Nighthawk is equipped.", context.ownership)

    assert owned[0].supported
    assert not equipped[0].supported
    assert context.ownership.source_for("Celestial Nighthawk", "equipped") is None


def test_wording_that_names_no_item_is_not_an_ownership_claim() -> None:
    context = BuildResponseValidationContext(is_build_request=True)

    for answer in (
        "You have high Grenade (100) but low Health (80).",
        "You have two weapons that overlap in role.",
        "You have Vanguard Ops available this week.",
        "Your Titan has a strong Solar kit.",
    ):
        assert find_ownership_claims(answer, context.ownership) == [], answer
        assert "unsupported_ownership_claim" not in codes(answer, context), answer


def test_the_old_false_positive_no_longer_blocks_a_grounded_critique() -> None:
    context = context_with("titan_veteran", ("analyze_current_build", ANALYZE))
    answer = (
        "Your weakest part is survivability: you have 80 Health against 100 Grenade, with "
        "Synthoceps equipped. Move one Recuperation mod toward Health."
    )

    assert not context.inventory_ownership_checked
    assert "unsupported_ownership_claim" not in codes(answer, context)


# --- Claim types stay separate ---------------------------------------------------------------


def test_possession_evidence_cannot_support_effect_or_meta_claims() -> None:
    context = context_with("titan_veteran", ("analyze_current_build", ANALYZE))

    assert set(get_args(ClaimType)) == {"possession", "equipped"}
    assert context.ownership.source_for("Synthoceps", "equipped") is not None
    answer = "You have Synthoceps equipped, and it is the best DPS in the current meta."
    found = codes(answer, context)
    assert "unsupported_ownership_claim" not in found
    assert "ungrounded_current_build_claim" in found
    # Effects are not ownership claims, and ownership evidence never vouches for them.
    assert find_ownership_claims("Synthoceps boosts your survivability.", context.ownership) == []
    summary = evidence_summary(context.ownership)
    assert "only possession and what is equipped" in summary


@pytest.mark.parametrize(
    "answer",
    [
        "Your weakest stat is Health.",
        "This slot contributes the least to the build.",
        "This weapon has poor synergy with the rest of the build.",
        "Your best DPS is the rocket you have equipped.",
        "Your strongest option right now is Rose.",
    ],
)
def test_ordinary_build_analysis_is_not_a_current_claim(answer: str) -> None:
    assert _has_current_claim(answer) is False


@pytest.mark.parametrize(
    "answer",
    [
        "Gjallarhorn is the best weapon in the current meta.",
        "Anything is the highest DPS this season.",
        "It is the top PvE option right now.",
        "The best DPS option is a rocket launcher.",
        "That was nerfed recently.",
    ],
)
def test_current_game_state_claims_still_need_live_grounding(answer: str) -> None:
    context = context_with("titan_veteran", ("analyze_current_build", ANALYZE))

    assert _has_current_claim(answer) is True
    assert "ungrounded_current_build_claim" in codes(answer, context)
    context.current_external_grounding = True
    assert "ungrounded_current_build_claim" not in codes(answer, context)


# --- Repair ------------------------------------------------------------------------------------


def test_repair_removes_only_the_unsupported_aside() -> None:
    context = context_with("titan_veteran", ("analyze_current_build", ANALYZE))
    answer = (
        "Your weakest stat is Health at 80 against 100 Grenade.\n"
        "You already have Witherhoard in your vault.\n"
        "Shift one Recuperation mod toward Health to even out survivability."
    )

    repaired, removed = VALIDATOR.repair_ownership(answer, context)

    assert [value.item for value in removed] == ["Witherhoard"]
    assert repaired == (
        "Your weakest stat is Health at 80 against 100 Grenade.\n"
        "Shift one Recuperation mod toward Health to even out survivability."
    )


def test_repair_removes_a_bullet_and_its_marker() -> None:
    context = context_with("titan_veteran", ("analyze_current_build", ANALYZE))
    answer = (
        "Your weakest stat is Health at 80 against 100 Grenade, so survivability suffers.\n"
        "- You have Witherhoard.\n"
        "- Shift one Recuperation mod toward Health to even out the stat spread."
    )

    repaired, removed = VALIDATOR.repair_ownership(answer, context)

    assert len(removed) == 1
    assert "Witherhoard" not in repaired and repaired.count("\n") == 1
    assert repaired.splitlines()[1].startswith("- Shift")


def test_a_recommendation_built_on_an_unsupported_claim_is_not_repairable() -> None:
    evidence = OwnershipEvidence()
    evidence.add("Synthoceps", "current_build_analysis", equipped=True)
    answer = (
        "Your weakest stat is Health at 80 against 100 Grenade. "
        "Swap in Witherhoard, which you have in your vault, for more damage."
    )

    repaired, removed = repair_unsupported_ownership(
        answer, find_ownership_claims(answer, evidence)
    )

    assert (repaired, removed) == (answer, [])


def test_repair_refuses_to_gut_the_answer() -> None:
    evidence = OwnershipEvidence()
    answer = "You already have Witherhoard. You also have Gjallarhorn. Short one."

    repaired, removed = repair_unsupported_ownership(
        answer, find_ownership_claims(answer, evidence)
    )

    assert (repaired, removed) == (answer, [])


# --- End to end: no extra rounds, correction without tools ------------------------------------


def chat(outputs: list[list[Any]]) -> tuple[Any, ScriptedBuildResponses, dict[str, Any]]:
    responses = ScriptedBuildResponses(outputs)
    service = RecommendationService(Settings(openai_api_key="test-key", enable_debug_tools=True))
    service.client = SimpleNamespace(responses=responses)
    answer = asyncio.run(service.chat(ChatRequest(message=MESSAGE), PROFILES["titan_veteran"]()))
    return answer, responses, service.latest_trace() or {}


GROUNDED_WITH_ASIDE = (
    "Your weakest stat is Health at 80 against 100 Grenade, so survivability suffers.\n"
    "You already have Witherhoard in your vault.\n"
    "Shift one Recuperation mod toward Health to even out the stat spread."
)
CLEAN = (
    "Your weakest stat is Health at 80 against 100 Grenade, so survivability suffers.\n"
    "Shift one Recuperation mod toward Health to even out the stat spread."
)


def test_one_removable_unsupported_phrase_does_not_trigger_a_fallback_or_extra_round() -> None:
    answer, responses, trace = chat(
        [[_call("analyze_current_build", ANALYZE, "analysis-1")], [_answer(GROUNDED_WITH_ASIDE)]]
    )

    assert answer.message == CLEAN
    assert len(responses.requests) == 2
    assert trace["planning_correction"]["attempted"] is False
    assert "fallback" not in trace["planning_correction"]
    validation = trace["ownership_validation"]
    assert validation["fallback_due_to_ownership"] is False
    (only_pass,) = validation["passes"]
    assert only_pass["stage"] == "initial" and only_pass["repaired"] is True
    assert only_pass["removed_claims"] == 1 and only_pass["remaining_violation_codes"] == []
    assert only_pass["claims"] == [
        {
            "item": "Witherhoard",
            "claim_type": "possession",
            "supported": False,
            "evidence": None,
            "action": "removed",
        }
    ]


def test_trace_reports_supported_claims_with_their_evidence_source() -> None:
    supported = (
        "You have Synthoceps equipped, but your weakest stat is Health at 80 against 100 "
        "Grenade. Shift one Recuperation mod toward Health to even out the stat spread."
    )
    answer, _, trace = chat(
        [[_call("analyze_current_build", ANALYZE, "analysis-1")], [_answer(supported)]]
    )

    assert answer.message == supported
    (only_pass,) = trace["ownership_validation"]["passes"]
    assert only_pass["claims"] == [
        {
            "item": "Synthoceps",
            "claim_type": "equipped",
            "supported": True,
            "evidence": "current_build_analysis",
            "action": "kept",
        }
    ]
    encoded = json.dumps(trace["ownership_validation"])
    assert "character_id" not in encoded and "inst-" not in encoded


UNREPAIRABLE = (
    "Your weakest stat is Health at 80 against 100 Grenade. "
    "Swap in Witherhoard, which you have in your vault, for more damage."
)


def test_a_genuinely_unsupported_recommendation_still_falls_back() -> None:
    answer, responses, trace = chat(
        [
            [_call("analyze_current_build", ANALYZE, "analysis-1")],
            [_answer(UNREPAIRABLE)],
            [_answer(UNREPAIRABLE)],
        ]
    )

    assert answer.message.startswith("I'd keep your current build as it is (")
    assert "you own a piece my draft relied on" in answer.message
    assert "Witherhoard" not in answer.message
    assert trace["planning_correction"]["fallback"] == "build_grounding"
    assert (
        "unsupported_ownership_claim" in trace["planning_correction"]["remaining_violation_codes"]
    )
    assert trace["ownership_validation"]["fallback_due_to_ownership"] is True
    assert [value["stage"] for value in trace["ownership_validation"]["passes"]] == [
        "initial",
        "after_correction",
    ]
    assert len(responses.requests) == 3


def test_the_correction_pass_calls_no_tools_and_carries_the_evidence() -> None:
    answer, responses, trace = chat(
        [
            [_call("analyze_current_build", ANALYZE, "analysis-1")],
            [_answer(UNREPAIRABLE)],
            [_answer(CLEAN)],
        ]
    )

    correction = responses.requests[2]
    developer = [
        value["content"]
        for value in correction["input"]
        if isinstance(value, dict) and value.get("role") == "developer"
    ][-1]
    assert answer.message == CLEAN
    assert trace["planning_correction"]["succeeded"] is True
    assert correction["tool_choice"] == "none"
    assert len(responses.requests) == 3  # No extra round beyond the one correction pass.
    assert trace["tool_budget"]["tool_calls_used"] == 1
    assert "Verified Guardian possession evidence" in developer
    assert "equipped:" in developer and "Synthoceps" in developer
    assert "Witherhoard" in developer  # Named as an unsupported claim to delete.
    assert "only possession and what is equipped" in developer


def test_correction_instruction_is_unchanged_without_a_build_context() -> None:
    violations = [
        SimpleNamespace(code="x", correction="Fix it."),
    ]

    plain = VALIDATOR.correction_instruction(violations)  # type: ignore[arg-type]
    non_build = VALIDATOR.correction_instruction(  # type: ignore[arg-type]
        violations, BuildResponseValidationContext()
    )

    assert plain == non_build and "Verified Guardian" not in plain


# --- Fail-closed behavior is intact ---------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "code"),
    [
        ("Replace Synthoceps with something stronger for this build.", "locked_item_replaced"),
        ("Build score: 87/100 for this setup.", "fake_build_score"),
        ("I equipped your Rose for you.", "implied_bungie_write"),
        ("Your character id is 2305843009213693999.", "raw_build_identifier"),
        ("Please sign in again to link your Bungie account.", "false_authentication_guidance"),
    ],
)
def test_other_grounding_and_safety_checks_still_fire(answer: str, code: str) -> None:
    context = context_with("titan_veteran", ("analyze_current_build", ANALYZE))
    context.locked_item_names = ["Synthoceps"]
    mode = response_mode_context(
        MESSAGE, session_planning=False, continuation=False, build_request=True, focused_build=False
    )

    violations = VALIDATOR.validate(
        answer, mode, MESSAGE, None, context, ReadOnlyActionValidationContext()
    )

    assert code in {value.code for value in violations}
    assert build_fallback_kind(violations) in {"evidence", "grounding_safety"}


def test_missing_current_build_evidence_still_fails_closed() -> None:
    context = BuildResponseValidationContext(
        is_build_request=True, requires_current_build_analysis=True
    )

    assert "missing_current_build_analysis" in codes(
        "Swap your rocket for something better.", context
    )


def test_ordinary_hyphenated_compounds_are_not_item_names() -> None:
    evidence = OwnershipEvidence()
    evidence.add("Synthoceps", "current_build_analysis", equipped=True)

    for answer in (
        "You have a Super-focused kit with Synthoceps equipped.",
        "You have Grenade-heavy stats.",
    ):
        claims = find_ownership_claims(answer, evidence)
        assert all(value.supported for value in claims), answer
    # Real item names (capitalized parts or digits) are still treated as items.
    for name in ("Wish-Ender", "Taipan-4fr", "Calus Mini-Tool"):
        claims = find_ownership_claims(f"You have {name}.", evidence)
        assert [(value.item, value.supported) for value in claims] == [(name, False)]


@pytest.mark.parametrize(
    "answer",
    [
        "Current equipped evidence (owned & equipped): Synthoceps; Rose.",
        "You can analyze armor you own and suggest exact mods (I’ll need a focused check).",
        "You have Synthoceps equipped. Let’s fix the stat spread; it’s the weakest part.",
    ],
)
def test_headers_contractions_and_trailing_clauses_are_not_item_claims(answer: str) -> None:
    evidence = OwnershipEvidence()
    evidence.add("Synthoceps", "current_build_analysis", equipped=True)
    evidence.add("Rose", "current_build_analysis", equipped=True)

    claims = find_ownership_claims(answer, evidence)

    assert all(value.supported for value in claims), [(v.item, v.supported) for v in claims]


def test_the_evidence_block_tells_the_model_not_to_recite_it() -> None:
    evidence = OwnershipEvidence()
    evidence.add("Synthoceps", "current_build_analysis", equipped=True)
    evidence.add("Fatebringer", "inventory_search", equipped=False)

    summary = evidence_summary(evidence)

    assert "equipped: Synthoceps." in summary and "owned, not equipped: Fatebringer." in summary
    assert "do not list, quote, or mention this evidence block" in summary


def test_a_shortened_reference_to_an_owned_item_is_that_item_not_a_new_one() -> None:
    evidence = OwnershipEvidence()
    evidence.add("Celestial Nighthawk", "current_build_analysis", equipped=True, character="Hunter")
    evidence.add("Calus Mini-Tool", "current_build_analysis", equipped=True, character="Hunter")
    answer = (
        "Put on a Nightstalker-focused Exotic chest you own (or switch to a Gunslinger "
        "subclass when you want to use Nighthawk)."
    )

    claims = find_ownership_claims(answer, evidence, "Hunter")

    assert all(value.supported for value in claims), [(v.item, v.supported) for v in claims]
    assert find_ownership_claims("You have Nighthawk equipped.", evidence, "Hunter")[0].item == (
        "Celestial Nighthawk"
    )


def test_partial_names_do_not_launder_unrelated_or_ambiguous_references() -> None:
    evidence = OwnershipEvidence()
    evidence.add("Celestial Nighthawk", "current_build_analysis", equipped=True)
    evidence.add("Mida Multi-Tool", "inventory_search", equipped=False)
    evidence.add("Calus Mini-Tool", "inventory_search", equipped=False)

    # An unrelated name stays unsupported; a word shared by two items is ambiguous; short
    # single words are never treated as references.
    for answer in ("You have Witherhoard.", "You have Tool.", "You have Sun."):
        claims = find_ownership_claims(answer, evidence)
        assert claims == [] or not any(value.supported for value in claims), answer


@pytest.mark.parametrize(
    "answer",
    [
        "Brief equipped loadout: Rose, Calus Mini-Tool and Apex Predator.",
        "Quick equipped summary for your Titan: Synthoceps with Rose.",
        "- Current equipped gear is solid.",
    ],
)
def test_a_capitalized_word_that_only_opens_the_sentence_is_not_an_item(answer: str) -> None:
    evidence = OwnershipEvidence()
    for name in ("Rose", "Calus Mini-Tool", "Apex Predator", "Synthoceps"):
        evidence.add(name, "current_build_analysis", equipped=True, character="Titan")

    claims = find_ownership_claims(answer, evidence, "Titan")

    assert all(value.supported for value in claims), [(v.item, v.supported) for v in claims]


def test_a_sentence_opening_item_claim_with_a_verb_is_still_caught() -> None:
    evidence = OwnershipEvidence()
    evidence.add("Rose", "current_build_analysis", equipped=True, character="Titan")

    claims = find_ownership_claims("Witherhoard is equipped on your Titan.", evidence, "Titan")

    assert [(value.item, value.supported) for value in claims] == [("Witherhoard", False)]


def test_alternatives_results_prove_the_characters_equipped_exotics() -> None:
    context = PROFILES["titan_veteran"]()
    evidence = OwnershipEvidence()
    result = GuardianToolService(context).execute("find_build_alternatives", _find_arguments())

    record_ownership_evidence(evidence, "find_build_alternatives", result)

    assert result["preservation"]["equipped_exotic_armor"] == ["Synthoceps"]
    assert evidence.source_for("Synthoceps", "equipped", "Titan") == "inventory_search"
    assert evidence.source_for("Synthoceps", "equipped", "Hunter") is None
    # Locked items are a user constraint, not evidence of anything equipped.
    locked_only = {"character_class": "Titan", "preservation": {"locked_items": ["Witherhoard"]}}
    fresh = OwnershipEvidence()
    record_ownership_evidence(fresh, "find_build_alternatives", locked_only)
    assert fresh.source_for("Witherhoard", "equipped", "Titan") is None
