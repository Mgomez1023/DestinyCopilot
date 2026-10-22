"""Repairs keep what is supported: only the offending sentence or clause changes.

Covers conflation repair, retrieved class/slot contradictions, filler openers and rewrites that
lose citations, a stated campaign premise, and inventory name matches hidden by filters.
"""

from app.ai import RecommendationService, states_completion_premise
from app.guardian_tools import GuardianToolService, InventorySearchRequest
from app.item_fact_claims import (
    collect_item_definitions,
    contradicted_item_facts,
    correct_item_facts,
)
from app.response_quality import (
    ResponseQualityValidator,
    loses_cited_facts,
    strip_filler_opener,
)
from tests.test_guardian_tools import guardian_context, item
from tests.test_inventory_perk_search import tools_with, vault_weapon
from tests.test_playtest_scenarios import (
    LIVE6,
    SOLO_HISTORY,
    Script,
    assistant_message,
    run,
    service,
    titan_with,
)

# --- B1: a conflating sentence is dropped; the plan stays ------------------------------------


def test_only_the_conflating_sentence_is_removed_from_a_plan() -> None:
    script = Script([assistant_message(LIVE6["turn2"]["draft"])], [assistant_message("unused")])
    value = service(script)
    response = run(value, LIVE6["turn2"]["message"], titan_with(), SOLO_HISTORY)
    assert "Launch Haunted Altars of Sorrow" in response.message
    assert "Pick up all engrams" in response.message
    assert "Waves of Altars of Sorrow" not in response.message


def test_a_plan_with_no_conflation_is_not_touched() -> None:
    script = Script(
        [assistant_message("Run Haunted Altars of Sorrow. It is available to your Titan.")],
        [assistant_message("unused")],
    )
    value = service(script)
    run(value, "What should I run next?", titan_with(), SOLO_HISTORY)
    assert "activity_conflation_repaired" not in value.latest_trace()


def test_repair_returns_none_when_only_the_conflation_would_remain() -> None:
    from app.session_planning import SessionPlanningContext

    assert (
        ResponseQualityValidator.repair_activity_conflations(
            "Nothing to repair here.", SessionPlanningContext.model_construct(candidates=[])
        )
        is None
    )


# --- B2: stated class and slot are compared with the retrieved definition --------------------


def _definitions(*rows: dict) -> dict:
    found: dict = {}
    collect_item_definitions(list(rows), found)
    return found


HELM = {"name": "Foxglove Crown", "class_restriction": "Warlock", "bucket": "Helmet"}
GUN = {"name": "Quillstorm", "class_restriction": "Any", "bucket": "Energy Weapons"}


def test_a_wrong_class_is_corrected_to_the_retrieved_class() -> None:
    answer = "Foxglove Crown is a Hunter helmet. It pairs well with your current build."
    defs = _definitions(HELM)
    assert contradicted_item_facts(answer, defs)
    fixed = correct_item_facts(answer, defs)
    assert fixed is not None and "Warlock helmet" in fixed and "Hunter" not in fixed
    assert "pairs well" in fixed


def test_an_appositive_class_and_slot_are_both_checked() -> None:
    defs = _definitions(HELM)
    answer = "Try Foxglove Crown (Exotic Hunter gauntlets) for the extra ability energy today."
    kinds = {kind for _, kind, _, _ in contradicted_item_facts(answer, defs)}
    assert kinds == {"class restriction", "equipment slot"}
    fixed = correct_item_facts(answer, defs)
    assert fixed is not None and "Warlock" in fixed and "helmet" in fixed
    assert "Hunter" not in fixed and "gauntlets" not in fixed


def test_a_slot_phrase_for_a_non_helmet_is_not_kept() -> None:
    defs = _definitions(GUN)
    answer = "Quillstorm refills with the helmet's ability on kills. Keep it equipped in Energy."
    assert contradicted_item_facts(answer, defs)
    fixed = correct_item_facts(answer, defs)
    assert fixed is None or "helmet's ability" not in fixed


def test_matching_claims_and_unrelated_phrases_are_not_flagged() -> None:
    defs = _definitions(HELM, GUN)
    assert not contradicted_item_facts("Foxglove Crown is a Warlock helmet.", defs)
    # A negated class mention agrees with the retrieved class.
    assert not contradicted_item_facts("Foxglove Crown isn't a Titan helmet.", defs)
    # Two known items in one sentence are not read as one claim.
    assert not contradicted_item_facts("Foxglove Crown and Quillstorm are Hunter picks.", defs)
    # A question is never a claim, and a helmet phrase that is not about the item is skipped.
    assert not contradicted_item_facts("Is Foxglove Crown a Hunter helmet?", defs)
    assert not contradicted_item_facts(
        "Equip Quillstorm, then check the helmet's ability on your armor.", defs
    )


def test_variants_with_several_classes_accept_any_of_them() -> None:
    defs = _definitions(
        {"name": "Foxglove Crown", "class_restriction": "Warlock"},
        {"name": "Foxglove Crown", "class_restriction": "Hunter"},
    )
    assert not contradicted_item_facts("Foxglove Crown is a Hunter helmet.", defs)


def test_a_definition_from_account_gear_counts_without_a_tool_call() -> None:
    tools = tools_with(vault_weapon("Quillstorm"))
    context = tools.context
    defs: dict = {}
    collect_item_definitions(
        [
            {"name": v.name, "class_restriction": v.class_restriction, "bucket": v.gear_slot()}
            for v in context.inventory.items
        ],
        defs,
    )
    assert "quillstorm" in defs


# --- I1: a rewrite keeps cited facts; a filler opener is only stripped -----------------------


def test_a_filler_opener_is_stripped_without_a_rewrite() -> None:
    answer = "Based on the available data, The Coil rotates weekly (source: example.com/coil)."
    assert strip_filler_opener(answer) == "The Coil rotates weekly (source: example.com/coil)."
    assert strip_filler_opener("Run Nightfall next.") is None
    assert strip_filler_opener("Short answer: no.") is None  # nothing substantive left


def test_a_rewrite_that_drops_the_link_or_date_is_detected() -> None:
    draft = "Reset was Oct 21, 2026 per https://example.com/reset. Run the Coil."
    assert loses_cited_facts(draft, "I can't check live rotations. Run the Coil.")
    assert not loses_cited_facts(
        draft, "Run the Coil. Reset was Oct 21, 2026 per https://example.com/reset."
    )
    assert not loses_cited_facts("Run the Coil.", "Run the Coil today.")


# --- I2: a stated premise gets at most one clause, never the disclaimer ----------------------


def test_a_stated_premise_gets_a_one_clause_note_not_the_disclaimer() -> None:
    answer = (
        "Since you finished The Final Shape, the Prismatic subclass unlock steps apply. "
        "Start with the Vestige chain."
    )
    assert states_completion_premise("I've finished The Final Shape, how do I unlock Facets?")
    result = RecommendationService._enforce_campaign_completion_safety(  # noqa: SLF001
        answer, "I've finished The Final Shape, how do I unlock Facets?"
    )
    assert "Start with the Vestige chain." in result
    assert "Since you finished The Final Shape" in result
    assert "going on your word" in result
    assert "I can't verify campaign completion" not in result
    assert len(result) < len(answer) + 120


def test_without_a_stated_premise_the_inference_is_still_blocked() -> None:
    answer = "Since Prismatic is equipped you have probably finished The Final Shape."
    result = RecommendationService._enforce_campaign_completion_safety(  # noqa: SLF001
        answer, "What should I do next?"
    )
    assert "can't verify campaign completion" in result


# --- I3: name matches excluded by filters are reported with their real place -----------------


def test_a_name_match_outside_the_filters_is_reported_with_its_real_bucket() -> None:
    tools = tools_with(vault_weapon("Quillstorm"))
    result = tools.search_inventory(InventorySearchRequest(query="quillstorm", subtype="Sniper"))
    assert result["items"] == []
    excluded = result["name_matches_excluded_by_filters"]
    assert excluded[0]["name"] == "Quillstorm"
    assert excluded[0]["subtype"] == "Auto Rifle"
    assert excluded[0]["location"] == "vault"
    assert "outside them" in result["search_note"]


def test_no_excluded_section_when_the_filters_hide_nothing() -> None:
    tools = tools_with(vault_weapon("Quillstorm"))
    result = tools.search_inventory(InventorySearchRequest(query="quillstorm"))
    assert "name_matches_excluded_by_filters" not in result
    assert names_of(result) == ["Quillstorm"]


def names_of(result: dict) -> list[str]:
    return [value["name"] for value in result["items"]]


def test_unrelated_names_are_not_reported_as_excluded() -> None:
    context = guardian_context()
    context.inventory.items.append(item("Other Gun", "Weapon", subtype="Auto Rifle"))
    result = GuardianToolService(context).search_inventory(
        InventorySearchRequest(query="quillstorm", subtype="Sniper")
    )
    assert "name_matches_excluded_by_filters" not in result


# --- I4: the loot correction names the claim and forbids a generic disclaimer ----------------


def test_the_loot_correction_quotes_the_claim_and_forbids_a_blanket_note() -> None:
    from app.response_quality import ReadOnlyActionValidationContext, response_mode_context

    read_only = ReadOnlyActionValidationContext(recommendation_turn=True)
    mode = response_mode_context("What next?", session_planning=True, continuation=False)
    violations = ResponseQualityValidator().validate(
        "Run Nightfall. Nightfall has the best loot of all.", mode, "next?", None, None, read_only
    )
    loot = next(v for v in violations if v.code == "unsupported_loot_claim")
    assert "best loot of all" in loot.correction
    assert "Do not add a general disclaimer" in loot.correction
