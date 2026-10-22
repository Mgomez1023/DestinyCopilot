"""Item-definition claims need a lookup, absence is not zero, and validators do not leak.

Offline checks only: the wording here uses made-up items and measures so nothing is keyed to an
evaluation prompt.
"""

import pytest

from app.absence_claims import (
    absent_measure_values,
    collect_evidence_terms,
    replace_absent_values,
)
from app.guardian_tools import InventorySearchRequest
from app.item_fact_claims import (
    collect_item_names,
    remove_unverified_item_claims,
    unverified_item_claims,
)
from app.ownership_evidence import OwnershipEvidence, record_ownership_evidence
from app.response_quality import ResponseQualityValidator, remove_correction_narration
from tests.test_inventory_perk_search import names, tools_with, vault_weapon
from tests.test_playtest_scenarios import (
    Script,
    StubKnowledge,
    assistant_message,
    call,
    run,
    service,
    titan_with,
)

# --- N1: definition facts about a named item rest on a retrieved result ------------------------


@pytest.mark.parametrize(
    "answer",
    [
        "Zenith Brace is a Hunter-only chest piece.",
        "Zenith Brace can only be equipped by Warlocks.",
        "Zenith Brace is class-locked to Titans.",
        "Zenith Brace isn't usable by a Titan.",
        "Zenith Brace goes in the kinetic slot.",
        "Zenith Brace's barrel is fixed, so every copy has it.",
        "The magazine on Zenith Brace can be re-rolled.",
    ],
)
def test_a_definition_fact_about_an_unretrieved_item_is_flagged(answer: str) -> None:
    assert unverified_item_claims(answer, set())


@pytest.mark.parametrize(
    "answer",
    [
        "Zenith Brace is a Hunter-only chest piece.",
        "Zenith Brace's barrel is fixed, so every copy has it.",
        "Zenith Brace goes in the kinetic slot.",
    ],
)
def test_the_same_fact_is_fine_once_the_item_was_retrieved(answer: str) -> None:
    retrieved: set[str] = set()
    collect_item_names({"item": {"name": "Zenith Brace"}}, retrieved)
    assert unverified_item_claims(answer, retrieved) == []


@pytest.mark.parametrize(
    "answer",
    [
        "Which Titan do you want to build around?",
        "Your Titan is the one you played last.",
        "Zenith Brace is a strong pick for add clear.",
        "Is Zenith Brace a Hunter-only chest piece?",
        "Kinetic weapons go in the kinetic slot.",
    ],
)
def test_ordinary_sentences_are_not_item_fact_claims(answer: str) -> None:
    assert unverified_item_claims(answer, set()) == []


def test_a_lookup_of_a_different_item_does_not_support_the_claim() -> None:
    retrieved: set[str] = set()
    collect_item_names({"items": [{"name": "Other Gun"}]}, retrieved)
    assert unverified_item_claims("Zenith Brace is a Hunter-only chest piece.", retrieved)


def test_unsupported_sentences_are_removed_and_the_rest_kept() -> None:
    answer = (
        "Zenith Brace is a Hunter-only chest piece. "
        "Pair your Exotic with a Solar subclass for steady ability uptime."
    )
    cleaned = remove_unverified_item_claims(answer, set())
    assert cleaned == "Pair your Exotic with a Solar subclass for steady ability uptime."
    # Nothing worth keeping is left: no repair is offered rather than an empty answer.
    assert (
        remove_unverified_item_claims("Zenith Brace is a Hunter-only chest piece.", set()) is None
    )
    assert remove_unverified_item_claims("A fine answer with no item facts at all.", set()) is None


def _knowledge() -> StubKnowledge:
    return StubKnowledge(
        {
            "get_item_details": {
                "item": {"name": "Zenith Brace", "class_restriction": {"name": "Titan"}}
            }
        }
    )


def test_a_remembered_class_restriction_forces_the_lookup_and_is_corrected() -> None:
    knowledge = _knowledge()
    script = Script(
        [assistant_message("Zenith Brace is a Hunter-only chest piece.")],
        [call("get_item_details", "i-1", item_name_or_hash="Zenith Brace")],
        [assistant_message("Zenith Brace is a Titan Exotic, so your Titan can wear it.")],
    )

    response = run(service(script, knowledge), "Who can wear Zenith Brace?", titan_with())

    assert response.message == "Zenith Brace is a Titan Exotic, so your Titan can wear it."
    assert knowledge.calls[0][0] == "get_item_details"
    assert script.requests[1]["tool_choice"] == {"type": "function", "name": "get_item_details"}


def test_a_claim_still_unsupported_after_the_retry_is_dropped_not_kept() -> None:
    knowledge = StubKnowledge({"get_item_details": {"item": {"name": "Other Gun"}}})
    wrong = (
        "Zenith Brace is a Hunter-only chest piece. "
        "Pair it with a Solar subclass for steady ability uptime."
    )
    script = Script(
        [assistant_message(wrong)],
        [call("get_item_details", "i-1", item_name_or_hash="Other Gun")],
        [assistant_message(wrong)],
        [assistant_message(wrong)],
    )

    response = run(service(script, knowledge), "Who can wear Zenith Brace?", titan_with())

    assert "Hunter-only" not in response.message
    assert "steady ability uptime" in response.message


# --- N2: missing data is unavailable, not zero -------------------------------------------------


def _terms(result: dict) -> set[str]:
    terms: set[str] = set()
    collect_evidence_terms(result, terms)
    return terms


RECENT = {"recent_activities": [], "note": "No recent activities were returned."}


@pytest.mark.parametrize(
    "answer",
    [
        "Your win streak is 0.",
        "Your current streak stands at zero.",
        "You have 0 wins in a row.",
        "Your current Gambit win streak is 0.",
        "You have 0 kills this season.",
        "You've got zero banked engrams.",
        "You're on a streak of 0.",
        "The Titan has 0 kills this season.",
    ],
)
def test_a_zero_for_a_measure_no_tool_returned_is_flagged(answer: str) -> None:
    assert absent_measure_values(answer, _terms(RECENT))


@pytest.mark.parametrize(
    "answer",
    [
        "Damage falloff is none for rockets.",
        "Recoil direction is 0 on this roll, which is ideal.",
        "Duality has 0 champions on Standard.",
        "Heavy ammo cost is zero in this mode.",
        "The Exotic class item has zero stat penalty.",
        "Streak: 0",
    ],
)
def test_general_game_facts_that_state_zero_are_not_player_values(answer: str) -> None:
    terms = _terms({"activities": [{"name": "Standard", "type": "Dungeon"}], "quest": "x"})
    assert absent_measure_values(answer, terms) == []
    assert absent_measure_values(answer, _terms(RECENT)) == []
    assert replace_absent_values(answer, terms) is None


def test_a_game_fact_next_to_a_player_value_flags_only_the_player_sentence() -> None:
    answer = "Duality has 0 champions on Standard. Your Gambit win streak is 0."
    flagged = absent_measure_values(answer, _terms(RECENT))
    assert [sentence for sentence, _ in flagged] == ["Your Gambit win streak is 0."]


@pytest.mark.parametrize(
    "answer",
    [
        "I can't see your win streak in the available data.",
        "There is no data on your streak, so I can't say it's 0.",
        "Your streak isn't available from the data I have.",
        "What is your streak?",
    ],
)
def test_stating_the_absence_is_not_flagged(answer: str) -> None:
    assert absent_measure_values(answer, _terms(RECENT)) == []


def test_a_zero_the_data_actually_returned_is_fine() -> None:
    result = {"objectives": [{"name": "Strikes", "progress": 0, "total": 100}]}
    assert absent_measure_values("Your progress is 0 of 100.", _terms(result)) == []
    assert absent_measure_values("You have 0 strikes done.", _terms(result)) == []


def test_without_any_retrieved_data_nothing_is_judged() -> None:
    assert absent_measure_values("Your streak is 0.", set()) == []


def test_the_flagged_sentence_becomes_a_statement_of_unavailability() -> None:
    answer = "Your win streak is 0. Try a Vanguard Op next."
    repaired = replace_absent_values(answer, _terms(RECENT))
    assert repaired == "I can't see your win streak in the available data. Try a Vanguard Op next."
    assert replace_absent_values("Try a Vanguard Op next.", _terms(RECENT)) is None


def test_a_zero_stated_after_an_empty_result_is_replaced_end_to_end() -> None:
    script = Script(
        [call("get_recent_activities", "r-1")],
        [assistant_message("Your win streak is 0. Try a Vanguard Op next.")],
        [assistant_message("Your win streak is 0. Try a Vanguard Op next.")],
    )

    response = run(service(script), "How long is my win streak?", titan_with())

    assert "streak is 0" not in response.message
    assert "I can't see your win streak in the available data." in response.message
    assert "Try a Vanguard Op next." in response.message


# --- I2: a scoped perk search names the matches held by other characters ----------------------


def test_other_character_matches_are_named_with_class_location_and_perk() -> None:
    held = vault_weapon("Elsewhere Gun", "Zephyr Loop", character_id="char-2", location="character")
    result = tools_with(held).search_inventory(
        InventorySearchRequest(query="zephyr loop", character_class="Titan")
    )
    assert names(result) == []
    assert result["matches_on_other_characters"] == 1
    [entry] = result["other_character_matches"]
    assert entry["name"] == "Elsewhere Gun"
    assert entry["location"] == "character"
    assert entry["matched_on"] == ["perk: Zephyr Loop"]
    assert "character_class" in entry


def test_unscoped_search_has_no_other_character_section() -> None:
    held = vault_weapon("Elsewhere Gun", "Zephyr Loop", character_id="char-2", location="character")
    result = tools_with(held).search_inventory(InventorySearchRequest(query="zephyr loop"))
    assert "other_character_matches" not in result


def test_other_character_matches_count_as_possession_evidence() -> None:
    evidence = OwnershipEvidence()
    record_ownership_evidence(
        evidence,
        "search_inventory",
        {
            "items": [],
            "other_character_matches": [
                {"name": "Elsewhere Gun", "character_class": "Hunter", "equipped": True}
            ],
        },
    )
    assert "elsewhere gun" in evidence.possessed
    assert evidence.equipped_on["elsewhere gun"] == ["Hunter"]


# --- I3: validators do not leak into the answer ------------------------------------------------


@pytest.mark.parametrize(
    ("draft", "expected"),
    [
        (
            "Quick correction: the Waves record belongs to the Moon patrol, not the seasonal "
            "activity.",
            "The Waves record belongs to the Moon patrol, not the seasonal activity.",
        ),
        (
            "**Correction:** Rose is a Legendary hand cannon.\nIt rolls Rampage.",
            "Rose is a Legendary hand cannon.\nIt rolls Rampage.",
        ),
        (
            "Run the seasonal activity. Don't mix them up. It has its own record.",
            "Run the seasonal activity. It has its own record.",
        ),
        ("Clarification — the patrol is on the Moon.", "The patrol is on the Moon."),
    ],
)
def test_correction_narration_is_removed_and_the_fact_kept(draft: str, expected: str) -> None:
    assert remove_correction_narration(draft) == expected


@pytest.mark.parametrize(
    "answer",
    [
        "Rose is a Legendary hand cannon.",
        "The correction fluid is in the Tower.",
        "Don't mix Solar and Void fragments on a Prismatic subclass.",
        "To clarify which record counts, open the Triumphs menu and read its path.",
        "Quick tip: pick one slot to improve first.",
    ],
)
def test_ordinary_sentences_are_left_alone(answer: str) -> None:
    assert remove_correction_narration(answer) == answer


def test_an_answer_that_is_only_narration_is_not_emptied() -> None:
    assert remove_correction_narration("Quick correction:") == "Quick correction:"


def test_the_correction_request_forbids_announcing_the_correction() -> None:
    from app.session_planning import REWRITE_RULE

    assert "do not announce" in REWRITE_RULE.casefold()
    assert "quick correction" in REWRITE_RULE.casefold()
    assert REWRITE_RULE in ResponseQualityValidator.correction_instruction([])


def test_loot_note_names_only_what_was_cut() -> None:
    cleaned = ResponseQualityValidator.repair_planning_claims(
        "Recommended: Haunted Altars of Sorrow, since it gives higher-tier rewards.\n- Launch it."
    )
    assert cleaned is not None
    assert cleaned.endswith("Loot quality isn't established by your data.")
    assert "solo" not in cleaned

    both = ResponseQualityValidator.repair_planning_claims(
        "Recommended: Haunted Altars of Sorrow (solo, ~30 min), since it gives "
        "higher-tier rewards.\n"
        "- Launch it.",
        {30},
    )
    assert both is not None
    assert both.endswith("Loot quality and solo clear time aren't established by your data.")


def test_the_note_is_not_repeated_when_the_answer_already_states_the_limit() -> None:
    cleaned = ResponseQualityValidator.repair_planning_claims(
        "Recommended: Haunted Altars of Sorrow, since it gives higher-tier rewards.\n- Launch it.\n"
        "Which loot is better isn't established by your data."
    )
    assert cleaned is not None
    assert cleaned.count("isn't established") == 1
