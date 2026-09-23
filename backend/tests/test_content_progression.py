import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.ai import RecommendationService
from app.content_progression import (
    MAX_CONTENT_RESULTS,
    MAX_EVIDENCE_ITEMS,
    ContentDefinition,
    ContentDefinitionSet,
    ContentProgressionResolver,
)
from app.guardian_tools import GUARDIAN_TOOL_DEFINITIONS, GuardianToolService
from app.models import (
    CharacterSummary,
    ChatTurn,
    DataAvailability,
    GuardianContext,
    ItemSummary,
    MilestoneSummary,
    ObjectiveSummary,
    ProgressionSummary,
    QuestSummary,
)
from app.response_quality import response_mode_context
from app.session_preferences import derive_session_preferences

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "guardian_copilot_evals.json"


def character(
    character_id: str,
    class_name: str,
    *,
    quests: list[QuestSummary] | None = None,
    milestones: list[MilestoneSummary] | None = None,
    progressions: list[ProgressionSummary] | None = None,
    subclass: ItemSummary | None = None,
) -> CharacterSummary:
    return CharacterSummary(
        character_id=character_id,
        class_name=class_name,
        race_name="Human",
        gender_name="Female",
        power=0,
        quests=quests or [],
        milestones=milestones or [],
        progressions=progressions or [],
        subclass=subclass,
    )


def context(
    *characters: CharacterSummary,
    unavailable: dict[str, str] | None = None,
) -> GuardianContext:
    return GuardianContext(
        bungie_display_name="Guardian#1234",
        membership_id="membership-private",
        membership_type=3,
        platform_name="Steam",
        characters=list(characters),
        data_availability=DataAvailability(
            fetched_at=datetime(2026, 9, 22, tzinfo=UTC),
            available_components=["Characters", "CharacterProgressions", "Records"],
            unavailable_components=unavailable or {},
        ),
    )


def quest(
    character_id: str,
    name: str,
    *,
    step: str | None = None,
    completed: bool = False,
    redeemed: bool = False,
    index: int = 1,
) -> QuestSummary:
    return QuestSummary(
        quest_hash=index,
        name=name,
        step_name=step,
        character_id=character_id,
        completed=completed,
        redeemed=redeemed,
        objectives=[
            ObjectiveSummary(
                objective_hash=1000 + index,
                name="Campaign objective",
                progress=2,
                completion_value=5,
                progress_percent=40,
            )
        ],
    )


def terminal_definitions() -> ContentDefinitionSet:
    return ContentDefinitionSet(
        version=2,
        contents=[
            ContentDefinition(
                canonical_name="The Final Shape",
                aliases=["The Final Shape", "Final Shape", "TFS"],
                match_names=["The Final Shape", "Final Shape"],
                completion_quest_names=["Final Shape Finale"],
                rule_reason="Test fixture terminal quest used to verify deterministic semantics.",
            )
        ],
    )


def resolved(
    guardian: GuardianContext,
    selected: CharacterSummary,
    content_name: str = "The Final Shape",
    definitions: ContentDefinitionSet | None = None,
):
    return ContentProgressionResolver(guardian, definitions).resolve(selected, content_name).content


def test_active_campaign_is_explicitly_in_progress() -> None:
    titan = character(
        "titan",
        "Titan",
        quests=[quest("titan", "The Final Shape", step="Temptation")],
    )
    result = resolved(context(titan), titan)

    assert result is not None
    assert result.status == "in_progress"
    assert result.evidence_state == "KNOWN_TRUE"
    assert result.active_quest_state == "KNOWN_TRUE"
    assert result.completion_state == "UNKNOWN"
    assert result.current_step == "Temptation"
    assert result.progress_summary == "Campaign objective: 2/5"


def test_maintained_terminal_rule_can_establish_completion() -> None:
    titan = character(
        "titan",
        "Titan",
        quests=[quest("titan", "Final Shape Finale", completed=True, redeemed=True)],
    )
    result = resolved(context(titan), titan, definitions=terminal_definitions())

    assert result is not None
    assert result.status == "completed"
    assert result.completion_state == "KNOWN_TRUE"
    assert result.active_quest_state == "NOT_RETURNED"


def test_supported_content_without_evidence_is_unknown() -> None:
    titan = character("titan", "Titan")
    result = resolved(context(titan), titan)

    assert result is not None
    assert result.status == "unknown"
    assert result.active_quest_state == "NOT_RETURNED"
    assert result.completion_state == "UNKNOWN"
    assert "does not establish completion" in result.limitations[0]


def test_completed_related_quest_without_terminal_rule_remains_unknown() -> None:
    titan = character(
        "titan",
        "Titan",
        quests=[quest("titan", "The Final Shape", completed=True, redeemed=True)],
    )
    result = resolved(context(titan), titan)

    assert result is not None
    assert result.evidence[0].state == "completed"
    assert result.status == "unknown"
    assert result.completion_state == "UNKNOWN"
    assert any("no maintained rule" in value for value in result.limitations)


def test_campaign_active_only_on_other_character_does_not_affect_titan() -> None:
    titan = character("titan", "Titan")
    hunter = character("hunter", "Hunter", quests=[quest("hunter", "The Final Shape")])
    guardian = context(titan, hunter)

    titan_result = resolved(guardian, titan)
    hunter_result = resolved(guardian, hunter)

    assert titan_result is not None and titan_result.status == "unknown"
    assert hunter_result is not None and hunter_result.status == "in_progress"


def test_three_character_states_remain_character_scoped() -> None:
    titan = character("titan", "Titan", quests=[quest("titan", "The Final Shape")])
    hunter = character(
        "hunter",
        "Hunter",
        quests=[quest("hunter", "Final Shape Finale", completed=True, redeemed=True)],
    )
    warlock = character("warlock", "Warlock")
    guardian = context(titan, hunter, warlock)
    resolver = ContentProgressionResolver(guardian, terminal_definitions())

    assert resolver.resolve(titan, "TFS").content.status == "in_progress"  # type: ignore[union-attr]
    assert resolver.resolve(hunter, "TFS").content.status == "completed"  # type: ignore[union-attr]
    assert resolver.resolve(warlock, "TFS").content.status == "unknown"  # type: ignore[union-attr]


def test_ambiguous_similarly_named_quest_is_not_matched() -> None:
    titan = character("titan", "Titan", quests=[quest("titan", "The Final Shapesmith")])
    result = resolved(context(titan), titan)

    assert result is not None
    assert result.status == "unknown"
    assert result.evidence == []


@pytest.mark.parametrize("alias", ["The Final Shape", "Final Shape", "TFS", "tfs"])
def test_supported_aliases_resolve_to_canonical_name(alias: str) -> None:
    titan = character("titan", "Titan")
    result = resolved(context(titan), titan, alias)

    assert result is not None
    assert result.supported is True
    assert result.canonical_name == "The Final Shape"


def test_unknown_content_name_fails_safely() -> None:
    titan = character("titan", "Titan")
    result = resolved(context(titan), titan, "Curse of Osiris")

    assert result is not None
    assert result.supported is False
    assert result.status == "unknown"
    assert result.evidence == []


def test_output_is_bounded() -> None:
    titan = character(
        "titan",
        "Titan",
        quests=[
            quest("titan", f"The Final Shape Quest {index}", index=index) for index in range(12)
        ],
    )
    guardian = context(titan)
    resolver = ContentProgressionResolver(guardian)
    one = resolver.resolve(titan, "Final Shape").content
    all_content = resolver.resolve(titan, None)

    assert one is not None and len(one.evidence) == MAX_EVIDENCE_ITEMS
    assert len(all_content.contents) == MAX_CONTENT_RESULTS


def test_missing_bungie_components_preserve_unknown() -> None:
    titan = character("titan", "Titan")
    result = resolved(
        context(
            titan,
            unavailable={
                "CharacterProgressions": "not returned",
                "Records": "not returned",
            },
        ),
        titan,
    )

    assert result is not None
    assert result.status == "unknown"
    assert any("components were not returned" in value for value in result.limitations)


def test_completed_content_can_be_resolved_without_active_quest() -> None:
    titan = character(
        "titan",
        "Titan",
        quests=[quest("titan", "Final Shape Finale", completed=True, redeemed=True)],
    )
    result = resolved(context(titan), titan, definitions=terminal_definitions())

    assert result is not None
    assert result.status == "completed"
    assert result.active_quest_state == "NOT_RETURNED"


def test_indirect_unlock_does_not_establish_campaign_completion() -> None:
    prismatic = ItemSummary(
        item_hash=55,
        name="Prismatic",
        item_type="Subclass",
        location="equipped",
        character_id="titan",
        is_equipped=True,
    )
    titan = character("titan", "Titan", subclass=prismatic)
    result = resolved(context(titan), titan)

    assert result is not None
    assert result.status == "unknown"
    assert result.completion_state == "UNKNOWN"


def test_multiple_relevant_signals_are_retained() -> None:
    objective = ObjectiveSummary(
        objective_hash=80,
        name="Finish missions",
        progress=3,
        completion_value=8,
    )
    titan = character(
        "titan",
        "Titan",
        quests=[quest("titan", "The Final Shape", step="Dissent")],
        milestones=[
            MilestoneSummary(
                milestone_hash=81,
                name="The Final Shape Campaign",
                character_id="titan",
                objectives=[objective],
            )
        ],
        progressions=[
            ProgressionSummary(
                progression_hash=82,
                name="The Final Shape",
                scope="character",
                character_id="titan",
                level=3,
                level_cap=8,
            )
        ],
    )
    result = resolved(context(titan), titan)

    assert result is not None
    assert result.status == "in_progress"
    assert {value.type for value in result.evidence} == {
        "quest",
        "milestone",
        "progression",
    }


def test_conflicting_terminal_and_active_evidence_fails_safely() -> None:
    titan = character(
        "titan",
        "Titan",
        quests=[
            quest("titan", "Final Shape Finale", completed=True, redeemed=True),
            quest("titan", "The Final Shape", step="Replay"),
        ],
    )
    result = resolved(context(titan), titan, definitions=terminal_definitions())

    assert result is not None
    assert result.status == "unknown"
    assert result.evidence_state == "UNKNOWN"
    assert result.completion_state == "UNKNOWN"
    assert "conflict" in result.limitations[0]


def test_guardian_tool_schema_and_result_are_strict_bounded_and_id_free() -> None:
    definition = next(
        value for value in GUARDIAN_TOOL_DEFINITIONS if value["name"] == "get_content_progression"
    )
    assert definition["strict"] is True
    assert definition["parameters"]["required"] == ["character_id", "content_name"]
    assert definition["parameters"]["additionalProperties"] is False

    titan = character("secret-character-id", "Titan")
    result = GuardianToolService(context(titan)).execute(
        "get_content_progression",
        {"character_id": "secret-character-id", "content_name": None},
    )

    assert len(result["contents"]) == MAX_CONTENT_RESULTS
    assert "secret-character-id" not in json.dumps(result)


def test_evaluation_fixture_matches_classification_and_tool_contracts() -> None:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    definitions = {value["name"] for value in GUARDIAN_TOOL_DEFINITIONS}

    assert payload["version"] == 5
    assert len(payload["cases"]) == 10
    for case in payload["cases"]:
        assert RecommendationService._is_session_planning(case["prompt"]) is case["planning"]
        assert RecommendationService._intent_category(case["prompt"]) == case["intent"]
        assert set(case["relevant_guardian_tools"]) <= definitions
        assert case["forbidden_behaviors"]
        assert case["response_style"]


def test_multi_turn_evaluation_fixtures_match_preference_extraction() -> None:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    assert len(payload["multi_turn_cases"]) == 8
    for case in payload["multi_turn_cases"]:
        history = [ChatTurn(role="user", content=value) for value in case["history"]]
        result = derive_session_preferences(history, case["current"]).preferences.model_dump()
        assert {key: result[key] for key in case["expected_preferences"]} == case[
            "expected_preferences"
        ]
        assert case["forbidden_behaviors"]


def test_response_style_evaluation_fixtures_match_mode_contracts() -> None:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    assert len(payload["response_style_cases"]) == 7
    for case in payload["response_style_cases"]:
        mode = response_mode_context(
            case["prompt"],
            session_planning=case["planning"],
            continuation=False,
        )
        assert mode.mode == case["response_mode"]
        assert mode.detail_level == case["detail_expectation"]
        assert mode.timed_itinerary_requested is case["timed_itinerary_allowed"]
        assert mode.source_limit == case["source_limit"]
        assert case["max_follow_up_count"] == 1
        assert case["suppress_account_metadata"] is True


def test_eval_campaign_cases_route_to_content_progression() -> None:
    cases = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["cases"]
    relevant = [value for value in cases if value["content_progression_relevant"]]

    assert len(relevant) == 3
    assert all("get_content_progression" in value["relevant_guardian_tools"] for value in relevant)


def test_build_evaluation_fixtures_match_build_tool_and_mode_contracts() -> None:
    cases = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["build_cases"]
    definitions = {value["name"] for value in GuardianToolService.definitions()}

    assert len(cases) == 9
    for case in cases:
        mode = response_mode_context(
            case["prompt"],
            session_planning=RecommendationService._is_session_planning(case["prompt"]),
            continuation=False,
        )
        assert mode.mode == case["response_mode"]
        assert set(case["relevant_guardian_tools"]) <= definitions
        assert case["forbidden_behaviors"]
