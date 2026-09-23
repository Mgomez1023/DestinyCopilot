from datetime import UTC, datetime

import pytest

from app.models import (
    CharacterSummary,
    ChatTurn,
    DataAvailability,
    GuardianContext,
    RecentActivitySummary,
)
from app.session_planning import (
    PlanningAnswerValidator,
    SessionPlanningContext,
    build_session_planning_context,
)
from app.session_preferences import SessionPreferenceContext, derive_session_preferences


def history(*messages: str) -> list[ChatTurn]:
    turns: list[ChatTurn] = []
    for message in messages:
        turns.extend(
            (
                ChatTurn(role="user", content=message),
                ChatTurn(
                    role="assistant",
                    content=(
                        "Acknowledged. A Warlock might also be useful, but this is not user input."
                    ),
                ),
            )
        )
    return turns


def test_chill_followup_retains_character_and_time() -> None:
    result = derive_session_preferences(
        history("I have an hour and want story on my Titan."),
        "Actually something chill.",
    )

    assert result.preferences.character == "Titan"
    assert result.preferences.time_minutes == 60
    assert result.preferences.primary_goal == "casual_chill"
    assert result.preferences.intensity == "chill"
    assert result.is_planning_followup is True


def test_character_override_retains_chill_preference() -> None:
    result = derive_session_preferences(
        history("Give me something chill on my Titan."),
        "Hunter instead.",
    )

    assert result.preferences.character == "Hunter"
    assert result.preferences.primary_goal == "casual_chill"
    assert result.preferences.intensity == "chill"


def test_new_explicit_time_replaces_stale_time() -> None:
    result = derive_session_preferences(
        history("I have 30 minutes."),
        "I have another hour now.",
    )

    assert result.preferences.time_minutes == 60
    assert result.preferences.duration_preference == "unspecified"


def test_new_goal_preserves_pvp_exclusion() -> None:
    result = derive_session_preferences(
        history("No PvP."),
        "I want better gear.",
    )

    assert result.preferences.primary_goal == "gear_rewards"
    assert result.preferences.activity_mode == "unspecified"
    assert result.preferences.exclusions == ["pvp"]


def test_relevant_reset_clears_only_story() -> None:
    result = derive_session_preferences(
        history("I have an hour and want story on my Titan."),
        "Forget the story preference.",
    )

    assert result.preferences.primary_goal == "unspecified"
    assert result.preferences.character == "Titan"
    assert result.preferences.time_minutes == 60


def test_current_turn_overrides_character_goal_and_time() -> None:
    result = derive_session_preferences(
        history("I have an hour and want story on my Titan."),
        "Give me a 20-minute gear run on Hunter.",
    )

    assert result.preferences.character == "Hunter"
    assert result.preferences.time_minutes == 20
    assert result.preferences.primary_goal == "gear_rewards"


@pytest.mark.parametrize(
    ("current", "field", "expected"),
    [
        ("something shorter", "duration_preference", "shorter"),
        ("what about story?", "primary_goal", "story_progression"),
        ("solo though", "fireteam", "solo"),
        ("no dungeons", "exclusions", ["dungeons"]),
        ("what if I only have 20 minutes?", "time_minutes", 20),
    ],
)
def test_followup_shorthand_refines_existing_context(
    current: str, field: str, expected: object
) -> None:
    result = derive_session_preferences(history("I have an hour on my Titan."), current)

    assert getattr(result.preferences, field) == expected
    assert result.preferences.character == "Titan"


def test_full_and_field_resets_are_explicit_and_bounded() -> None:
    time_reset = derive_session_preferences(
        history("I have an hour and want story on my Titan."),
        "Forget the time limit.",
    )
    character_reset = derive_session_preferences(
        history("I have an hour and want story on my Titan."),
        "Any character is fine.",
    )
    full_reset = derive_session_preferences(
        history("I have an hour and want story on my Titan."),
        "Start over.",
    )

    assert time_reset.preferences.time_minutes is None
    assert time_reset.preferences.character == "Titan"
    assert character_reset.preferences.character == "unspecified"
    assert character_reset.preferences.time_minutes == 60
    assert full_reset.preferences == SessionPreferenceContext()


def test_assistant_text_never_becomes_a_preference() -> None:
    result = derive_session_preferences(
        [
            ChatTurn(role="user", content="I have 30 minutes on my Titan."),
            ChatTurn(role="assistant", content="Try PvP on your Warlock for two hours."),
        ],
        "Something chill.",
    )

    assert result.preferences.character == "Titan"
    assert result.preferences.time_minutes == 30
    assert result.preferences.activity_mode == "unspecified"


def test_preference_derivation_defensively_bounds_history() -> None:
    turns = [ChatTurn(role="user", content="Use my Titan.")]
    turns.extend(
        ChatTurn(role="assistant", content=f"Bounded assistant turn {index}.")
        for index in range(12)
    )

    result = derive_session_preferences(turns, "Something chill.")

    assert result.preferences.character == "unspecified"
    assert result.preferences.primary_goal == "casual_chill"


def test_known_preferences_are_not_reasked() -> None:
    preferences = SessionPreferenceContext(
        character="Titan",
        time_minutes=45,
        primary_goal="story_progression",
    )
    context = SessionPlanningContext(
        preferences=preferences,
        requested_character_scope="Titan",
        requested_scope_explicit=True,
        character_class="Titan",
        intent_category="story",
        evidence_semantics={},
    )
    answer = "Which character? How much time do you have? Story or loot?"

    codes = {value.code for value in PlanningAnswerValidator().validate(answer, context)}

    assert {
        "reasked_known_character",
        "reasked_known_time",
        "reasked_known_goal",
        "too_many_followups",
    } <= codes


def test_explicit_exclusions_and_chill_intensity_outweigh_convenience() -> None:
    context = SessionPlanningContext(
        preferences=SessionPreferenceContext(
            primary_goal="casual_chill",
            intensity="chill",
            exclusions=["pvp"],
        ),
        requested_character_scope=None,
        requested_scope_explicit=False,
        character_class="Titan",
        intent_category="casual",
        evidence_semantics={},
    )

    codes = {
        value.code
        for value in PlanningAnswerValidator().validate(
            "I recommend Trials because it offers a useful reward.", context
        )
    }

    assert "excluded_pvp" in codes
    assert "chill_constraint" in codes


def test_explicit_answer_duration_cannot_exceed_known_window() -> None:
    context = SessionPlanningContext(
        preferences=SessionPreferenceContext(time_minutes=30),
        requested_character_scope=None,
        requested_scope_explicit=False,
        character_class="Titan",
        intent_category="time_limited",
        evidence_semantics={},
    )

    violations = PlanningAnswerValidator().validate(
        "I recommend the raid; it is a 90-minute activity with useful rewards.", context
    )

    assert "known_time_window_exceeded" in {value.code for value in violations}


@pytest.mark.parametrize(
    ("preferences", "answer", "expected_code"),
    [
        (
            SessionPreferenceContext(activity_mode="pve"),
            "I recommend Crucible because it offers a useful reward.",
            "activity_mode_constraint",
        ),
        (
            SessionPreferenceContext(fireteam="solo"),
            "I recommend a raid with a fireteam because it offers useful rewards.",
            "solo_constraint",
        ),
    ],
)
def test_explicit_style_constraints_are_validated(
    preferences: SessionPreferenceContext, answer: str, expected_code: str
) -> None:
    context = SessionPlanningContext(
        preferences=preferences,
        requested_character_scope=None,
        requested_scope_explicit=False,
        character_class="Titan",
        intent_category="general",
        evidence_semantics={},
    )

    codes = {value.code for value in PlanningAnswerValidator().validate(answer, context)}

    assert expected_code in codes


def test_known_session_preferences_are_not_reasked() -> None:
    context = SessionPlanningContext(
        preferences=SessionPreferenceContext(
            fireteam="either",
            activity_mode="pve",
            intensity="chill",
            duration_preference="shorter",
            exclusions=["raids"],
        ),
        requested_character_scope=None,
        requested_scope_explicit=False,
        character_class="Titan",
        intent_category="casual",
        evidence_semantics={},
    )
    answer = (
        "Do you prefer solo or matchmaking? PvE or PvP? Chill or challenging? "
        "How short should it be? Can you run raids with a fireteam?"
    )

    codes = {value.code for value in PlanningAnswerValidator().validate(answer, context)}

    assert {
        "reasked_known_fireteam",
        "reasked_known_activity_mode",
        "reasked_known_intensity",
        "reasked_known_duration",
        "reasked_excluded_raids",
        "too_many_followups",
    } <= codes


def test_guardian_activity_does_not_create_preferences() -> None:
    guardian = GuardianContext(
        bungie_display_name="Fixture",
        membership_id="fixture",
        membership_type=3,
        platform_name="Fixture",
        characters=[
            CharacterSummary(
                character_id="fixture-titan",
                class_name="Titan",
                race_name="Human",
                gender_name="Female",
                power=0,
                recent_activities=[
                    RecentActivitySummary(
                        activity_hash=1,
                        name="Crucible Control",
                        character_id="fixture-titan",
                    )
                ],
            )
        ],
        data_availability=DataAvailability(fetched_at=datetime(2026, 9, 22, tzinfo=UTC)),
    )
    preferences = derive_session_preferences([], "What should I do next?").preferences

    planning = build_session_planning_context(
        guardian,
        "What should I do next?",
        "general",
        None,
        preferences,
    )

    assert planning.preferences.activity_mode == "unspecified"
    assert planning.preferences.primary_goal == "unspecified"
    assert planning.preferences.exclusions == []
