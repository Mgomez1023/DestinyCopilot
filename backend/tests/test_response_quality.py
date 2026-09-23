import pytest

from app.response_quality import (
    BuildResponseValidationContext,
    ResponseQualityValidator,
    classify_response_mode,
    is_account_fact_only,
    response_mode_context,
    response_mode_instruction,
)


@pytest.mark.parametrize(
    ("message", "planning", "expected"),
    [
        ("What weapons do I have equipped?", False, "direct_fact"),
        ("What should I do next?", True, "recommendation"),
        ("Given my Titan, Final Shape or Edge?", True, "comparison"),
        ("How do I get Wish-Ender?", False, "walkthrough"),
        ("Help improve my Solar build.", False, "build_advice"),
        ("Give me an overview of my Guardian.", False, "account_summary"),
        ("Why is Bungie login not working?", False, "troubleshooting"),
    ],
)
def test_response_mode_classification(message: str, planning: bool, expected: str) -> None:
    assert classify_response_mode(message, session_planning=planning) == expected


def test_only_pure_account_fact_questions_suppress_external_research() -> None:
    assert is_account_fact_only("What weapons do I have equipped?") is True
    assert is_account_fact_only("What quests are active?") is True
    assert is_account_fact_only("Do I own Sunshot, and is it good in the current meta?") is False
    assert is_account_fact_only("Do I own a better heavy weapon?") is False
    assert is_account_fact_only("Have I started Edge of Fate, and what does it unlock?") is False


def test_mode_guidance_uses_approximate_ranges_without_global_limit() -> None:
    recommendation = response_mode_context(
        "What should I do next?", session_planning=True, continuation=False
    )
    walkthrough = response_mode_context(
        "Give me a detailed walkthrough.", session_planning=False, continuation=False
    )

    recommendation_instruction = response_mode_instruction(recommendation)
    walkthrough_instruction = response_mode_instruction(walkthrough)

    assert recommendation.target_min_words == 80
    assert recommendation.target_max_words == 180
    assert "guidance, not a truncation rule" in recommendation_instruction
    assert walkthrough.target_max_words is None
    assert "no fixed word target" in walkthrough_instruction


def test_moderately_detailed_recommendation_is_not_rejected() -> None:
    mode = response_mode_context(
        "I want better loot specifically on my Titan.",
        session_planning=True,
        continuation=False,
    )
    answer = (
        "Run matchmade Nightfalls on your Titan first. They give you a focused, repeatable route "
        "toward useful drops without requiring a raid team, and your current Vanguard objective "
        "can progress alongside them. Check each reward before dismantling it, prioritizing useful "
        "weapon rolls and armor that improves the stats your build needs. As a backup, use visible "
        "matchmade activities that overlap with an active quest, but don't choose one solely "
        "because it is tracked. Are you chasing a weapon or an armor upgrade?"
    )

    violations = ResponseQualityValidator().validate(answer, mode, "I want better loot.", None)

    assert violations == []


def test_unrequested_minute_by_minute_itinerary_is_rejected() -> None:
    mode = response_mode_context(
        "I have 30 minutes, what should I do on my Titan?",
        session_planning=True,
        continuation=False,
    )
    answer = "Spend 5 minutes prepping, then play for 20 minutes, then use 5 minutes to clean up."

    codes = {
        value.code
        for value in ResponseQualityValidator().validate(answer, mode, "I have 30 minutes.", None)
    }

    assert "unrequested_timed_itinerary" in codes


def test_explicit_timed_step_by_step_request_allows_timed_structure() -> None:
    message = "Give me a detailed step-by-step 30-minute plan."
    mode = response_mode_context(message, session_planning=True, continuation=False)
    answer = "Prep for 5 minutes, play the activity for 20 minutes, then spend 5 minutes reviewing."

    codes = {
        value.code for value in ResponseQualityValidator().validate(answer, mode, message, None)
    }

    assert mode.timed_itinerary_requested is True
    assert "unrequested_timed_itinerary" not in codes


@pytest.mark.parametrize(
    ("answer", "expected_code"),
    [
        ("Based on the available data, run Nightfall next.", "answer_not_first"),
        ("The Guardian tool returned your API response.", "internal_jargon"),
        (
            "Run Nightfall because it fits your goal. Run Nightfall because it fits your goal.",
            "repeated_content",
        ),
    ],
)
def test_clear_response_quality_violations_are_detected(answer: str, expected_code: str) -> None:
    message = "What should I do next?"
    mode = response_mode_context(message, session_planning=True, continuation=False)

    codes = {
        value.code for value in ResponseQualityValidator().validate(answer, mode, message, None)
    }

    assert expected_code in codes


def test_followup_should_adapt_instead_of_restarting_template() -> None:
    message = "I don't want raids. Solo or matchmaking activities preferably."
    mode = response_mode_context(message, session_planning=True, continuation=True)

    violations = ResponseQualityValidator().validate(
        "Goal: get better gear. Primary plan: run Nightfalls.", mode, message, None
    )

    assert "followup_restarts_plan" in {value.code for value in violations}


def test_extreme_verbosity_is_bounded_without_rejecting_moderate_detail() -> None:
    message = "What should I do next?"
    mode = response_mode_context(message, session_planning=True, continuation=False)
    answer = " ".join(["useful"] * 321)

    violations = ResponseQualityValidator().validate(answer, mode, message, None)

    assert "extreme_verbosity" in {value.code for value in violations}


@pytest.mark.parametrize(
    ("answer", "context", "expected"),
    [
        (
            "You already own Apex Predator, so equip it.",
            BuildResponseValidationContext(is_build_request=True),
            "unsupported_ownership_claim",
        ),
        (
            "Replace Sunshot with a pulse rifle.",
            BuildResponseValidationContext(
                is_build_request=True,
                locked_item_names=["Sunshot"],
            ),
            "locked_item_replaced",
        ),
        (
            "This is currently the best DPS meta loadout.",
            BuildResponseValidationContext(is_build_request=True),
            "ungrounded_current_build_claim",
        ),
        (
            "57 Resilience is bad and you need 100.",
            BuildResponseValidationContext(is_build_request=True),
            "unsupported_stat_threshold",
        ),
        (
            "Use character_id 123456789012345678 for this build.",
            BuildResponseValidationContext(is_build_request=True),
            "raw_build_identifier",
        ),
        (
            "Build score: 82/100. This is A Tier.",
            BuildResponseValidationContext(is_build_request=True),
            "fake_build_score",
        ),
        (
            "I've equipped Apex Predator and applied your mods.",
            BuildResponseValidationContext(is_build_request=True),
            "implied_bungie_write",
        ),
    ],
)
def test_build_quality_failures_are_detected(
    answer: str,
    context: BuildResponseValidationContext,
    expected: str,
) -> None:
    message = "Improve my build."
    mode = response_mode_context(message, session_planning=False, continuation=False)

    violations = ResponseQualityValidator().validate(answer, mode, message, None, context)

    assert expected in {value.code for value in violations}


def test_normal_build_advice_is_limited_to_three_changes_but_detailed_rebuild_is_not() -> None:
    answer = "\n".join(
        [
            "Keep Sunshot.",
            "- Swap your Heavy.",
            "- Replace one Fragment.",
            "- Change one armor mod.",
            "- Equip a different Special weapon.",
        ]
    )
    context = BuildResponseValidationContext(is_build_request=True, normal_change_limit=3)
    normal = response_mode_context("Improve my build.", session_planning=False, continuation=False)
    detailed = response_mode_context(
        "Give me a complete detailed build.", session_planning=False, continuation=False
    )

    normal_codes = {
        value.code
        for value in ResponseQualityValidator().validate(
            answer, normal, "Improve my build.", None, context
        )
    }
    detailed_codes = {
        value.code
        for value in ResponseQualityValidator().validate(
            answer, detailed, "Give me a complete detailed build.", None, context
        )
    }

    assert "too_many_build_changes" in normal_codes
    assert "too_many_build_changes" not in detailed_codes


def test_preserved_exotic_blocks_a_second_exotic_of_the_same_kind() -> None:
    message = "Build around Sunshot."
    mode = response_mode_context(message, session_planning=False, continuation=False)
    context = BuildResponseValidationContext(
        is_build_request=True,
        inventory_ownership_checked=True,
        locked_item_names=["Sunshot"],
        equipped_exotic_weapons=["Sunshot"],
        owned_exotic_weapons=["Sunshot", "Gjallarhorn"],
    )

    violations = ResponseQualityValidator().validate(
        "Keep Sunshot and equip Gjallarhorn for your Heavy.", mode, message, None, context
    )

    assert "incompatible_exotic_recommendation" in {value.code for value in violations}
