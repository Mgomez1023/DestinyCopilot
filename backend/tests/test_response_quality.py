import pytest

from app.response_quality import (
    BuildResponseValidationContext,
    ReadOnlyActionValidationContext,
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


@pytest.mark.parametrize(
    "message",
    [
        "What should I use if I'm jumping into Iron Banner?",
        "What gun should I run?",
        "Find me a good hand cannon that I own.",
        "What hand cannon that I own should I use?",
        "Recommend a shotgun that I own.",
        "Check my rolls.",
        "What should I equip for PvP?",
    ],
)
def test_natural_personalized_gear_prompts_use_build_response_mode(message: str) -> None:
    assert classify_response_mode(message, session_planning=False) == "build_advice"


@pytest.mark.parametrize(
    "message",
    [
        "Is Thorn good?",
        "What does Explosive Payload do?",
        "What hand cannons are good this season?",
    ],
)
def test_public_weapon_questions_do_not_use_build_response_mode(message: str) -> None:
    assert classify_response_mode(message, session_planning=False) == "direct_fact"


def test_focused_gear_guidance_is_shorter_than_full_build_review() -> None:
    focused = response_mode_context(
        "Anything better in my vault?",
        session_planning=False,
        continuation=True,
        build_request=True,
        focused_build=True,
    )
    detailed = response_mode_context(
        "Review my whole Titan build in detail.",
        session_planning=False,
        continuation=False,
        build_request=True,
        focused_build=False,
    )

    assert focused.mode == "build_advice"
    assert focused.focused_build is True
    assert (focused.target_min_words, focused.target_max_words) == (50, 120)
    assert focused.detail_level == "concise"
    assert "item choice and one or two grounded reasons" in response_mode_instruction(focused)
    assert detailed.focused_build is False
    assert (detailed.target_min_words, detailed.target_max_words) == (100, 220)
    assert detailed.detail_level == "detailed"


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
        ("Manifest match: that is the closest dungeon I found.", "internal_jargon"),
        ("It resolved to a canonical entity called Ibex.", "internal_jargon"),
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


def test_personalized_build_validation_requires_only_relevant_guardian_evidence() -> None:
    current_only = BuildResponseValidationContext(
        is_build_request=True,
        requires_current_build_analysis=True,
    )
    owned_alternatives = BuildResponseValidationContext(
        is_build_request=True,
        requires_current_build_analysis=True,
        requires_owned_inventory=True,
    )
    mode = response_mode_context(
        "Anything better in my vault?",
        session_planning=False,
        continuation=False,
        build_request=True,
        focused_build=True,
    )

    current_codes = {
        value.code
        for value in ResponseQualityValidator().validate(
            "Your current build looks good.",
            mode,
            "How does my current build look?",
            None,
            current_only,
        )
    }
    owned_codes = {
        value.code
        for value in ResponseQualityValidator().validate(
            "Use a better hand cannon from your vault.",
            mode,
            "Anything better in my vault?",
            None,
            owned_alternatives,
        )
    }

    assert current_codes == {"missing_current_build_analysis"}
    assert owned_codes == {
        "missing_current_build_analysis",
        "missing_owned_inventory_evidence",
    }

    current_only.current_build_analysis_used = True
    verified_codes = {
        value.code
        for value in ResponseQualityValidator().validate(
            "Your current build has a clear close-range focus.",
            mode,
            "How does my current build look?",
            None,
            current_only,
        )
    }
    assert "missing_current_build_analysis" not in verified_codes
    assert "missing_owned_inventory_evidence" not in verified_codes


def test_safe_build_answer_is_specific_to_missing_owned_evidence() -> None:
    answer = ResponseQualityValidator.safe_build_answer(
        BuildResponseValidationContext(
            is_build_request=True,
            requires_current_build_analysis=True,
            requires_owned_inventory=True,
        )
    )

    assert "grounded shortlist of your owned gear" in answer
    assert "don't want to guess" in answer
    assert "objectives" not in answer


def test_authenticated_build_failure_rejects_false_sign_in_guidance() -> None:
    context = BuildResponseValidationContext(
        is_build_request=True,
        requires_current_build_analysis=True,
    )
    mode = response_mode_context(
        "Check my Titan build.",
        session_planning=False,
        continuation=False,
        build_request=True,
    )

    violations = ResponseQualityValidator().validate(
        "I can't access your Titan. Sign in again or link your Bungie account.",
        mode,
        "Check my Titan build.",
        None,
        context,
    )

    assert "false_authentication_guidance" in {value.code for value in violations}


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


def test_resolved_read_only_lookup_rejects_permission_loop() -> None:
    message = "yes those are what I mean"
    mode = response_mode_context(message, session_planning=False, continuation=True)
    context = ReadOnlyActionValidationContext(lookups_performed=True)

    violations = ResponseQualityValidator().validate(
        "Would you like me to scan your Title progress now?",
        mode,
        message,
        None,
        None,
        context,
    )

    assert "unnecessary_read_only_permission" in {value.code for value in violations}


def test_unresolved_title_clarification_is_not_blocked_as_permission_loop() -> None:
    message = "What is the easiest title for me?"
    mode = response_mode_context(message, session_planning=False, continuation=False)
    context = ReadOnlyActionValidationContext()

    violations = ResponseQualityValidator().validate(
        "Do you mean a Destiny Triumph Title or something else?",
        mode,
        message,
        None,
        None,
        context,
    )

    assert "unnecessary_read_only_permission" not in {value.code for value in violations}


# --- Capability and initiative detectors (playtest follow-up) -------------------------------


def _codes(answer: str, **read_only: bool) -> set[str]:
    message = "Question?"
    mode = response_mode_context(message, session_planning=False, continuation=False)
    context = ReadOnlyActionValidationContext(**read_only)
    return {
        value.code
        for value in ResponseQualityValidator().validate(answer, mode, message, None, None, context)
    }


LOOKED_UP = {"lookups_performed": True, "guardian_lookup_performed": True}


@pytest.mark.parametrize(
    "answer",
    [
        "I can check official live data or community reports. Which do you prefer?",
        "Want me to search your vault for hand cannons?",
        "Should I look up the current drop table?",
        "If you'd like, I can compare your two Rose rolls.",
        "Let me know if you want me to verify that with Bungie's notes.",
    ],
)
def test_read_only_offers_are_flagged(answer: str) -> None:
    assert "unnecessary_read_only_permission" in _codes(answer, **LOOKED_UP)


@pytest.mark.parametrize(
    "answer",
    [
        "Which quest should I look up for you?",
        "Run the Nightfall. Are you chasing a weapon or an armor upgrade?",
        "Do you mean a Destiny Triumph Title or something else?",
        "I checked your vault: Fatebringer is there.",
    ],
)
def test_ordinary_questions_and_answers_are_not_read_only_offers(answer: str) -> None:
    assert "unnecessary_read_only_permission" not in _codes(answer, **LOOKED_UP)


def test_lookup_claims_and_unperformed_research_need_a_real_lookup() -> None:
    claim = "I checked Bungie's site and The Coil still drops it."
    assert "unperformed_lookup" in _codes(claim)
    assert "unperformed_lookup" not in _codes(claim, lookups_performed=True)
    assert "unperformed_lookup" in _codes("Yes, it drops there.", research_requested=True)


def test_asking_for_account_data_is_flagged_only_before_a_guardian_lookup() -> None:
    answer = "Type the fragments you have and I'll tell you which are missing."
    assert "account_data_requested" in _codes(answer)
    assert "account_data_requested" not in _codes(answer, guardian_lookup_performed=True)
    assert "account_data_requested" not in _codes(
        "Open the Quests tab and list the steps in your Director."
    )


@pytest.mark.parametrize(
    "answer",
    [
        "Upload a screenshot of your fragment screen.",
        "Send me a picture of your vault.",
        "Want me to move Rose to your vault?",
        "I can equip Sunshot for you if you don't mind.",
        "I'll swap the barrel on Sunshot.",
    ],
)
def test_unsupported_capability_offers_are_flagged(answer: str) -> None:
    assert "unsupported_capability" in _codes(answer, **LOOKED_UP)


@pytest.mark.parametrize(
    "answer",
    [
        "I can't view screenshots or move items, but here's how to do it in game.",
        "You can't send images in this chat, so describe the perk instead.",
        "Want me to move on to your Hunter?",
        "I'll lock in your solo preference for the rest of the session.",
        "Let me pull up your loadout.",
    ],
)
def test_denials_and_unrelated_phrases_are_not_capability_offers(answer: str) -> None:
    assert "unsupported_capability" not in _codes(answer, **LOOKED_UP)


def test_claiming_a_completed_item_change_is_flagged_on_any_turn() -> None:
    assert "implied_bungie_write" in _codes("Done! I've moved Rose to your vault.", **LOOKED_UP)
    assert "implied_bungie_write" not in _codes("You've equipped Rose already.", **LOOKED_UP)
    assert "implied_bungie_write" not in _codes("I've locked in your solo plan.", **LOOKED_UP)


def test_unsupported_capability_sentences_are_removed_and_denials_kept() -> None:
    from app.response_quality import (
        UNSUPPORTED_CAPABILITY_FALLBACK,
        remove_unsupported_capability_offers,
    )

    assert (
        remove_unsupported_capability_offers(
            "Rose is in your kinetic slot. Want me to move it to your vault?\n- Next: Vanguard Ops."
        )
        == "Rose is in your kinetic slot.\n- Next: Vanguard Ops."
    )
    honest = "I can't move items. In game, hover Rose and transfer it."
    assert remove_unsupported_capability_offers(honest) == honest
    assert (
        remove_unsupported_capability_offers("Upload a screenshot.")
        == UNSUPPORTED_CAPABILITY_FALLBACK
    )


def _drop(answer: str) -> str | None:
    return ResponseQualityValidator.trim_trailing_handback(answer, offers=True, choices=False)


def test_a_closing_lookup_offer_is_dropped_only_when_the_rest_stands_alone() -> None:
    drop = _drop
    assert (
        drop("Mint Retrograde drops from The Coil. Want me to check if you own one?")
        == "Mint Retrograde drops from The Coil."
    )
    assert drop("Would you like me to scan your Title progress now?") is None
    assert drop("Want me to check your vault? Rose is equipped.") is None


@pytest.mark.parametrize(
    ("answer", "flagged", "kept"),
    [
        # In-game advice to take a screenshot is not a request to send one to Copilot.
        ("Take a screenshot of the symbols so you remember the order.", False, None),
        # A denial in an earlier clause does not excuse the request that follows it.
        (
            "I can't see your unlocked fragments, so send me a screenshot of the fragment screen.",
            True,
            "I can't see your unlocked fragments.",
        ),
        # Information before an offer survives; only the offering clause goes.
        ("Rose is in your vault, and I can equip it for you.", True, "Rose is in your vault."),
    ],
)
def test_capability_checks_work_per_clause(answer: str, flagged: bool, kept: str | None) -> None:
    from app.response_quality import remove_unsupported_capability_offers

    assert ("unsupported_capability" in _codes(answer, **LOOKED_UP)) is flagged
    assert remove_unsupported_capability_offers(answer) == (kept or answer)


def test_a_closing_sentence_with_information_or_a_clarification_is_never_dropped() -> None:
    drop = _drop
    # Only the offering clause goes; the information before it stays.
    assert (
        drop("Rose is equipped. Rose has Rampage, and I can compare it if you'd like.")
        == "Rose is equipped. Rose has Rampage."
    )
    assert drop("I see three active quests. Which quest do you want me to look up?") is None
    assert drop("Rose is equipped. Do you want me to compare its roll?") == "Rose is equipped."


def test_putting_a_wanted_item_on_is_not_replacing_it() -> None:
    from app.build_analysis import replaces_item

    assert not replaces_item("Swap Rose into your kinetic slot for this.", "Rose")
    assert not replaces_item("Swap Rose back in once you have Rampage.", "Rose")
    assert replaces_item("Swap out Rose for Fatebringer.", "Rose")
    assert replaces_item("Use Fatebringer instead of Rose.", "Rose")
