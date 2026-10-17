"""Behavioral property helpers and scoring: each property fires on a violating answer and
stays quiet on a correct one (CI-safe, no model calls)."""

import json
from dataclasses import replace
from typing import Any

import pytest

from evals.judge import SCHEMA, evidence
from evals.model import Expectations, Scenario
from evals.profiles import PROFILES, profile_facts
from evals.properties import evaluate_properties
from evals.report import record, summary_markdown
from evals.rubric import DIMENSIONS, MAX_SCORE, aggregate, score_card
from evals.runtime import Observation, ToolCallRecord, TurnRecord


def observe(
    answer: str,
    *,
    profile: str = "titan_veteran",
    prompt: str = "Question?",
    tools: list[ToolCallRecord] | None = None,
    trace: dict[str, Any] | None = None,
    error: str | None = None,
    **expect: Any,
) -> tuple[Observation, Any]:
    scenario = Scenario("test", profile, (prompt,), Expectations(**expect))
    context = PROFILES[profile]()
    turn = TurnRecord(
        message=prompt,
        answer=answer,
        source="openai",
        trace=trace or {"grounding": {}},
        tool_calls=tools or [],
        error=error,
    )
    return Observation(scenario, context, [turn]), profile_facts(context)


def result(answer: str, name: str, **kwargs: Any) -> Any:
    obs, facts = observe(answer, **kwargs)
    return next(value for value in evaluate_properties(obs, facts) if value.name == name)


def assert_flag(answer: str, name: str, flag: str | None, **kwargs: Any) -> None:
    value = result(answer, name, **kwargs)
    assert value.applies and not value.passed, value
    assert value.flag == flag, value


def assert_pass(answer: str, name: str, **kwargs: Any) -> None:
    value = result(answer, name, **kwargs)
    assert value.passed, value


# --- Ownership and class ----------------------------------------------------------------


def test_claiming_an_unowned_item_is_an_ownership_hallucination() -> None:
    assert_flag(
        "Equip your Necrotic Grip for this.",
        "never_claims_unowned_items",
        "ownership_hallucination",
    )
    assert_flag(
        "You already have Witherhoard in your vault.",
        "never_claims_unowned_items",
        "ownership_hallucination",
    )
    assert_pass("Equip your Synthoceps and Rose.", "never_claims_unowned_items")
    assert_pass("You don't own Witherhoard; farm it later.", "never_claims_unowned_items")


def test_only_owned_flags_unowned_recommendations() -> None:
    kwargs = {"only_owned": True}
    assert_flag(
        "Run Witherhoard in the kinetic slot.",
        "only_owned_recommendations",
        "ownership_hallucination",
        **kwargs,
    )
    assert_pass("Run Rose with Apex Predator.", "only_owned_recommendations", **kwargs)
    assert_pass(
        "Witherhoard is great, but you don't own it; chase it later.",
        "only_owned_recommendations",
        **kwargs,
    )


def test_equipped_only_and_user_said_not_owned() -> None:
    assert_flag(
        "Swap to Fatebringer.",
        "only_owned_recommendations",
        "ownership_hallucination",
        equipped_only=True,
    )
    assert_flag(
        "Use Gjallarhorn for bosses.",
        "only_owned_recommendations",
        "ownership_hallucination",
        profile="new_player",
        not_owned=("Gjallarhorn",),
    )


def test_wrong_class_armor_is_flagged_unless_marked_unusable() -> None:
    kwargs = {"character": "Titan"}
    assert_flag(
        "Equip Celestial Nighthawk for big damage.",
        "never_recommends_wrong_class_armor",
        "wrong_character",
        **kwargs,
    )
    assert_pass(
        "Celestial Nighthawk can't be used on a Titan.",
        "never_recommends_wrong_class_armor",
        **kwargs,
    )
    assert_pass(
        "Celestial Nighthawk is Hunter-only, so keep Synthoceps.",
        "never_recommends_wrong_class_armor",
        **kwargs,
    )


def test_character_scope_uses_tool_arguments_and_missing_classes() -> None:
    wrong_tool = [ToolCallRecord("analyze_current_build", {"character_class": "Titan"}, {})]
    assert_flag(
        "Here is your plan.",
        "uses_requested_character",
        "wrong_character",
        character="Hunter",
        tools=wrong_tool,
    )
    assert_pass(
        "On your Hunter, run the Nightfall.",
        "uses_requested_character",
        character="Hunter",
        tools=[ToolCallRecord("get_active_quests", {"character_class": "Hunter"}, {})],
    )
    assert_flag(
        "Run Gambit on Hunter.",
        "uses_requested_character",
        "wrong_character",
        profile="missing_class",
        missing_class="Hunter",
    )
    assert_pass(
        "You don't have a Hunter character on this account.",
        "uses_requested_character",
        profile="missing_class",
        missing_class="Hunter",
    )


def test_inactive_quest_claims() -> None:
    kwargs = {"inactive_quests": ("A Single Thread",)}
    assert_flag(
        "Finish A Single Thread tonight.",
        "no_unsupported_active_quest_claims",
        "unsupported_active_quest",
        **kwargs,
    )
    assert_pass(
        "A Single Thread isn't active on your Titan, so start with Edge of Fate.",
        "no_unsupported_active_quest_claims",
        **kwargs,
    )


# --- Knowledge --------------------------------------------------------------------------


def test_ambiguity_must_name_candidates_or_ask() -> None:
    kwargs = {"ambiguity": "altars"}
    assert_flag(
        "Altars of Sorrow is on the Moon.",
        "ambiguity_not_silently_resolved",
        "ambiguity_silently_resolved",
        **kwargs,
    )
    assert_pass(
        "It can mean Altars of Sorrow or the Altar of Reflection.",
        "ambiguity_not_silently_resolved",
        **kwargs,
    )
    assert_pass(
        "Do you mean Altars of Sorrow on the Moon? Which one?",
        "ambiguity_not_silently_resolved",
        **kwargs,
    )


def test_location_conflicts() -> None:
    kwargs = {"grounded_entities": ("Altars of Sorrow",)}
    assert_flag(
        "Altars of Sorrow is located on Nessus.",
        "no_location_conflicts",
        "wrong_destination",
        **kwargs,
    )
    assert_pass(
        "Altars of Sorrow is in Sorrow's Harbor on the Moon.", "no_location_conflicts", **kwargs
    )
    traced = {"grounding": {}, "core_knowledge": {"destination_conflicts": [{"entity_id": "x"}]}}
    assert_flag("Anything.", "no_location_conflicts", "wrong_destination", trace=traced)


def test_unknown_fields_must_not_be_asserted() -> None:
    kwargs = {"unknown_fields": (("Altars of Sorrow", "matchmaking"),)}
    assert_flag(
        "Altars of Sorrow is not matchmade.",
        "unknown_fields_not_filled_from_memory",
        "hallucinated_fact",
        **kwargs,
    )
    assert_pass(
        "Whether Altars of Sorrow is matchmade isn't established; Haunted Altars of Sorrow is "
        "matchmade.",
        "unknown_fields_not_filled_from_memory",
        **kwargs,
    )


def test_live_claims_need_evidence_or_hedging() -> None:
    assert_flag(
        "This week's Nightfall is The Lightblade.",
        "live_claims_need_live_evidence",
        "stale_live_claim",
        live_sensitive=True,
    )
    assert_pass(
        "I can't confirm this week's Nightfall without live data.",
        "live_claims_need_live_evidence",
        live_sensitive=True,
    )
    assert_pass(
        "It is The Lightblade.",
        "live_claims_need_live_evidence",
        live_sensitive=True,
        trace={"grounding": {"live_provider": True}},
    )


def test_fuzzy_manifest_match_presented_as_exact() -> None:
    fuzzy = [
        ToolCallRecord(
            "get_activity_details",
            {},
            {
                "resolution": {
                    "exact": False,
                    "query": "Altars of Sorrow",
                    "matched_name": "Haunted Altars of Sorrow",
                }
            },
        )
    ]
    assert_flag(
        "Haunted Altars of Sorrow is a matchmade activity.",
        "fuzzy_match_not_presented_as_exact",
        "hallucinated_fact",
        tools=fuzzy,
    )
    assert_pass(
        "Haunted Altars of Sorrow is a different, seasonal variant.",
        "fuzzy_match_not_presented_as_exact",
        tools=fuzzy,
    )
    typo = [
        ToolCallRecord(
            "get_item_details",
            {},
            {"resolution": {"exact": False, "query": "Gjallarhrn", "matched_name": "Gjallarhorn"}},
        )
    ]
    assert not result(
        "Gjallarhorn is a rocket launcher.", "fuzzy_match_not_presented_as_exact", tools=typo
    ).applies


# --- Loop health and quality ------------------------------------------------------------


def test_loop_failures_and_false_refusals() -> None:
    assert_flag("", "tool_loop_completed", "tool_loop_failed", error="RuntimeError: x")
    assert_flag(
        "Safe answer.",
        "no_false_refusal",
        "false_refusal",
        trace={"grounding": {}, "planning_correction": {"fallback": "build_grounding"}},
    )
    assert_flag(
        "I couldn't retrieve your build.",
        "no_false_refusal",
        "false_refusal",
        tools=[ToolCallRecord("analyze_current_build", {}, {"character": {}})],
    )


def test_response_quality_properties() -> None:
    long_answer = " ".join(["word"] * 200) + "."
    assert_flag(long_answer, "concise", "excessive_verbosity")
    assert result(" ".join(["word"] * 150) + ".", "concise").flag is None
    assert not result("Sure! It is on the Moon.", "answer_first").passed
    assert not result("Great question. It is on the Moon.", "no_filler").passed
    assert not result("A? B? C?", "limited_follow_up_questions").passed
    assert result(
        "Which one do you mean? Sorrow or Summoning?",
        "limited_follow_up_questions",
        question_allowed=True,
    ).passed
    assert not result(
        "Where is Altars of Sorrow?", "does_not_repeat_prompt", prompt="Where is Altars of Sorrow?"
    ).passed
    plan = "\n".join(f"{index}. Option" for index in range(1, 7))
    assert not result(plan, "one_primary_and_brief_backup", style="planning").passed


# --- Scoring and reporting --------------------------------------------------------------


def test_serious_flags_stay_visible_even_with_a_perfect_judge_score() -> None:
    obs, facts = observe("Equip your Witherhoard.", only_owned=True)
    results = evaluate_properties(obs, facts)
    perfect = {
        "scores": dict.fromkeys(DIMENSIONS, 2),
        "flags": [],
        "grounded_claims": [],
        "ungrounded_claims": [],
        "notes": "",
    }

    card = score_card(results, obs.scenario.expect, obs.final.answer, facts, perfect)

    assert card.total == MAX_SCORE
    assert "ownership_hallucination" in card.flags
    assert card.passed is False
    assert aggregate([card])["passed_without_flags"] == 0


def test_deterministic_scores_cover_every_dimension_and_penalize_flags() -> None:
    good_obs, facts = observe("Altars of Sorrow is on the Moon.", mentions_any=("Moon",))
    bad_obs, _ = observe("Altars of Sorrow is on Nessus.", grounded_entities=("Altars of Sorrow",))

    good = score_card(
        evaluate_properties(good_obs, facts), good_obs.scenario.expect, good_obs.final.answer, facts
    )
    bad = score_card(
        evaluate_properties(bad_obs, facts), bad_obs.scenario.expect, bad_obs.final.answer, facts
    )

    assert set(good.scores) == set(DIMENSIONS)
    assert good.total == MAX_SCORE and good.passed
    assert bad.scores["factual_correctness"] == 0 and "wrong_destination" in bad.flags


def test_judge_evidence_carries_ground_truth_and_schema_is_strict() -> None:
    obs, facts = observe("Run Rose.", tools=[ToolCallRecord("search_inventory", {}, {"items": []})])
    payload = json.loads(evidence(obs, facts))

    assert payload["account_ground_truth"]["owned_items"]["Rose"]["class"] == "Any"
    assert payload["tool_results_this_turn"][0]["tool"] == "search_inventory"
    assert SCHEMA["additionalProperties"] is False
    assert set(SCHEMA["properties"]["scores"]["required"]) == set(DIMENSIONS)


def test_records_and_summary_are_serializable() -> None:
    obs, facts = observe("Altars of Sorrow is on the Moon.", mentions_any=("Moon",))
    results = evaluate_properties(obs, facts)
    card = score_card(results, obs.scenario.expect, obs.final.answer, facts)
    value = record(obs, results, card)
    meta = {
        "subset": "test",
        "model": "m",
        "judge": False,
        "live_provider": False,
        "started_at": "now",
    }

    json.dumps(value)
    text = summary_markdown(meta, [value], [card])
    assert "| test | 16 |" in text


@pytest.mark.parametrize("style", ["fact", "planning", "build", "explain"])
def test_every_style_has_a_word_limit(style: str) -> None:
    obs, facts = observe("Short.", style=style)
    assert replace(obs.scenario.expect).style == style
    assert result("Short.", "concise", style=style).passed


def test_live_sensitive_rewards_must_not_be_asserted() -> None:
    kwargs = {"unknown_fields": (("Altars of Sorrow", "rewards"),)}
    assert_flag(
        "Altars of Sorrow grants XP and Exotic engrams.",
        "unknown_fields_not_filled_from_memory",
        "invented_reward_or_mechanic",
        **kwargs,
    )
    assert_pass(
        "Current Altars of Sorrow rewards aren't established, so check in-game.",
        "unknown_fields_not_filled_from_memory",
        **kwargs,
    )


def test_planning_context_counts_as_guardian_evidence_only_for_planning() -> None:
    planning_trace = {"grounding": {}, "planning_context": {"included": True}}
    assert_pass(
        "Run Vanguard Ops.",
        "calls_expected_evidence_tool",
        style="planning",
        tools_any=("get_active_quests",),
        trace=planning_trace,
    )
    assert not result(
        "Swap your rocket.",
        "calls_expected_evidence_tool",
        style="build",
        tools_any=("analyze_current_build",),
        trace=planning_trace,
    ).passed


def test_missing_class_phrasings_and_clarifying_questions_are_not_false_positives() -> None:
    assert_pass(
        "I couldn't find a Hunter on your account.",
        "uses_requested_character",
        profile="missing_class",
        missing_class="Hunter",
    )
    assert_pass(
        'Do you mean the quest "A Single Thread," and do you want to finish it for the story?',
        "no_unsupported_active_quest_claims",
        inactive_quests=("A Single Thread",),
    )


def test_judge_only_flags_are_reported_separately() -> None:
    obs, facts = observe("Altars of Sorrow is on the Moon.", mentions_any=("Moon",))
    verdict = {
        "scores": dict.fromkeys(DIMENSIONS, 2),
        "flags": ["ambiguity_silently_resolved"],
        "grounded_claims": [],
        "ungrounded_claims": [],
        "notes": "",
    }
    card = score_card(
        evaluate_properties(obs, facts), obs.scenario.expect, obs.final.answer, facts, verdict
    )

    assert card.deterministic_flags == []
    assert card.judge_only_flags == ["ambiguity_silently_resolved"]
    totals = aggregate([card])
    assert totals["passed_without_deterministic_flags"] == 1
    assert totals["flag_counts_judge_only"] == {"ambiguity_silently_resolved": 1}


def conversation(
    answers: list[str],
    scopes: list[str],
    *,
    wanted: tuple[str | None, ...],
    tools: list[list[ToolCallRecord]] | None = None,
    **expect: Any,
) -> Any:
    scenario = Scenario(
        "convo",
        "titan_veteran",
        tuple(f"turn {index}" for index in range(len(answers))),
        Expectations(turn_characters=wanted, **expect),
    )
    turns = [
        TurnRecord(
            message=f"turn {index}",
            answer=answer,
            source="openai",
            trace={"grounding": {}, "conversation_scope": {"active": scope}},
            tool_calls=(tools or [[] for _ in answers])[index],
        )
        for index, (answer, scope) in enumerate(zip(answers, scopes, strict=True))
    ]
    context = PROFILES["titan_veteran"]()
    obs = Observation(scenario, context, turns)
    results = evaluate_properties(obs, profile_facts(context))
    return next(value for value in results if value.name == "conversation_scope_isolated")


def test_conversation_isolation_accepts_a_clean_switch() -> None:
    result = conversation(
        ["Run the Kepler Uplink on your Titan.", "Run Gambit on your Hunter."],
        ["Titan", "Hunter"],
        wanted=("Titan", "Hunter"),
        leak_terms=("Kepler",),
    )

    assert result.applies and result.passed, result


def test_conversation_isolation_flags_a_previous_character_leak() -> None:
    result = conversation(
        ["Do Kepler Uplink on your Titan.", "Finish the Kepler Uplink mission next."],
        ["Titan", "Hunter"],
        wanted=("Titan", "Hunter"),
        leak_terms=("Kepler",),
    )

    assert not result.passed and result.flag == "cross_character_leak"


def test_conversation_isolation_flags_wrong_scope_and_wrong_tool_character() -> None:
    wrong_scope = conversation(
        ["Hi.", "Run Gambit."], ["Titan", "Titan"], wanted=("Titan", "Hunter")
    )
    wrong_tool = conversation(
        ["Hi.", "Run Gambit."],
        ["Titan", "Hunter"],
        wanted=("Titan", "Hunter"),
        tools=[[], [ToolCallRecord("get_active_quests", {"character_class": "Titan"}, {})]],
    )

    assert wrong_scope.flag == "wrong_character" and wrong_tool.flag == "wrong_character"


def test_conversation_isolation_checks_equipped_claims_against_the_active_character() -> None:
    leaked = conversation(
        ["You have Synthoceps equipped, so keep your Health up."],
        ["Hunter"],
        wanted=("Hunter",),
    )
    own = conversation(
        ["You have Celestial Nighthawk equipped, so keep your Health up."],
        ["Hunter"],
        wanted=("Hunter",),
    )

    assert leaked.flag == "ownership_hallucination" and own.passed


def test_equipped_only_means_equipped_on_the_requested_character() -> None:
    obs, facts = observe(
        "Run Synthoceps for your melee.",
        profile="titan_veteran",
        equipped_only=True,
        character="Hunter",
    )
    wrong = next(
        v for v in evaluate_properties(obs, facts) if v.name == "only_owned_recommendations"
    )
    obs, facts = observe(
        "Run Celestial Nighthawk for your supers.",
        profile="titan_veteran",
        equipped_only=True,
        character="Hunter",
    )
    right = next(
        v for v in evaluate_properties(obs, facts) if v.name == "only_owned_recommendations"
    )

    assert wrong.flag == "ownership_hallucination" and right.passed


def test_ground_truth_treats_mods_as_equipped_with_the_character_that_wears_them() -> None:
    result = conversation(
        ["You have Recuperation equipped, so your Grenade stat is high."],
        ["Hunter"],
        wanted=("Hunter",),
    )

    assert result.passed, result
