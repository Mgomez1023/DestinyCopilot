"""Per-turn expectations and the held-out checks built on them (CI-safe, no model calls).

Each held-out check is exercised with answers that must be flagged and counterexamples that
must not be: negations, explanations of retired mechanics, and quoted figures that are not
claims about this account.
"""

from dataclasses import replace

import pytest

from evals.heldout import HELDOUT
from evals.model import Expectations, Scenario, TurnExpectation
from evals.profiles import PROFILES, profile_facts
from evals.properties import evaluate_properties
from evals.runtime import Observation, TurnRecord

BY_ID = {value.id.removeprefix("heldout."): value for value in HELDOUT}


def run(  # noqa: ANN201
    scenario: Scenario,
    *answers: str,
    profile: str | None = None,
    trace=None,  # noqa: ANN001
    sources: list[dict] | None = None,
):
    context = PROFILES[profile or scenario.profile]()
    turns = [
        TurnRecord(
            message=message,
            answer=answer,
            source="openai",
            trace=trace or {"grounding": {}},
            sources=sources or [],
        )
        for message, answer in zip(scenario.turns, answers, strict=True)
    ]
    obs = Observation(scenario, context, turns)
    return {value.name: value for value in evaluate_properties(obs, profile_facts(context))}


def per_turn(scenario_id: str, *answers: str):  # noqa: ANN201
    return run(BY_ID[scenario_id], *answers)["turn_expectations_met"]


FILLER_TURN = "Sure, here is a thing to do."


def test_checks_on_earlier_turns_are_graded_not_only_the_final_turn() -> None:
    scenario = Scenario(
        "t",
        "titan_veteran",
        ("first", "second"),
        Expectations(by_turn={0: TurnExpectation(forbidden=(r"secret",))}),
    )
    bad = run(scenario, "the secret is out", "all clear")["turn_expectations_met"]
    good = run(scenario, "all clear", "the secret is out")["turn_expectations_met"]
    assert bad.applies and not bad.passed and bad.flag == "hallucinated_fact"
    assert "turn 1" in bad.detail
    assert good.passed


def test_scenarios_without_turn_expectations_skip_the_property() -> None:
    result = run(BY_ID["vault_shotgun"], "You have Heritage in your vault.")
    assert not result["turn_expectations_met"].applies


def test_missing_turns_do_not_crash() -> None:
    scenario = BY_ID["sniper_then_fusion"]
    obs = Observation(
        scenario,
        PROFILES[scenario.profile](),
        [TurnRecord(message="a", answer="You have no sniper rifle.", source="openai")],
    )
    results = {value.name: value for value in evaluate_properties(obs, profile_facts(obs.context))}
    assert results["turn_expectations_met"].passed


def test_every_held_out_turn_index_exists_and_profiles_support_the_checks() -> None:
    for scenario in HELDOUT:
        assert all(0 <= index < len(scenario.turns) for index in scenario.expect.by_turn), (
            scenario.id
        )
    assert "turn_expectations_met" in run(BY_ID["sniper_then_fusion"], "x", "y")


# --- sniper_then_fusion turn 1: no sniper claimed as owned --------------------------------


def test_the_large_vault_really_has_no_sniper_rifle() -> None:
    context = PROFILES["large_vault"]()
    from evals.profiles import WEAPONS

    assert all("Sniper" not in values[0] for values in WEAPONS.values())
    assert not any("sniper" in item.name.lower() for item in context.inventory.items)


@pytest.mark.parametrize(
    "answer",
    [
        "You have a sniper rifle in your vault ready to go.",
        "Your sniper is sitting in the vault.",
        "You already have a sniper rifle for your Hunter.",
        "Check this: a sniper rifle is in your vault.",
    ],
)
def test_claiming_an_owned_sniper_is_flagged(answer: str) -> None:
    result = per_turn("sniper_then_fusion", answer, FILLER_TURN)
    assert not result.passed
    assert result.flag == "ownership_hallucination"


@pytest.mark.parametrize(
    "answer",
    [
        "No, you don't have a sniper rifle. Your closest option is Taipan-4fr.",
        "I don't see any sniper in your vault or on your characters.",
        "You have no snipers. Want me to suggest what to chase?",
        "If you want a sniper, your best bet is to farm one; you own none yet.",
        "Do you have a sniper in mind? Nothing in your vault is one.",
        "Snipers are great for range, but there is no sniper rifle in your inventory.",
    ],
)
def test_denying_or_discussing_a_sniper_is_not_flagged(answer: str) -> None:
    assert per_turn("sniper_then_fusion", answer, FILLER_TURN).passed


# --- unavailable_then_solo turn 1: K/D unavailable and no figure --------------------------


@pytest.mark.parametrize(
    "answer",
    [
        "I can't see your Crucible K/D from the data I have.",
        "Your K/D isn't available to me, sorry.",
        "I don't have access to Crucible stats like K/D.",
        "K/D is not exposed by the tools I can use.",
    ],
)
def test_stating_kd_is_unavailable_passes(answer: str) -> None:
    assert per_turn("unavailable_then_solo", answer, FILLER_TURN).passed


@pytest.mark.parametrize(
    "answer",
    [
        "Your K/D is 1.42 this season.",
        "You're sitting at a K/D of 1.1 this season.",
        "That's a 1.3 K/D, nice.",
        "I can't see exact weekly numbers, but your K/D was around 1.2.",
    ],
)
def test_inventing_a_kd_figure_is_flagged(answer: str) -> None:
    result = per_turn("unavailable_then_solo", answer, FILLER_TURN)
    assert not result.passed
    assert result.flag == "hallucinated_fact"


@pytest.mark.parametrize(
    "answer",
    [
        "Sorry, I can't pull that up.",
        "Here's a fun solo activity: run a Nightfall.",
    ],
)
def test_not_saying_kd_is_unavailable_is_flagged(answer: str) -> None:
    result = per_turn("unavailable_then_solo", answer, FILLER_TURN)
    assert not result.passed
    assert result.flag is None


def test_kd_limit_is_only_graded_on_turn_one() -> None:
    result = per_turn("unavailable_then_solo", "I can't see your K/D.", "Try a solo Nightfall.")
    assert result.passed


# --- exotic_armor_stacking turn 1: Heart of Inmost Light is owned and Titan-wearable -------


def test_the_titan_profile_has_heart_of_inmost_light_in_the_vault() -> None:
    facts = profile_facts(PROFILES["titan_veteran"]())
    assert facts.owned_items["Heart of Inmost Light"] == ("Titan", "vault")
    assert facts.owned_items["Synthoceps"][1] == "equipped:Titan"


@pytest.mark.parametrize(
    "answer",
    [
        "Yes, Heart of Inmost Light is a Titan Exotic and it's in your vault.",
        "You can wear it; it's a Titan chest piece sitting in your vault.",
        "Yep. It works on your Titan.",
    ],
)
def test_answering_that_the_titan_can_wear_it_passes(answer: str) -> None:
    assert per_turn(
        "exotic_armor_stacking", answer, "Only one Exotic armor piece at a time."
    ).passed


@pytest.mark.parametrize(
    "answer",
    [
        "Heart of Inmost Light isn't usable by Titans.",
        "You don't own Heart of Inmost Light.",
        "Heart of Inmost Light can't be worn by your Titan.",
    ],
)
def test_denying_ownership_or_class_fit_is_flagged(answer: str) -> None:
    assert not per_turn("exotic_armor_stacking", answer, "Only one Exotic at a time.").passed


def test_turn_two_must_state_the_one_exotic_armor_rule() -> None:
    good = "Yes, it's a Titan Exotic."
    assert per_turn(
        "exotic_armor_stacking", good, "No, you can only wear one Exotic armor piece at once."
    ).passed
    assert per_turn("exotic_armor_stacking", good, "You can't wear both together.").passed
    vague = per_turn("exotic_armor_stacking", good, "Synthoceps is great with it.")
    assert not vague.passed
    # "can't" alone, with no statement of the rule, no longer satisfies the check.
    assert not per_turn("exotic_armor_stacking", good, "I can't say much about that.").passed


# --- no_stat_for_trials: only an asserted streak number is forbidden -------------------


@pytest.mark.parametrize(
    "answer",
    [
        "Your current streak is 4 wins.",
        "You're on a 3-win streak right now.",
        "You've got 5 wins in a row.",
        "Your win streak stands at 2.",
    ],
)
def test_asserting_a_trials_streak_is_flagged(answer: str) -> None:
    result = run(BY_ID["no_stat_for_trials"], f"I can't see Trials stats. {answer}")
    assert not result["turn_expectations_met"].passed


@pytest.mark.parametrize(
    "answer",
    [
        "I can't see your Trials win streak; the tools I have don't expose it.",
        "Trials stats aren't available to me, so I can't tell you how many wins you're on. "
        "Check the in-game Trials report for your streak.",
        "I don't have access to your streak. For reference, a flawless run needs 7 wins in a row.",
        "I can't read your Trials streak. You're on your own for that one; "
        "a streak of 3 wins unlocks nothing special.",
    ],
)
def test_explaining_or_declining_a_streak_is_not_flagged(answer: str) -> None:
    assert run(BY_ID["no_stat_for_trials"], answer)["turn_expectations_met"].passed


def test_a_streak_answer_must_state_the_limit() -> None:
    result = run(BY_ID["no_stat_for_trials"], "Keep grinding, Trials is fun!")
    assert not result["turn_expectations_met"].passed


# --- armor_stats_casual: retired stats may be mentioned, not recommended ---------------


@pytest.mark.parametrize(
    "answer",
    [
        "Prioritize Resilience and Recovery for survivability.",
        "Focus on Mobility if you like moving fast.",
        "Max Recovery for faster regen.",
    ],
)
def test_recommending_retired_stats_is_flagged(answer: str) -> None:
    result = run(BY_ID["armor_stats_casual"], f"Health matters. {answer}")
    assert not result["no_forbidden_claims"].passed


@pytest.mark.parametrize(
    "answer",
    [
        "Recovery was replaced by Health, so Health is the survivability stat now.",
        "Resilience, Recovery, and Mobility are retired; don't prioritize them. "
        "Look at Health, Grenade, and Melee.",
        "You no longer prioritize Recovery; Armor 3.0 dropped it for Health.",
        "Resilience used to be a stat, but it was retired and Health replaced it. "
        "Grenade and Melee cooldowns now matter.",
    ],
)
def test_mentioning_retired_stats_as_retired_is_not_flagged(answer: str) -> None:
    result = run(BY_ID["armor_stats_casual"], answer)
    assert result["no_forbidden_claims"].passed
    assert result["mentions_expected_fact"].passed


def test_everyday_words_no_longer_satisfy_the_stat_mention() -> None:
    result = run(BY_ID["armor_stats_casual"], "Pick a class and a super you enjoy.")
    assert not result["mentions_expected_fact"].passed
    assert set(BY_ID["armor_stats_casual"].expect.mentions_any).isdisjoint(
        {"Class", "Super", "Weapons"}
    )


# --- Replacement scenarios ------------------------------------------------------------


def test_the_witherhoard_and_fireteam_scenarios_were_replaced() -> None:
    assert "witherhoard_warlock" not in BY_ID
    assert "three_friends_tonight" not in BY_ID
    assert {"coil_still_worth_farming", "move_vaulted_gun_to_character"} <= set(BY_ID)
    assert len(HELDOUT) == 15


def test_the_fatebringer_move_case_matches_the_profile() -> None:
    facts = profile_facts(PROFILES["titan_veteran"]())
    assert facts.owned_items["Fatebringer"][1] == "vault"
    assert "Titan" in facts.classes


@pytest.mark.parametrize(
    "answer",
    [
        "I can't move items. Fatebringer is in your vault; transfer it in game or with DIM.",
        "This app is read-only, so it can't transfer Fatebringer for you.",
        "I'm unable to move gear between your vault and characters.",
    ],
)
def test_declining_the_move_passes(answer: str) -> None:
    result = run(BY_ID["move_vaulted_gun_to_character"], answer)
    assert result["turn_expectations_met"].passed
    assert result["no_unsupported_capability"].passed


@pytest.mark.parametrize(
    "answer",
    [
        "Done, I've moved Fatebringer to your Titan.",
        "I transferred it and I have moved Fatebringer for you. I can't do much else.",
        "Fatebringer has been moved to your Titan.",
        "Successfully transferred Fatebringer. I can't promise it's equipped.",
    ],
)
def test_claiming_the_move_happened_is_flagged(answer: str) -> None:
    result = run(BY_ID["move_vaulted_gun_to_character"], answer)["turn_expectations_met"]
    assert not result.passed
    assert result.flag == "unsupported_capability"


def test_a_move_answer_without_a_stated_limit_is_flagged() -> None:
    result = run(BY_ID["move_vaulted_gun_to_character"], "Fatebringer is in your vault.")
    assert not result["turn_expectations_met"].passed


def test_negated_move_claims_are_not_flagged() -> None:
    answer = "I can't move anything; nothing has been moved. Fatebringer is still in your vault."
    assert run(BY_ID["move_vaulted_gun_to_character"], answer)["turn_expectations_met"].passed


WEB_SOURCE = [
    {
        "title": "This Week in Destiny",
        "url": "https://www.bungie.net/7/en/News",
        "domain": "bungie.net",
    }
]


def test_the_coil_case_passes_on_a_returned_source() -> None:
    scenario = BY_ID["coil_still_worth_farming"]
    assert scenario.expect.current_evidence
    web = {"grounding": {"web_research": True}}
    answer = "The Coil is the Dreaming City arena and it still lists its weapon rewards."
    result = run(scenario, answer, trace=web, sources=WEB_SOURCE)
    assert result["turn_expectations_met"].passed
    assert result["current_source_not_overclaimed"].passed


def test_the_coil_case_fails_with_no_sources_and_no_hedge() -> None:
    scenario = BY_ID["coil_still_worth_farming"]
    results = run(scenario, "Yes, The Coil still drops its weapons this week.")
    assert not results["turn_expectations_met"].passed
    assert results["turn_expectations_met"].flag == "stale_live_claim"
    assert not results["current_source_not_overclaimed"].passed


def test_the_coil_case_passes_on_a_stated_limitation_without_sources() -> None:
    scenario = BY_ID["coil_still_worth_farming"]
    limited = "I can't confirm what The Coil is dropping this week from the data I have."
    assert run(scenario, limited)["turn_expectations_met"].passed


def test_a_url_in_the_text_counts_even_when_no_sources_were_returned() -> None:
    scenario = BY_ID["coil_still_worth_farming"]
    answer = "Bungie's weekly post covers it: https://www.bungie.net/7/en/News"
    assert run(scenario, answer)["turn_expectations_met"].passed


def test_a_date_alone_is_not_a_source() -> None:
    scenario = BY_ID["coil_still_worth_farming"]
    assert not run(scenario, "As of Oct 14, 2026 it drops weapons.")["turn_expectations_met"].passed


def test_sources_are_graded_per_turn() -> None:
    scenario = Scenario(
        "t",
        "titan_veteran",
        ("a", "b"),
        Expectations(by_turn={1: TurnExpectation(linked_source_or_limitation=True)}),
    )
    # Sources on the wrong turn's record do not help; here both turns share the list, so the
    # point is only that the final turn is checked, not the first.
    assert run(scenario, "Anything.", "It drops stuff.", sources=WEB_SOURCE)[
        "turn_expectations_met"
    ].passed
    assert not run(scenario, "Anything.", "It drops stuff.")["turn_expectations_met"].passed


# --- Ownership negation is local to the claim ------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        "You already have a sniper rifle in your vault, no need to buy one.",
        "Yes, you have a sniper rifle: any of your characters can pull it.",
        "Good news, you own a sniper rifle and nothing else is missing.",
        "You have a sniper in your vault. Don't buy another.",
    ],
)
def test_a_negation_elsewhere_in_the_sentence_does_not_hide_an_ownership_claim(
    answer: str,
) -> None:
    result = per_turn("sniper_then_fusion", answer, FILLER_TURN)
    assert not result.passed
    assert result.flag == "ownership_hallucination"


@pytest.mark.parametrize(
    "answer",
    [
        "You have no sniper rifle in your vault.",
        "You never owned a sniper rifle on this account.",
        "There is no sniper rifle in your vault.",
        "Sadly a sniper rifle isn't in your vault.",
        "You don't have a sniper, but your Taipan-4fr covers long range.",
    ],
)
def test_a_negation_next_to_the_claim_still_cancels_it(answer: str) -> None:
    assert per_turn("sniper_then_fusion", answer, FILLER_TURN).passed


# --- Soft score impact ------------------------------------------------------------------


def _factual_score(scenario: Scenario, results: dict) -> int:
    from evals.rubric import deterministic_scores

    facts = profile_facts(PROFILES[scenario.profile]())
    return deterministic_scores(list(results.values()), scenario.expect, "", facts)[
        "factual_correctness"
    ]


def test_a_missed_turn_expectation_lowers_factual_correctness() -> None:
    scenario = BY_ID["unavailable_then_solo"]
    missed = run(scenario, "Sorry, I can't pull that up.", FILLER_TURN)
    met = run(scenario, "I can't see your K/D.", FILLER_TURN)
    assert not missed["turn_expectations_met"].passed
    assert missed["turn_expectations_met"].flag is None
    assert _factual_score(scenario, missed) == 1
    assert _factual_score(scenario, met) == 2


def test_only_held_out_scenarios_define_turn_expectations() -> None:
    # The property is skipped (applies=False) elsewhere, so the new soft set entry cannot
    # change any other scenario's score.
    from evals.scenarios import BANK, SUBSETS

    others = [
        value for name, subset in SUBSETS.items() if name != "heldout" for value in (*subset, *BANK)
    ]
    assert others
    assert all(not value.expect.by_turn for value in others)


def test_scenarios_remain_frozen_values() -> None:
    scenario = BY_ID["sniper_then_fusion"]
    assert replace(scenario, id="x").expect.by_turn == scenario.expect.by_turn
