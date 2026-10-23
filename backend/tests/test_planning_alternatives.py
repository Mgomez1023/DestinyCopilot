import json

import pytest

from app.models import ChatTurn
from app.planning_alternatives import apply_alternatives, derive_alternatives, offered_candidates
from app.session_planning import (
    PlanningAnswerValidator,
    PlanningCandidate,
    PlanningObjective,
    SessionPlanningContext,
)
from app.session_preferences import SessionPreferenceContext
from tests.test_conversation_scope import chat, turns
from tests.test_conversation_scope import planning_json as _after_marker


def candidate(
    name: str, *, activity_type: str | None = None, objective: str | None = None
) -> PlanningCandidate:
    return PlanningCandidate(
        name=name,
        source_kinds=["available_activity"],
        classification="unknown_significance",
        active_quest_status="NOT_RETURNED",
        completion_status="KNOWN_FALSE",
        available_to_character="KNOWN_TRUE",
        activity_type=activity_type,
        objectives=(
            [
                PlanningObjective(
                    name=f'Complete the mission "{objective}"', completion_status="KNOWN_FALSE"
                )
            ]
            if objective
            else []
        ),
    )


CANDIDATES = [
    candidate("Ember Relay", objective="Spark Ascent"),
    candidate("Harbor Patrol", activity_type="Patrol"),
    candidate("Glass Descent: Standard", activity_type="Dungeon"),
    candidate("Iron Trial", activity_type="Dungeon"),
    candidate("Tide Sweep"),
]
FIRST_REPLY = (
    "Do Ember Relay first: run Spark Ascent. Backup: Harbor Patrol, then Tide Sweep if you "
    "have time."
)


def history(*pairs: tuple[str, str]) -> list[ChatTurn]:
    return turns(*pairs)


def context(candidates: list[PlanningCandidate]) -> SessionPlanningContext:
    return SessionPlanningContext(
        preferences=SessionPreferenceContext(),
        requested_character_scope="Titan",
        requested_scope_explicit=True,
        character_class="Titan",
        intent_category="general",
        evidence_semantics={},
        candidates=candidates,
    )


def names(items: list[PlanningCandidate]) -> list[str]:
    return [value.name for value in items]


def state_after(message: str, *pairs: tuple[str, str]):
    return derive_alternatives(history(*pairs), message, CANDIDATES)


# --- Resolution --------------------------------------------------------------------------------


def test_offered_candidates_follow_the_order_of_the_reply() -> None:
    assert offered_candidates(FIRST_REPLY, CANDIDATES) == [
        "Ember Relay",
        "Harbor Patrol",
        "Tide Sweep",
    ]


def test_a_negated_mention_is_not_an_offer() -> None:
    reply = "No Iron Trial tonight. Try Tide Sweep for something short."

    assert offered_candidates(reply, CANDIDATES) == ["Tide Sweep"]


@pytest.mark.parametrize(
    "message", ["I don't want to do Ember Relay.", "I don't want to do Spark Ascent."]
)
def test_rejecting_a_named_objective_excludes_its_candidate(message: str) -> None:
    state = state_after(message, ("user", "What next?"), ("assistant", FIRST_REPLY))

    assert state.rejected == ["Ember Relay"]
    assert state.rejection_detected is True


def test_rejecting_an_activity_type_excludes_every_candidate_of_that_type() -> None:
    state = state_after(
        "I don't want to do dungeons.", ("user", "What next?"), ("assistant", FIRST_REPLY)
    )

    assert state.rejected == ["Glass Descent: Standard", "Iron Trial"]


@pytest.mark.parametrize(
    "message", ["I don't want to do that.", "Not that one.", "Nah, pass on it."]
)
def test_a_bare_rejection_excludes_the_replys_primary_choice(message: str) -> None:
    state = state_after(message, ("user", "What next?"), ("assistant", FIRST_REPLY))

    assert state.rejected == ["Ember Relay"]


def test_rejecting_those_excludes_everything_the_reply_offered() -> None:
    state = state_after(
        "I don't want to do any of those.", ("user", "What next?"), ("assistant", FIRST_REPLY)
    )

    assert state.rejected == ["Ember Relay", "Harbor Patrol", "Tide Sweep"]


def test_multiple_rejections_accumulate_across_the_exchange() -> None:
    state = state_after(
        "I don't want to do that either.",
        ("user", "What next?"),
        ("assistant", FIRST_REPLY),
        ("user", "I don't want to do Ember Relay."),
        ("assistant", "Then Harbor Patrol is the best fit."),
    )

    assert state.rejected == ["Ember Relay", "Harbor Patrol"]


def test_a_rejection_that_cannot_be_resolved_is_flagged_not_guessed() -> None:
    state = derive_alternatives([], "I don't want to do that.", CANDIDATES)

    assert state.rejected == []
    assert state.unresolved_rejection is True


def test_rejecting_something_that_is_not_a_candidate_changes_nothing() -> None:
    state = state_after(
        "I don't want to use LFG.", ("user", "What next?"), ("assistant", FIRST_REPLY)
    )

    assert state.rejected == []


def test_an_exclusion_can_be_reversed() -> None:
    state = state_after(
        "Actually Ember Relay is fine, I'll do it.",
        ("user", "What next?"),
        ("assistant", FIRST_REPLY),
        ("user", "I don't want to do Ember Relay."),
        ("assistant", "Then Harbor Patrol."),
    )

    assert state.rejected == []
    assert state.reinstated == ["Ember Relay"]


def test_asking_what_else_sets_aside_only_the_previous_primary() -> None:
    state = state_after("What else could I do?", ("user", "What next?"), ("assistant", FIRST_REPLY))

    assert state.alternative_requested is True
    assert state.rejected == []
    assert state.already_recommended == ["Ember Relay"]


def test_a_character_switch_does_not_carry_the_previous_primary() -> None:
    state = derive_alternatives(
        history(("user", "What next?"), ("assistant", FIRST_REPLY)),
        "What else could I do?",
        CANDIDATES,
        character_switched=True,
    )

    assert state.already_recommended == []


# --- Applying them -----------------------------------------------------------------------------


def test_rejected_candidates_are_removed_and_the_rest_are_kept() -> None:
    state = state_after(
        "I don't want to do Ember Relay.", ("user", "What next?"), ("assistant", FIRST_REPLY)
    )

    result = apply_alternatives(context(list(CANDIDATES)), state)

    assert "Ember Relay" not in names(result.candidates)
    assert names(result.candidates) == [
        "Harbor Patrol",
        "Glass Descent: Standard",
        "Iron Trial",
        "Tide Sweep",
    ]
    assert result.excluded_by_user == ["Ember Relay"]
    assert "Do not ask a clarifying question" in (result.alternative_guidance or "")


def test_an_alternative_request_removes_the_previous_primary_from_the_choices() -> None:
    state = state_after("What else?", ("user", "What next?"), ("assistant", FIRST_REPLY))

    result = apply_alternatives(context(list(CANDIDATES)), state)

    assert "Ember Relay" not in names(result.candidates)
    assert result.already_recommended == ["Ember Relay"]
    assert result.excluded_by_user == []


def test_no_remaining_candidates_yields_a_limitation_instruction_and_answer() -> None:
    only = [candidate("Ember Relay")]
    state = derive_alternatives(
        history(("user", "What next?"), ("assistant", "Do Ember Relay.")),
        "I don't want to do that.",
        only,
    )

    result = apply_alternatives(context(only), state)

    assert result.candidates == []
    assert "No grounded candidate remains" in (result.alternative_guidance or "")
    fallback = PlanningAnswerValidator.safe_answer(result)
    assert fallback.startswith("I've set aside Ember Relay.")
    assert "don't see another grounded option" in fallback


def test_when_only_the_previous_primary_remains_it_is_kept_and_flagged() -> None:
    only = [candidate("Ember Relay")]
    state = derive_alternatives(
        history(("user", "What next?"), ("assistant", "Do Ember Relay.")), "What else?", only
    )

    result = apply_alternatives(context(only), state)

    assert names(result.candidates) == ["Ember Relay"]
    assert result.already_recommended == []
    assert "already recommended" in (result.alternative_guidance or "")


def test_the_context_is_untouched_when_nothing_was_rejected_or_requested() -> None:
    state = derive_alternatives([], "What should I do next?", CANDIDATES)
    original = context(list(CANDIDATES))

    assert apply_alternatives(original, state) is original


def test_specific_fallback_names_one_verified_pick_and_the_other_options() -> None:
    state = state_after(
        "I don't want to do Ember Relay.", ("user", "What next?"), ("assistant", FIRST_REPLY)
    )
    result = apply_alternatives(context(list(CANDIDATES)), state)

    fallback = PlanningAnswerValidator.safe_answer(result)

    assert fallback.startswith("I've set aside Ember Relay.")
    # One pick the data supports, an honest limit, and the rest, not a "tell me which one" list.
    assert "Harbor Patrol is available to you" in fallback
    assert "Also available: Glass Descent: Standard, Iron Trial." in fallback
    assert "not ranking it" in fallback and "tell me which one" not in fallback
    assert "can't make a grounded next-step recommendation" not in fallback


# --- Validator ---------------------------------------------------------------------------------


def validate(answer: str, excluded: list[str]) -> list[str]:
    base = context(list(CANDIDATES))
    base = base.model_copy(update={"excluded_by_user": excluded})
    return [value.code for value in PlanningAnswerValidator().validate(answer, base)]


def test_recommending_a_rejected_candidate_is_a_violation() -> None:
    assert "rejected_candidate_recommended" in validate(
        "You should run Ember Relay next.", ["Ember Relay"]
    )


@pytest.mark.parametrize(
    "answer",
    [
        "Since you passed on Ember Relay, try Harbor Patrol instead of Ember Relay.",
        "Skipping Ember Relay, you should run Harbor Patrol.",
        "You should run Harbor Patrol, which is shorter than Ember Relay.".replace(
            "shorter than", "an option other than"
        ),
    ],
)
def test_mentioning_a_rejected_candidate_to_set_it_aside_is_allowed(answer: str) -> None:
    assert "rejected_candidate_recommended" not in validate(answer, ["Ember Relay"])


# --- Through the service -----------------------------------------------------------------------

PLAN = "I have 20 minutes and want to play solo, no Crucible. What should I do?"
PLAN_REPLY = (
    'Finish The Edge of Fate by completing the mission "Kepler Uplink". '
    "Backup: Vanguard Ops for the weekly milestone."
)


def planning_json(instructions: str) -> str:
    text = _after_marker(instructions)
    return json.dumps(json.JSONDecoder().raw_decode(text[text.index("{") :])[0])


def planning_data(responses) -> dict:
    text = _after_marker(responses.requests[0]["instructions"])
    return json.JSONDecoder().raw_decode(text[text.index("{") :])[0]


def plan_history():
    return turns(("user", PLAN), ("assistant", PLAN_REPLY))


def test_what_else_keeps_time_solo_and_exclusions_and_sets_aside_the_last_primary() -> None:
    _, responses, trace = chat(plan_history(), "What else could I do?")

    data = json.loads(planning_json(responses.requests[0]["instructions"]).split("\n", 1)[0])
    assert data["preferences"]["time_minutes"] == 20
    assert data["preferences"]["fireteam"] == "solo"
    assert data["preferences"]["exclusions"] == ["pvp"]
    assert data["already_recommended"] == ["The Edge of Fate"]
    assert "The Edge of Fate" not in [value["name"] for value in data["candidates"]]
    assert trace["planning_intent"]["alternatives"]["already_recommended"] == ["The Edge of Fate"]


def test_rejecting_a_recommended_quest_removes_it_and_keeps_the_other_constraints() -> None:
    _, responses, trace = chat(plan_history(), "I don't want to do Kepler Uplink.")

    data = json.loads(planning_json(responses.requests[0]["instructions"]).split("\n", 1)[0])
    assert data["excluded_by_user"] == ["The Edge of Fate"]
    assert "The Edge of Fate" not in [value["name"] for value in data["candidates"]]
    assert data["preferences"]["time_minutes"] == 20
    assert data["preferences"]["fireteam"] == "solo"
    assert trace["planning_intent"]["alternatives"]["rejection_detected"] is True


def test_rejecting_a_recommended_activity_type_removes_all_of_that_type() -> None:
    _, responses, _ = chat(plan_history(), "I don't want to do dungeons.")

    data = json.loads(planning_json(responses.requests[0]["instructions"]).split("\n", 1)[0])
    assert all(value.get("activity_type") != "Dungeon" for value in data["candidates"])
    # "Don't want to do dungeons" is also a stated session exclusion, so the dungeon is pruned
    # with the other excluded activity types before rejection handling runs.
    assert "Duality: Standard" in data["excluded_by_preference"]


def test_a_reversed_rejection_brings_the_candidate_back() -> None:
    rejected = [
        *plan_history(),
        *turns(("user", "I don't want to do Kepler Uplink."), ("assistant", "OK.")),
    ]

    _, responses, _ = chat(rejected, "Actually The Edge of Fate is fine.")

    data = json.loads(planning_json(responses.requests[0]["instructions"]).split("\n", 1)[0])
    assert data["excluded_by_user"] == []
    assert "The Edge of Fate" in [value["name"] for value in data["candidates"]]


def test_an_unrelated_question_ends_the_exclusions() -> None:
    moved_on = [
        *plan_history(),
        *turns(
            ("user", "I don't want to do Kepler Uplink."),
            ("assistant", "OK."),
            ("user", "How does Sunshot work?"),
            ("assistant", "It fires solar bolts."),
        ),
    ]

    _, responses, trace = chat(moved_on, "What should I do next?")

    data = json.loads(planning_json(responses.requests[0]["instructions"]).split("\n", 1)[0])
    assert data["excluded_by_user"] == []
    assert "The Edge of Fate" in [value["name"] for value in data["candidates"]]
    assert trace["planning_intent"]["stale_constraints_dropped"] is True


def test_a_switch_after_a_rejection_does_not_carry_the_old_characters_primary() -> None:
    _, responses, trace = chat(plan_history(), "Switch to my Hunter.")

    data = json.loads(planning_json(responses.requests[0]["instructions"]).split("\n", 1)[0])
    assert data["character_class"] == "Hunter"
    assert data["already_recommended"] == []
    assert "The Edge of Fate" not in json.dumps(data["candidates"])
    assert trace["planning_intent"]["character_switch_continuation"] is True


def test_a_draft_that_recommends_a_rejected_quest_ends_in_a_specific_fallback() -> None:
    from tests.test_build_analysis import _answer

    draft = "You should run The Edge of Fate next; it is your active campaign quest."
    answer, _, trace = chat(
        plan_history(),
        "I don't want to do Kepler Uplink.",
        [[_answer(draft, 1)], [_answer(draft, 2)]],
    )

    assert answer.message.startswith("I've set aside The Edge of Fate.")
    assert "can't make a grounded next-step recommendation" not in answer.message
    assert "rejected_candidate_recommended" in trace["planning_correction"]["violation_codes"]
