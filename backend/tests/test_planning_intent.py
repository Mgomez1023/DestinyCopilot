import pytest

from app.models import ChatTurn
from app.planning_intent import (
    classify_planning_intent,
    is_alternatives_request,
    is_standalone_planning,
)
from tests.test_conversation_scope import chat, planning_json, turns


def classify(history: list[ChatTurn], message: str, *, switched: bool = False):
    return classify_planning_intent(history, message, character_switched=switched)


def exchange(*user_messages: str) -> list[ChatTurn]:
    pairs: list[tuple[str, str]] = []
    for text in user_messages:
        pairs += [("user", text), ("assistant", "Here is a reply.")]
    return turns(*pairs)


# --- Standalone recognition -------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "What should I do next?",
        "I've got an hour to play. What can I do?",
        "What can I do for an hour?",
        "What else could I do?",
        "Anything worth doing solo?",
        "I want to make some progress tonight.",
        "Give me something quick.",
        "I don't feel like playing Crucible.",
        "What can I work toward on my Hunter?",
        "We have about 45 minutes, what can we run?",
        "Where should I start tonight?",
        "I'm in the mood for something chill.",
    ],
)
def test_natural_planning_phrasings_are_recognized(message: str) -> None:
    assert is_standalone_planning(message) is True
    assert classify([], message).kind == "new_request"


@pytest.mark.parametrize(
    "message",
    [
        "What is a dungeon?",
        "How long is a raid?",
        "What does this perk do?",
        "I spent an hour in Crucible yesterday.",
        "What can I do with this perk?",
        "Switch to my Hunter.",
        "How does Sunshot work?",
        "Tell me about the Whisper mission.",
        "Is my Titan build good?",
    ],
)
def test_unrelated_or_informational_messages_are_not_planning(message: str) -> None:
    assert is_standalone_planning(message) is False
    assert classify([], message).is_planning is False


def test_narrative_time_mentions_do_not_set_a_constraint() -> None:
    intent = classify([], "I spent an hour in Crucible yesterday.")

    assert intent.is_planning is False
    assert intent.added == []


@pytest.mark.parametrize(
    "message", ["What else could I do?", "Anything else?", "Any other options?", "Something else"]
)
def test_alternative_requests_are_detected(message: str) -> None:
    assert is_alternatives_request(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "What can I do with Sunshot?",
        "What can I do with Synthoceps?",
        "What can I do with this perk?",
        "What should I do with my Apex Predator?",
        "What can I do with Riskrunner and Arc Soul?",
    ],
)
def test_use_questions_about_an_object_are_not_session_planning(message: str) -> None:
    assert is_standalone_planning(message) is False
    assert classify([], message).is_planning is False


@pytest.mark.parametrize(
    "message",
    [
        "What can I do with 30 minutes?",
        "What can I do with an hour?",
        "What can I do on my Hunter?",
        "What can I do with my Hunter in 30 minutes?",
        "What should I do with my evening?",
        "What can we do with friends tonight?",
    ],
)
def test_session_resources_after_with_still_mean_planning(message: str) -> None:
    assert is_standalone_planning(message) is True
    assert classify([], message).kind == "new_request"


def test_an_object_question_inside_a_plan_ends_the_plan() -> None:
    history = exchange("I have 30 minutes. What should I do?", "What can I do with Sunshot?")

    assert classify(history, "Anything else?").is_planning is False


# --- Continuation and constraints -------------------------------------------------------------


def test_alternatives_after_a_plan_keep_the_applicable_constraints() -> None:
    history = exchange("I have 45 minutes and want to play solo. What should I do?")

    intent = classify(history, "Anything else?")

    assert (intent.kind, intent.signal) == ("continuation", "alternatives")
    assert intent.preferences.time_minutes == 45
    assert intent.preferences.fireteam == "solo"
    assert sorted(intent.inherited) == ["fireteam", "time_minutes"]


def test_alternatives_with_no_planning_context_are_not_planning() -> None:
    assert classify(exchange("How does Sunshot work?"), "Anything else?").is_planning is False


def test_a_new_time_replaces_the_old_one_and_keeps_the_rest() -> None:
    history = exchange("I have 30 minutes, solo. What should I do?")

    intent = classify(history, "Actually I have 2 hours.")

    assert intent.kind == "continuation"
    assert intent.preferences.time_minutes == 120
    assert intent.preferences.fireteam == "solo"
    assert intent.replaced == ["time_minutes"]
    assert intent.inherited == ["fireteam"]


def test_a_dropped_time_limit_is_reported_as_cleared() -> None:
    history = exchange("I have 30 minutes. What should I do?")

    intent = classify(history, "Forget the time limit.")

    assert intent.preferences.time_minutes is None
    assert intent.cleared == ["time_minutes"]


@pytest.mark.parametrize(
    "follow_up",
    ["I don't feel like playing Crucible.", "No Crucible.", "Skip PvP tonight."],
)
def test_avoiding_an_activity_type_becomes_an_exclusion_not_a_preference(follow_up: str) -> None:
    history = exchange("What should I do next?")

    intent = classify(history, follow_up)

    assert intent.preferences.exclusions == ["pvp"]
    assert intent.preferences.activity_mode == "unspecified"
    assert intent.kind == "continuation"


def test_missing_constraints_are_never_inferred() -> None:
    intent = classify([], "What can I do for an hour?")

    assert intent.preferences.time_minutes is None or intent.preferences.time_minutes == 60
    assert intent.preferences.fireteam == "unspecified"
    assert intent.preferences.activity_mode == "unspecified"
    assert intent.preferences.exclusions == []


def test_a_constraint_stated_before_the_ask_joins_the_plan() -> None:
    history = exchange("I only want solo stuff")

    intent = classify(history, "What should I do next?")

    assert intent.kind == "new_request"
    assert intent.preferences.fireteam == "solo"


# --- Topic boundaries --------------------------------------------------------------------------


def test_a_topic_change_ends_the_plan_and_its_constraints() -> None:
    history = exchange(
        "I have 30 minutes, solo. What should I do?",
        "How does Sunshot work?",
    )

    assert classify(history, "Anything else?").is_planning is False
    fresh = classify(history, "What should I do next?")
    assert fresh.kind == "new_request"
    assert fresh.preferences.time_minutes is None
    assert fresh.preferences.fireteam == "unspecified"
    assert fresh.stale_constraints_dropped is True


def test_a_constraint_only_reply_after_a_topic_change_is_not_planning() -> None:
    history = exchange("I have 30 minutes. What should I do?", "What is a dungeon?")

    assert classify(history, "Make it shorter.").is_planning is False


def test_a_new_request_inside_a_plan_keeps_that_plans_constraints() -> None:
    history = exchange("I have 30 minutes, solo. What should I do?")

    intent = classify(history, "What should I do next?")

    assert intent.preferences.time_minutes == 30
    assert intent.preferences.fireteam == "solo"
    assert intent.stale_constraints_dropped is False


# --- Character switches ------------------------------------------------------------------------


def test_a_switch_during_planning_continues_planning() -> None:
    history = exchange("I have 30 minutes. What should I do?")

    intent = classify(history, "Switch to my Hunter.", switched=True)

    assert intent.kind == "character_switch"
    assert intent.preferences.time_minutes == 30


def test_a_switch_after_an_unrelated_conversation_is_not_planning() -> None:
    history = exchange("Is my Titan build good?")

    assert classify(history, "Switch to my Hunter.", switched=True).is_planning is False


def test_a_switch_after_a_stray_constraint_is_not_planning() -> None:
    history = exchange("Solo only please.")

    assert classify(history, "Switch to my Warlock.", switched=True).is_planning is False


def test_a_switch_after_planning_then_an_unrelated_question_is_not_planning() -> None:
    history = exchange("What should I do next?", "What does this perk do?")

    assert classify(history, "Switch to my Hunter.", switched=True).is_planning is False


def test_a_bare_switch_in_the_middle_does_not_end_a_plan() -> None:
    history = exchange("I have 30 minutes. What should I do?", "Switch to my Hunter.")

    assert classify(history, "Anything else?").kind == "continuation"


# --- Trace ---------------------------------------------------------------------------------------


def test_the_trace_records_decisions_without_text_or_identifiers() -> None:
    history = exchange("I have 30 minutes, solo. What should I do?")

    trace = classify(history, "Actually I have 2 hours.").trace(context_rebuilt=False)

    assert trace == {
        "classification": "standalone",
        "kind": "continuation",
        "is_planning": True,
        "constraints": {
            "inherited": ["fireteam"],
            "replaced": ["time_minutes"],
            "added": [],
            "cleared": [],
        },
        "stale_constraints_dropped": False,
        "context_rebuilt": False,
        "character_switch_continuation": False,
    }


# --- Through the service ---


def test_service_supplies_planning_context_for_a_natural_request() -> None:
    _, responses, trace = chat([], "What can I do for an hour?")

    assert trace["session_planning"] is True
    assert trace["planning_intent"]["kind"] == "new_request"
    assert "session planning context follows" in responses.requests[0]["instructions"]


def test_service_does_not_enter_planning_for_an_unrelated_question_or_switch() -> None:
    _, responses, trace = chat([], "What is a dungeon?")
    assert trace["session_planning"] is False
    assert "session planning context follows" not in responses.requests[0]["instructions"]

    history = turns(("user", "What is a dungeon?"), ("assistant", "A dungeon is an activity."))
    _, _, switched = chat(history, "Switch to my Hunter.")
    assert switched["session_planning"] is False
    assert switched["conversation_scope"]["active"] == "Hunter"
    assert switched["planning_intent"]["character_switch_continuation"] is False


def test_service_carries_constraints_into_alternatives_and_drops_them_after_a_topic_change() -> (
    None
):
    history = exchange("I have 30 minutes and want to play solo. What should I do?")
    _, responses, trace = chat(history, "What else could I do?")
    context = planning_json(responses.requests[0]["instructions"])
    assert sorted(trace["planning_intent"]["constraints"]["inherited"]) == [
        "fireteam",
        "time_minutes",
    ]
    assert '"time_minutes":30' in context.replace(" ", "")

    moved_on = history + exchange("How does Sunshot work?")
    _, later, later_trace = chat(moved_on, "What should I do next?")
    assert later_trace["planning_intent"]["stale_constraints_dropped"] is True
    assert '"time_minutes":30' not in planning_json(later.requests[0]["instructions"]).replace(
        " ", ""
    )


def test_service_rebuilds_planning_for_the_new_character_after_a_planning_switch() -> None:
    history = exchange("I have 30 minutes. What should I do?")

    _, _, trace = chat(history, "Switch to my Hunter.")

    assert trace["session_planning"] is True
    assert trace["planning_intent"]["character_switch_continuation"] is True
    assert trace["planning_intent"]["context_rebuilt"] is True
    assert trace["conversation_scope"]["active"] == "Hunter"


@pytest.mark.parametrize(
    "message", ["Not that one.", "Nah, pass.", "Actually, the first one is fine."]
)
def test_feedback_on_a_suggestion_continues_an_active_plan_only(message: str) -> None:
    planned = exchange("What should I do next?")

    assert classify(planned, message).kind == "continuation"
    assert classify(exchange("How does Sunshot work?"), message).is_planning is False
    assert classify([], message).is_planning is False
