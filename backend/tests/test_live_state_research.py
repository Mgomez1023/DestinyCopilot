"""Current-state questions use web research when it is available; the canned limit is the floor."""

import asyncio
from types import SimpleNamespace

from app.ai import RecommendationService
from app.config import Settings
from app.live_knowledge import volatile_topic
from app.models import ChatRequest
from tests.test_guardian_tools import ScriptedWebResponses, guardian_context, web_call
from tests.test_source_retention import URL, cited

QUESTION = "What is in the weekly rotation this week?"


def ask(outputs: list, texts: list[str], message: str = QUESTION):  # noqa: ANN201
    scripted = ScriptedWebResponses(outputs, texts)
    service = RecommendationService(Settings(openai_api_key="test-key"))
    service.client = SimpleNamespace(responses=scripted)
    response = asyncio.run(service.chat(ChatRequest(message=message), guardian_context()))
    return response, service.latest_trace(), scripted


def test_the_question_is_a_live_state_question() -> None:
    assert volatile_topic(QUESTION) is not None


def test_web_research_answers_a_live_state_question_with_its_source() -> None:
    text = "Bungie's update dated Oct 14, 2026 lists the rotation; it does not say what changed."
    response, trace, scripted = ask([[web_call([URL]), cited(text)]], [text])
    assert response.source == "openai"
    assert response.message == text
    assert [source.url for source in response.sources] == [URL]
    assert trace["grounding"]["web_research"] is True
    assert trace["answer"] == "generated"
    instructions = scripted.requests[0]["instructions"]
    assert "Use web research now" in instructions
    assert "unconfirmed" in instructions


def test_no_web_research_keeps_the_canned_limit() -> None:
    plain = SimpleNamespace(
        type="message",
        id="m",
        role="assistant",
        status="completed",
        content=[
            SimpleNamespace(type="output_text", text="It is probably the usual.", annotations=[])
        ],
    )
    response, trace, _ = ask([[plain]], ["It is probably the usual."])
    assert response.source == "local"
    assert "can't" in response.message
    assert "usual" not in response.message
    assert trace["answer"] == "unavailable_live_data"


def test_a_stable_question_gets_no_live_state_instruction() -> None:
    text = "A dungeon is a three-player activity."
    plain = cited(text)
    _, _, scripted = ask([[plain]], [text], message="What is a dungeon?")
    assert "Use web research now" not in scripted.requests[0]["instructions"]


def test_without_a_model_client_the_canned_limit_is_returned_before_anything_else() -> None:
    service = RecommendationService(Settings(openai_api_key=""))
    service.client = None
    response = asyncio.run(service.chat(ChatRequest(message=QUESTION), guardian_context()))
    assert response.source == "local"
    assert service.latest_trace()["answer"] == "unavailable_live_data"
