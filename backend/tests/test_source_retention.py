"""Safety rewrites keep the sources an answer still rests on and drop them only with the answer."""

import asyncio
from types import SimpleNamespace
from typing import Any

from app.ai import RecommendationService, replace_flagged_sentences, retains_answer
from app.config import Settings
from app.models import ChatRequest
from tests.test_guardian_tools import ScriptedWebResponses, guardian_context, web_call

URL = "https://news.example.com/this-week"


def cited(text: str, url: str = URL, title: str = "Weekly update") -> Any:
    return SimpleNamespace(
        type="message",
        id="m1",
        role="assistant",
        status="completed",
        content=[
            SimpleNamespace(
                type="output_text",
                text=text,
                annotations=[
                    SimpleNamespace(
                        type="url_citation", url=url, title=title, start_index=0, end_index=4
                    )
                ],
            )
        ],
    )


def ask(text: str, message: str = "What is in the weekly update?"):  # noqa: ANN201
    scripted = ScriptedWebResponses([[web_call([URL]), cited(text)]], [text])
    service = RecommendationService(Settings(openai_api_key="test-key"))
    service.client = SimpleNamespace(responses=scripted)
    response = asyncio.run(service.chat(ChatRequest(message=message), guardian_context()))
    return response, service.latest_trace()


def test_a_surgical_safety_rewrite_keeps_the_web_citation() -> None:
    text = (
        "The update lists three new rewards this week. Your Power must be at least 2000 for "
        "it. The update is dated this week."
    )
    response, _ = ask(text)
    assert "Power eligibility is unknown" in response.message
    assert "three new rewards" in response.message
    assert "dated this week" in response.message
    assert "must be at least 2000" not in response.message
    assert [source.url for source in response.sources] == [URL]


def test_an_unrewritten_answer_keeps_its_sources() -> None:
    response, _ = ask("The update lists three new rewards this week.")
    assert [source.url for source in response.sources] == [URL]


def test_a_rewrite_that_replaces_the_whole_answer_drops_the_sources() -> None:
    response, _ = ask("Your Power must be at least 2000 for it.")
    assert "Power eligibility is unknown" in response.message
    assert response.sources == []


def test_a_campaign_inference_sentence_is_replaced_without_dropping_the_rest() -> None:
    from app.ai import RecommendationService as Service

    message = (
        "The Final Shape has no active quest, so the campaign must be completed and unavailable "
        "to you. The update page covers the weekly rewards."
    )
    result = Service._enforce_campaign_completion_safety(message)  # noqa: SLF001
    assert "does not establish" in result
    assert "The update page covers the weekly rewards." in result
    assert "must be completed" not in result


def test_retains_answer_requires_at_least_half_of_the_sentences() -> None:
    original = "One thing. Another thing. A third thing. A fourth thing."
    assert retains_answer(original, "One thing. Another thing. Replacement.")
    assert retains_answer(original, "One thing. Another thing.")
    assert not retains_answer(original, "One thing. Totally different text.")
    assert not retains_answer(original, "Canned limitation text.")
    assert not retains_answer("", "anything")


def test_replace_flagged_sentences_preserves_other_sentences_and_line_breaks() -> None:
    message = "Alpha is fine.\nBeta is bad. Gamma is bad too. Delta is fine."
    result = replace_flagged_sentences(
        message, lambda sentence: "LIMIT." if "bad" in sentence else None
    )
    assert result == "Alpha is fine.\nLIMIT. Delta is fine."
    assert replace_flagged_sentences("All fine here.", lambda _: None) is None
