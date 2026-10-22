"""Equip/combination rules questions carry the curated rule even when they name no rule alias."""

import pytest

from app.core_knowledge import DestinyKnowledgeEngine, asks_equip_rule
from tests.test_core_knowledge import FIXTURE
from tests.test_core_knowledge_integration import chat

ENGINE = DestinyKnowledgeEngine.from_file(FIXTURE)


@pytest.mark.parametrize(
    "message",
    [
        "can my titan wear zorblax chestplate",
        "and keep that pair of gauntlets on at the same time?",
        "am I allowed to equip both of these at once",
        "is it possible to slot two of them together",
        "how many can I wear?",
        "can you stack these on one character",
    ],
)
def test_equip_rule_questions_are_recognised_without_naming_a_concept(message: str) -> None:
    assert asks_equip_rule(message)


@pytest.mark.parametrize(
    "message",
    [
        "what should I do tonight",
        "can you run a dungeon with me",
        "I wear glasses and play together with friends",
        "equip",
        "what does this perk do",
        "",
    ],
)
def test_ordinary_messages_are_not_rules_questions(message: str) -> None:
    assert not asks_equip_rule(message)


def test_a_rules_question_with_no_concept_alias_still_carries_the_rule() -> None:
    bundle = ENGINE.bundle("and keep zorblax gauntlets on at the same time?")
    assert bundle is not None
    assert [match.id for match in bundle.matches] == ["rule.exotic_equip_limit"]
    assert bundle.matches[0].origin == "rules_question"
    assert "max_equipped_exotic_armor" in bundle.text


def test_the_rule_is_not_duplicated_when_an_alias_already_matched() -> None:
    bundle = ENGINE.bundle("can I wear two exotics at once")
    assert bundle is not None
    assert [match.id for match in bundle.matches].count("rule.exotic_equip_limit") == 1
    assert bundle.matches[0].origin == "message"


def test_non_rules_messages_are_unchanged() -> None:
    assert ENGINE.bundle("I want something chill for an hour.") is None
    assert ENGINE.bundle("I wear glasses and play together") is None


def test_the_attached_rule_is_the_only_rules_the_corpus_defines() -> None:
    bundle = ENGINE.bundle("can I equip both at the same time")
    assert bundle is not None
    assert bundle.payload["rules"] and "entities" not in bundle.payload


def test_the_chat_turn_attaches_the_core_bundle_for_such_a_question() -> None:
    _, responses, trace = chat("and keep these gauntlets on at the same time?")
    core = trace["core_knowledge"]
    assert core["bundle_attached"] is True
    assert core["matched_entity_ids"]
    assert trace["grounding"]["core_knowledge"] is True
    request = responses.requests[0] if hasattr(responses, "requests") else None
    if request is not None:
        assert "max_equipped_exotic_armor" in str(request)
