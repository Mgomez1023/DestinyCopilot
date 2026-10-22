"""Guide resolution must not substitute a different subject that shares only a generic word."""

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest

from app.config import Settings
from app.guide_knowledge import (
    GuideKnowledgeProvider,
    MemoryGuideCache,
    shares_distinctive_token,
)
from app.knowledge_models import GuideRecord


def record(
    guide_id: str, subject: str, aliases: list[str], guide_type: str = "activity_walkthrough"
):
    return GuideRecord(
        guide_id=guide_id,
        title=f"{subject} walkthrough",
        subject=subject,
        guide_type=guide_type,  # type: ignore[arg-type]
        aliases=aliases,
        freshness="stable",
        summary=f"A dungeon guide with several encounters. {subject} is one of them.",
    )


RECORDS = [
    record("g1", "Sunken Vault of Ibex", ["Ibex dungeon"]),
    record("g2", "Hollow Spire of Marrow", ["Marrow raid", "Hollow Spire"]),
    record("g3", "Quillfeather Crossing", ["Quillfeather"], "quest_walkthrough"),
]


class EmptyManifest:
    async def search_entities(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"results": []}


class Source:
    version = "synthetic"

    async def records(self) -> list[GuideRecord]:
        return RECORDS


def provider() -> GuideKnowledgeProvider:
    return GuideKnowledgeProvider(
        EmptyManifest(),  # type: ignore[arg-type]
        Settings(),
        source=Source(),  # type: ignore[call-arg]
        cache=MemoryGuideCache(),
        now=lambda: datetime(2026, 9, 12, tzinfo=UTC),
    )


@pytest.mark.parametrize(
    "query",
    [
        "Gloamwick Depths dungeon",
        "how do I beat the Zephyr Hollows dungeon",
        "dungeon guide for Thornmere Keep",
        "Thornmere raid walkthrough",
        "the Tidewrack strike guide",
    ],
)
def test_an_unknown_subject_sharing_only_a_generic_word_does_not_resolve(query: str) -> None:
    guides = provider()
    result = asyncio.run(guides.get_guide(query, "activity_walkthrough"))
    assert result["found"] is False
    assert result["resolved_canonical_entity"] is None
    search = asyncio.run(guides.search_guides(query, None, None, 5))
    assert search["resolved_canonical_entity"] is None
    assert search["results"] == []


@pytest.mark.parametrize(
    ("query", "subject"),
    [
        ("Ibex", "Sunken Vault of Ibex"),
        ("walk me through the Ibex dungeon", "Sunken Vault of Ibex"),
        ("Sunken Vault", "Sunken Vault of Ibex"),
        ("how do I do hollow spire", "Hollow Spire of Marrow"),
        ("how do I finish quillfether crossing", "Quillfeather Crossing"),
    ],
)
def test_partial_and_misspelled_names_still_resolve(query: str, subject: str) -> None:
    result = asyncio.run(provider().get_guide(query))
    assert result["found"] is True
    assert result["guide"]["subject"] == subject


def test_generic_words_never_count_as_shared_subject_words() -> None:
    assert not shares_distinctive_token("the dungeon guide", "The Dungeon Guide")
    assert not shares_distinctive_token("some other dungeon", "Sunken Vault of Ibex dungeon")
    assert shares_distinctive_token("sunken vaults", "Sunken Vault of Ibex")
    assert shares_distinctive_token("quillfether", "Quillfeather")
    assert not shares_distinctive_token("ibex", "ibis")
