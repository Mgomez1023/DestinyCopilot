"""Guide resolution must not substitute a different subject that shares only a generic word."""

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest

from app.config import Settings
from app.guide_knowledge import (
    GuideKnowledgeProvider,
    MemoryGuideCache,
    alias_covers,
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
    assert not alias_covers("the dungeon guide", "The Dungeon Guide")
    assert not alias_covers("some other dungeon", "Sunken Vault of Ibex dungeon")
    assert alias_covers("sunken vaults", "Sunken Vault of Ibex")
    assert alias_covers("quillfether", "Quillfeather")
    assert not alias_covers("ibex", "ibis")


def test_names_sharing_one_distinctive_word_do_not_cover_each_other() -> None:
    assert not alias_covers("Spire of the Watcher", "Spire of Stars")
    assert not alias_covers("Spire of Stars", "Spire of the Watcher")
    assert not alias_covers("Sunken Vault of Ibex", "Sunken Vault of Marrow")


def test_a_variant_and_its_base_name_do_not_cover_each_other() -> None:
    assert not alias_covers("Haunted Quillfeather Crossing", "Quillfeather Crossing")
    assert not alias_covers("Quillfeather Crossing", "Haunted Quillfeather Crossing")
    assert not alias_covers("how do I do the heroic Ibex", "Ibex")
    assert alias_covers("Haunted Quillfeather", "Haunted Quillfeather Crossing")


def test_questions_containing_a_whole_alias_or_a_partial_name_still_cover() -> None:
    assert alias_covers("how do I finish the Sunken Vault of Ibex quest", "Sunken Vault of Ibex")
    assert alias_covers("sunken vault", "Sunken Vault of Ibex")
    assert alias_covers("hollow spire", "Hollow Spire of Marrow")


def test_the_unqualified_name_does_not_resolve_a_variant_only_guide() -> None:
    variant = RECORDS + [record("g4", "Haunted Gloam Hollow", ["Haunted Gloam"])]

    class VariantSource(Source):
        async def records(self) -> list[GuideRecord]:
            return variant

    guides = GuideKnowledgeProvider(
        EmptyManifest(),  # type: ignore[arg-type]
        Settings(),
        source=VariantSource(),  # type: ignore[call-arg]
        cache=MemoryGuideCache(),
        now=lambda: datetime(2026, 9, 12, tzinfo=UTC),
    )
    assert asyncio.run(guides.get_guide("Gloam Hollow"))["found"] is False
    assert asyncio.run(guides.get_guide("Haunted Gloam Hollow"))["found"] is True
    assert asyncio.run(guides.get_guide("Haunted Ibex"))["found"] is False
    assert asyncio.run(guides.get_guide("Ibex"))["found"] is True
