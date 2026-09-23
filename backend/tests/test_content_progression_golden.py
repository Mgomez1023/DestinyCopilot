import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from app.content_progression import (
    ContentProgressionResolver,
    load_content_definitions,
    validate_terminal_mappings,
)
from app.guardian_tools import GuardianToolService
from app.models import (
    CharacterSummary,
    DataAvailability,
    GuardianContext,
    RecordProgressSummary,
    RecordSummary,
)
from app.session_planning import build_session_planning_context

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
PROFILE_FIXTURES = FIXTURE_ROOT / "content_progression_profiles.json"
MANIFEST_FIXTURES = FIXTURE_ROOT / "content_progression_manifest.json"


class SnapshotManifestResolver:
    def __init__(self, definitions: dict[str, dict[str, Any]]) -> None:
        self.definitions = definitions

    async def resolve_many(
        self, entity_type: str, entity_hashes: set[int]
    ) -> dict[int, dict[str, Any]]:
        values = self.definitions.get(entity_type, {})
        return {
            entity_hash: values[str(entity_hash)]
            for entity_hash in entity_hashes
            if str(entity_hash) in values
        }


def load_cases() -> list[dict[str, Any]]:
    payload = json.loads(PROFILE_FIXTURES.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    return payload["cases"]


def guardian_from_case(case: dict[str, Any]) -> tuple[GuardianContext, CharacterSummary]:
    characters = [
        CharacterSummary.model_validate(
            {
                "race_name": "Human",
                "gender_name": "Female",
                "power": 0,
                **value,
            }
        )
        for value in case["characters"]
    ]
    unavailable = case.get("unavailable_components", {})
    guardian = GuardianContext(
        bungie_display_name="Sanitized Fixture",
        membership_id="fixture-membership",
        membership_type=3,
        platform_name="Fixture",
        characters=characters,
        records=RecordProgressSummary(
            records=[RecordSummary.model_validate(value) for value in case["records"]]
        ),
        data_availability=DataAvailability(
            fetched_at=datetime(2026, 9, 22, tzinfo=UTC),
            available_components=[] if "Records" in unavailable else ["Records"],
            unavailable_components=unavailable,
        ),
    )
    selected = next(value for value in characters if value.class_name == case["query_character"])
    return guardian, selected


@pytest.mark.parametrize("case", load_cases(), ids=lambda value: value["id"])
def test_golden_progression_snapshots(case: dict[str, Any]) -> None:
    guardian, selected = guardian_from_case(case)
    response = ContentProgressionResolver(guardian).resolve(selected, case["content_name"])

    if "expected_contents" in case:
        by_name = {value.canonical_name: value for value in response.contents}
        for content_name, expected in case["expected_contents"].items():
            actual = by_name[content_name].model_dump()
            assert {key: actual[key] for key in expected} == expected
        return

    assert response.content is not None
    actual = response.content.model_dump()
    expected = case["expected"]
    assert {key: actual[key] for key in expected} == expected


def test_manifest_snapshot_verifies_every_terminal_mapping() -> None:
    manifest = json.loads(MANIFEST_FIXTURES.read_text(encoding="utf-8"))
    definitions = load_content_definitions()
    mappings = [
        (content.canonical_name, rule)
        for content in definitions.contents
        for rule in content.terminal_evidence
    ]

    assert len(mappings) == 4
    assert len({(rule.entity_type, rule.hash) for _, rule in mappings}) == len(mappings)
    assert all(rule.expected_scope in {"profile", "character"} for _, rule in mappings)
    assert all(rule.semantic == "campaign_completion" for _, rule in mappings)
    assert asyncio.run(validate_terminal_mappings(SnapshotManifestResolver(manifest))) == []


@pytest.mark.parametrize("failure", ["unavailable", "name", "description", "scope"])
def test_terminal_mapping_verification_fails_safely(failure: str) -> None:
    manifest = json.loads(MANIFEST_FIXTURES.read_text(encoding="utf-8"))
    records = manifest["DestinyRecordDefinition"]
    target = records["1580882372"]
    if failure == "unavailable":
        records.pop("1580882372")
    elif failure == "name":
        target["displayProperties"]["name"] = "Different Triumph"
    elif failure == "description":
        target["displayProperties"]["description"] = "Different meaning"
    else:
        target["scope"] = 1

    errors = asyncio.run(validate_terminal_mappings(SnapshotManifestResolver(manifest)))

    assert any("The Final Shape:1580882372" in value for value in errors)


def test_runtime_rejects_mismatched_manifest_identity() -> None:
    case = next(
        value for value in load_cases() if value["id"] == "profile_scoped_terminal_completion"
    )
    case["records"][0]["name"] = "Different Triumph"
    guardian, selected = guardian_from_case(case)

    result = ContentProgressionResolver(guardian).resolve(selected, "The Final Shape").content

    assert result is not None
    assert result.status == "unknown"
    assert result.profile_completion_state == "UNKNOWN"
    assert any("could not be validated" in value for value in result.limitations)


def test_guardian_tools_expose_relevant_scope_without_retained_record_dump() -> None:
    case = next(
        value for value in load_cases() if value["id"] == "profile_scoped_terminal_completion"
    )
    guardian, selected = guardian_from_case(case)
    tools = GuardianToolService(guardian)

    content = tools.get_content_progression(selected.character_id, "The Final Shape")
    general = tools.get_progression(selected.character_id)
    serialized = json.dumps(content)

    assert content["content"]["profile_completion_state"] == "KNOWN_TRUE"
    assert content["content"]["completion_state"] == "UNKNOWN"
    assert selected.character_id not in serialized
    assert guardian.membership_id not in serialized
    assert "records" not in general["records"]


def test_planning_context_preserves_profile_vs_character_completion() -> None:
    case = next(
        value for value in load_cases() if value["id"] == "final_shape_vs_edge_different_scopes"
    )
    guardian, _ = guardian_from_case(case)

    planning = build_session_planning_context(
        guardian,
        "Given my Titan, should I complete The Final Shape or The Edge of Fate next?",
        "story",
        "Titan",
    )
    by_name = {value.content: value for value in planning.named_content}

    assert by_name["The Final Shape"].completion_status == "UNKNOWN"
    assert by_name["The Final Shape"].profile_completion_status == "KNOWN_TRUE"
    assert by_name["The Final Shape"].completion_scope == "profile"
    assert by_name["The Edge of Fate"].status == "in_progress"
    assert "fixture-titan" not in planning.model_dump_json()
