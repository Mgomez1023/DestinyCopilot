import json
import shutil
import sqlite3
import zipfile
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from app.bungie.manifest import DefinitionResolver, _signed_hash
from app.config import Settings
from app.guardian_refresh import GuardianRefreshService


def definition(name: str, entity_hash: int) -> dict[str, Any]:
    return {
        "displayProperties": {"name": name, "description": f"{name} description"},
        "hash": entity_hash,
    }


def create_mobile_manifest(
    root: Path, tables: dict[str, dict[int, dict[str, Any]]], name: str
) -> Path:
    database_path = root / f"{name}.sqlite3"
    archive_path = root / f"{name}.content"
    with closing(sqlite3.connect(database_path)) as database:
        for table_name, values in tables.items():
            database.execute(f'CREATE TABLE "{table_name}" (id INTEGER PRIMARY KEY, json TEXT)')
            database.executemany(
                f'INSERT INTO "{table_name}" (id, json) VALUES (?, ?)',
                [
                    (_signed_hash(entity_hash), json.dumps(value))
                    for entity_hash, value in values.items()
                ],
            )
        database.commit()
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(database_path, arcname="world_sql_content.content")
    database_path.unlink()
    return archive_path


class FakeManifestClient:
    def __init__(
        self,
        settings: Settings,
        archive_path: Path,
        *,
        version: str = "manifest-v1",
        fail_download: bool = False,
    ) -> None:
        self.settings = settings
        self.archive_path = archive_path
        self.version = version
        self.fail_download = fail_download
        self.download_calls = 0
        self.definition_calls: list[tuple[str, int]] = []
        self.public_json_calls = 0

    async def get_manifest(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "mobileWorldContentPaths": {"en": "/manifest/world.content"},
        }

    async def download_public_file(self, _path: str, destination: Path) -> int:
        self.download_calls += 1
        if self.fail_download:
            raise OSError("manifest storage unavailable")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.archive_path, destination)
        return destination.stat().st_size

    async def get_public_json(self, _path: str) -> dict[str, Any]:
        self.public_json_calls += 1
        raise AssertionError("The component JSON path must not be used")

    async def get_definition(self, entity_type: str, entity_hash: int) -> dict[str, Any]:
        self.definition_calls.append((entity_type, entity_hash))
        raise RuntimeError("single-definition fallback unavailable")


def manifest_settings(
    cache_dir: Path, *, cache_size: int = 1024, metadata_ttl: int = 900
) -> Settings:
    return Settings(
        _env_file=None,
        manifest_cache_dir=cache_dir,
        manifest_definition_cache_size=cache_size,
        manifest_metadata_ttl_seconds=metadata_ttl,
    )


@pytest.mark.asyncio
async def test_sqlite_definition_lookup_uses_signed_hash_and_bounded_lru(tmp_path: Path) -> None:
    large_hash = 4_000_000_000
    archive = create_mobile_manifest(
        tmp_path,
        {
            "DestinyInventoryItemDefinition": {
                1: definition("First Item", 1),
                large_hash: definition("Signed Hash Item", large_hash),
            }
        },
        "lookup",
    )
    client = FakeManifestClient(manifest_settings(tmp_path / "cache", cache_size=1), archive)
    resolver = DefinitionResolver(client)  # type: ignore[arg-type]

    values = await resolver.resolve_many(
        "DestinyInventoryItemDefinition", {1, large_hash}
    )
    assert values[1]["displayProperties"]["name"] == "First Item"
    assert values[large_hash]["displayProperties"]["name"] == "Signed Hash Item"
    assert client.download_calls == 1
    assert client.definition_calls == []
    assert client.public_json_calls == 0
    assert len(resolver._cache) == 1
    assert not hasattr(resolver, "_tables")

    await resolver.resolve_many("DestinyInventoryItemDefinition", {1})
    assert client.download_calls == 1
    assert len(resolver._cache) == 1


@pytest.mark.asyncio
async def test_manifest_database_cache_and_version_invalidation(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    first_archive = create_mobile_manifest(
        tmp_path,
        {"DestinyInventoryItemDefinition": {1: definition("Version One", 1)}},
        "version-one",
    )
    first_client = FakeManifestClient(
        manifest_settings(cache_dir, metadata_ttl=0), first_archive, version="version-one"
    )
    first_resolver = DefinitionResolver(first_client)  # type: ignore[arg-type]
    assert (await first_resolver.resolve("DestinyInventoryItemDefinition", 1))["hash"] == 1
    assert first_client.download_calls == 1

    warm_client = FakeManifestClient(
        manifest_settings(cache_dir), first_archive, version="version-one"
    )
    warm_resolver = DefinitionResolver(warm_client)  # type: ignore[arg-type]
    await warm_resolver.resolve("DestinyInventoryItemDefinition", 1)
    assert warm_client.download_calls == 0

    second_archive = create_mobile_manifest(
        tmp_path,
        {"DestinyInventoryItemDefinition": {1: definition("Version Two", 1)}},
        "version-two",
    )
    first_client.version = "version-two"
    first_client.archive_path = second_archive
    value = await first_resolver.resolve("DestinyInventoryItemDefinition", 1)

    assert value["displayProperties"]["name"] == "Version Two"
    assert first_client.download_calls == 2
    assert len(list(cache_dir.glob("world-content-*.sqlite3"))) == 1


@pytest.mark.asyncio
async def test_iter_definitions_streams_database_rows_in_batches(tmp_path: Path) -> None:
    archive = create_mobile_manifest(
        tmp_path,
        {
            "DestinyInventoryItemDefinition": {
                value: definition(f"Item {value}", value) for value in range(7)
            }
        },
        "stream",
    )
    client = FakeManifestClient(manifest_settings(tmp_path / "cache"), archive)
    resolver = DefinitionResolver(client)  # type: ignore[arg-type]

    values = [
        value
        async for value in resolver.iter_definitions(
            "DestinyInventoryItemDefinition", batch_size=2
        )
    ]

    assert len(values) == 7
    assert {value["hash"] for value in values} == set(range(7))
    assert client.public_json_calls == 0


class FakeGuardianClient(FakeManifestClient):
    async def get_current_memberships(self, _access_token: str) -> dict[str, Any]:
        return {
            "primaryMembershipId": "membership",
            "destinyMemberships": [
                {
                    "membershipId": "membership",
                    "membershipType": 3,
                    "bungieGlobalDisplayName": "Guardian",
                    "bungieGlobalDisplayNameCode": 7,
                }
            ],
        }

    async def get_profile(
        self,
        _membership_type: int,
        _membership_id: str,
        _access_token: str,
        *,
        components: tuple[str, ...] | set[str] | None = None,
    ) -> dict[str, Any]:
        del components
        return {
            "profile": {"data": {"minutesPlayedTotal": "10"}},
            "characters": {
                "data": {
                    "character": {
                        "characterId": "character",
                        "classHash": 10,
                        "classType": 2,
                        "raceType": 3,
                        "genderType": 2,
                        "light": 200,
                        "minutesPlayedTotal": "10",
                    }
                }
            },
        }

    async def get_activity_history(
        self,
        _membership_type: int,
        _membership_id: str,
        _character_id: str,
        _access_token: str,
        *,
        count: int = 5,
    ) -> dict[str, Any]:
        del count
        return {"activities": []}


@pytest.mark.asyncio
async def test_guardian_cold_refresh_uses_sqlite_without_component_json(tmp_path: Path) -> None:
    from app.bungie.guardian import GuardianService

    archive = create_mobile_manifest(
        tmp_path,
        {"DestinyClassDefinition": {10: definition("Warlock", 10)}},
        "guardian",
    )
    settings = manifest_settings(tmp_path / "cache")
    client = FakeGuardianClient(settings, archive)
    resolver = DefinitionResolver(client)  # type: ignore[arg-type]
    guardian = GuardianService(client, resolver)  # type: ignore[arg-type]
    refresh = GuardianRefreshService(guardian, settings)

    result = await refresh.get_context("account", "access-token")

    assert result.context.characters[0].class_name == "Warlock"
    assert client.download_calls == 1
    assert client.public_json_calls == 0
    assert list((tmp_path / "cache").glob("world-content-*.sqlite3"))


@pytest.mark.asyncio
async def test_guardian_refresh_degrades_when_manifest_preparation_fails(tmp_path: Path) -> None:
    from app.bungie.guardian import GuardianService

    archive = create_mobile_manifest(tmp_path, {}, "failed")
    settings = manifest_settings(tmp_path / "cache")
    client = FakeGuardianClient(settings, archive, fail_download=True)
    resolver = DefinitionResolver(client)  # type: ignore[arg-type]
    guardian = GuardianService(client, resolver)  # type: ignore[arg-type]
    refresh = GuardianRefreshService(guardian, settings)

    result = await refresh.get_context("account", "access-token")

    assert result.context.bungie_display_name == "Guardian#0007"
    assert result.context.characters[0].class_name == "Warlock"
    assert client.download_calls == 1
    assert client.definition_calls == [("DestinyClassDefinition", 10)]


@pytest.mark.asyncio
async def test_large_lookup_does_not_fan_out_when_manifest_is_unavailable(
    tmp_path: Path,
) -> None:
    archive = create_mobile_manifest(tmp_path, {}, "failed-large")
    client = FakeManifestClient(
        manifest_settings(tmp_path / "cache"), archive, fail_download=True
    )
    resolver = DefinitionResolver(client)  # type: ignore[arg-type]

    values = await resolver.resolve_many(
        "DestinyInventoryItemDefinition", set(range(100, 125))
    )

    assert values == {}
    assert client.download_calls == 1
    assert client.definition_calls == []
