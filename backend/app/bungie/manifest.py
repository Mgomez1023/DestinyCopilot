import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from app.bungie.client import BungieAPIError, BungieClient

logger = logging.getLogger(__name__)


class DefinitionResolver:
    """Resolve hashes from Bungie's versioned, localized JSON component manifest.

    Component files are cached on disk and in memory. This is content metadata only,
    not player data, and avoids hundreds of calls to the single-definition endpoint.
    """

    def __init__(self, client: BungieClient) -> None:
        self.client = client
        self._cache: dict[tuple[str, int], dict[str, Any]] = {}
        self._tables: dict[str, dict[str, Any]] = {}
        self._table_locks: dict[str, asyncio.Lock] = {}
        self._manifest: dict[str, Any] | None = None
        self._manifest_lock = asyncio.Lock()

    async def _get_manifest(self) -> dict[str, Any]:
        if self._manifest is not None:
            return self._manifest
        async with self._manifest_lock:
            if self._manifest is None:
                self._manifest = await self.client.get_manifest()
        return self._manifest

    async def manifest_version(self) -> str:
        """Return the version used to key local derived Manifest indexes."""
        return str((await self._get_manifest()).get("version", "unknown"))

    async def load_table(self, entity_type: str) -> dict[str, Any]:
        """Load a localized definition table for internal normalized consumers."""
        return await self._load_table(entity_type)

    async def _load_table(self, entity_type: str) -> dict[str, Any]:
        if entity_type in self._tables:
            return self._tables[entity_type]
        lock = self._table_locks.setdefault(entity_type, asyncio.Lock())
        async with lock:
            if entity_type in self._tables:
                return self._tables[entity_type]
            manifest = await self._get_manifest()
            paths = manifest.get("jsonWorldComponentContentPaths", {}).get("en", {})
            component_path = paths.get(entity_type)
            if not component_path:
                raise BungieAPIError(f"The English JSON manifest has no {entity_type} component.")

            version = str(manifest.get("version", "unknown"))
            version_key = hashlib.sha256(version.encode()).hexdigest()[:12]
            cache_dir = self.client.settings.resolve_local_path(
                self.client.settings.manifest_cache_dir
            )
            cache_file = cache_dir / f"{entity_type}-{version_key}.json"
            table = await self._read_cached_table(cache_file)
            if table is None:
                logger.info("Downloading manifest component %s", entity_type)
                table = await self.client.get_public_json(component_path)
                await asyncio.to_thread(self._write_cached_table, cache_file, table)
            self._tables[entity_type] = table
            return table

    @staticmethod
    async def _read_cached_table(path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None

        def read() -> dict[str, Any]:
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            if not isinstance(value, dict):
                raise ValueError("Manifest cache is not an object")
            return value

        try:
            return await asyncio.to_thread(read)
        except (OSError, ValueError, json.JSONDecodeError):
            logger.warning("Ignoring unreadable manifest cache file %s", path)
            return None

    @staticmethod
    def _write_cached_table(path: Path, table: dict[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(table, handle, separators=(",", ":"))
            temporary.replace(path)
        except OSError:
            logger.warning("Could not persist manifest cache file %s", path)

    async def resolve(self, entity_type: str, entity_hash: int) -> dict[str, Any]:
        key = (entity_type, entity_hash)
        if key in self._cache:
            return self._cache[key]
        try:
            table = await self._load_table(entity_type)
            definition = table.get(str(entity_hash))
            if definition is None:
                signed_hash = entity_hash - 2**32 if entity_hash >= 2**31 else entity_hash
                definition = table.get(str(signed_hash))
            if not isinstance(definition, dict):
                raise KeyError(entity_hash)
        except (BungieAPIError, KeyError):
            definition = await self.client.get_definition(entity_type, entity_hash)
        self._cache[key] = definition
        return definition

    async def resolve_many(
        self, entity_type: str, entity_hashes: set[int]
    ) -> dict[int, dict[str, Any]]:
        if not entity_hashes:
            return {}
        if len(entity_hashes) <= 12:
            pairs = await asyncio.gather(
                *(self._safe_fetch_single(entity_type, value) for value in entity_hashes)
            )
            return {key: value for key, value in pairs if value}
        try:
            table = await self._load_table(entity_type)
        except BungieAPIError:
            pairs = await asyncio.gather(
                *(self._safe_fetch_single(entity_type, value) for value in entity_hashes)
            )
            return {key: value for key, value in pairs if value}

        definitions: dict[int, dict[str, Any]] = {}
        for entity_hash in entity_hashes:
            value = table.get(str(entity_hash))
            if value is None:
                signed_hash = entity_hash - 2**32 if entity_hash >= 2**31 else entity_hash
                value = table.get(str(signed_hash))
            if isinstance(value, dict):
                definitions[entity_hash] = value
                self._cache[(entity_type, entity_hash)] = value
        return definitions

    def release_table(self, entity_type: str) -> None:
        """Drop a bulk table after requested definitions have been copied to the hash cache."""
        self._tables.pop(entity_type, None)

    async def _safe_fetch_single(
        self, entity_type: str, entity_hash: int
    ) -> tuple[int, dict[str, Any]]:
        cached = self._cache.get((entity_type, entity_hash))
        if cached is not None:
            return entity_hash, cached
        try:
            definition = await self.client.get_definition(entity_type, entity_hash)
        except Exception:
            return entity_hash, {}
        self._cache[(entity_type, entity_hash)] = definition
        return entity_hash, definition

    async def display_name(self, entity_type: str, entity_hash: int, fallback: str) -> str:
        try:
            definition = await self.resolve(entity_type, entity_hash)
        except Exception:
            return fallback
        return definition.get("displayProperties", {}).get("name") or fallback
