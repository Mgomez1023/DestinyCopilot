import asyncio
import hashlib
import json
import logging
import re
import shutil
import sqlite3
import time
import zipfile
from collections import OrderedDict
from collections.abc import AsyncIterator, Iterable
from contextlib import closing
from pathlib import Path
from typing import Any

from app.bungie.client import BungieAPIError, BungieClient

logger = logging.getLogger(__name__)

_ENTITY_TYPE = re.compile(r"^Destiny[A-Za-z0-9]+Definition$")
_SQLITE_BATCH_SIZE = 400
_INDEX_BATCH_SIZE = 500
_FALLBACK_CONCURRENCY = 8
_MAX_DEGRADED_FALLBACKS = 12
_PREPARE_RETRY_SECONDS = 60.0


def _signed_hash(entity_hash: int) -> int:
    value = entity_hash & 0xFFFFFFFF
    return value - 2**32 if value >= 2**31 else value


class DefinitionResolver:
    """Resolve Manifest hashes from Bungie's on-disk mobile SQLite database.

    The official database is downloaded as a ZIP-wrapped ``.content`` artifact and
    cached by Manifest version. Queries parse only the requested rows. A bounded LRU
    retains hot definitions; no complete definition table is held in process memory.
    """

    def __init__(self, client: BungieClient) -> None:
        self.client = client
        self._cache: OrderedDict[tuple[str, int], dict[str, Any]] = OrderedDict()
        self._manifest: dict[str, Any] | None = None
        self._manifest_fetched_at = 0.0
        self._manifest_lock = asyncio.Lock()
        self._database_lock = asyncio.Lock()
        self._database_path: Path | None = None
        self._database_version: str | None = None
        self._prepare_failure_version: str | None = None
        self._prepare_failure_at = 0.0
        self._lookup_count = 0

    async def _get_manifest(self) -> dict[str, Any]:
        now = time.monotonic()
        ttl = self.client.settings.manifest_metadata_ttl_seconds
        if self._manifest is not None and now - self._manifest_fetched_at < ttl:
            return self._manifest
        async with self._manifest_lock:
            now = time.monotonic()
            if self._manifest is not None and now - self._manifest_fetched_at < ttl:
                return self._manifest
            previous_version = (
                str(self._manifest.get("version", "unknown")) if self._manifest else None
            )
            manifest = await self.client.get_manifest()
            current_version = str(manifest.get("version", "unknown"))
            self._manifest = manifest
            self._manifest_fetched_at = now
            if previous_version is not None and previous_version != current_version:
                self._cache.clear()
                self._prepare_failure_version = None
                logger.info(
                    "Manifest version changed previous=%s current=%s; definition LRU cleared",
                    previous_version,
                    current_version,
                )
        return self._manifest

    async def manifest_version(self) -> str:
        """Return the version used to key local Manifest artifacts and indexes."""
        return str((await self._get_manifest()).get("version", "unknown"))

    def _cache_get(self, key: tuple[str, int]) -> dict[str, Any] | None:
        value = self._cache.pop(key, None)
        if value is not None:
            self._cache[key] = value
        return value

    def _cache_put(self, key: tuple[str, int], value: dict[str, Any]) -> None:
        limit = self.client.settings.manifest_definition_cache_size
        if limit == 0:
            return
        self._cache.pop(key, None)
        self._cache[key] = value
        while len(self._cache) > limit:
            self._cache.popitem(last=False)

    async def _ensure_database(self) -> tuple[Path, str]:
        manifest = await self._get_manifest()
        version = str(manifest.get("version", "unknown"))
        if (
            self._database_version == version
            and self._database_path is not None
            and self._database_path.is_file()
        ):
            return self._database_path, version
        if (
            self._prepare_failure_version == version
            and time.monotonic() - self._prepare_failure_at < _PREPARE_RETRY_SECONDS
        ):
            raise BungieAPIError("The local Destiny manifest database is temporarily unavailable.")

        async with self._database_lock:
            if (
                self._database_version == version
                and self._database_path is not None
                and self._database_path.is_file()
            ):
                return self._database_path, version

            started = time.perf_counter()
            mobile_paths = manifest.get("mobileWorldContentPaths") or {}
            artifact_path = mobile_paths.get("en") if isinstance(mobile_paths, dict) else None
            if not artifact_path:
                self._record_prepare_failure(version)
                raise BungieAPIError("The English mobile Destiny manifest is unavailable.")

            version_key = hashlib.sha256(version.encode()).hexdigest()[:12]
            root = self.client.settings.resolve_local_path(
                self.client.settings.manifest_cache_dir
            )
            database_path = root / f"world-content-{version_key}.sqlite3"
            cache_hit = await asyncio.to_thread(self._database_is_valid, database_path)
            logger.info(
                "Manifest database prepare version=%s cache=%s storage=%s",
                version,
                "hit" if cache_hit else "miss",
                database_path,
            )
            try:
                if not cache_hit:
                    await self._download_database(artifact_path, database_path)
                if self._database_version is not None and self._database_version != version:
                    self._cache.clear()
                self._database_path = database_path
                self._database_version = version
                self._prepare_failure_version = None
                await asyncio.to_thread(self._remove_old_artifacts, root, database_path)
            except Exception:
                self._record_prepare_failure(version)
                logger.warning(
                    "Manifest database prepare failed version=%s elapsed_ms=%d",
                    version,
                    int((time.perf_counter() - started) * 1000),
                )
                raise

            logger.info(
                "Manifest database ready version=%s cache=%s storage=%s elapsed_ms=%d",
                version,
                "hit" if cache_hit else "miss",
                database_path,
                int((time.perf_counter() - started) * 1000),
            )
            return database_path, version

    def _record_prepare_failure(self, version: str) -> None:
        self._prepare_failure_version = version
        self._prepare_failure_at = time.monotonic()

    async def _download_database(self, artifact_path: str, database_path: Path) -> None:
        archive_path = database_path.with_suffix(".download")
        extracting_path = database_path.with_suffix(".extracting")
        archive_path.unlink(missing_ok=True)
        extracting_path.unlink(missing_ok=True)
        logger.info("Downloading mobile Manifest database storage=%s", database_path)
        downloaded = await self.client.download_public_file(artifact_path, archive_path)
        try:
            await asyncio.to_thread(self._extract_database, archive_path, extracting_path)
            if not await asyncio.to_thread(self._database_is_valid, extracting_path):
                raise BungieAPIError("The downloaded Destiny manifest database is invalid.")
            extracting_path.replace(database_path)
            logger.info(
                "Mobile Manifest database stored bytes=%d storage=%s",
                downloaded,
                database_path,
            )
        except (OSError, sqlite3.Error, zipfile.BadZipFile) as exc:
            raise BungieAPIError("The Destiny manifest database could not be prepared.") from exc
        finally:
            archive_path.unlink(missing_ok=True)
            extracting_path.unlink(missing_ok=True)

    @staticmethod
    def _extract_database(archive_path: Path, destination: Path) -> None:
        with zipfile.ZipFile(archive_path) as archive:
            members = [member for member in archive.infolist() if not member.is_dir()]
            if not members:
                raise zipfile.BadZipFile("The mobile Manifest archive is empty.")
            member = max(members, key=lambda value: value.file_size)
            with archive.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)

    @staticmethod
    def _database_is_valid(path: Path) -> bool:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        try:
            with closing(
                sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
            ) as database:
                row = database.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' LIMIT 1"
                ).fetchone()
            return row is not None
        except (OSError, sqlite3.Error):
            return False

    @staticmethod
    def _remove_old_artifacts(root: Path, current: Path) -> None:
        try:
            for candidate in root.glob("world-content-*"):
                if candidate != current and candidate.is_file():
                    candidate.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove an obsolete Manifest cache artifact")

    @staticmethod
    def _validate_entity_type(entity_type: str) -> str:
        if not _ENTITY_TYPE.fullmatch(entity_type):
            raise BungieAPIError("Invalid Destiny definition type.")
        return entity_type

    @classmethod
    def _query_database(
        cls, database_path: Path, entity_type: str, entity_hashes: Iterable[int]
    ) -> dict[int, dict[str, Any]]:
        table = cls._validate_entity_type(entity_type)
        requested = list(dict.fromkeys(int(value) for value in entity_hashes))
        signed_to_hash = {_signed_hash(value): value for value in requested}
        definitions: dict[int, dict[str, Any]] = {}
        with closing(
            sqlite3.connect(f"{database_path.resolve().as_uri()}?mode=ro", uri=True)
        ) as database:
            database.execute("PRAGMA query_only = ON")
            signed_keys = list(signed_to_hash)
            for offset in range(0, len(signed_keys), _SQLITE_BATCH_SIZE):
                signed_values = signed_keys[offset : offset + _SQLITE_BATCH_SIZE]
                placeholders = ",".join("?" for _ in signed_values)
                rows = database.execute(
                    f'SELECT id, json FROM "{table}" WHERE id IN ({placeholders})',
                    signed_values,
                ).fetchall()
                for row_id, raw_json in rows:
                    try:
                        value = json.loads(raw_json)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        continue
                    entity_hash = signed_to_hash.get(int(row_id))
                    if entity_hash is not None and isinstance(value, dict):
                        definitions[entity_hash] = value
        return definitions

    @classmethod
    def _read_batch(
        cls,
        database_path: Path,
        entity_type: str,
        after_id: int | None,
        limit: int,
    ) -> tuple[list[dict[str, Any]], int | None]:
        table = cls._validate_entity_type(entity_type)
        where = "" if after_id is None else "WHERE id > ?"
        parameters: tuple[int, ...] = () if after_id is None else (after_id,)
        with closing(
            sqlite3.connect(f"{database_path.resolve().as_uri()}?mode=ro", uri=True)
        ) as database:
            database.execute("PRAGMA query_only = ON")
            rows = database.execute(
                f'SELECT id, json FROM "{table}" {where} ORDER BY id LIMIT ?',
                (*parameters, limit),
            ).fetchall()
        definitions: list[dict[str, Any]] = []
        for _, raw_json in rows:
            try:
                value = json.loads(raw_json)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(value, dict):
                definitions.append(value)
        return definitions, int(rows[-1][0]) if rows else None

    async def iter_definitions(
        self, entity_type: str, *, batch_size: int = _INDEX_BATCH_SIZE
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream one definition table in bounded batches for derived index builds."""
        database_path, version = await self._ensure_database()
        started = time.perf_counter()
        count = 0
        after_id: int | None = None
        logger.info(
            "Manifest component scan start version=%s component=%s batch_size=%d",
            version,
            entity_type,
            batch_size,
        )
        while True:
            definitions, next_id = await asyncio.to_thread(
                self._read_batch, database_path, entity_type, after_id, batch_size
            )
            for definition in definitions:
                count += 1
                yield definition
            if next_id is None:
                break
            after_id = next_id
        logger.info(
            "Manifest component scan complete version=%s component=%s definitions=%d elapsed_ms=%d",
            version,
            entity_type,
            count,
            int((time.perf_counter() - started) * 1000),
        )

    async def resolve(self, entity_type: str, entity_hash: int) -> dict[str, Any]:
        values = await self.resolve_many(entity_type, {entity_hash})
        definition = values.get(entity_hash)
        if definition is None:
            raise BungieAPIError(f"Destiny definition {entity_type}/{entity_hash} is unavailable.")
        return definition

    async def resolve_many(
        self, entity_type: str, entity_hashes: set[int]
    ) -> dict[int, dict[str, Any]]:
        if not entity_hashes:
            return {}
        await self._get_manifest()
        started = time.perf_counter()
        definitions: dict[int, dict[str, Any]] = {}
        missing: set[int] = set()
        cache_hits = 0
        for entity_hash in entity_hashes:
            cached = self._cache_get((entity_type, entity_hash))
            if cached is None:
                missing.add(entity_hash)
            else:
                cache_hits += 1
                definitions[entity_hash] = cached

        database_hits = 0
        database_failed = False
        if missing:
            try:
                database_path, _ = await self._ensure_database()
                found = await asyncio.to_thread(
                    self._query_database, database_path, entity_type, missing
                )
                database_hits = len(found)
                definitions.update(found)
                missing.difference_update(found)
            except (BungieAPIError, OSError, sqlite3.Error) as exc:
                database_failed = True
                logger.warning(
                    "Manifest database lookup unavailable component=%s error=%s",
                    entity_type,
                    type(exc).__name__,
                )

        fallback_hits = 0
        fallback_skipped = 0
        if database_failed and len(missing) > _MAX_DEGRADED_FALLBACKS:
            fallback_skipped = len(missing)
            logger.warning(
                "Manifest single-definition fallback skipped component=%s definitions=%d",
                entity_type,
                fallback_skipped,
            )
        elif missing:
            fallback = await self._fetch_fallback(entity_type, missing)
            fallback_hits = len(fallback)
            definitions.update(fallback)

        for entity_hash, definition in definitions.items():
            self._cache_put((entity_type, entity_hash), definition)
        self._lookup_count += len(entity_hashes)
        logger.info(
            "Manifest definition lookup component=%s requested=%d cache_hits=%d "
            "database_hits=%d fallback_hits=%d fallback_skipped=%d database_failed=%s "
            "total_lookups=%d elapsed_ms=%d",
            entity_type,
            len(entity_hashes),
            cache_hits,
            database_hits,
            fallback_hits,
            fallback_skipped,
            database_failed,
            self._lookup_count,
            int((time.perf_counter() - started) * 1000),
        )
        return definitions

    async def _fetch_fallback(
        self, entity_type: str, entity_hashes: set[int]
    ) -> dict[int, dict[str, Any]]:
        definitions: dict[int, dict[str, Any]] = {}
        values = list(entity_hashes)
        for offset in range(0, len(values), _FALLBACK_CONCURRENCY):
            pairs = await asyncio.gather(
                *(
                    self._safe_fetch_single(entity_type, entity_hash)
                    for entity_hash in values[offset : offset + _FALLBACK_CONCURRENCY]
                )
            )
            definitions.update({key: value for key, value in pairs if value})
        return definitions

    async def _safe_fetch_single(
        self, entity_type: str, entity_hash: int
    ) -> tuple[int, dict[str, Any]]:
        cached = self._cache_get((entity_type, entity_hash))
        if cached is not None:
            return entity_hash, cached
        try:
            definition = await self.client.get_definition(entity_type, entity_hash)
        except Exception:
            return entity_hash, {}
        self._cache_put((entity_type, entity_hash), definition)
        return entity_hash, definition

    def release_table(self, entity_type: str) -> None:
        """Compatibility no-op: SQLite lookups never retain complete tables."""
        del entity_type

    async def display_name(self, entity_type: str, entity_hash: int, fallback: str) -> str:
        try:
            definition = await self.resolve(entity_type, entity_hash)
        except Exception:
            return fallback
        return definition.get("displayProperties", {}).get("name") or fallback
