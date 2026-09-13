import asyncio
import hashlib
import json
import re
import time
import unicodedata
from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings
from app.knowledge_models import (
    FreshnessClass,
    GuideRecord,
    GuideType,
    KnowledgeConflict,
)
from app.live_knowledge import unavailable_live_data, volatile_topic

GUIDE_TYPES = (
    "exotic_acquisition",
    "quest_walkthrough",
    "activity_walkthrough",
    "encounter_mechanics",
    "item_farming",
    "catalyst",
    "build_mechanics",
    "general",
)


def _normalized(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return " ".join(re.findall(r"[a-z0-9]+", ascii_value.casefold()))


def _strict_tool(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    }


GUIDE_KNOWLEDGE_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    _strict_tool(
        "search_destiny_guides",
        (
            "Search concise, source-backed Destiny walkthrough and acquisition guides. "
            "Volatile/live questions are identified but not answered from guide data."
        ),
        {
            "query": {"type": "string", "description": "Practical Destiny question."},
            "entity_name": {
                "type": ["string", "null"],
                "description": "Optional item, quest, activity, encounter, or NPC name.",
            },
            "guide_type": {
                "type": ["string", "null"],
                "enum": [*GUIDE_TYPES, None],
                "description": "Optional normalized guide category.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "description": "Maximum compact results.",
            },
        },
    ),
    _strict_tool(
        "get_destiny_guide",
        "Retrieve one practical, normalized, source-backed Destiny guide.",
        {
            "entity_or_query": {
                "type": "string",
                "description": "Canonical entity name, alias, misspelling, or practical question.",
            },
            "guide_type": {
                "type": ["string", "null"],
                "enum": [*GUIDE_TYPES, None],
                "description": "Optional normalized guide category.",
            },
        },
    ),
]


class GuideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchGuidesRequest(GuideRequest):
    query: str = Field(min_length=1, max_length=300)
    entity_name: str | None = Field(default=None, max_length=160)
    guide_type: GuideType | None = None
    limit: int = Field(default=5, ge=1, le=10)


class GetGuideRequest(GuideRequest):
    entity_or_query: str = Field(min_length=1, max_length=300)
    guide_type: GuideType | None = None


class CanonicalEntityResolver(Protocol):
    async def search_entities(
        self, query: str, entity_types: list[str] | None, limit: int
    ) -> dict[str, Any]: ...


class GuideCache(Protocol):
    async def get(self, key: str) -> dict[str, Any] | None: ...

    async def set(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None: ...


class MemoryGuideCache:
    def __init__(self) -> None:
        self.values: dict[str, tuple[float, dict[str, Any]]] = {}

    async def get(self, key: str) -> dict[str, Any] | None:
        cached = self.values.get(key)
        if cached is None or cached[0] <= time.time():
            self.values.pop(key, None)
            return None
        return deepcopy(cached[1])

    async def set(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None:
        if ttl_seconds > 0:
            self.values[key] = (time.time() + ttl_seconds, deepcopy(value))


class FileGuideCache:
    """Small JSON cache with an interface that can be replaced for deployment."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._locks: dict[str, asyncio.Lock] = {}

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self.root / f"{digest}.json"

    async def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)

        def read() -> dict[str, Any] | None:
            if not path.is_file():
                return None
            try:
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if float(payload.get("expires_at", 0)) <= time.time():
                    return None
                value = payload.get("value")
                return value if isinstance(value, dict) else None
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                return None

        value = await asyncio.to_thread(read)
        return deepcopy(value) if value is not None else None

    async def set(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            return
        path = self._path(key)
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:

            def write() -> None:
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix(".tmp")
                    with temporary.open("w", encoding="utf-8") as handle:
                        json.dump(
                            {"expires_at": time.time() + ttl_seconds, "value": value},
                            handle,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    temporary.replace(path)
                except OSError:
                    # Retrieval should still work if a deployment cache is read-only.
                    return

            await asyncio.to_thread(write)


class CuratedGuideSource:
    """Versioned factual extracts maintained in-repo; no runtime web scraping."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._records: list[GuideRecord] | None = None
        self.version = "unknown"
        self._lock = asyncio.Lock()

    async def records(self) -> list[GuideRecord]:
        if self._records is not None:
            return self._records
        async with self._lock:
            if self._records is None:

                def read() -> tuple[str, list[GuideRecord]]:
                    with self.path.open("r", encoding="utf-8") as handle:
                        payload = json.load(handle)
                    return str(payload["version"]), [
                        GuideRecord.model_validate(value) for value in payload["guides"]
                    ]

                self.version, self._records = await asyncio.to_thread(read)
        return self._records


class GuideKnowledgeProvider:
    source_name = "curated_destiny_guides"
    knowledge_category = "guide"
    tool_names = frozenset(value["name"] for value in GUIDE_KNOWLEDGE_TOOL_DEFINITIONS)

    def __init__(
        self,
        manifest: CanonicalEntityResolver,
        settings: Settings,
        *,
        source: CuratedGuideSource | None = None,
        cache: GuideCache | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.manifest = manifest
        self.settings = settings
        source_path = settings.resolve_local_path(settings.guide_corpus_file)
        self.source = source or CuratedGuideSource(source_path)
        cache_root = settings.resolve_local_path(settings.guide_cache_dir)
        self.cache = cache or FileGuideCache(cache_root)
        self._now = now or (lambda: datetime.now(UTC))

    @staticmethod
    def definitions() -> list[dict[str, Any]]:
        return GUIDE_KNOWLEDGE_TOOL_DEFINITIONS

    def handles(self, name: str) -> bool:
        return name in self.tool_names

    async def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "search_destiny_guides":
            request = SearchGuidesRequest.model_validate(arguments)
            return await self.search_guides(
                request.query, request.entity_name, request.guide_type, request.limit
            )
        if name == "get_destiny_guide":
            request = GetGuideRequest.model_validate(arguments)
            return await self.get_guide(request.entity_or_query, request.guide_type)
        raise ValueError(f"Unsupported guide knowledge tool: {name}")

    @staticmethod
    def _is_volatile(query: str) -> bool:
        return volatile_topic(query) is not None

    @staticmethod
    def _record_text(record: GuideRecord) -> str:
        return _normalized(
            " ".join(
                [
                    record.title,
                    record.subject,
                    record.summary,
                    *record.aliases,
                    *(value.name for value in record.associated_entities),
                ]
            )
        )

    @staticmethod
    def _score(query: str, record: GuideRecord) -> float:
        normalized_query = _normalized(query)
        candidates = [record.subject, record.title, *record.aliases]
        ratios = [
            SequenceMatcher(None, normalized_query, _normalized(value)).ratio()
            for value in candidates
        ]
        text = GuideKnowledgeProvider._record_text(record)
        tokens = set(normalized_query.split())
        overlap = len(tokens & set(text.split())) / max(1, len(tokens))
        containment = (
            0.55
            if any(
                len(normalized := _normalized(value)) >= 4 and normalized in normalized_query
                for value in candidates
            )
            else 0
        )
        return min(1.0, max(ratios, default=0) * 0.45 + overlap * 0.35 + containment)

    async def _resolve_canonical(
        self, phrase: str, records: list[GuideRecord]
    ) -> dict[str, Any] | None:
        normalized_phrase = _normalized(phrase)
        aliases = [
            (value, record) for record in records for value in [record.subject, *record.aliases]
        ]
        if not aliases:
            return None
        contained = [
            (value, record)
            for value, record in aliases
            if _normalized(value) and _normalized(value) in normalized_phrase
        ]
        if contained:
            matched_name, matched_record = max(contained, key=lambda value: len(value[0]))
        else:
            matched_name, matched_record = max(
                aliases,
                key=lambda value: SequenceMatcher(
                    None, normalized_phrase, _normalized(value[0])
                ).ratio(),
            )
            ratio = SequenceMatcher(None, normalized_phrase, _normalized(matched_name)).ratio()
            if ratio < 0.45:
                return None

        manifest_result = await self.manifest.search_entities(
            matched_record.subject,
            [value.entity_type for value in matched_record.associated_entities] or None,
            5,
        )
        results = manifest_result.get("results", [])
        exact = next(
            (
                value
                for value in results
                if _normalized(str(value.get("name", ""))) == _normalized(matched_record.subject)
            ),
            None,
        )
        return {
            "input": phrase,
            "matched_alias": matched_name,
            "name": exact.get("name") if exact else matched_record.subject,
            "entity_type": exact.get("entity_type") if exact else None,
            "definition_type": exact.get("definition_type") if exact else None,
            "hash": exact.get("hash") if exact else None,
            "manifest_resolved": bool(exact),
        }

    def _ttl(self, freshness: FreshnessClass) -> int:
        if freshness == "stable":
            return self.settings.guide_cache_stable_ttl_seconds
        if freshness == "semi_stable":
            return self.settings.guide_cache_semi_stable_ttl_seconds
        return 0

    def _freshness_warnings(self, record: GuideRecord) -> list[str]:
        if record.freshness == "volatile":
            return ["This information is volatile and requires the future LiveDestinyProvider."]
        dates = [value.updated_at or value.published_at for value in record.sources]
        known_dates = [value for value in dates if value is not None]
        if not known_dates:
            return ["Source update date is unknown; treat current availability as uncertain."]
        age_days = (self._now() - max(known_dates)).days
        threshold = 730 if record.freshness == "stable" else 180
        if age_days > threshold:
            return [
                f"The newest supporting source is {age_days} days old; current availability "
                "should be re-verified."
            ]
        return []

    @staticmethod
    def _conflicts(record: GuideRecord) -> list[KnowledgeConflict]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for evidence in record.claim_evidence:
            key = str(evidence.get("claim_key") or "").strip()
            value = str(evidence.get("value") or "").strip()
            if key and value:
                grouped.setdefault(key, []).append(evidence)
        conflicts: list[KnowledgeConflict] = []
        authority = {"official": 3, "structured_community": 2, "editorial": 1}
        for key, evidence in grouped.items():
            values = list(dict.fromkeys(str(value["value"]) for value in evidence))
            if len(values) < 2:
                continue
            ranked = sorted(
                evidence,
                key=lambda value: (
                    authority.get(str(value.get("authority")), 0),
                    str(value.get("updated_at") or ""),
                ),
                reverse=True,
            )
            top_rank = authority.get(str(ranked[0].get("authority")), 0)
            next_rank = authority.get(str(ranked[1].get("authority")), 0)
            resolved = top_rank > next_rank
            conflicts.append(
                KnowledgeConflict(
                    claim_key=key,
                    values=values,
                    source_ids=list(
                        dict.fromkeys(str(value.get("source_id")) for value in evidence)
                    ),
                    status="resolved" if resolved else "unresolved",
                    resolution=(
                        f"Preferred {ranked[0]['source_id']} because it has higher "
                        "source authority."
                        if resolved
                        else None
                    ),
                )
            )
        return conflicts

    async def _matching_records(
        self, query: str, guide_type: GuideType | None
    ) -> tuple[list[tuple[GuideRecord, float]], dict[str, Any] | None]:
        records = await self.source.records()
        canonical = await self._resolve_canonical(query, records)
        search_phrase = canonical["name"] if canonical else query
        scored = [
            (record, self._score(search_phrase, record))
            for record in records
            if guide_type is None or record.guide_type == guide_type
        ]
        threshold = 0.28 if canonical else 0.4
        scored = [value for value in scored if value[1] >= threshold]
        scored.sort(key=lambda value: (value[1], value[0].title), reverse=True)
        return scored, canonical

    def _cache_key(self, operation: str, entity: str, guide_type: str | None) -> str:
        return ":".join((self.source.version, operation, _normalized(entity), guide_type or "any"))

    @staticmethod
    def _source_summary(record: GuideRecord) -> list[dict[str, Any]]:
        return [
            {
                "source_id": value.source_id,
                "title": value.title,
                "provider": value.provider,
                "source_url": value.source_url,
                "retrieved_at": value.retrieved_at.isoformat(),
                "published_at": value.published_at.isoformat() if value.published_at else None,
                "updated_at": value.updated_at.isoformat() if value.updated_at else None,
                "relevant_section": value.relevant_section,
                "confidence": value.confidence,
                "relevance_score": value.relevance_score,
                "time_sensitive": value.time_sensitive,
                "factual_claims": value.factual_claims,
            }
            for value in record.sources
        ]

    async def search_guides(
        self,
        query: str,
        entity_name: str | None = None,
        guide_type: GuideType | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        combined = " ".join(value for value in (entity_name, query) if value)
        if self._is_volatile(combined):
            return self._volatile_result(query)
        records = await self.source.records()
        canonical = await self._resolve_canonical(entity_name or query, records)
        key = (
            self._cache_key("search", canonical["name"] if canonical else combined, guide_type)
            + f":limit={limit}"
        )
        if cached := await self.cache.get(key):
            cached["cache"] = {"status": "hit", "key": key}
            return cached
        matches, canonical = await self._matching_records(combined, guide_type)
        result = {
            "query": query,
            "resolved_canonical_entity": canonical,
            "results": [
                {
                    "guide_id": record.guide_id,
                    "title": record.title,
                    "subject": record.subject,
                    "guide_type": record.guide_type,
                    "provider": self.source_name,
                    "relevant_summary": record.summary,
                    "associated_entities": [
                        value.model_dump() for value in record.associated_entities
                    ],
                    "freshness": record.freshness,
                    "updated_at": (
                        newest.isoformat()
                        if (
                            newest := max(
                                (
                                    value.updated_at or value.published_at
                                    for value in record.sources
                                    if value.updated_at or value.published_at
                                ),
                                default=None,
                            )
                        )
                        else None
                    ),
                    "relevance_score": round(score, 3),
                    "confidence": (
                        "high"
                        if any(value.confidence == "high" for value in record.sources)
                        else "medium"
                    ),
                    "sources": self._source_summary(record),
                    "conflicts": [value.model_dump() for value in self._conflicts(record)],
                    "warnings": record.warnings + self._freshness_warnings(record),
                }
                for record, score in matches[:limit]
            ],
            "count": min(len(matches), limit),
            "provider": self.source_name,
            "source": self.source_name,
            "requires_live_provider": False,
            "cache": {"status": "miss", "key": key},
        }
        ttl = min((self._ttl(value.freshness) for value, _ in matches[:limit]), default=0)
        await self.cache.set(key, result, ttl)
        return result

    async def get_guide(
        self, entity_or_query: str, guide_type: GuideType | None = None
    ) -> dict[str, Any]:
        if self._is_volatile(entity_or_query):
            return self._volatile_result(entity_or_query)
        records = await self.source.records()
        canonical = await self._resolve_canonical(entity_or_query, records)
        key = self._cache_key(
            "guide", canonical["name"] if canonical else entity_or_query, guide_type
        )
        if cached := await self.cache.get(key):
            cached["cache"] = {"status": "hit", "key": key}
            return cached
        matches, canonical = await self._matching_records(entity_or_query, guide_type)
        if not matches:
            return {
                "found": False,
                "query": entity_or_query,
                "resolved_canonical_entity": canonical,
                "provider": self.source_name,
                "source": self.source_name,
                "requires_live_provider": False,
                "warnings": ["No maintained detailed guide matched this query."],
                "cache": {"status": "miss", "key": key},
            }
        record, score = matches[0]
        conflicts = self._conflicts(record)
        unresolved = [value for value in conflicts if value.status == "unresolved"]
        warnings = record.warnings + self._freshness_warnings(record)
        if unresolved:
            warnings.append(
                "Supporting sources conflict on one or more claims; do not present those claims "
                "as certain."
            )
        guide = {
            "subject": record.subject,
            "guide_type": record.guide_type,
            "summary": record.summary,
            "associated_entities": [value.model_dump() for value in record.associated_entities],
        }
        for field in ("prerequisites", "steps", "reward", "encounters", "farming"):
            value = getattr(record, field)
            if value:
                guide[field] = value
        result = {
            "found": True,
            "query": entity_or_query,
            "resolved_canonical_entity": canonical,
            "provider": self.source_name,
            "source": self.source_name,
            "guide": guide,
            "freshness": record.freshness,
            "relevance_score": round(score, 3),
            "factual_claims": record.factual_claims,
            "sources": self._source_summary(record),
            "conflicts": [value.model_dump() for value in conflicts],
            "warnings": warnings,
            "requires_live_provider": record.freshness == "volatile",
            "cache": {"status": "miss", "key": key},
        }
        await self.cache.set(key, result, self._ttl(record.freshness))
        return result

    def _volatile_result(self, query: str) -> dict[str, Any]:
        result = unavailable_live_data(query)
        return {
            **result,
            "found": False,
            "resolved_canonical_entity": None,
            "provider": self.source_name,
            "source": self.source_name,
            "warnings": [result["warning"]],
            "cache": {"status": "bypass", "key": None},
        }

    async def status(self) -> dict[str, Any]:
        records = await self.source.records()
        return {
            "source": self.source_name,
            "guide_count": len(records),
            "corpus_version": self.source.version,
            "guide_types": sorted({value.guide_type for value in records}),
            "cache": type(self.cache).__name__,
            "stable_ttl_seconds": self.settings.guide_cache_stable_ttl_seconds,
            "semi_stable_ttl_seconds": self.settings.guide_cache_semi_stable_ttl_seconds,
        }
