import asyncio
import hashlib
import json
import logging
import re
import unicodedata
from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.bungie.client import BungieClient
from app.bungie.manifest import DefinitionResolver
from app.config import Settings
from app.knowledge_models import LiveEffectiveWindow, LiveKnowledgeSource
from app.live_knowledge import LIVE_PROVIDER_CATEGORY, unavailable_live_data, volatile_topic

logger = logging.getLogger(__name__)

WEEKLY_CATEGORIES = ("dungeon", "raid", "nightfall", "exotic_mission", "general")
XUR_HASH = 2190858386
AUTHORITY_RANK = {"official": 4, "structured_community": 3, "editorial": 2, "derived": 1}


def _normalized(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return " ".join(re.findall(r"[a-z0-9]+", ascii_value.casefold()))


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(UTC)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat().replace("+00:00", "Z") if value else None


def _display(definition: dict[str, Any]) -> tuple[str, str]:
    display = definition.get("displayProperties") or {}
    return str(display.get("name") or ""), str(display.get("description") or "")


def _reset_window(now: datetime, cadence: str) -> LiveEffectiveWindow:
    now = now.astimezone(UTC)
    if cadence == "daily":
        start = now.replace(hour=17, minute=0, second=0, microsecond=0)
        if start > now:
            start -= timedelta(days=1)
        end = start + timedelta(days=1)
        reset_cadence = "daily"
    else:
        days_since_tuesday = (now.weekday() - 1) % 7
        start = (now - timedelta(days=days_since_tuesday)).replace(
            hour=17, minute=0, second=0, microsecond=0
        )
        if start > now:
            start -= timedelta(days=7)
        end = start + timedelta(days=7)
        reset_cadence = "weekly"
    return LiveEffectiveWindow(
        effective_from=start,
        effective_until=end,
        reset_cadence=reset_cadence,
        stale_after=end,
        valid_at_retrieval=start <= now < end,
    )


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


LIVE_DESTINY_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    _strict_tool(
        "get_live_destiny_status",
        "Get source health, reset windows, and a compact overview of current live data.",
        {},
    ),
    _strict_tool(
        "get_weekly_rotation",
        "Get a weekly rotation only when a current source explicitly establishes it.",
        {
            "category": {
                "type": ["string", "null"],
                "enum": [*WEEKLY_CATEGORIES, None],
                "description": "Dungeon, raid, Nightfall, exotic mission, or general.",
            }
        },
    ),
    _strict_tool(
        "get_vendor_status",
        "Get a current public vendor location and inventory when reliably exposed.",
        {"vendor": {"type": "string", "description": "Vendor name, such as Xur."}},
    ),
    _strict_tool(
        "get_current_activity_status",
        (
            "Check whether an activity is explicitly represented by current public live data. "
            "This does not infer weekly featured status."
        ),
        {
            "activity_name_or_hash": {
                "type": "string",
                "description": "Activity name or Manifest hash.",
            }
        },
    ),
    _strict_tool(
        "search_live_destiny",
        "Search current normalized Destiny data while preserving live-source limitations.",
        {"query": {"type": "string", "description": "Current or rotating-state question."}},
    ),
]


class LiveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WeeklyRotationRequest(LiveRequest):
    category: str | None = None


class VendorStatusRequest(LiveRequest):
    vendor: str = Field(min_length=1, max_length=100)


class ActivityStatusRequest(LiveRequest):
    activity_name_or_hash: str = Field(min_length=1, max_length=180)


class SearchLiveRequest(LiveRequest):
    query: str = Field(min_length=1, max_length=300)


class LiveCache(Protocol):
    async def get(self, key: str, now: datetime) -> dict[str, Any] | None: ...

    async def set(self, key: str, value: dict[str, Any], stale_after: datetime) -> None: ...


class MemoryLiveCache:
    def __init__(self) -> None:
        self.values: dict[str, tuple[datetime, dict[str, Any]]] = {}

    async def get(self, key: str, now: datetime) -> dict[str, Any] | None:
        cached = self.values.get(key)
        if cached is None or cached[0] <= now:
            self.values.pop(key, None)
            return None
        return deepcopy(cached[1])

    async def set(self, key: str, value: dict[str, Any], stale_after: datetime) -> None:
        self.values[key] = (stale_after, deepcopy(value))


class FileLiveCache:
    """Reset-aware JSON cache with a replaceable production-facing interface."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._locks: dict[str, asyncio.Lock] = {}

    def _path(self, key: str) -> Path:
        return self.root / f"{hashlib.sha256(key.encode()).hexdigest()}.json"

    async def get(self, key: str, now: datetime) -> dict[str, Any] | None:
        path = self._path(key)

        def read() -> dict[str, Any] | None:
            if not path.is_file():
                return None
            try:
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                stale_after = _parse_datetime(payload.get("stale_after"))
                value = payload.get("value")
                if stale_after is None or stale_after <= now or not isinstance(value, dict):
                    return None
                return value
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                return None

        value = await asyncio.to_thread(read)
        return deepcopy(value) if value is not None else None

    async def set(self, key: str, value: dict[str, Any], stale_after: datetime) -> None:
        path = self._path(key)
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:

            def write() -> None:
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix(".tmp")
                    with temporary.open("w", encoding="utf-8") as handle:
                        json.dump(
                            {"stale_after": _iso(stale_after), "value": value},
                            handle,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    temporary.replace(path)
                except OSError:
                    logger.warning("Could not persist live Destiny cache %s", path)

            await asyncio.to_thread(write)


class LiveSnapshotSource(Protocol):
    source_name: str
    authority: str

    async def retrieve(self, now: datetime) -> dict[str, Any]: ...


class BungiePublicLiveSource:
    """Official public current-state endpoints, normalized without player payloads."""

    source_name = "bungie_public_live"
    authority = "official"

    def __init__(self, client: BungieClient, resolver: DefinitionResolver) -> None:
        self.client = client
        self.resolver = resolver

    async def retrieve(self, now: datetime) -> dict[str, Any]:
        milestone_result, vendor_result = await asyncio.gather(
            self.client.get_public_milestones(),
            self.client.get_public_vendors(),
            return_exceptions=True,
        )
        warnings: list[str] = []
        milestones: list[dict[str, Any]] = []
        vendors: list[dict[str, Any]] = []
        if isinstance(milestone_result, Exception):
            warnings.append("Bungie's public milestones endpoint was unavailable.")
        else:
            milestones = await self._milestones(milestone_result)
        if isinstance(vendor_result, Exception):
            warnings.append("Bungie's public vendors endpoint was unavailable.")
        else:
            vendors = await self._vendors(vendor_result)
            warnings.append(
                "Bungie's public vendors endpoint is a preview with a limited public subset."
            )

        stale_candidates = [now + timedelta(minutes=5)]
        stale_candidates.extend(
            value
            for value in (
                *(_parse_datetime(item.get("effective_until")) for item in milestones),
                *(_parse_datetime(item.get("next_refresh")) for item in vendors),
            )
            if value is not None and value > now
        )
        stale_after = min(stale_candidates)
        rotations = self._explicit_rotations(milestones)
        source = LiveKnowledgeSource(
            source_id="bungie-public-platform",
            title="Bungie public Destiny milestones",
            provider=self.source_name,
            source_url="https://www.bungie.net/Platform/Destiny2/Milestones/",
            authority="official",
            retrieved_at=now,
        )
        vendor_source = LiveKnowledgeSource(
            source_id="bungie-public-vendors",
            title="Bungie public Destiny vendors",
            provider=self.source_name,
            source_url=("https://www.bungie.net/Platform/Destiny2/Vendors/?components=400,401,402"),
            authority="official",
            retrieved_at=now,
        )
        return {
            "source": source.model_dump(mode="json"),
            "vendor_source": vendor_source.model_dump(mode="json"),
            "sources": [
                source.model_dump(mode="json"),
                vendor_source.model_dump(mode="json"),
            ],
            "retrieved_at": _iso(now),
            "stale_after": _iso(stale_after),
            "milestones": milestones,
            "vendors": vendors,
            "rotations": rotations,
            "warnings": warnings,
        }

    async def _milestones(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        milestone_hashes = {int(value) for value in payload}
        milestone_defs = await self.resolver.resolve_many(
            "DestinyMilestoneDefinition", milestone_hashes
        )
        activity_hashes = {
            int(activity["activityHash"])
            for milestone in payload.values()
            for activity in milestone.get("activities", [])
            if activity.get("activityHash") is not None
        }
        activity_defs = await self.resolver.resolve_many(
            "DestinyActivityDefinition", activity_hashes
        )
        modifier_hashes = {
            int(value)
            for milestone in payload.values()
            for activity in milestone.get("activities", [])
            for value in activity.get("modifierHashes", [])
        }
        modifier_defs = await self.resolver.resolve_many(
            "DestinyActivityModifierDefinition", modifier_hashes
        )
        normalized: list[dict[str, Any]] = []
        for hash_text, current in payload.items():
            milestone_hash = int(hash_text)
            definition = milestone_defs.get(milestone_hash, {})
            name, description = _display(definition)
            activities: list[dict[str, Any]] = []
            for activity in current.get("activities", []):
                activity_hash = int(activity.get("activityHash") or 0)
                activity_name, activity_description = _display(activity_defs.get(activity_hash, {}))
                modifiers = []
                for modifier_hash in activity.get("modifierHashes", []):
                    modifier_name, modifier_description = _display(
                        modifier_defs.get(int(modifier_hash), {})
                    )
                    modifiers.append(
                        {
                            "hash": int(modifier_hash),
                            "name": modifier_name or None,
                            "description": modifier_description or None,
                        }
                    )
                activities.append(
                    {
                        "hash": activity_hash,
                        "name": activity_name or None,
                        "description": activity_description or None,
                        "modifiers": modifiers,
                    }
                )
            normalized.append(
                {
                    "hash": milestone_hash,
                    "name": name or definition.get("friendlyName") or f"Milestone {hash_text}",
                    "description": description or None,
                    "effective_from": current.get("startDate"),
                    "effective_until": current.get("endDate"),
                    "activities": activities,
                }
            )
        return normalized

    async def _vendors(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        vendor_data = (payload.get("vendors") or {}).get("data") or {}
        sales_data = (payload.get("sales") or {}).get("data") or {}
        vendor_defs = await self.resolver.resolve_many(
            "DestinyVendorDefinition", {int(value) for value in vendor_data}
        )
        item_hashes = {
            int(sale["itemHash"])
            for values in sales_data.values()
            for sale in (values.get("saleItems") or {}).values()
            if sale.get("itemHash") is not None
        }
        item_defs = await self.resolver.resolve_many("DestinyInventoryItemDefinition", item_hashes)
        destination_hashes = {
            int(location["destinationHash"])
            for definition in vendor_defs.values()
            for location in definition.get("locations", [])
            if location.get("destinationHash") is not None
        }
        destination_defs = await self.resolver.resolve_many(
            "DestinyDestinationDefinition", destination_hashes
        )
        vendors: list[dict[str, Any]] = []
        for hash_text, current in vendor_data.items():
            vendor_hash = int(hash_text)
            definition = vendor_defs.get(vendor_hash, {})
            name, description = _display(definition)
            possible_locations = []
            for location in definition.get("locations", []):
                destination_hash = int(location.get("destinationHash") or 0)
                destination_name, _ = _display(destination_defs.get(destination_hash, {}))
                possible_locations.append(
                    {"destination_hash": destination_hash, "destination": destination_name or None}
                )
            location_index = current.get("vendorLocationIndex")
            current_locations = []
            if isinstance(location_index, int) and 0 <= location_index < len(possible_locations):
                current_locations = [possible_locations[location_index]]
            inventory = []
            for sale in (sales_data.get(hash_text, {}).get("saleItems") or {}).values():
                item_hash = int(sale.get("itemHash") or 0)
                item_name, item_description = _display(item_defs.get(item_hash, {}))
                inventory.append(
                    {
                        "item_hash": item_hash,
                        "name": item_name or None,
                        "description": item_description or None,
                    }
                )
            vendors.append(
                {
                    "vendor_hash": vendor_hash,
                    "name": name or f"Vendor {hash_text}",
                    "description": description or None,
                    "enabled": bool(current.get("enabled", True)),
                    "available": bool(current.get("enabled", True)),
                    "locations": current_locations or possible_locations,
                    "location_precision": (
                        "current_destination"
                        if current_locations
                        else "definition_possible_destination"
                        if possible_locations
                        else "not_exposed"
                    ),
                    "inventory": inventory,
                    "next_refresh": current.get("nextRefreshDate"),
                }
            )
        return vendors

    @staticmethod
    def _explicit_rotations(milestones: list[dict[str, Any]]) -> dict[str, Any]:
        """Only classify source text that explicitly names a current rotation category."""
        rotations: dict[str, Any] = {}
        patterns = {
            "dungeon": re.compile(r"\bfeatured\s+dungeon\b"),
            "raid": re.compile(r"\bfeatured\s+raid\b"),
            "nightfall": re.compile(r"\bnightfall\b"),
            "exotic_mission": re.compile(r"\bexotic\s+mission\b"),
        }
        for category, pattern in patterns.items():
            entries: list[dict[str, Any]] = []
            for milestone in milestones:
                text = _normalized(
                    " ".join(
                        [
                            str(milestone.get("name") or ""),
                            str(milestone.get("description") or ""),
                        ]
                    )
                )
                if not pattern.search(text):
                    continue
                activities = milestone.get("activities") or []
                if activities:
                    entries.extend(
                        {
                            "name": value.get("name"),
                            "hash": value.get("hash"),
                            "effective_from": milestone.get("effective_from"),
                            "effective_until": milestone.get("effective_until"),
                        }
                        for value in activities
                        if value.get("name")
                    )
                else:
                    entries.append(
                        {
                            "name": milestone.get("name"),
                            "hash": milestone.get("hash"),
                            "effective_from": milestone.get("effective_from"),
                            "effective_until": milestone.get("effective_until"),
                        }
                    )
            if entries:
                rotations[category] = {
                    "entries": entries,
                    "explicit": True,
                    "confidence": "high",
                }
        return rotations


class LiveDestinyProvider:
    source_name = "live_destiny"
    knowledge_category = LIVE_PROVIDER_CATEGORY
    tool_names = frozenset(value["name"] for value in LIVE_DESTINY_TOOL_DEFINITIONS)

    def __init__(
        self,
        client: BungieClient,
        resolver: DefinitionResolver,
        settings: Settings,
        *,
        canonical: Any | None = None,
        sources: list[LiveSnapshotSource] | None = None,
        cache: LiveCache | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = settings
        self.canonical = canonical
        self.sources = sources or [BungiePublicLiveSource(client, resolver)]
        cache_root = settings.resolve_local_path(settings.live_cache_dir)
        self.cache = cache or FileLiveCache(cache_root)
        self._now = now or (lambda: datetime.now(UTC))

    @staticmethod
    def definitions() -> list[dict[str, Any]]:
        return LIVE_DESTINY_TOOL_DEFINITIONS

    def handles(self, name: str) -> bool:
        return name in self.tool_names

    async def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "get_live_destiny_status":
            LiveRequest.model_validate(arguments)
            return await self.get_status()
        if name == "get_weekly_rotation":
            request = WeeklyRotationRequest.model_validate(arguments)
            return await self.get_weekly_rotation(request.category)
        if name == "get_vendor_status":
            request = VendorStatusRequest.model_validate(arguments)
            return await self.get_vendor_status(request.vendor)
        if name == "get_current_activity_status":
            request = ActivityStatusRequest.model_validate(arguments)
            return await self.get_current_activity_status(request.activity_name_or_hash)
        if name == "search_live_destiny":
            request = SearchLiveRequest.model_validate(arguments)
            return await self.search_live_destiny(request.query)
        raise ValueError(f"Unknown live Destiny tool: {name}")

    async def _snapshots(self) -> tuple[list[dict[str, Any]], str]:
        now = self._now()
        values: list[dict[str, Any]] = []
        statuses: list[str] = []
        for source in self.sources:
            key = f"live-snapshot:v2:{source.source_name}"
            try:
                cached = await self.cache.get(key, now)
            except Exception:
                logger.exception("Live Destiny cache read failed key=%s", key)
                cached = None
                statuses.append("cache_error")
            if cached is not None:
                values.append(cached)
                statuses.append("hit")
                continue
            try:
                snapshot = await source.retrieve(now)
            except Exception:
                logger.exception("Live Destiny source failed source=%s", source.source_name)
                statuses.append("error")
                continue
            configured_ttl = self.settings.live_cache_ttl_seconds
            fallback_stale = now + timedelta(seconds=configured_ttl)
            source_stale = _parse_datetime(snapshot.get("stale_after"))
            stale_after = min(
                fallback_stale,
                source_stale if source_stale and source_stale > now else fallback_stale,
            )
            if configured_ttl > 0:
                snapshot = deepcopy(snapshot)
                snapshot["cache_stale_after"] = _iso(stale_after)
                try:
                    await self.cache.set(key, snapshot, stale_after)
                except Exception:
                    logger.exception("Live Destiny cache write failed key=%s", key)
            values.append(snapshot)
            statuses.append("miss")
        cache_status = statuses[0] if len(set(statuses)) == 1 and statuses else "mixed"
        return values, cache_status

    def _base(
        self,
        snapshots: list[dict[str, Any]],
        cache_status: str,
        supported_topics: list[str],
        live_available: bool,
    ) -> dict[str, Any]:
        now = self._now()
        retrieved = [
            value
            for snapshot in snapshots
            if (value := _parse_datetime(snapshot.get("retrieved_at"))) is not None
        ]
        stale = [
            value
            for snapshot in snapshots
            if (
                value := _parse_datetime(
                    snapshot.get("cache_stale_after") or snapshot.get("stale_after")
                )
            )
            is not None
        ]
        stale_after = min(stale) if stale else now
        authorities = {
            str((snapshot.get("source") or {}).get("authority")) for snapshot in snapshots
        }
        window = LiveEffectiveWindow(
            effective_from=None,
            effective_until=stale_after,
            reset_cadence="unknown",
            stale_after=stale_after,
            valid_at_retrieval=bool(stale_after > now),
        )
        return {
            "source": self.source_name,
            "provider": self.source_name,
            "knowledge_category": self.knowledge_category,
            "freshness": "volatile",
            "live_data_available": live_available,
            "requires_live_provider": not live_available,
            "supported_live_topics": supported_topics,
            "source_confidence": (
                "high" if "official" in authorities else "medium" if snapshots else "unknown"
            ),
            "retrieved_at": _iso(max(retrieved) if retrieved else now),
            "effective_window": window.model_dump(mode="json"),
            "cache": {"status": cache_status, "reset_aware": True},
            "sources": [
                source
                for snapshot in snapshots
                for source in (
                    snapshot.get("sources")
                    or ([snapshot.get("source")] if snapshot.get("source") else [])
                )
            ],
            "warnings": [
                warning for snapshot in snapshots for warning in snapshot.get("warnings", [])
            ],
        }

    async def get_status(self) -> dict[str, Any]:
        snapshots, cache_status = await self._snapshots()
        milestones = [item for value in snapshots for item in value.get("milestones", [])]
        vendors = [item for value in snapshots for item in value.get("vendors", [])]
        rotations: dict[str, Any] = {}
        conflicts: list[dict[str, Any]] = []
        supported = ["live_status"] if snapshots else []
        topic_by_category = {
            "dungeon": "featured_dungeon",
            "raid": "featured_raid",
            "nightfall": "weekly_nightfall",
            "exotic_mission": "current_exotic_mission",
        }
        for category, topic in topic_by_category.items():
            claims = [
                {
                    **snapshot["rotations"][category],
                    "source": snapshot.get("source"),
                }
                for snapshot in snapshots
                if category in snapshot.get("rotations", {})
                and snapshot["rotations"][category].get("explicit") is True
            ]
            resolved, conflict = self._resolve_rotation_claim(category, claims)
            rotations[category] = resolved
            if resolved.get("known") is True:
                supported.append(topic)
            if conflict:
                conflicts.append(conflict)
        result = self._base(snapshots, cache_status, supported, bool(snapshots))
        result.update(
            {
                "status": "current" if snapshots else "unavailable",
                "weekly_rotation": rotations,
                "public_milestones": milestones,
                "public_vendors": [
                    {
                        "vendor_hash": value.get("vendor_hash"),
                        "name": value.get("name"),
                        "enabled": value.get("enabled"),
                        "next_refresh": value.get("next_refresh"),
                    }
                    for value in vendors
                ],
                "reset_windows": {
                    "daily": _reset_window(self._now(), "daily").model_dump(mode="json"),
                    "weekly": _reset_window(self._now(), "weekly").model_dump(mode="json"),
                    "derivation": (
                        "Calendar boundaries only; this does not identify what changed at reset."
                    ),
                },
                "conflicts": conflicts,
                "limitations": [
                    "Public milestone presence does not by itself mean an activity is featured.",
                    "This overview is not a complete inventory of everything available in-game.",
                ],
            }
        )
        return result

    async def get_weekly_rotation(self, category: str | None = None) -> dict[str, Any]:
        category = category or "general"
        if category not in WEEKLY_CATEGORIES:
            raise ValueError(f"Unsupported weekly rotation category: {category}")
        snapshots, cache_status = await self._snapshots()
        categories = (
            ["dungeon", "raid", "nightfall", "exotic_mission"]
            if category == "general"
            else [category]
        )
        rotations: dict[str, Any] = {}
        conflicts: list[dict[str, Any]] = []
        supported: list[str] = []
        for current_category in categories:
            claims = [
                {
                    **snapshot["rotations"][current_category],
                    "source": snapshot.get("source"),
                }
                for snapshot in snapshots
                if current_category in snapshot.get("rotations", {})
                and snapshot["rotations"][current_category].get("explicit") is True
            ]
            resolved, conflict = self._resolve_rotation_claim(current_category, claims)
            rotations[current_category] = resolved
            if conflict:
                conflicts.append(conflict)
            if resolved.get("known") is True:
                supported.append(
                    {
                        "dungeon": "featured_dungeon",
                        "raid": "featured_raid",
                        "nightfall": "weekly_nightfall",
                        "exotic_mission": "current_exotic_mission",
                    }[current_category]
                )
        if category == "general" and len(supported) == len(categories):
            supported.append("weekly_rotation")
        live_available = bool(supported)
        result = self._base(snapshots, cache_status, supported, live_available)
        rotation_entries = [
            entry for rotation in rotations.values() for entry in rotation.get("entries", [])
        ]
        effective_from = [
            parsed
            for entry in rotation_entries
            if (parsed := _parse_datetime(entry.get("effective_from"))) is not None
        ]
        effective_until = [
            parsed
            for entry in rotation_entries
            if (parsed := _parse_datetime(entry.get("effective_until"))) is not None
        ]
        current_stale = _parse_datetime(result["effective_window"]["stale_after"])
        rotation_until = min(effective_until) if effective_until else current_stale
        if current_stale and rotation_until:
            rotation_until = min(rotation_until, current_stale)
        result["effective_window"] = LiveEffectiveWindow(
            effective_from=min(effective_from) if effective_from else None,
            effective_until=min(effective_until) if effective_until else None,
            reset_cadence="weekly",
            stale_after=rotation_until or self._now(),
            valid_at_retrieval=bool(rotation_until and rotation_until > self._now()),
        ).model_dump(mode="json")
        result.update(
            {
                "status": "current" if live_available else "unsupported_by_current_sources",
                "requested_category": category,
                "rotations": rotations,
                "conflicts": conflicts,
                "authoritative_vs_derived": (
                    "Only explicitly labeled current source values qualify."
                ),
                "limitations": [
                    "An activity appearing in public milestones is not treated as featured.",
                    "Farmability is unknown unless a current source explicitly supplies it.",
                ],
            }
        )
        if not live_available:
            result["warnings"].append(
                "Connected live sources do not explicitly establish the requested rotation."
            )
        return result

    @staticmethod
    def _resolve_rotation_claim(
        category: str, claims: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        if not claims:
            return {
                "known": False,
                "entries": [],
                "reason": "No current source explicitly labels this rotation category.",
            }, None
        grouped: dict[tuple[tuple[str, int | None], ...], list[dict[str, Any]]] = {}
        for claim in claims:
            identity = tuple(
                sorted(
                    (str(value.get("name") or ""), value.get("hash"))
                    for value in claim.get("entries", [])
                )
            )
            grouped.setdefault(identity, []).append(claim)
        if len(grouped) == 1:
            best = max(
                claims,
                key=lambda value: AUTHORITY_RANK.get(
                    str((value.get("source") or {}).get("authority")), 0
                ),
            )
            return {
                "known": True,
                "entries": best.get("entries", []),
                "confidence": best.get("confidence", "medium"),
                "source_ids": [(value.get("source") or {}).get("source_id") for value in claims],
            }, None
        ranked = sorted(
            claims,
            key=lambda value: AUTHORITY_RANK.get(
                str((value.get("source") or {}).get("authority")), 0
            ),
            reverse=True,
        )
        first_rank = AUTHORITY_RANK.get(str((ranked[0].get("source") or {}).get("authority")), 0)
        second_rank = AUTHORITY_RANK.get(str((ranked[1].get("source") or {}).get("authority")), 0)
        source_ids = [(value.get("source") or {}).get("source_id") for value in claims]
        conflict_values = [
            [entry.get("name") for entry in value.get("entries", [])] for value in claims
        ]
        if first_rank > second_rank:
            conflict = {
                "claim_key": f"weekly_rotation.{category}",
                "values": conflict_values,
                "status": "resolved",
                "source_ids": source_ids,
                "resolution": "Higher-authority current source selected.",
            }
            return {
                "known": True,
                "entries": ranked[0].get("entries", []),
                "confidence": ranked[0].get("confidence", "medium"),
                "source_ids": [(ranked[0].get("source") or {}).get("source_id")],
            }, conflict
        return {
            "known": False,
            "entries": [],
            "reason": "Equally authoritative current sources conflict.",
        }, {
            "claim_key": f"weekly_rotation.{category}",
            "values": conflict_values,
            "status": "unresolved",
            "source_ids": source_ids,
            "resolution": None,
        }

    async def get_vendor_status(self, vendor: str) -> dict[str, Any]:
        snapshots, cache_status = await self._snapshots()
        query = _normalized(vendor).replace("xur", "xur")
        claims = [
            {
                "vendor": item,
                "source": snapshot.get("vendor_source") or snapshot.get("source") or {},
            }
            for snapshot in snapshots
            for item in snapshot.get("vendors", [])
            if query in _normalized(str(item.get("name") or ""))
            or str(item.get("vendor_hash")) == vendor.strip()
        ]
        conflicts: list[dict[str, Any]] = []
        selected: dict[str, Any] | None = claims[0]["vendor"] if claims else None
        if len(claims) > 1:
            identities = {
                (
                    bool(claim["vendor"].get("enabled")),
                    tuple(
                        sorted(
                            str(value.get("destination") or "")
                            for value in claim["vendor"].get("locations", [])
                        )
                    ),
                    tuple(
                        sorted(
                            str(value.get("name") or "")
                            for value in claim["vendor"].get("inventory", [])
                        )
                    ),
                )
                for claim in claims
            }
            if len(identities) > 1:
                ranked = sorted(
                    claims,
                    key=lambda claim: AUTHORITY_RANK.get(str(claim["source"].get("authority")), 0),
                    reverse=True,
                )
                first_rank = AUTHORITY_RANK.get(str(ranked[0]["source"].get("authority")), 0)
                second_rank = AUTHORITY_RANK.get(str(ranked[1]["source"].get("authority")), 0)
                resolved = first_rank > second_rank
                conflicts.append(
                    {
                        "claim_key": f"vendor.{query}.current_state",
                        "values": [list(identity) for identity in identities],
                        "source_ids": [claim["source"].get("source_id") for claim in claims],
                        "status": "resolved" if resolved else "unresolved",
                        "resolution": (
                            "Higher-authority current source selected." if resolved else None
                        ),
                    }
                )
                selected = ranked[0]["vendor"] if resolved else None
        supported = []
        if selected:
            supported = ["current_vendor_inventory"]
            if query == "xur":
                supported.extend(["xur_inventory", "xur_status"])
                if selected.get("location_precision") == "current_destination":
                    supported.append("xur_location")
        result = self._base(snapshots, cache_status, supported, bool(selected))
        current_stale = _parse_datetime(result["effective_window"]["stale_after"])
        vendor_stale_candidates = [
            value for value in (current_stale,) if value is not None and value > self._now()
        ]
        vendor_stale = min(vendor_stale_candidates) if vendor_stale_candidates else self._now()
        if selected:
            result["effective_window"] = LiveEffectiveWindow(
                effective_from=_parse_datetime(result.get("retrieved_at")),
                effective_until=current_stale,
                reset_cadence="vendor",
                stale_after=vendor_stale,
                valid_at_retrieval=vendor_stale > self._now(),
            ).model_dump(mode="json")
        result.update(
            {
                "status": "current" if selected else "not_exposed_by_current_sources",
                "vendor_query": vendor,
                "vendor": selected,
                "claim_confidence": "high" if selected and not conflicts else "unknown",
                "conflicts": conflicts,
                "limitations": [
                    "Public vendor data is a smaller subset than authenticated vendor data.",
                    "A destination-only location does not establish an exact map position.",
                    "next_refresh is an inventory refresh time, not a visibility end time.",
                    "Inventory omits owned-item comparisons and may omit roll-specific details.",
                ],
            }
        )
        if not selected:
            result["warnings"].append(
                "Current vendor sources conflict; do not present either state as current."
                if claims
                else "The requested vendor is not in Bungie's public vendor set."
            )
        return result

    async def get_current_activity_status(self, value: str) -> dict[str, Any]:
        snapshots, cache_status = await self._snapshots()
        resolved_entity: dict[str, Any] | None = None
        if self.canonical is not None:
            search = await self.canonical.search_entities(value, ["activity"], 3)
            results = search.get("results", [])
            if results:
                resolved_entity = results[0]
        queries = {_normalized(value)}
        hashes = {value.strip()}
        if resolved_entity:
            queries.add(_normalized(str(resolved_entity.get("name") or "")))
            hashes.add(str(resolved_entity.get("hash")))
        matches = []
        for snapshot in snapshots:
            for milestone in snapshot.get("milestones", []):
                milestone_match = (
                    any(query in _normalized(str(milestone.get("name") or "")) for query in queries)
                    or str(milestone.get("hash")) in hashes
                )
                activities = [
                    activity
                    for activity in milestone.get("activities", [])
                    if any(
                        query in _normalized(str(activity.get("name") or "")) for query in queries
                    )
                    or str(activity.get("hash")) in hashes
                ]
                if milestone_match or activities:
                    matches.append(
                        {
                            "milestone": {
                                "hash": milestone.get("hash"),
                                "name": milestone.get("name"),
                                "effective_from": milestone.get("effective_from"),
                                "effective_until": milestone.get("effective_until"),
                            },
                            "activities": activities,
                            "listed_in_public_milestones": True,
                            "featured": None,
                            "farmable": None,
                        }
                    )
        supported = ["current_activity_status"] if matches else []
        result = self._base(snapshots, cache_status, supported, bool(matches))
        if matches:
            starts = [
                parsed
                for match in matches
                if (parsed := _parse_datetime(match["milestone"].get("effective_from"))) is not None
            ]
            ends = [
                parsed
                for match in matches
                if (parsed := _parse_datetime(match["milestone"].get("effective_until")))
                is not None
            ]
            current_stale = _parse_datetime(result["effective_window"]["stale_after"])
            activity_stale = min([*ends, current_stale or self._now()])
            result["effective_window"] = LiveEffectiveWindow(
                effective_from=min(starts) if starts else None,
                effective_until=min(ends) if ends else None,
                reset_cadence="weekly",
                stale_after=activity_stale,
                valid_at_retrieval=activity_stale > self._now(),
            ).model_dump(mode="json")
        result.update(
            {
                "status": "current_public_milestone" if matches else "not_established",
                "activity_query": value,
                "resolved_canonical_entity": resolved_entity,
                "matches": matches,
                "claim_confidence": "high" if matches else "unknown",
                "conflicts": [],
                "limitations": [
                    "No match does not prove that the activity is unavailable in the Director.",
                    "Public milestone presence does not establish featured or farmable status.",
                ],
            }
        )
        return result

    async def search_live_destiny(self, query: str) -> dict[str, Any]:
        topic = volatile_topic(query)
        if topic == "featured_dungeon":
            return await self.get_weekly_rotation("dungeon")
        if topic == "featured_raid":
            return await self.get_weekly_rotation("raid")
        if topic == "weekly_nightfall":
            return await self.get_weekly_rotation("nightfall")
        if topic in {"xur_location", "xur_inventory", "xur_status"}:
            return await self.get_vendor_status("Xur")
        if topic:
            result = unavailable_live_data(query, topic)
            snapshots, cache_status = await self._snapshots()
            result.update(self._base(snapshots, cache_status, [], False))
            result["status"] = "unsupported_by_current_sources"
            result["query"] = query
            return result

        snapshots, cache_status = await self._snapshots()
        normalized_query = _normalized(query)
        matches = []
        for snapshot in snapshots:
            for milestone in snapshot.get("milestones", []):
                if normalized_query in _normalized(str(milestone.get("name") or "")):
                    matches.append({"kind": "public_milestone", **milestone})
                for activity in milestone.get("activities", []):
                    if normalized_query in _normalized(str(activity.get("name") or "")):
                        matches.append(
                            {
                                "kind": "public_milestone_activity",
                                "milestone": milestone.get("name"),
                                **activity,
                            }
                        )
                    for modifier in activity.get("modifiers", []):
                        if normalized_query in _normalized(str(modifier.get("name") or "")):
                            matches.append(
                                {
                                    "kind": "current_activity_modifier",
                                    "milestone": milestone.get("name"),
                                    "activity": activity.get("name"),
                                    **modifier,
                                }
                            )
            for vendor in snapshot.get("vendors", []):
                if normalized_query in _normalized(str(vendor.get("name") or "")):
                    matches.append({"kind": "public_vendor", **vendor})
                for item in vendor.get("inventory", []):
                    if normalized_query in _normalized(str(item.get("name") or "")):
                        matches.append(
                            {
                                "kind": "public_vendor_item",
                                "vendor": vendor.get("name"),
                                **item,
                            }
                        )
        result = self._base(
            snapshots,
            cache_status,
            ["live_search"] if matches else [],
            bool(matches),
        )
        result.update(
            {
                "status": "current_matches" if matches else "no_current_match",
                "query": query,
                "results": matches[:20],
                "conflicts": [],
                "limitations": ["Search covers only the connected normalized live sources."],
            }
        )
        return result

    async def status(self) -> dict[str, Any]:
        return {
            "source": self.source_name,
            "knowledge_category": self.knowledge_category,
            "providers": [
                {"source": value.source_name, "authority": value.authority}
                for value in self.sources
            ],
            "cache": "reset_aware_file_or_injected",
            "tool_count": len(self.tool_names),
        }
