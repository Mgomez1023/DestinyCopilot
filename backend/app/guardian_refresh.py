"""Centralized, slice-aware cache and refresh policy for normalized Guardian state."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Protocol

from app.bungie.guardian import GuardianService
from app.config import Settings
from app.content_progression import mentioned_content_names
from app.models import (
    CharacterSummary,
    CollectionProgressSummary,
    CraftingProgressSummary,
    DataAvailability,
    GuardianContext,
    InventorySummary,
    RecordProgressSummary,
)

logger = logging.getLogger(__name__)
CACHE_SCHEMA_VERSION = 2


class GuardianSlice(StrEnum):
    PROFILE = "profile"
    EQUIPMENT = "equipment"
    INVENTORY = "inventory"
    QUESTS_PROGRESS = "quests_progress"
    ACTIVITY_HISTORY = "activity_history"
    COLLECTIONS = "collections"


class RefreshIntent(StrEnum):
    CURRENT_LOADOUT = "current_loadout"
    QUEST_STATUS = "quest_status"
    CURRENT_INVENTORY = "current_inventory"
    RECENT_ACTIVITY = "recent_activity"
    WHAT_SHOULD_I_DO = "what_should_i_do"
    BUILD_ANALYSIS = "build_analysis"


ALL_SLICES = tuple(GuardianSlice)

SLICE_COMPONENTS: dict[GuardianSlice, set[str]] = {
    GuardianSlice.PROFILE: {"Profiles", "Characters"},
    GuardianSlice.EQUIPMENT: {
        "Characters",
        "CharacterEquipment",
        "ItemInstances",
        "ItemSockets",
        "ItemStats",
    },
    GuardianSlice.INVENTORY: {
        "Characters",
        "CharacterEquipment",
        "ProfileInventories",
        "ProfileCurrencies",
        "CharacterInventories",
        "ItemInstances",
        "ItemSockets",
        "ItemStats",
    },
    GuardianSlice.QUESTS_PROGRESS: {
        "Profiles",
        "Characters",
        "CharacterInventories",
        "CharacterProgressions",
        "CharacterActivities",
        "ProfileProgression",
        "ItemInstances",
        "ItemObjectives",
        "Records",
        "Craftables",
    },
    GuardianSlice.ACTIVITY_HISTORY: set(),
    GuardianSlice.COLLECTIONS: {"Profiles", "Characters", "Collectibles"},
}

TOOL_SLICE_DEPENDENCIES: dict[str, set[GuardianSlice]] = {
    "get_character_summary": {GuardianSlice.PROFILE, GuardianSlice.EQUIPMENT},
    "get_equipped_loadout": {GuardianSlice.EQUIPMENT},
    "get_active_quests": {GuardianSlice.QUESTS_PROGRESS},
    "get_content_progression": {GuardianSlice.PROFILE, GuardianSlice.QUESTS_PROGRESS},
    "get_available_activities": {GuardianSlice.PROFILE, GuardianSlice.QUESTS_PROGRESS},
    "get_recent_activities": {GuardianSlice.ACTIVITY_HISTORY},
    "get_progression": {
        GuardianSlice.PROFILE,
        GuardianSlice.QUESTS_PROGRESS,
        GuardianSlice.COLLECTIONS,
    },
    "search_inventory": {
        GuardianSlice.INVENTORY,
        GuardianSlice.EQUIPMENT,
    },
    "get_build_details": {
        GuardianSlice.EQUIPMENT,
        GuardianSlice.INVENTORY,
    },
    "analyze_current_build": {
        GuardianSlice.EQUIPMENT,
        GuardianSlice.INVENTORY,
    },
    "find_build_alternatives": {
        GuardianSlice.EQUIPMENT,
        GuardianSlice.INVENTORY,
    },
}

INTENT_SLICE_DEPENDENCIES: dict[RefreshIntent, set[GuardianSlice]] = {
    RefreshIntent.CURRENT_LOADOUT: {GuardianSlice.EQUIPMENT},
    RefreshIntent.QUEST_STATUS: {GuardianSlice.QUESTS_PROGRESS},
    RefreshIntent.CURRENT_INVENTORY: {
        GuardianSlice.INVENTORY,
        GuardianSlice.EQUIPMENT,
    },
    RefreshIntent.RECENT_ACTIVITY: {GuardianSlice.ACTIVITY_HISTORY},
    RefreshIntent.WHAT_SHOULD_I_DO: {
        GuardianSlice.EQUIPMENT,
        GuardianSlice.QUESTS_PROGRESS,
        GuardianSlice.ACTIVITY_HISTORY,
    },
    RefreshIntent.BUILD_ANALYSIS: {
        GuardianSlice.EQUIPMENT,
        GuardianSlice.INVENTORY,
    },
}


@dataclass
class SliceMetadata:
    fetched_at: datetime
    stale_after: datetime
    last_accessed_at: datetime
    reason: str
    source_components: list[str]
    generation: int
    last_success_at: datetime
    last_error: str | None = None
    last_error_at: datetime | None = None
    cache_hits: int = 0
    cache_misses: int = 0


@dataclass
class CachedGuardianSlice:
    payload: dict[str, Any]
    metadata: SliceMetadata


@dataclass
class GuardianCacheEntry:
    account_key: str
    membership_id: str
    membership_type: int
    slices: dict[GuardianSlice, CachedGuardianSlice]
    created_at: datetime
    last_accessed_at: datetime
    last_app_activity_at: datetime
    generation: int = 1
    schema_version: int = CACHE_SCHEMA_VERSION
    last_app_open_level: int | None = None


class GuardianStateCache(Protocol):
    async def get(self, key: str) -> GuardianCacheEntry | None: ...

    async def set(self, key: str, entry: GuardianCacheEntry) -> None: ...

    async def delete(self, key: str) -> None: ...


class MemoryGuardianStateCache:
    """Process-local implementation behind a production-replaceable interface."""

    def __init__(self) -> None:
        self._entries: dict[str, GuardianCacheEntry] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> GuardianCacheEntry | None:
        async with self._lock:
            value = self._entries.get(key)
            return deepcopy(value) if value is not None else None

    async def set(self, key: str, entry: GuardianCacheEntry) -> None:
        async with self._lock:
            self._entries[key] = deepcopy(entry)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._entries.pop(key, None)


@dataclass
class RefreshOutcome:
    context: GuardianContext
    refreshed_slices: list[GuardianSlice] = field(default_factory=list)
    cache_hit: bool = False
    stale_served: bool = False
    background_refresh: bool = False
    reason: str = "cache"


class GuardianRefreshService:
    """Owns Guardian cache lifetime, invalidation, deduplication, and refresh policy."""

    def __init__(
        self,
        guardian: GuardianService,
        settings: Settings,
        cache: GuardianStateCache | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.guardian = guardian
        self.settings = settings
        self.cache = cache or MemoryGuardianStateCache()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._locks: dict[str, asyncio.Lock] = {}
        self._background: dict[str, asyncio.Task[None]] = {}
        self._pending: dict[str, set[GuardianSlice]] = {}
        self._inflight: dict[str, set[GuardianSlice]] = {}

    def _ttl(self, slice_name: GuardianSlice) -> int:
        return {
            GuardianSlice.PROFILE: self.settings.guardian_cache_profile_ttl_seconds,
            GuardianSlice.EQUIPMENT: self.settings.guardian_cache_equipment_ttl_seconds,
            GuardianSlice.INVENTORY: self.settings.guardian_cache_inventory_ttl_seconds,
            GuardianSlice.QUESTS_PROGRESS: self.settings.guardian_cache_quests_progress_ttl_seconds,
            GuardianSlice.ACTIVITY_HISTORY: (
                self.settings.guardian_cache_activity_history_ttl_seconds
            ),
            GuardianSlice.COLLECTIONS: self.settings.guardian_cache_collections_ttl_seconds,
        }[slice_name]

    def _minimum(self, slice_name: GuardianSlice) -> int:
        return {
            GuardianSlice.PROFILE: self.settings.guardian_soft_profile_min_seconds,
            GuardianSlice.EQUIPMENT: self.settings.guardian_soft_equipment_min_seconds,
            GuardianSlice.INVENTORY: self.settings.guardian_soft_inventory_min_seconds,
            GuardianSlice.QUESTS_PROGRESS: self.settings.guardian_soft_quests_progress_min_seconds,
            GuardianSlice.ACTIVITY_HISTORY: (
                self.settings.guardian_soft_activity_history_min_seconds
            ),
            GuardianSlice.COLLECTIONS: self.settings.guardian_soft_collections_min_seconds,
        }[slice_name]

    async def get_context(
        self,
        session_id: str,
        access_token: str,
        *,
        stale_while_revalidate: bool = True,
        reason: str = "guardian_load",
    ) -> RefreshOutcome:
        entry = await self.cache.get(session_id)
        if entry is not None and entry.schema_version != CACHE_SCHEMA_VERSION:
            await self.clear(session_id)
            entry = None
        if entry is not None and GuardianSlice.PROFILE not in entry.slices:
            await self.clear(session_id)
            entry = None
        if entry is None:
            return await self.refresh(
                session_id, access_token, slices=set(ALL_SLICES), reason="cache_miss"
            )

        now = self._clock()
        stale = {name for name, value in entry.slices.items() if now >= value.metadata.stale_after}
        for value in entry.slices.values():
            value.metadata.last_accessed_at = now
            value.metadata.cache_hits += 1
        entry.last_accessed_at = now
        await self.cache.set(session_id, entry)
        logger.info(
            "Guardian cache hit session=%s stale_slices=%s reason=%s generation=%s",
            session_id[:8],
            sorted(value.value for value in stale),
            reason,
            entry.generation,
        )
        if stale and stale_while_revalidate:
            self._schedule(session_id, access_token, stale, reason)
        return RefreshOutcome(
            context=self._assemble(entry),
            cache_hit=True,
            stale_served=bool(stale),
            background_refresh=bool(stale and stale_while_revalidate),
            reason=reason,
        )

    async def app_open(self, session_id: str, access_token: str) -> RefreshOutcome:
        entry = await self.cache.get(session_id)
        if entry is not None and entry.schema_version != CACHE_SCHEMA_VERSION:
            await self.clear(session_id)
            entry = None
        if entry is None:
            outcome = await self.refresh(
                session_id, access_token, slices=set(ALL_SLICES), reason="app_open_cold"
            )
            created = await self.cache.get(session_id)
            if created:
                created.last_app_open_level = 2
                await self.cache.set(session_id, created)
            return outcome
        now = self._clock()
        idle = max(0.0, (now - entry.last_app_activity_at).total_seconds())
        entry.last_app_activity_at = now
        if idle < self.settings.guardian_app_open_recent_seconds:
            entry.last_app_open_level = 0
            await self.cache.set(session_id, entry)
            return RefreshOutcome(
                context=self._assemble(entry), cache_hit=True, reason="app_open_level_0"
            )
        if idle >= self.settings.guardian_app_open_full_seconds:
            slices = set(ALL_SLICES)
            reason = "app_open_level_2"
            entry.last_app_open_level = 2
        else:
            slices = {
                GuardianSlice.EQUIPMENT,
                GuardianSlice.QUESTS_PROGRESS,
                GuardianSlice.ACTIVITY_HISTORY,
            }
            slices.update(
                name for name, value in entry.slices.items() if now >= value.metadata.stale_after
            )
            reason = "app_open_level_1"
            entry.last_app_open_level = 1
        await self.cache.set(session_id, entry)
        self._schedule(session_id, access_token, slices, reason)
        return RefreshOutcome(
            context=self._assemble(entry),
            cache_hit=True,
            stale_served=True,
            background_refresh=True,
            reason=reason,
        )

    async def for_tool(self, session_id: str, access_token: str, tool_name: str) -> RefreshOutcome:
        dependencies = TOOL_SLICE_DEPENDENCIES.get(tool_name, {GuardianSlice.PROFILE})
        return await self.refresh(
            session_id,
            access_token,
            slices=dependencies,
            force=False,
            soft=True,
            reason=f"tool:{tool_name}",
        )

    async def for_prompt(self, session_id: str, access_token: str, prompt: str) -> RefreshOutcome:
        intents = self.classify_prompt(prompt)
        slices = {
            slice_name for intent in intents for slice_name in INTENT_SLICE_DEPENDENCIES[intent]
        }
        if not slices:
            return await self.get_context(
                session_id, access_token, stale_while_revalidate=False, reason="prompt_no_refresh"
            )
        return await self.refresh(
            session_id,
            access_token,
            slices=slices,
            soft=True,
            reason="prompt:" + "+".join(sorted(value.value for value in intents)),
        )

    @staticmethod
    def classify_prompt(prompt: str) -> set[RefreshIntent]:
        text = prompt.casefold()
        intents: set[RefreshIntent] = set()
        if re.search(
            r"\b(what should i do|what next|do next|work on|worth doing|"
            r"i have \d+ minutes?)\b",
            text,
        ):
            intents.add(RefreshIntent.WHAT_SHOULD_I_DO)
        if re.search(r"\b(build|build analysis)\b", text):
            intents.add(RefreshIntent.BUILD_ANALYSIS)
        elif re.search(r"\b(equipped|loadout|wearing|currently using|current gear)\b", text):
            intents.add(RefreshIntent.CURRENT_LOADOUT)
        if re.search(
            r"\b(quest|objective|quest progress|did i finish|have i (?:completed|finished)|"
            r"how far am i|campaign)\b",
            text,
        ) or mentioned_content_names(prompt):
            intents.add(RefreshIntent.QUEST_STATUS)
        if re.search(r"\b(inventory|vault|do i own|did i get|weapon|armor)\b", text):
            intents.add(RefreshIntent.CURRENT_INVENTORY)
        if re.search(
            r"\b(recent|last activity|just completed|just finish|just did|"
            r"what did i just|played)\b",
            text,
        ):
            intents.add(RefreshIntent.RECENT_ACTIVITY)
        return intents

    async def refresh(
        self,
        session_id: str,
        access_token: str,
        *,
        slices: set[GuardianSlice],
        force: bool = False,
        soft: bool = False,
        reason: str = "manual_normal",
    ) -> RefreshOutcome:
        observed = await self.cache.get(session_id)
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            entry = await self.cache.get(session_id)
            now = self._clock()
            if entry is None:
                return await self._full_refresh(session_id, access_token, reason)
            if force and (
                (observed is None)
                or (
                    entry.generation > observed.generation
                    and all(
                        name in entry.slices
                        and entry.slices[name].metadata.generation > observed.generation
                        for name in slices
                    )
                )
            ):
                return RefreshOutcome(
                    context=self._assemble(entry), cache_hit=True, reason=f"{reason}:deduplicated"
                )

            entry.last_accessed_at = now
            for name in slices:
                cached = entry.slices.get(name)
                if cached:
                    cached.metadata.last_accessed_at = now

            eligible: set[GuardianSlice] = set()
            for name in slices:
                cached = entry.slices.get(name)
                if cached is None:
                    eligible.add(name)
                    continue
                age = (now - cached.metadata.fetched_at).total_seconds()
                backed_off = bool(
                    cached.metadata.last_error_at
                    and (now - cached.metadata.last_error_at).total_seconds()
                    < self.settings.guardian_refresh_failure_backoff_seconds
                )
                if backed_off and not force:
                    continue
                if (
                    force
                    or (soft and age >= self._minimum(name))
                    or now >= cached.metadata.stale_after
                ):
                    eligible.add(name)
            if not eligible:
                await self.cache.set(session_id, entry)
                logger.info(
                    "Guardian refresh skipped session=%s slices=%s reason=%s generation=%s",
                    session_id[:8],
                    sorted(value.value for value in slices),
                    reason,
                    entry.generation,
                )
                return RefreshOutcome(context=self._assemble(entry), cache_hit=True, reason=reason)
            if eligible == set(ALL_SLICES):
                try:
                    return await self._full_refresh(session_id, access_token, reason)
                except Exception as exc:
                    self._record_failure(entry, eligible, now, exc)
                    await self.cache.set(session_id, entry)
                    return RefreshOutcome(
                        context=self._assemble(entry),
                        cache_hit=True,
                        stale_served=True,
                        reason=reason,
                    )
            self._inflight[session_id] = set(eligible)
            started = time.perf_counter()
            logger.info(
                "Guardian refresh start session=%s slices=%s reason=%s",
                session_id[:8],
                sorted(value.value for value in eligible),
                reason,
            )
            try:
                base = self._assemble(entry)
                profile_slices = eligible - {GuardianSlice.ACTIVITY_HISTORY}
                if profile_slices:
                    components = {"Characters"}
                    for name in profile_slices:
                        components.update(SLICE_COMPONENTS[name])
                    partial = await self.guardian.load_partial_profile(
                        access_token,
                        base,
                        components,
                        refresh_identity=GuardianSlice.PROFILE in profile_slices,
                    )
                    if partial.membership_id != entry.membership_id:
                        await self.cache.delete(session_id)
                        return await self._full_refresh(session_id, access_token, "account_changed")
                    for name in profile_slices:
                        entry.generation += 1
                        entry.slices[name] = self._make_slice(
                            name, self._extract(name, partial), now, reason, entry.generation
                        )
                if GuardianSlice.ACTIVITY_HISTORY in eligible:
                    histories = await self.guardian.load_activity_history(access_token, base)
                    entry.generation += 1
                    entry.slices[GuardianSlice.ACTIVITY_HISTORY] = self._make_slice(
                        GuardianSlice.ACTIVITY_HISTORY,
                        {
                            "characters": histories,
                            "_availability": {
                                "available_components": ["RecentActivityHistory"],
                                "unavailable_components": {},
                                "notes": [],
                            },
                        },
                        now,
                        reason,
                        entry.generation,
                    )
                    self._invalidate_after_activity(entry, histories, now)
                entry.last_accessed_at = now
                await self.cache.set(session_id, entry)
                logger.info(
                    "Guardian refresh complete session=%s slices=%s reason=%s "
                    "generation=%s duration_ms=%d",
                    session_id[:8],
                    sorted(value.value for value in eligible),
                    reason,
                    entry.generation,
                    int((time.perf_counter() - started) * 1000),
                )
                return RefreshOutcome(
                    context=self._assemble(entry),
                    refreshed_slices=sorted(eligible, key=str),
                    reason=reason,
                )
            except Exception as exc:
                self._record_failure(entry, eligible, now, exc)
                await self.cache.set(session_id, entry)
                logger.warning(
                    "Guardian refresh failed session=%s slices=%s reason=%s error=%s "
                    "duration_ms=%d",
                    session_id[:8],
                    sorted(value.value for value in eligible),
                    reason,
                    type(exc).__name__,
                    int((time.perf_counter() - started) * 1000),
                )
                if entry.slices:
                    return RefreshOutcome(
                        context=self._assemble(entry),
                        cache_hit=True,
                        stale_served=True,
                        reason=reason,
                    )
                raise
            finally:
                self._inflight.pop(session_id, None)

    async def normal_refresh(self, session_id: str, access_token: str) -> RefreshOutcome:
        return await self.refresh(
            session_id,
            access_token,
            slices={
                GuardianSlice.PROFILE,
                GuardianSlice.EQUIPMENT,
                GuardianSlice.INVENTORY,
                GuardianSlice.QUESTS_PROGRESS,
                GuardianSlice.ACTIVITY_HISTORY,
            },
            force=True,
            reason="manual_normal",
        )

    async def full_refresh(self, session_id: str, access_token: str) -> RefreshOutcome:
        return await self.refresh(
            session_id, access_token, slices=set(ALL_SLICES), force=True, reason="manual_full"
        )

    async def clear(self, session_id: str) -> None:
        task = self._background.pop(session_id, None)
        if task and not task.done():
            task.cancel()
        self._inflight.pop(session_id, None)
        self._pending.pop(session_id, None)
        await self.cache.delete(session_id)
        logger.info("Guardian cache cleared session=%s", session_id[:8])

    async def status(self, session_id: str) -> dict[str, Any]:
        entry = await self.cache.get(session_id)
        now = self._clock()
        if entry is None:
            return {"cached": False, "refreshing": False, "slices": {}}
        statuses: dict[str, Any] = {}
        inflight = self._inflight.get(session_id, set())
        for name in ALL_SLICES:
            cached = entry.slices.get(name)
            statuses[name.value] = (
                {
                    "present": False,
                    "fresh": False,
                    "in_flight": name in inflight,
                }
                if cached is None
                else {
                    "present": True,
                    "fresh": now < cached.metadata.stale_after,
                    "ttl_seconds": self._ttl(name),
                    "age_seconds": max(0, int((now - cached.metadata.fetched_at).total_seconds())),
                    "fetched_at": cached.metadata.fetched_at.isoformat(),
                    "stale_after": cached.metadata.stale_after.isoformat(),
                    "last_accessed_at": cached.metadata.last_accessed_at.isoformat(),
                    "reason": cached.metadata.reason,
                    "source_components": cached.metadata.source_components,
                    "generation": cached.metadata.generation,
                    "last_success_at": cached.metadata.last_success_at.isoformat(),
                    "last_error": cached.metadata.last_error,
                    "last_result": "error" if cached.metadata.last_error else "success",
                    "cache_hits": cached.metadata.cache_hits,
                    "cache_misses": cached.metadata.cache_misses,
                    "in_flight": name in inflight,
                }
            )
        return {
            "cached": True,
            "refreshing": bool(inflight),
            "generation": entry.generation,
            "membership_id": entry.membership_id,
            "last_accessed_at": entry.last_accessed_at.isoformat(),
            "last_app_activity_at": entry.last_app_activity_at.isoformat(),
            "last_app_open_level": entry.last_app_open_level,
            "slices": statuses,
        }

    async def wait_for_background(self, session_id: str) -> None:
        task = self._background.get(session_id)
        if task:
            await task

    async def close(self) -> None:
        tasks = [task for task in self._background.values() if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._background.clear()
        self._pending.clear()
        self._inflight.clear()

    async def _full_refresh(
        self, session_id: str, access_token: str, reason: str
    ) -> RefreshOutcome:
        now = self._clock()
        previous = await self.cache.get(session_id)
        generation = previous.generation + 1 if previous else 1
        self._inflight[session_id] = set(ALL_SLICES)
        started = time.perf_counter()
        logger.info("Guardian full refresh start session=%s reason=%s", session_id[:8], reason)
        try:
            context = await self.guardian.load(access_token)
            slices = {
                name: self._make_slice(name, self._extract(name, context), now, reason, generation)
                for name in ALL_SLICES
            }
            entry = GuardianCacheEntry(
                account_key=session_id,
                membership_id=context.membership_id,
                membership_type=context.membership_type,
                slices=slices,
                created_at=previous.created_at if previous else now,
                last_accessed_at=now,
                last_app_activity_at=previous.last_app_activity_at if previous else now,
                generation=generation,
                last_app_open_level=previous.last_app_open_level if previous else None,
            )
            await self.cache.set(session_id, entry)
            logger.info(
                "Guardian full refresh complete session=%s reason=%s generation=%s duration_ms=%d",
                session_id[:8],
                reason,
                generation,
                int((time.perf_counter() - started) * 1000),
            )
            return RefreshOutcome(
                context=self._assemble(entry), refreshed_slices=list(ALL_SLICES), reason=reason
            )
        except Exception as exc:
            logger.warning(
                "Guardian full refresh failed session=%s reason=%s error=%s duration_ms=%d",
                session_id[:8],
                reason,
                type(exc).__name__,
                int((time.perf_counter() - started) * 1000),
            )
            raise
        finally:
            self._inflight.pop(session_id, None)

    def _make_slice(
        self,
        name: GuardianSlice,
        payload: dict[str, Any],
        now: datetime,
        reason: str,
        generation: int,
    ) -> CachedGuardianSlice:
        return CachedGuardianSlice(
            payload=payload,
            metadata=SliceMetadata(
                fetched_at=now,
                stale_after=now + timedelta(seconds=self._ttl(name)),
                last_accessed_at=now,
                reason=reason,
                source_components=sorted(SLICE_COMPONENTS[name]),
                generation=generation,
                last_success_at=now,
                cache_misses=1,
            ),
        )

    @staticmethod
    def _extract(name: GuardianSlice, context: GuardianContext) -> dict[str, Any]:
        characters = context.characters
        if name == GuardianSlice.PROFILE:
            payload = {
                "guardian": context.model_dump(
                    mode="json",
                    include={
                        "bungie_display_name",
                        "membership_id",
                        "membership_type",
                        "platform_name",
                        "last_played",
                        "total_minutes_played",
                        "current_season_hash",
                        "current_season_name",
                    },
                ),
                "characters": {
                    value.character_id: value.model_dump(
                        mode="json",
                        include={
                            "character_id",
                            "class_name",
                            "race_name",
                            "gender_name",
                            "power",
                            "last_played",
                            "minutes_played_total",
                            "emblem_url",
                            "emblem_background_url",
                        },
                    )
                    for value in characters
                },
            }
        elif name == GuardianSlice.EQUIPMENT:
            payload = {
                "characters": {
                    value.character_id: value.model_dump(
                        mode="json", include={"subclass", "equipped_gear"}
                    )
                    for value in characters
                }
            }
        elif name == GuardianSlice.INVENTORY:
            payload = {
                "inventory": context.inventory.model_dump(mode="json"),
                "currencies": [value.model_dump(mode="json") for value in context.currencies],
            }
        elif name == GuardianSlice.QUESTS_PROGRESS:
            payload = {
                "characters": {
                    value.character_id: value.model_dump(
                        mode="json",
                        include={"quests", "milestones", "progressions", "available_activities"},
                    )
                    for value in characters
                },
                "profile_progressions": [
                    value.model_dump(mode="json") for value in context.profile_progressions
                ],
                "records": context.records.model_dump(mode="json"),
                "crafting": context.crafting.model_dump(mode="json"),
            }
        elif name == GuardianSlice.ACTIVITY_HISTORY:
            payload = {
                "characters": {
                    value.character_id: [
                        item.model_dump(mode="json") for item in value.recent_activities
                    ]
                    for value in characters
                }
            }
        else:
            payload = {"collectibles": context.collectibles.model_dump(mode="json")}

        relevant_components = set(SLICE_COMPONENTS[name])
        if name == GuardianSlice.ACTIVITY_HISTORY:
            relevant_components.add("RecentActivityHistory")
        availability = context.data_availability
        payload["_availability"] = {
            "available_components": [
                value for value in availability.available_components if value in relevant_components
            ],
            "unavailable_components": {
                key: value
                for key, value in availability.unavailable_components.items()
                if key in relevant_components
            },
            "notes": availability.notes,
        }
        return payload

    def _assemble(self, entry: GuardianCacheEntry) -> GuardianContext:
        now = self._clock()
        profile = entry.slices[GuardianSlice.PROFILE].payload
        character_values = deepcopy(profile.get("characters", {}))
        for character_id, value in character_values.items():
            value.update(
                {
                    "subclass": None,
                    "equipped_gear": [],
                    "quests": [],
                    "milestones": [],
                    "progressions": [],
                    "available_activities": [],
                    "recent_activities": [],
                }
            )
            equipment = entry.slices.get(GuardianSlice.EQUIPMENT)
            if equipment:
                value.update(equipment.payload.get("characters", {}).get(character_id, {}))
            quests = entry.slices.get(GuardianSlice.QUESTS_PROGRESS)
            if quests:
                value.update(quests.payload.get("characters", {}).get(character_id, {}))
            history = entry.slices.get(GuardianSlice.ACTIVITY_HISTORY)
            if history:
                value["recent_activities"] = history.payload.get("characters", {}).get(
                    character_id, []
                )

        inventory = entry.slices.get(GuardianSlice.INVENTORY)
        quests = entry.slices.get(GuardianSlice.QUESTS_PROGRESS)
        collections = entry.slices.get(GuardianSlice.COLLECTIONS)
        stale = [
            name.value for name, value in entry.slices.items() if now >= value.metadata.stale_after
        ]
        errors = [name.value for name, value in entry.slices.items() if value.metadata.last_error]
        fetched = max(value.metadata.fetched_at for value in entry.slices.values())
        availability_rows = [
            value.payload.get("_availability", {}) for value in entry.slices.values()
        ]
        available = sorted(
            {
                component
                for availability in availability_rows
                for component in availability.get("available_components", [])
            }
        )
        unavailable = {
            key: detail
            for availability in availability_rows
            for key, detail in availability.get("unavailable_components", {}).items()
        }
        notes = ["Guardian data is assembled from independently refreshed normalized cache slices."]
        notes.extend(
            dict.fromkeys(
                note for availability in availability_rows for note in availability.get("notes", [])
            )
        )
        if stale:
            notes.append(f"Stale cached slices served: {', '.join(sorted(stale))}.")
        if errors:
            notes.append(
                f"Latest refresh failed for: {', '.join(sorted(errors))}; cached data was retained."
            )
        return GuardianContext(
            **profile["guardian"],
            characters=[
                CharacterSummary.model_validate(value) for value in character_values.values()
            ],
            inventory=InventorySummary.model_validate(
                inventory.payload["inventory"] if inventory else {}
            ),
            currencies=(inventory.payload.get("currencies", []) if inventory else []),
            profile_progressions=(quests.payload.get("profile_progressions", []) if quests else []),
            collectibles=CollectionProgressSummary.model_validate(
                collections.payload["collectibles"] if collections else {}
            ),
            records=RecordProgressSummary.model_validate(
                quests.payload["records"] if quests else {}
            ),
            crafting=CraftingProgressSummary.model_validate(
                quests.payload["crafting"] if quests else {}
            ),
            data_availability=DataAvailability(
                fetched_at=fetched,
                available_components=available,
                unavailable_components=unavailable,
                notes=notes,
            ),
            data_scope=available,
        )

    @staticmethod
    def _invalidate_after_activity(
        entry: GuardianCacheEntry,
        histories: dict[str, list[Any]],
        now: datetime,
    ) -> None:
        newest = max(
            (
                item.period + timedelta(seconds=item.duration_seconds or 0)
                for values in histories.values()
                for item in values
                if item.completed and item.period is not None
            ),
            default=None,
        )
        if newest is None:
            return
        for name in (
            GuardianSlice.EQUIPMENT,
            GuardianSlice.INVENTORY,
            GuardianSlice.QUESTS_PROGRESS,
        ):
            cached = entry.slices.get(name)
            if cached and newest > cached.metadata.fetched_at:
                cached.metadata.stale_after = min(cached.metadata.stale_after, now)

    def _schedule(
        self, session_id: str, access_token: str, slices: set[GuardianSlice], reason: str
    ) -> None:
        current = self._background.get(session_id)
        if current and not current.done():
            self._pending.setdefault(session_id, set()).update(slices)
            self._inflight.setdefault(session_id, set()).update(slices)
            return

        self._pending.setdefault(session_id, set()).update(slices)
        self._inflight.setdefault(session_id, set()).update(slices)

        async def run() -> None:
            try:
                while pending := self._pending.pop(session_id, set()):
                    await self.refresh(
                        session_id,
                        access_token,
                        slices=pending,
                        force=True,
                        reason=reason,
                    )
            except Exception:
                logger.exception("Background Guardian refresh failed session=%s", session_id[:8])
            finally:
                self._background.pop(session_id, None)

        self._background[session_id] = asyncio.create_task(run())

    @staticmethod
    def _record_failure(
        entry: GuardianCacheEntry,
        slices: set[GuardianSlice],
        now: datetime,
        exc: Exception,
    ) -> None:
        for name in slices:
            cached = entry.slices.get(name)
            if cached:
                cached.metadata.last_error = type(exc).__name__
                cached.metadata.last_error_at = now
                cached.metadata.stale_after = min(cached.metadata.stale_after, now)
