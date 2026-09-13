import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from test_guardian_tools import guardian_context

from app.config import Settings
from app.guardian_refresh import (
    ALL_SLICES,
    TOOL_SLICE_DEPENDENCIES,
    GuardianRefreshService,
    GuardianSlice,
    MemoryGuardianStateCache,
)
from app.models import GuardianContext, RecentActivitySummary


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 9, 12, 12, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: int) -> None:
        self.now += timedelta(seconds=seconds)


class FakeGuardianService:
    def __init__(self, context: GuardianContext) -> None:
        self.context = context
        self.full_calls = 0
        self.partial_calls: list[set[str]] = []
        self.history_calls = 0
        self.partial_delay = 0.0
        self.full_delay = 0.0
        self.fail_partial = False
        self.fail_full = False

    async def load(self, access_token: str) -> GuardianContext:
        assert access_token
        self.full_calls += 1
        if self.full_delay:
            await asyncio.sleep(self.full_delay)
        if self.fail_full:
            raise RuntimeError("Bungie unavailable on cold load")
        return self.context.model_copy(deep=True)

    async def load_partial_profile(
        self,
        access_token: str,
        base: GuardianContext,
        components: set[str],
        *,
        refresh_identity: bool = False,
    ) -> GuardianContext:
        assert access_token and base.membership_id
        self.partial_calls.append(set(components))
        if self.partial_delay:
            await asyncio.sleep(self.partial_delay)
        if self.fail_partial:
            raise RuntimeError("Bungie temporarily unavailable")
        return self.context.model_copy(deep=True)

    async def load_activity_history(
        self, access_token: str, base: GuardianContext
    ) -> dict[str, list[RecentActivitySummary]]:
        assert access_token and base.membership_id
        self.history_calls += 1
        return {value.character_id: value.recent_activities for value in self.context.characters}


def make_service(
    *,
    clock: Clock | None = None,
    context: GuardianContext | None = None,
    **overrides: int,
) -> tuple[GuardianRefreshService, FakeGuardianService, Clock]:
    test_clock = clock or Clock()
    guardian = FakeGuardianService(context or guardian_context())
    values = {
        "guardian_cache_profile_ttl_seconds": 900,
        "guardian_cache_equipment_ttl_seconds": 180,
        "guardian_cache_inventory_ttl_seconds": 600,
        "guardian_cache_quests_progress_ttl_seconds": 300,
        "guardian_cache_activity_history_ttl_seconds": 600,
        "guardian_cache_collections_ttl_seconds": 1800,
        **overrides,
    }
    settings = Settings(**values)
    return (
        GuardianRefreshService(
            guardian, settings, cache=MemoryGuardianStateCache(), clock=test_clock
        ),
        guardian,
        test_clock,
    )


@pytest.mark.asyncio
async def test_cache_miss_loads_once_and_fresh_hit_reuses_normalized_slices() -> None:
    service, guardian, _ = make_service()

    cold = await service.get_context("session-a", "token")
    warm = await service.get_context("session-a", "token")

    assert guardian.full_calls == 1
    assert not cold.cache_hit
    assert warm.cache_hit
    assert warm.context == cold.context
    status = await service.status("session-a")
    assert set(status["slices"]) == {value.value for value in ALL_SLICES}
    assert all(value["fresh"] for value in status["slices"].values())
    entry = await service.cache.get("session-a")
    assert entry is not None
    serialized = repr(entry)
    assert "token" not in serialized
    assert "Response" not in serialized


@pytest.mark.asyncio
async def test_stale_while_revalidate_returns_cached_then_refreshes_only_stale_slice() -> None:
    service, guardian, clock = make_service(
        guardian_cache_equipment_ttl_seconds=10,
    )
    await service.get_context("session-a", "token")
    clock.advance(11)

    result = await service.get_context("session-a", "token")
    assert result.cache_hit and result.stale_served and result.background_refresh
    assert guardian.partial_calls == []

    await service.wait_for_background("session-a")
    assert len(guardian.partial_calls) == 1
    assert "CharacterEquipment" in guardian.partial_calls[0]
    assert "CharacterActivities" not in guardian.partial_calls[0]


@pytest.mark.asyncio
async def test_tool_refresh_dependencies_do_not_fetch_unrelated_vault_data() -> None:
    service, guardian, clock = make_service()
    await service.get_context("session-a", "token")
    clock.advance(301)

    result = await service.for_tool("session-a", "token", "get_active_quests")

    assert {value.value for value in result.refreshed_slices} == {"quests_progress"}
    assert "CharacterActivities" in guardian.partial_calls[-1]
    assert "ProfileInventories" not in guardian.partial_calls[-1]
    assert TOOL_SLICE_DEPENDENCIES["get_active_quests"] == {
        GuardianSlice.QUESTS_PROGRESS,
    }


@pytest.mark.asyncio
async def test_prompt_what_should_i_do_refreshes_progress_not_inventory() -> None:
    service, guardian, clock = make_service()
    await service.get_context("session-a", "token")
    clock.advance(301)

    await service.for_prompt("session-a", "token", "What should I do next?")

    assert "CharacterActivities" in guardian.partial_calls[-1]
    assert "ProfileInventories" not in guardian.partial_calls[-1]
    assert guardian.history_calls == 1
    assert "CharacterEquipment" in guardian.partial_calls[-1]


@pytest.mark.asyncio
async def test_soft_refresh_floors_and_intents_are_selective() -> None:
    service, guardian, clock = make_service()
    await service.get_context("session-a", "token")

    clock.advance(20)
    await service.for_prompt("session-a", "token", "Look at my current Titan build")
    assert guardian.partial_calls == []

    clock.advance(41)
    await service.for_prompt("session-a", "token", "Look at my current Titan build")
    assert len(guardian.partial_calls) == 1
    assert "CharacterEquipment" in guardian.partial_calls[-1]
    assert "ProfileInventories" in guardian.partial_calls[-1]
    assert "CharacterActivities" not in guardian.partial_calls[-1]

    await service.for_prompt("session-a", "token", "What is The Shattered Throne?")
    assert len(guardian.partial_calls) == 1
    assert guardian.history_calls == 0


@pytest.mark.asyncio
async def test_current_loadout_returns_equipment_changed_after_soft_floor() -> None:
    service, guardian, clock = make_service()
    initial = await service.get_context("session-a", "token")
    old_name = initial.context.characters[0].equipped_gear[0].name
    guardian.context.characters[0].equipped_gear[0].name = "Newly equipped weapon"
    clock.advance(46)

    refreshed = await service.for_prompt("session-a", "token", "What am I currently using?")

    assert old_name != "Newly equipped weapon"
    assert refreshed.context.characters[0].equipped_gear[0].name == "Newly equipped weapon"
    assert len(guardian.partial_calls) == 1
    assert guardian.history_calls == 0


@pytest.mark.asyncio
async def test_individual_tool_intents_refresh_only_due_slices() -> None:
    service, guardian, clock = make_service()
    await service.get_context("session-a", "token")
    clock.advance(121)

    await service.for_tool("session-a", "token", "get_active_quests")
    assert len(guardian.partial_calls) == 1
    assert "CharacterActivities" in guardian.partial_calls[-1]
    assert "ProfileInventories" not in guardian.partial_calls[-1]

    await service.for_tool("session-a", "token", "get_recent_activities")
    assert guardian.history_calls == 1

    await service.for_tool("session-a", "token", "search_inventory")
    assert len(guardian.partial_calls) == 2
    assert "ProfileInventories" in guardian.partial_calls[-1]
    assert "CharacterActivities" not in guardian.partial_calls[-1]

    clock.advance(20)
    await service.for_tool("session-a", "token", "search_inventory")
    assert len(guardian.partial_calls) == 2


@pytest.mark.asyncio
async def test_concurrent_tool_refreshes_are_deduplicated() -> None:
    service, guardian, clock = make_service()
    await service.get_context("session-a", "token")
    clock.advance(181)
    guardian.partial_delay = 0.02

    await asyncio.gather(
        service.for_tool("session-a", "token", "get_equipped_loadout"),
        service.for_tool("session-a", "token", "get_equipped_loadout"),
        service.for_tool("session-a", "token", "get_equipped_loadout"),
    )

    assert len(guardian.partial_calls) == 1


@pytest.mark.asyncio
async def test_concurrent_cold_and_forced_refreshes_are_deduplicated() -> None:
    service, guardian, _ = make_service()
    guardian.partial_delay = 0.02
    guardian.full_delay = 0.02

    await asyncio.gather(
        service.get_context("session-a", "token"),
        service.get_context("session-a", "token"),
        service.get_context("session-a", "token"),
    )
    assert guardian.full_calls == 1

    await asyncio.gather(
        service.refresh(
            "session-a",
            "token",
            slices={GuardianSlice.EQUIPMENT},
            force=True,
            reason="manual",
        ),
        service.refresh(
            "session-a",
            "token",
            slices={GuardianSlice.EQUIPMENT},
            force=True,
            reason="manual",
        ),
    )
    assert len(guardian.partial_calls) == 1


@pytest.mark.asyncio
async def test_failed_refresh_retains_stale_cache_and_applies_backoff() -> None:
    service, guardian, clock = make_service(
        guardian_cache_equipment_ttl_seconds=10,
        guardian_refresh_failure_backoff_seconds=60,
    )
    initial = await service.get_context("session-a", "token")
    clock.advance(11)
    guardian.fail_partial = True

    failed = await service.refresh(
        "session-a",
        "token",
        slices={GuardianSlice.EQUIPMENT},
        reason="test_failure",
    )
    backed_off = await service.refresh(
        "session-a",
        "token",
        slices={GuardianSlice.EQUIPMENT},
        reason="test_backoff",
    )

    assert failed.stale_served
    assert failed.context.characters[0].equipped_gear == initial.context.characters[0].equipped_gear
    assert backed_off.cache_hit
    assert len(guardian.partial_calls) == 1
    status = await service.status("session-a")
    assert status["slices"]["equipment"]["last_error"] == "RuntimeError"


@pytest.mark.asyncio
async def test_missing_slice_is_replaced_without_full_account_fetch() -> None:
    service, guardian, _ = make_service()
    await service.get_context("session-a", "token")
    entry = await service.cache.get("session-a")
    assert entry is not None
    entry.slices.pop(GuardianSlice.COLLECTIONS)
    await service.cache.set("session-a", entry)

    result = await service.refresh(
        "session-a", "token", slices={GuardianSlice.COLLECTIONS}, reason="missing"
    )

    assert guardian.full_calls == 1
    assert len(guardian.partial_calls) == 1
    assert result.context.collectibles == guardian.context.collectibles


@pytest.mark.asyncio
async def test_no_cache_bungie_failure_is_not_hidden() -> None:
    service, guardian, _ = make_service()
    guardian.fail_full = True

    with pytest.raises(RuntimeError, match="cold load"):
        await service.get_context("session-a", "token")

    assert not (await service.status("session-a"))["cached"]


@pytest.mark.asyncio
async def test_failed_full_refresh_preserves_every_cached_slice() -> None:
    service, guardian, _ = make_service()
    initial = await service.get_context("session-a", "token")
    guardian.fail_full = True

    result = await service.full_refresh("session-a", "token")

    assert result.stale_served
    assert result.context.characters == initial.context.characters
    status = await service.status("session-a")
    assert all(value["last_result"] == "error" for value in status["slices"].values())


@pytest.mark.asyncio
async def test_cache_is_isolated_by_session_and_clear_removes_only_target() -> None:
    service, guardian, _ = make_service()
    await service.get_context("session-a", "token-a")
    await service.get_context("session-b", "token-b")
    assert guardian.full_calls == 2

    await service.clear("session-a")

    assert not (await service.status("session-a"))["cached"]
    assert (await service.status("session-b"))["cached"]


@pytest.mark.asyncio
async def test_membership_change_discards_old_account_state() -> None:
    service, guardian, clock = make_service()
    await service.get_context("session-a", "token")
    replacement = guardian.context.model_copy(deep=True)
    replacement.membership_id = "different-membership"
    replacement.bungie_display_name = "Different Guardian"
    guardian.context = replacement
    clock.advance(301)

    result = await service.for_tool("session-a", "token", "get_active_quests")

    assert result.context.membership_id == "different-membership"
    assert result.context.bungie_display_name == "Different Guardian"
    assert guardian.full_calls == 2


@pytest.mark.asyncio
async def test_cache_schema_change_forces_rebuild() -> None:
    service, guardian, _ = make_service()
    await service.get_context("session-a", "token")
    entry = await service.cache.get("session-a")
    assert entry is not None
    entry.schema_version = 0
    await service.cache.set("session-a", entry)

    await service.get_context("session-a", "token")

    assert guardian.full_calls == 2


@pytest.mark.asyncio
async def test_app_open_uses_three_idle_levels() -> None:
    service, guardian, clock = make_service()
    await service.get_context("session-a", "token")

    recent = await service.app_open("session-a", "token")
    assert recent.reason == "app_open_level_0"
    clock.advance(301)
    medium = await service.app_open("session-a", "token")
    assert medium.reason == "app_open_level_1" and medium.background_refresh
    await service.wait_for_background("session-a")
    assert guardian.history_calls == 1
    clock.advance(7201)
    old = await service.app_open("session-a", "token")
    assert old.reason == "app_open_level_2" and old.background_refresh
    await service.wait_for_background("session-a")
    assert guardian.full_calls == 2


@pytest.mark.asyncio
async def test_recent_completion_invalidates_older_dynamic_slices() -> None:
    clock = Clock()
    context = guardian_context()
    character = context.characters[0]
    character.recent_activities = [
        RecentActivitySummary(
            activity_hash=1,
            name="Completed activity",
            character_id=character.character_id,
            period=clock.now + timedelta(seconds=20),
            completed=True,
        )
    ]
    service, _, _ = make_service(clock=clock, context=context)
    await service.get_context("session-a", "token")
    clock.advance(10)

    await service.refresh(
        "session-a",
        "token",
        slices={GuardianSlice.ACTIVITY_HISTORY},
        force=True,
        reason="activity_check",
    )

    status = await service.status("session-a")
    assert not status["slices"]["equipment"]["fresh"]
    assert not status["slices"]["inventory"]["fresh"]
    assert not status["slices"]["quests_progress"]["fresh"]


@pytest.mark.asyncio
async def test_manual_normal_skips_collections_but_full_replaces_every_slice() -> None:
    service, guardian, _ = make_service()
    await service.get_context("session-a", "token")

    normal = await service.normal_refresh("session-a", "token")
    full = await service.full_refresh("session-a", "token")

    assert GuardianSlice.COLLECTIONS not in normal.refreshed_slices
    assert set(full.refreshed_slices) == set(ALL_SLICES)
    assert guardian.full_calls == 2
    assert len(guardian.partial_calls) == 1
    assert guardian.history_calls == 1
    status = await service.status("session-a")
    assert status["generation"] > 1
    assert {value["generation"] for value in status["slices"].values()} == {status["generation"]}
    assert all(value["reason"] == "manual_full" for value in status["slices"].values())
