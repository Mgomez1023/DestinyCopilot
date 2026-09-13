import asyncio
import logging
from collections import Counter
from collections.abc import Iterable
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from app.bungie.client import BUNGIE_ROOT_URL, BungieAPIError, BungieClient
from app.bungie.manifest import DefinitionResolver
from app.models import (
    AvailableActivitySummary,
    CharacterSummary,
    CollectionProgressSummary,
    CraftingProgressSummary,
    CurrencySummary,
    DataAvailability,
    GuardianContext,
    InventorySummary,
    ItemStatSummary,
    ItemSummary,
    MilestoneSummary,
    ObjectiveSummary,
    ProgressionSummary,
    QuestSummary,
    RecentActivitySummary,
    RecordProgressSummary,
    SocketedPlugSummary,
)

logger = logging.getLogger(__name__)

PLATFORM_NAMES = {
    1: "Xbox",
    2: "PlayStation",
    3: "Steam",
    4: "Blizzard",
    5: "Stadia",
    6: "Epic Games",
    10: "Demon",
}
CLASS_NAMES = {0: "Titan", 1: "Hunter", 2: "Warlock", 3: "Unknown class"}
RACE_NAMES = {0: "Human", 1: "Awoken", 2: "Exo", 3: "Unknown race"}
GENDER_NAMES = {0: "Male", 1: "Female", 2: "Unknown gender"}
ITEM_TYPE_NAMES = {
    1: "Currency",
    2: "Armor",
    3: "Weapon",
    12: "Quest step",
    15: "Quest",
    16: "Subclass",
    19: "Mod",
    20: "Dummy",
    26: "Bounty",
    27: "Wrapper",
    28: "Seasonal artifact",
    29: "Finisher",
    30: "Pattern",
}
QUEST_ITEM_TYPES = {12, 15, 26}
ACTIVITY_LIMIT_PER_CHARACTER = 30
RECENT_ACTIVITY_LIMIT_PER_CHARACTER = 25
NEAR_RECORD_LIMIT = 12
ACTIVITY_DIFFICULTY_NAMES = {
    0: "Trivial",
    1: "Easy",
    2: "Normal",
    3: "Challenging",
    4: "Hard",
    5: "Brave",
    6: "Almost impossible",
    7: "Impossible",
}

REQUESTED_COMPONENT_KEYS = {
    "Profiles": "profile",
    "ProfileInventories": "profileInventory",
    "ProfileCurrencies": "profileCurrencies",
    "ProfileProgression": "profileProgression",
    "Characters": "characters",
    "CharacterInventories": "characterInventories",
    "CharacterProgressions": "characterProgressions",
    "CharacterActivities": "characterActivities",
    "CharacterEquipment": "characterEquipment",
    "Collectibles": "profileCollectibles",
    "Records": "profileRecords",
    "Craftables": "characterCraftables",
}


class GuardianNotFoundError(ValueError):
    pass


class GuardianNormalizationError(RuntimeError):
    pass


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def select_membership(membership_data: dict[str, Any]) -> dict[str, Any]:
    memberships = membership_data.get("destinyMemberships") or []
    if not memberships:
        raise GuardianNotFoundError("No Destiny 2 membership is linked to this Bungie account.")
    primary_id = membership_data.get("primaryMembershipId")
    if primary_id:
        for membership in memberships:
            if str(membership.get("membershipId")) == str(primary_id):
                return membership
    return memberships[0]


def _bungie_asset(path: str | None) -> str | None:
    return f"{BUNGIE_ROOT_URL}{path}" if path else None


def _component_data(payload: dict[str, Any], key: str, default: Any) -> Any:
    component = payload.get(key)
    if not isinstance(component, dict):
        return default
    data = component.get("data")
    return default if data is None else data


def _item_component_data(payload: dict[str, Any], key: str) -> dict[str, Any]:
    item_components = payload.get("itemComponents") or {}
    component = item_components.get(key) or {}
    return component.get("data") or {}


def _display(definition: dict[str, Any]) -> tuple[str, str | None, str | None]:
    display = definition.get("displayProperties") or {}
    name = display.get("name") or "Unknown"
    description = display.get("description") or None
    icon = _bungie_asset(display.get("icon"))
    return name, description, icon


def _iter_component_map(
    payload: dict[str, Any], profile_key: str, character_key: str, leaf_key: str
) -> Iterable[tuple[str, dict[str, Any]]]:
    profile_values = _component_data(payload, profile_key, {}).get(leaf_key, {})
    if isinstance(profile_values, dict):
        yield from ((str(key), value) for key, value in profile_values.items())
    character_components = _component_data(payload, character_key, {})
    if isinstance(character_components, dict):
        for component in character_components.values():
            values = (component or {}).get(leaf_key, {})
            if isinstance(values, dict):
                yield from ((str(key), value) for key, value in values.items())


def _collect_hashes(value: Any, key_name: str) -> set[int]:
    found: set[int] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key == key_name and child:
                with suppress(TypeError, ValueError):
                    found.add(int(child))
            else:
                found.update(_collect_hashes(child, key_name))
    elif isinstance(value, list):
        for child in value:
            found.update(_collect_hashes(child, key_name))
    return found


def _progress_percent(progress: int | None, completion: int) -> float | None:
    if progress is None or completion <= 0:
        return None
    return round(min(100.0, max(0.0, progress / completion * 100)), 1)


def _optional_nonnegative(value: Any) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    return parsed if parsed >= 0 else None


def _near_record_candidates(
    profile: dict[str, Any], limit: int
) -> list[tuple[float, int, dict[str, Any]]]:
    unique: dict[tuple[int, int], tuple[float, int, dict[str, Any]]] = {}
    for record_hash, value in _iter_component_map(
        profile, "profileRecords", "characterRecords", "records"
    ):
        state = int(value.get("state", 0))
        if state & 16 or not state & 4:
            continue
        objectives = (value.get("objectives", []) or []) + (
            value.get("intervalObjectives", []) or []
        )
        for objective in objectives:
            progress = objective.get("progress")
            completion = int(objective.get("completionValue", 0) or 0)
            percent = _progress_percent(int(progress), completion) if progress is not None else None
            if percent is not None and 0 < percent < 100 and objective.get("visible", True):
                key = (int(record_hash), int(objective.get("objectiveHash", 0)))
                previous = unique.get(key)
                candidate = (percent, int(record_hash), objective)
                if previous is None or candidate[0] > previous[0]:
                    unique[key] = candidate
    candidates = list(unique.values())
    candidates.sort(key=lambda value: value[0], reverse=True)
    return candidates[:limit]


class GuardianService:
    def __init__(self, client: BungieClient, resolver: DefinitionResolver) -> None:
        self.client = client
        self.resolver = resolver

    async def load(self, access_token: str) -> GuardianContext:
        membership_data = await self.client.get_current_memberships(access_token)
        membership = select_membership(membership_data)
        membership_id = str(membership["membershipId"])
        membership_type = int(membership["membershipType"])
        profile = await self.client.get_profile(membership_type, membership_id, access_token)
        character_ids = list((_component_data(profile, "characters", {}) or {}).keys())
        histories = await asyncio.gather(
            *(
                self._safe_activity_history(
                    membership_type, membership_id, str(character_id), access_token
                )
                for character_id in character_ids
            )
        )
        history_by_character = dict(zip(map(str, character_ids), histories, strict=True))
        normalizer = GuardianNormalizer(self.resolver)
        try:
            return await normalizer.normalize(
                membership_data, membership, profile, history_by_character
            )
        except ValidationError as exc:
            logger.exception("Bungie profile normalization failed")
            raise GuardianNormalizationError(
                "Bungie returned player data that Guardian Copilot could not normalize."
            ) from exc

    async def load_partial_profile(
        self,
        access_token: str,
        base: GuardianContext,
        components: set[str],
        *,
        refresh_identity: bool = False,
    ) -> GuardianContext:
        """Normalize selected profile components for refresh-service slice merging."""
        if refresh_identity:
            membership_data = await self.client.get_current_memberships(access_token)
            membership = select_membership(membership_data)
        else:
            display_name, _, display_code = base.bungie_display_name.partition("#")
            membership = {
                "membershipId": base.membership_id,
                "membershipType": base.membership_type,
                "bungieGlobalDisplayName": display_name,
                "bungieGlobalDisplayNameCode": (
                    int(display_code) if display_code.isdigit() else None
                ),
            }
            membership_data = {
                "destinyMemberships": [membership],
                "primaryMembershipId": base.membership_id,
            }
        membership_id = str(membership["membershipId"])
        membership_type = int(membership["membershipType"])
        profile = await self.client.get_profile(
            membership_type,
            membership_id,
            access_token,
            components=components,
        )
        normalizer = GuardianNormalizer(self.resolver)
        try:
            return await normalizer.normalize(membership_data, membership, profile, histories={})
        except ValidationError as exc:
            logger.exception("Partial Bungie profile normalization failed")
            raise GuardianNormalizationError(
                "Bungie returned partial player data that could not be normalized."
            ) from exc

    async def load_activity_history(
        self, access_token: str, base: GuardianContext
    ) -> dict[str, list[RecentActivitySummary]]:
        histories = await asyncio.gather(
            *(
                self._safe_activity_history(
                    base.membership_type,
                    base.membership_id,
                    character.character_id,
                    access_token,
                )
                for character in base.characters
            )
        )
        history_by_character = dict(
            zip(
                (value.character_id for value in base.characters),
                histories,
                strict=True,
            )
        )
        normalizer = GuardianNormalizer(self.resolver)
        await normalizer._resolve_referenced_definitions({}, history_by_character)
        return {
            character_id: normalizer._normalize_recent_activities(character_id, history)
            for character_id, history in history_by_character.items()
        }

    async def _safe_activity_history(
        self,
        membership_type: int,
        membership_id: str,
        character_id: str,
        access_token: str,
    ) -> dict[str, Any] | None:
        try:
            return await self.client.get_activity_history(
                membership_type,
                membership_id,
                character_id,
                access_token,
                count=RECENT_ACTIVITY_LIMIT_PER_CHARACTER,
            )
        except BungieAPIError as exc:
            logger.info(
                "Recent activity history unavailable for character=%s: %s",
                character_id,
                exc,
            )
            return None


class GuardianNormalizer:
    """Pure normalization boundary apart from manifest definition resolution."""

    def __init__(self, resolver: DefinitionResolver) -> None:
        self.resolver = resolver
        self.definitions: dict[str, dict[int, dict[str, Any]]] = {}

    async def normalize(
        self,
        membership_data: dict[str, Any],
        membership: dict[str, Any],
        profile: dict[str, Any],
        histories: dict[str, dict[str, Any] | None] | None = None,
    ) -> GuardianContext:
        histories = histories or {}
        await self._resolve_referenced_definitions(profile, histories)

        membership_id = str(membership["membershipId"])
        membership_type = int(membership["membershipType"])
        profile_data = _component_data(profile, "profile", {})
        raw_characters = _component_data(profile, "characters", {})
        characters = [
            self._normalize_character(str(character_id), raw, profile, histories)
            for character_id, raw in raw_characters.items()
        ]
        characters.sort(
            key=lambda character: character.last_played or datetime.min.replace(tzinfo=UTC),
            reverse=True,
        )

        display_name = (
            membership.get("bungieGlobalDisplayName")
            or membership.get("displayName")
            or membership_data.get("bungieNetUser", {}).get("displayName")
            or "Guardian"
        )
        display_code = membership.get("bungieGlobalDisplayNameCode")
        if membership.get("bungieGlobalDisplayName") and display_code is not None:
            display_name = f"{display_name}#{int(display_code):04d}"

        current_season_hash = profile_data.get("currentSeasonHash")
        current_season = self._definition("DestinySeasonDefinition", current_season_hash)
        current_season_name = _display(current_season)[0] if current_season else None
        availability = self._data_availability(profile, histories)
        return GuardianContext(
            bungie_display_name=display_name,
            membership_id=membership_id,
            membership_type=membership_type,
            platform_name=PLATFORM_NAMES.get(membership_type, "Destiny 2"),
            last_played=_parse_datetime(profile_data.get("dateLastPlayed")),
            total_minutes_played=int(profile_data.get("minutesPlayedTotal", 0)),
            current_season_hash=int(current_season_hash) if current_season_hash else None,
            current_season_name=current_season_name,
            characters=characters,
            inventory=self._normalize_inventory(profile),
            currencies=self._normalize_currencies(profile),
            profile_progressions=self._normalize_profile_progressions(profile),
            collectibles=self._normalize_collectibles(profile),
            records=self._normalize_records(profile),
            crafting=self._normalize_crafting(profile),
            data_availability=availability,
            data_scope=availability.available_components,
        )

    async def _resolve_referenced_definitions(
        self,
        profile: dict[str, Any],
        histories: dict[str, dict[str, Any] | None],
    ) -> None:
        inventory_items = list(self._raw_inventory_items(profile, include_equipped=True))
        item_hashes = {
            int(item.get("itemHash", 0)) for _, _, item in inventory_items if item.get("itemHash")
        }
        item_hashes.update(_collect_hashes(profile, "questHash"))
        item_hashes.update(_collect_hashes(profile, "stepHash"))
        item_hashes.update(_collect_hashes(profile, "questItemHash"))
        for component in _component_data(profile, "characterCraftables", {}).values():
            item_hashes.update(int(item_hash) for item_hash in (component.get("craftables") or {}))
        socket_data = _item_component_data(profile, "sockets")
        for instance_sockets in socket_data.values():
            item_hashes.update(_collect_hashes(instance_sockets, "plugHash"))

        bucket_hashes = {
            int(item.get("bucketHash", 0))
            for _, _, item in inventory_items
            if item.get("bucketHash")
        }
        objective_sources = {
            key: profile.get(key)
            for key in (
                "characterProgressions",
                "characterActivities",
                "itemComponents",
                "characterUninstancedItemComponents",
            )
        }
        near_records = _near_record_candidates(profile, NEAR_RECORD_LIMIT)
        objective_sources["nearRecords"] = [value[2] for value in near_records]
        objective_hashes = _collect_hashes(objective_sources, "objectiveHash")
        activity_sources = {
            "progressions": profile.get("characterProgressions"),
            "activities": profile.get("characterActivities"),
        }
        activity_hashes = _collect_hashes(activity_sources, "activityHash")
        destination_hashes = _collect_hashes(objective_sources, "destinationHash")
        milestone_hashes: set[int] = set()
        progression_hashes = _collect_hashes(profile, "progressionHash")
        faction_hashes: set[int] = set()
        class_hashes: set[int] = set()
        race_hashes: set[int] = set()
        gender_hashes: set[int] = set()
        stat_hashes: set[int] = set()
        for entry in (_item_component_data(profile, "stats")).values():
            stat_hashes.update(int(value) for value in (entry.get("stats") or {}))
        damage_hashes = _collect_hashes(
            _item_component_data(profile, "instances"), "damageTypeHash"
        )
        season_hashes: set[int] = set()
        record_hashes = {value[1] for value in near_records}

        for raw in _component_data(profile, "characters", {}).values():
            if raw.get("classHash"):
                class_hashes.add(int(raw["classHash"]))
            if raw.get("raceHash"):
                race_hashes.add(int(raw["raceHash"]))
            if raw.get("genderHash"):
                gender_hashes.add(int(raw["genderHash"]))

        profile_data = _component_data(profile, "profile", {})
        if profile_data.get("currentSeasonHash"):
            season_hashes.add(int(profile_data["currentSeasonHash"]))

        for component in _component_data(profile, "characterProgressions", {}).values():
            milestone_hashes.update(
                int(value) for value in (component.get("milestones") or {}) if value
            )
            faction_hashes.update(
                int(value) for value in (component.get("factions") or {}) if value
            )

        for history in histories.values():
            if history:
                activity_hashes.update(_collect_hashes(history, "referenceId"))
                activity_hashes.update(_collect_hashes(history, "directorActivityHash"))

        calls = {
            "DestinyInventoryItemDefinition": item_hashes,
            "DestinyInventoryBucketDefinition": bucket_hashes,
            "DestinyObjectiveDefinition": objective_hashes,
            "DestinyActivityDefinition": activity_hashes,
            "DestinyDestinationDefinition": destination_hashes,
            "DestinyMilestoneDefinition": milestone_hashes,
            "DestinyProgressionDefinition": progression_hashes,
            "DestinyFactionDefinition": faction_hashes,
            "DestinyClassDefinition": class_hashes,
            "DestinyRaceDefinition": race_hashes,
            "DestinyGenderDefinition": gender_hashes,
            "DestinyStatDefinition": stat_hashes,
            "DestinyDamageTypeDefinition": damage_hashes,
            "DestinySeasonDefinition": season_hashes,
            "DestinyRecordDefinition": record_hashes,
        }
        self.definitions = {}
        for entity_type, hashes in calls.items():
            self.definitions[entity_type] = await self.resolver.resolve_many(entity_type, hashes)
            self.resolver.release_table(entity_type)

        activity_definitions = self.definitions.get("DestinyActivityDefinition", {})
        referenced_destinations = {
            int(value["destinationHash"])
            for value in activity_definitions.values()
            if value.get("destinationHash")
        }
        referenced_activity_types = {
            int(value["activityTypeHash"])
            for value in activity_definitions.values()
            if value.get("activityTypeHash")
        }
        self.definitions.setdefault("DestinyDestinationDefinition", {}).update(
            await self.resolver.resolve_many(
                "DestinyDestinationDefinition", referenced_destinations
            )
        )
        self.resolver.release_table("DestinyDestinationDefinition")
        self.definitions["DestinyActivityTypeDefinition"] = await self.resolver.resolve_many(
            "DestinyActivityTypeDefinition", referenced_activity_types
        )
        self.resolver.release_table("DestinyActivityTypeDefinition")

    def _definition(self, entity_type: str, value: Any) -> dict[str, Any]:
        try:
            entity_hash = int(value)
        except (TypeError, ValueError):
            return {}
        return self.definitions.get(entity_type, {}).get(entity_hash, {})

    def _normalize_character(
        self,
        character_id: str,
        raw: dict[str, Any],
        profile: dict[str, Any],
        histories: dict[str, dict[str, Any] | None],
    ) -> CharacterSummary:
        class_name = self._definition_name(
            "DestinyClassDefinition",
            raw.get("classHash"),
            CLASS_NAMES.get(int(raw.get("classType", 3)), "Unknown class"),
        )
        race_name = self._definition_name(
            "DestinyRaceDefinition",
            raw.get("raceHash"),
            RACE_NAMES.get(int(raw.get("raceType", 3)), "Unknown race"),
        )
        gender_name = self._definition_name(
            "DestinyGenderDefinition",
            raw.get("genderHash"),
            GENDER_NAMES.get(int(raw.get("genderType", 2)), "Unknown gender"),
        )
        equipment_component = (
            _component_data(profile, "characterEquipment", {}).get(character_id, {}) or {}
        )
        raw_equipment = equipment_component.get("items", []) or []
        equipped = [
            self._normalize_item(item, "equipped", profile, character_id) for item in raw_equipment
        ]
        subclass = next((item for item in equipped if item.item_type == "Subclass"), None)
        return CharacterSummary(
            character_id=character_id,
            class_name=class_name,
            race_name=race_name,
            gender_name=gender_name,
            power=int(raw.get("light", 0)),
            last_played=_parse_datetime(raw.get("dateLastPlayed")),
            minutes_played_total=int(raw.get("minutesPlayedTotal", 0)),
            emblem_url=_bungie_asset(raw.get("emblemPath")),
            emblem_background_url=_bungie_asset(raw.get("emblemBackgroundPath")),
            subclass=subclass,
            equipped_gear=equipped,
            quests=self._normalize_quests(character_id, profile),
            milestones=self._normalize_milestones(character_id, profile),
            progressions=self._normalize_character_progressions(character_id, profile),
            available_activities=self._normalize_available_activities(character_id, profile),
            recent_activities=self._normalize_recent_activities(
                character_id, histories.get(character_id)
            ),
        )

    def _definition_name(self, entity_type: str, value: Any, fallback: str) -> str:
        definition = self._definition(entity_type, value)
        return _display(definition)[0] if definition else fallback

    def _raw_inventory_items(
        self, profile: dict[str, Any], *, include_equipped: bool
    ) -> Iterable[tuple[str, str | None, dict[str, Any]]]:
        profile_items = _component_data(profile, "profileInventory", {}).get("items", [])
        for item in profile_items or []:
            location = "vault" if int(item.get("location", 0)) == 2 else "profile"
            yield location, None, item
        for character_id, component in _component_data(profile, "characterInventories", {}).items():
            for item in (component or {}).get("items", []) or []:
                yield "character", str(character_id), item
        if include_equipped:
            for character_id, component in _component_data(
                profile, "characterEquipment", {}
            ).items():
                for item in (component or {}).get("items", []) or []:
                    yield "equipped", str(character_id), item

    def _normalize_item(
        self,
        raw: dict[str, Any],
        location: str,
        profile: dict[str, Any],
        character_id: str | None,
        *,
        include_details: bool = True,
    ) -> ItemSummary:
        item_hash = int(raw.get("itemHash", 0))
        definition = self._definition("DestinyInventoryItemDefinition", item_hash)
        name, description, icon = _display(definition)
        item_type_value = int(definition.get("itemType", 0))
        type_name = ITEM_TYPE_NAMES.get(item_type_value)
        if not type_name:
            type_name = definition.get("itemTypeDisplayName") or "Unknown"
        subtype = definition.get("itemTypeDisplayName") or None
        inventory_definition = definition.get("inventory") or {}
        tier = inventory_definition.get("tierTypeName") or None
        bucket_definition = self._definition(
            "DestinyInventoryBucketDefinition", raw.get("bucketHash")
        )
        bucket_name = _display(bucket_definition)[0] if bucket_definition else None

        instance_id = str(raw["itemInstanceId"]) if raw.get("itemInstanceId") else None
        instances = _item_component_data(profile, "instances")
        instance = instances.get(instance_id, {}) if instance_id else {}
        primary_stat = instance.get("primaryStat") or {}
        primary_value = _optional_nonnegative(primary_stat.get("value"))
        power = primary_value if item_type_value in {2, 3} and primary_value is not None else None
        damage_type = (
            self._definition_name("DestinyDamageTypeDefinition", instance.get("damageTypeHash"), "")
            or None
        )
        energy = instance.get("energy") or {}
        state = int(raw.get("state", 0))

        stats: list[ItemStatSummary] = []
        stats_component = _item_component_data(profile, "stats")
        raw_stats = (
            (stats_component.get(instance_id, {}) or {}).get("stats", {}) if instance_id else {}
        )
        for stat_hash, stat in raw_stats.items() if include_details else []:
            stat_name = self._definition_name("DestinyStatDefinition", stat_hash, "Unknown stat")
            stats.append(ItemStatSummary(name=stat_name, value=int(stat.get("value", 0))))
        stats.sort(key=lambda value: value.name)

        socketed_plugs: list[str] = []
        socketed_plug_details: list[SocketedPlugSummary] = []
        socket_components = _item_component_data(profile, "sockets")
        sockets = (
            (socket_components.get(instance_id, {}) or {}).get("sockets", []) if instance_id else []
        )
        for socket in sockets if include_details else []:
            plug_hash = socket.get("plugHash")
            if not plug_hash:
                continue
            plug_name = self._definition_name(
                "DestinyInventoryItemDefinition", plug_hash, "Unknown plug"
            )
            if plug_name not in socketed_plugs:
                socketed_plugs.append(plug_name)
            plug_definition = self._definition("DestinyInventoryItemDefinition", plug_hash)
            _, plug_description, _ = _display(plug_definition)
            plug_block = plug_definition.get("plug") or {}
            plug_type_value = int(plug_definition.get("itemType", 0))
            socketed_plug_details.append(
                SocketedPlugSummary(
                    item_hash=int(plug_hash),
                    name=plug_name,
                    description=plug_description,
                    item_type=(
                        ITEM_TYPE_NAMES.get(plug_type_value)
                        or plug_definition.get("itemTypeDisplayName")
                        or None
                    ),
                    category_identifier=plug_block.get("plugCategoryIdentifier") or None,
                )
            )

        return ItemSummary(
            item_hash=item_hash,
            instance_id=instance_id,
            name=name,
            description=description if include_details else None,
            item_type=type_name,
            item_subtype=subtype if subtype != type_name else None,
            tier=tier,
            icon_url=icon,
            bucket_name=bucket_name,
            quantity=max(0, int(raw.get("quantity", 1))),
            power=power,
            damage_type=damage_type,
            is_equipped=bool(instance.get("isEquipped")) or location == "equipped",
            is_locked=bool(state & 1),
            is_crafted=bool(state & 8),
            energy_capacity=_optional_nonnegative(energy.get("energyCapacity")),
            energy_used=_optional_nonnegative(energy.get("energyUsed")),
            stats=stats,
            socketed_plugs=socketed_plugs,
            socketed_plug_details=socketed_plug_details,
            location=location,
            character_id=character_id,
        )

    def _normalize_inventory(self, profile: dict[str, Any]) -> InventorySummary:
        raw_items = list(self._raw_inventory_items(profile, include_equipped=False))
        items = []
        for location, character_id, raw in raw_items:
            definition = self._definition("DestinyInventoryItemDefinition", raw.get("itemHash"))
            # Weapons and armor keep their already-returned stats/sockets so the
            # bounded inventory tool can answer perk and build questions. Other
            # inventory entries stay compact.
            include_details = int(definition.get("itemType", 0) or 0) in {2, 3}
            items.append(
                self._normalize_item(
                    raw,
                    location,
                    profile,
                    character_id,
                    include_details=include_details,
                )
            )
        by_type = Counter(item.item_type for item in items)
        tier_score = {"Exotic": 4, "Legendary": 3, "Rare": 2, "Uncommon": 1}
        items.sort(
            key=lambda item: (
                tier_score.get(item.tier or "", 0),
                item.power or 0,
                item.name,
            ),
            reverse=True,
        )
        return InventorySummary(
            total_items=len(items),
            vault_items=sum(item.location == "vault" for item in items),
            character_items=sum(item.location == "character" for item in items),
            unique_item_hashes=len({item.item_hash for item in items}),
            by_type=dict(sorted(by_type.items())),
            items=items,
            returned_items=len(items),
            truncated=False,
        )

    def _normalize_objective(self, raw: dict[str, Any], *, prefix: str = "") -> ObjectiveSummary:
        objective_hash = int(raw.get("objectiveHash", 0))
        definition = self._definition("DestinyObjectiveDefinition", objective_hash)
        name, description, _ = _display(definition)
        progress = raw.get("progress")
        progress_value = int(progress) if progress is not None else None
        completion = int(raw.get("completionValue", definition.get("completionValue", 0)) or 0)
        activity_name = (
            self._definition_name("DestinyActivityDefinition", raw.get("activityHash"), "") or None
        )
        destination_name = (
            self._definition_name("DestinyDestinationDefinition", raw.get("destinationHash"), "")
            or None
        )
        return ObjectiveSummary(
            objective_hash=objective_hash,
            name=f"{prefix}: {name}" if prefix else name,
            description=description,
            progress=progress_value,
            completion_value=completion,
            progress_percent=_progress_percent(progress_value, completion),
            complete=bool(raw.get("complete")),
            visible=bool(raw.get("visible", True)),
            activity_name=activity_name,
            destination_name=destination_name,
        )

    def _item_objectives(
        self,
        profile: dict[str, Any],
        character_id: str,
        raw_item: dict[str, Any],
    ) -> list[ObjectiveSummary]:
        instance_id = str(raw_item["itemInstanceId"]) if raw_item.get("itemInstanceId") else None
        values: list[dict[str, Any]] = []
        if instance_id:
            entry = _item_component_data(profile, "objectives").get(instance_id, {})
            values = entry.get("objectives", []) or []
        else:
            item_hash = str(raw_item.get("itemHash", 0))
            character_set = (profile.get("characterUninstancedItemComponents") or {}).get(
                character_id, {}
            )
            entry = character_set.get("objectives", {}).get("data", {}).get(item_hash, {})
            values = entry.get("objectives", []) or []
        return [self._normalize_objective(value) for value in values if value.get("visible", True)]

    def _normalize_quests(self, character_id: str, profile: dict[str, Any]) -> list[QuestSummary]:
        progression = (
            _component_data(profile, "characterProgressions", {}).get(character_id, {}) or {}
        )
        statuses = progression.get("quests", []) or []
        quests: list[QuestSummary] = []
        seen_instances: set[str] = set()
        seen_hashes: set[int] = set()
        for status in statuses:
            quest_hash = int(status.get("questHash", 0))
            step_hash = int(status.get("stepHash", 0)) or None
            quest_definition = self._definition("DestinyInventoryItemDefinition", quest_hash)
            step_definition = self._definition("DestinyInventoryItemDefinition", step_hash)
            name, description, icon = _display(quest_definition)
            step_name = _display(step_definition)[0] if step_definition else None
            objectives = [
                self._normalize_objective(value)
                for value in status.get("stepObjectives", [])
                if value.get("visible", True)
            ]
            quests.append(
                QuestSummary(
                    quest_hash=quest_hash,
                    step_hash=step_hash,
                    name=name,
                    step_name=step_name,
                    description=description,
                    icon_url=icon,
                    character_id=character_id,
                    tracked=bool(status.get("tracked")),
                    started=bool(status.get("started", True)),
                    completed=bool(status.get("completed")),
                    redeemed=bool(status.get("redeemed")),
                    objectives=objectives,
                )
            )
            if status.get("itemInstanceId"):
                seen_instances.add(str(status["itemInstanceId"]))
            seen_hashes.update({value for value in (quest_hash, step_hash) if value})

        inventory_component = (
            _component_data(profile, "characterInventories", {}).get(character_id, {}) or {}
        )
        raw_items = inventory_component.get("items", []) or []
        for item in raw_items:
            item_hash = int(item.get("itemHash", 0))
            definition = self._definition("DestinyInventoryItemDefinition", item_hash)
            if int(definition.get("itemType", 0)) not in QUEST_ITEM_TYPES:
                continue
            instance_id = str(item["itemInstanceId"]) if item.get("itemInstanceId") else ""
            if instance_id in seen_instances or item_hash in seen_hashes:
                continue
            name, description, icon = _display(definition)
            objectives = self._item_objectives(profile, character_id, item)
            quests.append(
                QuestSummary(
                    quest_hash=item_hash,
                    step_hash=item_hash,
                    name=name,
                    step_name=name,
                    description=description,
                    icon_url=icon,
                    character_id=character_id,
                    tracked=bool(int(item.get("state", 0)) & 2),
                    completed=bool(objectives) and all(value.complete for value in objectives),
                    objectives=objectives,
                )
            )
        return sorted(quests, key=lambda value: (value.completed, value.name))[:80]

    def _normalize_milestones(
        self, character_id: str, profile: dict[str, Any]
    ) -> list[MilestoneSummary]:
        progression_component = (
            _component_data(profile, "characterProgressions", {}).get(character_id, {}) or {}
        )
        raw_values = progression_component.get("milestones", {}) or {}
        milestones: list[MilestoneSummary] = []
        for milestone_hash_text, raw in raw_values.items():
            milestone_hash = int(milestone_hash_text)
            definition = self._definition("DestinyMilestoneDefinition", milestone_hash)
            name, description, _ = _display(definition)
            activity_names = sorted(
                {
                    self._definition_name("DestinyActivityDefinition", value, "Unknown activity")
                    for value in _collect_hashes(raw, "activityHash")
                }
            )
            quest_names = sorted(
                {
                    self._definition_name("DestinyInventoryItemDefinition", value, "Unknown quest")
                    for value in _collect_hashes(raw, "questItemHash")
                }
            )
            objective_values = self._visible_objective_dicts(raw)
            milestones.append(
                MilestoneSummary(
                    milestone_hash=milestone_hash,
                    name=name,
                    description=description,
                    character_id=character_id,
                    start_date=_parse_datetime(raw.get("startDate")),
                    end_date=_parse_datetime(raw.get("endDate")),
                    activity_names=activity_names,
                    quest_names=quest_names,
                    objectives=[self._normalize_objective(value) for value in objective_values],
                )
            )
        return sorted(milestones, key=lambda value: value.name)[:30]

    @staticmethod
    def _visible_objective_dicts(value: Any) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        if isinstance(value, dict):
            if value.get("objectiveHash") and value.get("visible", True):
                found.append(value)
            else:
                for child in value.values():
                    found.extend(GuardianNormalizer._visible_objective_dicts(child))
        elif isinstance(value, list):
            for child in value:
                found.extend(GuardianNormalizer._visible_objective_dicts(child))
        unique: dict[tuple[int, int | None], dict[str, Any]] = {}
        for raw in found:
            key = (int(raw["objectiveHash"]), raw.get("progress"))
            unique[key] = raw
        return list(unique.values())

    def _normalize_progression(
        self,
        raw: dict[str, Any],
        scope: str,
        character_id: str | None,
        *,
        definition_type: str = "DestinyProgressionDefinition",
        definition_hash: int | None = None,
    ) -> ProgressionSummary:
        progression_hash = int(raw.get("progressionHash", 0))
        lookup_hash = int(definition_hash or progression_hash)
        definition = self._definition(definition_type, lookup_hash)
        name, description, _ = _display(definition)
        return ProgressionSummary(
            progression_hash=progression_hash,
            faction_hash=definition_hash if scope == "faction" else None,
            name=name,
            description=description,
            scope=scope,
            character_id=character_id,
            level=max(0, int(raw.get("level", 0))),
            level_cap=max(0, int(raw.get("levelCap", 0))),
            current_progress=max(0, int(raw.get("currentProgress", 0))),
            progress_to_next_level=max(0, int(raw.get("progressToNextLevel", 0))),
            next_level_at=max(0, int(raw.get("nextLevelAt", 0))),
            daily_progress=max(0, int(raw.get("dailyProgress", 0))),
            daily_limit=max(0, int(raw.get("dailyLimit", 0))),
            weekly_progress=max(0, int(raw.get("weeklyProgress", 0))),
            weekly_limit=max(0, int(raw.get("weeklyLimit", 0))),
            current_reset_count=max(0, int(raw.get("currentResetCount", 0))),
        )

    def _normalize_character_progressions(
        self, character_id: str, profile: dict[str, Any]
    ) -> list[ProgressionSummary]:
        raw = _component_data(profile, "characterProgressions", {}).get(character_id, {}) or {}
        values = [
            self._normalize_progression(value, "character", character_id)
            for value in (raw.get("progressions") or {}).values()
        ]
        values.extend(
            self._normalize_progression(
                value,
                "faction",
                character_id,
                definition_type="DestinyFactionDefinition",
                definition_hash=int(faction_hash),
            )
            for faction_hash, value in (raw.get("factions") or {}).items()
        )
        meaningful = [
            value
            for value in values
            if value.level
            or value.current_progress
            or value.progress_to_next_level
            or value.current_reset_count
        ]
        return sorted(meaningful, key=lambda value: value.name)[:50]

    def _normalize_profile_progressions(self, profile: dict[str, Any]) -> list[ProgressionSummary]:
        raw = _component_data(profile, "profileProgression", {})
        artifact = raw.get("seasonalArtifact") or {}
        candidates = [
            value
            for key in ("powerBonusProgression", "pointProgression")
            if isinstance((value := artifact.get(key)), dict) and value.get("progressionHash")
        ]
        return [self._normalize_progression(value, "profile", None) for value in candidates]

    def _normalize_available_activities(
        self, character_id: str, profile: dict[str, Any]
    ) -> list[AvailableActivitySummary]:
        raw = _component_data(profile, "characterActivities", {}).get(character_id, {}) or {}
        values: list[AvailableActivitySummary] = []
        for activity in raw.get("availableActivities", []) or []:
            activity_hash = int(activity.get("activityHash", 0))
            definition = self._definition("DestinyActivityDefinition", activity_hash)
            name, description, _ = _display(definition)
            activity_type = (
                self._definition_name(
                    "DestinyActivityTypeDefinition", definition.get("activityTypeHash"), ""
                )
                or None
            )
            destination = (
                self._definition_name(
                    "DestinyDestinationDefinition", definition.get("destinationHash"), ""
                )
                or None
            )
            difficulty_tier = activity.get("difficultyTier")
            objectives = [
                self._normalize_objective(value)
                for value in self._visible_objective_dicts(activity.get("challenges", []))
            ]
            values.append(
                AvailableActivitySummary(
                    activity_hash=activity_hash,
                    name=name,
                    description=description,
                    character_id=character_id,
                    activity_type=activity_type,
                    destination=destination,
                    difficulty=(
                        ACTIVITY_DIFFICULTY_NAMES.get(int(difficulty_tier))
                        if difficulty_tier is not None
                        else None
                    ),
                    display_level=_optional_nonnegative(activity.get("displayLevel")),
                    recommended_power=_optional_nonnegative(activity.get("recommendedLight")),
                    is_new=bool(activity.get("isNew")),
                    can_lead=bool(activity.get("canLead")),
                    can_join=bool(activity.get("canJoin")),
                    is_visible=bool(activity.get("isVisible", True)),
                    is_completed=bool(activity.get("isCompleted")),
                    objectives=objectives,
                )
            )
        values.sort(
            key=lambda value: (
                bool(value.objectives and not all(item.complete for item in value.objectives)),
                value.is_new,
                not value.is_completed,
                value.name,
            ),
            reverse=True,
        )
        return values[:ACTIVITY_LIMIT_PER_CHARACTER]

    def _normalize_recent_activities(
        self, character_id: str, history: dict[str, Any] | None
    ) -> list[RecentActivitySummary]:
        if not history:
            return []
        values: list[RecentActivitySummary] = []
        for activity in history.get("activities", []) or []:
            details = activity.get("activityDetails") or {}
            activity_hash = int(
                details.get("directorActivityHash") or details.get("referenceId") or 0
            )
            definition = self._definition("DestinyActivityDefinition", activity_hash)
            name, description, _ = _display(definition)
            activity_type = (
                self._definition_name(
                    "DestinyActivityTypeDefinition", definition.get("activityTypeHash"), ""
                )
                or None
            )
            destination = (
                self._definition_name(
                    "DestinyDestinationDefinition", definition.get("destinationHash"), ""
                )
                or None
            )
            difficulty_tier = definition.get("tier")
            stats = activity.get("values") or {}
            completed_raw = (stats.get("completed") or {}).get("basic", {}).get("value")
            duration_raw = (
                (stats.get("activityDurationSeconds") or {}).get("basic", {}).get("value")
            )
            values.append(
                RecentActivitySummary(
                    activity_hash=activity_hash,
                    name=name,
                    description=description,
                    character_id=character_id,
                    period=_parse_datetime(activity.get("period")),
                    instance_id=(str(details["instanceId"]) if details.get("instanceId") else None),
                    mode=int(details["mode"]) if details.get("mode") is not None else None,
                    activity_type=activity_type,
                    destination=destination,
                    difficulty=(
                        ACTIVITY_DIFFICULTY_NAMES.get(int(difficulty_tier))
                        if difficulty_tier is not None
                        else None
                    ),
                    recommended_power=_optional_nonnegative(definition.get("activityLightLevel")),
                    completed=bool(completed_raw) if completed_raw is not None else None,
                    duration_seconds=(
                        max(0, int(duration_raw)) if duration_raw is not None else None
                    ),
                )
            )
        return values

    def _normalize_currencies(self, profile: dict[str, Any]) -> list[CurrencySummary]:
        items = _component_data(profile, "profileCurrencies", {}).get("items", []) or []
        values: list[CurrencySummary] = []
        for item in items:
            item_hash = int(item.get("itemHash", 0))
            definition = self._definition("DestinyInventoryItemDefinition", item_hash)
            name, description, icon = _display(definition)
            max_stack = (definition.get("inventory") or {}).get("maxStackSize")
            values.append(
                CurrencySummary(
                    item_hash=item_hash,
                    name=name,
                    description=description,
                    quantity=max(0, int(item.get("quantity", 0))),
                    max_stack_size=_optional_nonnegative(max_stack),
                    icon_url=icon,
                )
            )
        return sorted(values, key=lambda value: value.name)

    def _normalize_collectibles(self, profile: dict[str, Any]) -> CollectionProgressSummary:
        states: dict[str, list[int]] = {}
        for collectible_hash, value in _iter_component_map(
            profile, "profileCollectibles", "characterCollectibles", "collectibles"
        ):
            states.setdefault(collectible_hash, []).append(int(value.get("state", 0)))
        visible = [values for values in states.values() if any(not state & 4 for state in values)]
        return CollectionProgressSummary(
            total_visible=len(visible),
            acquired=sum(any(not state & 1 for state in values) for values in visible),
        )

    def _normalize_records(self, profile: dict[str, Any]) -> RecordProgressSummary:
        records: dict[str, list[dict[str, Any]]] = {}
        for record_hash, value in _iter_component_map(
            profile, "profileRecords", "characterRecords", "records"
        ):
            records.setdefault(record_hash, []).append(value)
        visible = [
            (record_hash, values)
            for record_hash, values in records.items()
            if any(not int(value.get("state", 0)) & 16 for value in values)
        ]
        completed = sum(
            any(not int(value.get("state", 0)) & 4 for value in values) for _, values in visible
        )

        near_completion: list[ObjectiveSummary] = []
        for _, record_hash, objective in _near_record_candidates(profile, NEAR_RECORD_LIMIT):
            record_definition = self._definition("DestinyRecordDefinition", record_hash)
            record_name = _display(record_definition)[0] if record_definition else "Record"
            near_completion.append(self._normalize_objective(objective, prefix=record_name))
        return RecordProgressSummary(
            total_visible=len(visible),
            completed=completed,
            near_completion=near_completion,
        )

    def _normalize_crafting(self, profile: dict[str, Any]) -> CraftingProgressSummary:
        craftables: dict[str, list[dict[str, Any]]] = {}
        character_components = _component_data(profile, "characterCraftables", {})
        for component in character_components.values():
            for item_hash, value in ((component or {}).get("craftables") or {}).items():
                craftables.setdefault(str(item_hash), []).append(value)
        visible = {
            item_hash: values
            for item_hash, values in craftables.items()
            if any(value.get("visible", True) for value in values)
        }
        met = {
            item_hash
            for item_hash, values in visible.items()
            if any(not (value.get("failedRequirementIndexes") or []) for value in values)
        }
        incomplete_names = sorted(
            self._definition_name("DestinyInventoryItemDefinition", item_hash, "Unknown pattern")
            for item_hash in visible
            if item_hash not in met
        )
        return CraftingProgressSummary(
            total_visible=len(visible),
            requirements_met=len(met),
            incomplete_pattern_names=incomplete_names[:20],
        )

    def _data_availability(
        self,
        profile: dict[str, Any],
        histories: dict[str, dict[str, Any] | None],
    ) -> DataAvailability:
        available: list[str] = []
        unavailable: dict[str, str] = {}
        for component_name, key in REQUESTED_COMPONENT_KEYS.items():
            component = profile.get(key)
            has_profile_data = isinstance(component, dict) and component.get("data") is not None
            has_character_data = component_name in {"Collectibles", "Records"} and any(
                (profile.get(character_key) or {}).get("data") is not None
                for character_key in (
                    f"character{component_name}",
                    f"character{component_name[:-1]}s",
                )
            )
            if has_profile_data or has_character_data:
                available.append(component_name)
            else:
                unavailable[component_name] = (
                    "Bungie omitted this component; privacy settings, OAuth scope, "
                    "or the current profile may not permit it."
                )
        item_components = profile.get("itemComponents") or {}
        for name, key in (
            ("ItemInstances", "instances"),
            ("ItemObjectives", "objectives"),
            ("ItemSockets", "sockets"),
            ("ItemStats", "stats"),
        ):
            if (item_components.get(key) or {}).get("data") is not None:
                available.append(name)
            else:
                unavailable[name] = "Bungie did not return this item component."
        if histories and all(value is not None for value in histories.values()):
            available.append("RecentActivityHistory")
        elif histories:
            unavailable["RecentActivityHistory"] = (
                "Activity history was unavailable for one or more characters."
            )
        return DataAvailability(
            fetched_at=datetime.now(UTC),
            available_components=available,
            unavailable_components=unavailable,
            notes=[
                "The complete compact normalized inventory is available to bounded search tools.",
                (
                    "Available activities are capped per character and prioritize new, "
                    "incomplete, or challenge-bearing entries."
                ),
                (
                    "CharacterActivities reflects Bungie's server-known activity state and "
                    "is not treated as a complete Director listing."
                ),
                "Invisible collectible and record states are excluded from totals.",
            ],
        )
