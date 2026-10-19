from datetime import datetime
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator, model_serializer, model_validator


class ObjectiveSummary(BaseModel):
    objective_hash: int
    name: str
    description: str | None = None
    progress: int | None = None
    completion_value: int = 0
    progress_percent: float | None = Field(default=None, ge=0, le=100)
    complete: bool = False
    visible: bool = True
    activity_name: str | None = None
    destination_name: str | None = None


class RecordObjectiveSummary(ObjectiveSummary):
    """A near-complete Record objective with the Record's own Manifest context."""

    record_name: str | None = Field(default=None, max_length=200)
    record_description: str | None = Field(default=None, max_length=300)
    context_path: list[str] = Field(default_factory=list, max_length=6)


class ItemStatSummary(BaseModel):
    name: str
    value: int


# Plug categories that never affect a build; one definition for normalization and the tools.
COSMETIC_PLUG_MARKERS = ("shader", "skins", "ornament", "memento", "tracker", "holofoil")

# Evidence states for one kind of plug information. "Missing" and "absent" differ on purpose:
# `empty_in_manifest` / `none_declared` mean the Manifest definition was read and holds nothing
# for that field (verified absence); `definition_unavailable` / `unresolved` mean the
# definition was not retrieved, so nothing can be said; an empty `PlugEvidence` means the
# plug was normalized before this evidence was collected.
PlugDescriptionState = Literal["present", "empty_in_manifest", "definition_unavailable"]
PlugSandboxState = Literal[
    "present", "declared_without_text", "none_declared", "unresolved", "definition_unavailable"
]
PlugStatState = Literal["present", "none_declared", "definition_unavailable"]


class SandboxPerkSummary(BaseModel):
    """A linked DestinySandboxPerkDefinition: Bungie's short effect text for a plug."""

    perk_hash: int
    name: str | None = None
    description: str | None = None
    is_displayable: bool | None = None


class PlugStatEffect(BaseModel):
    """One investment stat delta declared by the plug definition."""

    name: str
    value: int
    conditional: bool = False


class PlugEvidence(BaseModel):
    """Which plug information was read, and from where. Provenance is always the Manifest."""

    source: Literal["manifest"] = "manifest"
    description: PlugDescriptionState
    sandbox_perks: PlugSandboxState
    stat_effects: PlugStatState


_PLUG_EVIDENCE_FIELDS = (
    "socket_index",
    "socket_category",
    "is_enabled",
    "is_visible",
    "sandbox_perks",
    "stat_effects",
    "energy_cost",
    "energy_type",
    "unnamed",
    "evidence",
)


class SocketedPlugSummary(BaseModel):
    item_hash: int
    name: str
    description: str | None = None
    item_type: str | None = None
    plug_type: str | None = None
    category_identifier: str | None = None
    # M1 plug evidence, collected for equipped gear only. All defaults mean "not collected".
    socket_index: int | None = None
    socket_category: str | None = None
    is_enabled: bool | None = None
    is_visible: bool | None = None
    sandbox_perks: list[SandboxPerkSummary] = Field(default_factory=list)
    stat_effects: list[PlugStatEffect] = Field(default_factory=list)
    energy_cost: int | None = None
    energy_type: str | None = None
    unnamed: bool = False
    evidence: PlugEvidence | None = None

    @model_serializer(mode="wrap")
    def _omit_unset_plug_evidence(self, handler: Any) -> dict[str, Any]:
        """Leave out empty M1 evidence fields so inventory and API payloads stay compact.

        Only fields added for plug evidence are dropped, and only when empty; validation
        restores the same defaults, so cached and older payloads round-trip unchanged.
        """

        data = handler(self)
        for key in _PLUG_EVIDENCE_FIELDS:
            value = data.get(key)
            # False is real information for the state flags; it is only noise for `unnamed`.
            if value is None or value == [] or (key == "unnamed" and value is False):
                data.pop(key, None)
        return data


ClassRestriction = Literal["Titan", "Hunter", "Warlock", "Any"]


class SocketConfigSummary(BaseModel):
    """What an item *definition* declares about one socket (see socket_configuration).

    Manifest structure only: it never says what a particular copy holds or what a player can
    change. Empty fields are omitted on serialization.
    """

    index: int
    category: str | None = None
    classification: str
    accepts: str | None = None
    plug: str | None = None
    initial_plug: str | None = None
    pool_kind: str | None = None
    options: int | None = None
    rollable: int | None = None
    distinct_names: int | None = None
    set_hash: int | None = None
    display_only: bool = False
    sample: list[str] = Field(default_factory=list)
    plug_sources: list[str] = Field(default_factory=list)

    @model_serializer(mode="wrap")
    def _omit_empty(self, handler: Any) -> dict[str, Any]:
        data = handler(self)
        return {
            key: value
            for key, value in data.items()
            if key in {"index", "classification"} or value not in (None, [], False)
        }


SocketConfigStatus = Literal["verified", "partial", "no_sockets_declared"]
EquipmentSlotState = Literal["known", "not_equippable", "unresolved", "not_collected"]
_LOCATION_BUCKET_LOCATIONS = ("equipped", "character")


class ItemSummary(BaseModel):
    item_hash: int
    instance_id: str | None = None
    name: str
    description: str | None = None
    item_type: str
    item_subtype: str | None = None
    tier: str | None = None
    class_restriction: ClassRestriction | None = None
    icon_url: str | None = None
    # Where this copy is stored right now ("General" for a vaulted item). It is NOT the
    # equipment slot; use `equipment_slot` / `gear_slot()` for that.
    bucket_name: str | None = None
    # The slot from the item's Manifest definition (Helmet, Energy Weapons, ...). "known" is
    # implied by a set slot; "not_collected" marks payloads normalized before slots existed.
    equipment_slot: str | None = None
    equipment_slot_state: EquipmentSlotState = "not_collected"
    # Manifest socket structure for equipped weapons and Exotic armor only; unset elsewhere.
    socket_configuration: list[SocketConfigSummary] = Field(default_factory=list)
    socket_configuration_status: SocketConfigStatus | None = None
    quantity: int = Field(default=1, ge=0)
    power: int | None = Field(default=None, ge=0)
    damage_type: str | None = None
    is_equipped: bool = False
    is_locked: bool = False
    is_crafted: bool = False
    energy_capacity: int | None = Field(default=None, ge=0)
    energy_used: int | None = Field(default=None, ge=0)
    instance_data_available: bool | None = None
    socket_data_available: bool | None = None
    stat_data_available: bool | None = None
    socket_count: int | None = Field(default=None, ge=0)
    empty_socket_count: int | None = Field(default=None, ge=0)
    stats: list[ItemStatSummary] = Field(default_factory=list)
    socketed_plugs: list[str] = Field(default_factory=list)
    socketed_plug_details: list[SocketedPlugSummary] = Field(default_factory=list)
    location: Literal["equipped", "character", "vault", "profile"]
    character_id: str | None = None

    @model_validator(mode="after")
    def _slot_state_follows_slot(self) -> "ItemSummary":
        if self.equipment_slot and self.equipment_slot_state == "not_collected":
            self.equipment_slot_state = "known"
        return self

    @model_serializer(mode="wrap")
    def _omit_implied_slot_fields(self, handler: Any) -> dict[str, Any]:
        """Keep payloads compact: a set slot implies "known", and unset means not collected."""

        data = handler(self)
        if data.get("equipment_slot") is None:
            data.pop("equipment_slot", None)
        if data.get("equipment_slot_state") in ("known", "not_collected"):
            data.pop("equipment_slot_state", None)
        if not data.get("socket_configuration"):
            data.pop("socket_configuration", None)
        if data.get("socket_configuration_status") is None:
            data.pop("socket_configuration_status", None)
        return data

    def gear_slot(self) -> str | None:
        """The verified equipment slot, or None when it is not verified.

        A payload normalized before slots existed falls back to the storage bucket only for
        equipped and character-inventory copies, whose bucket is a slot bucket. A vaulted copy
        never falls back: "General" is not a slot.
        """

        if self.equipment_slot:
            return self.equipment_slot
        if self.equipment_slot_state == "not_collected" and (
            self.location in _LOCATION_BUCKET_LOCATIONS
        ):
            return self.bucket_name
        return None

    def slot_status(self) -> str:
        """known, not_equippable, unresolved, or legacy_location_bucket / unverified."""

        if self.equipment_slot:
            return "known"
        if self.equipment_slot_state != "not_collected":
            return self.equipment_slot_state
        return "legacy_location_bucket" if self.gear_slot() is not None else "unverified"


class QuestSummary(BaseModel):
    quest_hash: int
    step_hash: int | None = None
    name: str
    step_name: str | None = None
    description: str | None = None
    icon_url: str | None = None
    character_id: str
    tracked: bool = False
    started: bool = True
    completed: bool = False
    redeemed: bool = False
    objectives: list[ObjectiveSummary] = Field(default_factory=list)
    # The Manifest's own specific type label, such as "Duality Order"; null when absent.
    item_type_name: str | None = Field(default=None, max_length=80)
    # Reward names declared in the Manifest definition's value.itemValue, in Manifest order.
    reward_names: list[str] = Field(default_factory=list, max_length=4)
    reward_total: int = Field(default=0, ge=0)
    rewards_unresolved: int = Field(default=0, ge=0)


class MilestoneSummary(BaseModel):
    milestone_hash: int
    name: str
    description: str | None = None
    character_id: str
    start_date: datetime | None = None
    end_date: datetime | None = None
    activity_names: list[str] = Field(default_factory=list)
    quest_names: list[str] = Field(default_factory=list)
    objectives: list[ObjectiveSummary] = Field(default_factory=list)


class ProgressionSummary(BaseModel):
    progression_hash: int
    faction_hash: int | None = None
    name: str
    description: str | None = None
    scope: Literal["profile", "character", "faction"]
    character_id: str | None = None
    level: int = Field(default=0, ge=0)
    level_cap: int = Field(default=0, ge=0)
    current_progress: int = Field(default=0, ge=0)
    progress_to_next_level: int = Field(default=0, ge=0)
    next_level_at: int = Field(default=0, ge=0)
    daily_progress: int = Field(default=0, ge=0)
    daily_limit: int = Field(default=0, ge=0)
    weekly_progress: int = Field(default=0, ge=0)
    weekly_limit: int = Field(default=0, ge=0)
    current_reset_count: int = Field(default=0, ge=0)


class CurrencySummary(BaseModel):
    item_hash: int
    name: str
    description: str | None = None
    quantity: int = Field(default=0, ge=0)
    max_stack_size: int | None = Field(default=None, ge=0)
    icon_url: str | None = None


class RecentActivitySummary(BaseModel):
    activity_hash: int
    name: str
    description: str | None = None
    character_id: str
    period: datetime | None = None
    instance_id: str | None = None
    mode: int | None = None
    activity_type: str | None = None
    destination: str | None = None
    place: str | None = None
    difficulty: str | None = None
    recommended_power: int | None = None
    completed: bool | None = None
    duration_seconds: int | None = Field(default=None, ge=0)


class AvailableActivitySummary(BaseModel):
    activity_hash: int
    name: str
    description: str | None = None
    character_id: str
    activity_type: str | None = None
    destination: str | None = None
    place: str | None = None
    difficulty: str | None = None
    display_level: int | None = None
    recommended_power: int | None = None
    is_new: bool = False
    can_lead: bool = False
    can_join: bool = False
    is_visible: bool = True
    is_completed: bool = False
    objectives: list[ObjectiveSummary] = Field(default_factory=list)


class InventorySummary(BaseModel):
    total_items: int = Field(default=0, ge=0)
    vault_items: int = Field(default=0, ge=0)
    character_items: int = Field(default=0, ge=0)
    unique_item_hashes: int = Field(default=0, ge=0)
    by_type: dict[str, int] = Field(default_factory=dict)
    items: list[ItemSummary] = Field(default_factory=list)
    returned_items: int = Field(default=0, ge=0)
    truncated: bool = False


class CollectionProgressSummary(BaseModel):
    total_visible: int = Field(default=0, ge=0)
    acquired: int = Field(default=0, ge=0)


class RecordSummary(BaseModel):
    record_hash: int
    name: str
    description: str | None = Field(default=None, max_length=500)
    scope: Literal["profile", "character", "unknown"]
    character_id: str | None = None
    completed: bool = False
    redeemed: bool = False
    objectives_complete: bool | None = None
    objectives: list[ObjectiveSummary] = Field(default_factory=list, max_length=4)
    manifest_resolved: bool = False


class TitleRecordProgressSummary(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    completed: bool | None = None
    objectives: list[ObjectiveSummary] = Field(default_factory=list, max_length=4)


class TitleProgressSummary(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    completed: bool | None = None
    completed_records: int = Field(default=0, ge=0)
    total_records: int = Field(default=0, ge=0)
    remaining_records: int = Field(default=0, ge=0)
    completion_percent: float | None = Field(default=None, ge=0, le=100)
    remaining_objective_progress_percent: float | None = Field(default=None, ge=0, le=100)
    remaining: list[TitleRecordProgressSummary] = Field(default_factory=list, max_length=32)
    remaining_truncated: bool = False
    data_complete: bool = False


class RecordProgressSummary(BaseModel):
    total_visible: int = Field(default=0, ge=0)
    completed: int = Field(default=0, ge=0)
    near_completion: list[RecordObjectiveSummary] = Field(default_factory=list)
    records: list[RecordSummary] = Field(default_factory=list, max_length=64)
    records_truncated: bool = False
    titles: list[TitleProgressSummary] = Field(default_factory=list, max_length=96)
    titles_truncated: bool = False
    title_data_available: bool = False


class CraftingProgressSummary(BaseModel):
    total_visible: int = Field(default=0, ge=0)
    requirements_met: int = Field(default=0, ge=0)
    incomplete_pattern_names: list[str] = Field(default_factory=list)


class CharacterSummary(BaseModel):
    character_id: str
    class_name: str
    race_name: str
    gender_name: str
    power: int = Field(ge=0)
    last_played: datetime | None = None
    minutes_played_total: int = Field(default=0, ge=0)
    emblem_url: str | None = None
    emblem_background_url: str | None = None
    subclass: ItemSummary | None = None
    equipped_gear: list[ItemSummary] = Field(default_factory=list)
    quests: list[QuestSummary] = Field(default_factory=list)
    milestones: list[MilestoneSummary] = Field(default_factory=list)
    progressions: list[ProgressionSummary] = Field(default_factory=list)
    available_activities: list[AvailableActivitySummary] = Field(default_factory=list)
    recent_activities: list[RecentActivitySummary] = Field(default_factory=list)


class DataAvailability(BaseModel):
    fetched_at: datetime
    available_components: list[str] = Field(default_factory=list)
    unavailable_components: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class GuardianContext(BaseModel):
    bungie_display_name: str
    membership_id: str
    membership_type: int
    platform_name: str
    last_played: datetime | None = None
    total_minutes_played: int = Field(default=0, ge=0)
    current_season_hash: int | None = None
    current_season_name: str | None = None
    characters: list[CharacterSummary] = Field(default_factory=list)
    inventory: InventorySummary = Field(default_factory=InventorySummary)
    currencies: list[CurrencySummary] = Field(default_factory=list)
    profile_progressions: list[ProgressionSummary] = Field(default_factory=list)
    collectibles: CollectionProgressSummary = Field(default_factory=CollectionProgressSummary)
    records: RecordProgressSummary = Field(default_factory=RecordProgressSummary)
    crafting: CraftingProgressSummary = Field(default_factory=CraftingProgressSummary)
    data_availability: DataAvailability
    data_scope: list[str] = Field(default_factory=list)


class AuthStatus(BaseModel):
    configured: bool
    authenticated: bool
    debug_enabled: bool = False
    message: str | None = None


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=12)


class ChatSource(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    url: str = Field(min_length=1, max_length=2048)
    domain: str | None = Field(default=None, min_length=1, max_length=253)

    @field_validator("title", "url", "domain", mode="before")
    @classmethod
    def strip_source_fields(cls, value: str | None) -> str | None:
        return value.strip() if isinstance(value, str) else value

    @field_validator("url")
    @classmethod
    def validate_external_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or any(ord(character) < 32 for character in value)
        ):
            raise ValueError("Source URL must be an HTTP(S) URL without credentials.")
        return value


class ChatResponse(BaseModel):
    message: str
    source: Literal["openai", "local"]
    sources: list[ChatSource] = Field(default_factory=list, max_length=8)
