from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


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


class ItemStatSummary(BaseModel):
    name: str
    value: int


class SocketedPlugSummary(BaseModel):
    item_hash: int
    name: str
    description: str | None = None
    item_type: str | None = None
    category_identifier: str | None = None


class ItemSummary(BaseModel):
    item_hash: int
    instance_id: str | None = None
    name: str
    description: str | None = None
    item_type: str
    item_subtype: str | None = None
    tier: str | None = None
    icon_url: str | None = None
    bucket_name: str | None = None
    quantity: int = Field(default=1, ge=0)
    power: int | None = Field(default=None, ge=0)
    damage_type: str | None = None
    is_equipped: bool = False
    is_locked: bool = False
    is_crafted: bool = False
    energy_capacity: int | None = Field(default=None, ge=0)
    energy_used: int | None = Field(default=None, ge=0)
    stats: list[ItemStatSummary] = Field(default_factory=list)
    socketed_plugs: list[str] = Field(default_factory=list)
    socketed_plug_details: list[SocketedPlugSummary] = Field(default_factory=list)
    location: Literal["equipped", "character", "vault", "profile"]
    character_id: str | None = None


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


class RecordProgressSummary(BaseModel):
    total_visible: int = Field(default=0, ge=0)
    completed: int = Field(default=0, ge=0)
    near_completion: list[ObjectiveSummary] = Field(default_factory=list)


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


class ChatResponse(BaseModel):
    message: str
    source: Literal["openai", "local"]
