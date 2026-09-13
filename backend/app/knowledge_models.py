from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

GuideType = Literal[
    "exotic_acquisition",
    "quest_walkthrough",
    "activity_walkthrough",
    "encounter_mechanics",
    "item_farming",
    "catalyst",
    "build_mechanics",
    "general",
]
FreshnessClass = Literal["stable", "semi_stable", "volatile"]
Confidence = Literal["low", "medium", "high"]
LiveAuthority = Literal["official", "structured_community", "editorial", "derived"]


class AssociatedDestinyEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    entity_type: str
    entity_hash: int | None = None


class KnowledgeSource(BaseModel):
    """Normalized provenance retained for grounding and developer inspection."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    title: str
    provider: str
    source_url: str | None = None
    retrieved_at: datetime
    published_at: datetime | None = None
    updated_at: datetime | None = None
    relevant_section: str | None = None
    confidence: Confidence = "medium"
    relevance_score: float = Field(default=1.0, ge=0, le=1)
    time_sensitive: bool = False
    factual_claims: list[str] = Field(default_factory=list, max_length=30)


class KnowledgeConflict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_key: str
    values: list[str]
    source_ids: list[str]
    status: Literal["resolved", "unresolved"]
    resolution: str | None = None


class LiveKnowledgeSource(BaseModel):
    """Provenance for one current-state source, never a raw upstream payload."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    title: str
    provider: str
    source_url: str
    authority: LiveAuthority
    retrieved_at: datetime


class LiveEffectiveWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    effective_from: datetime | None = None
    effective_until: datetime | None = None
    reset_cadence: Literal["daily", "weekly", "vendor", "unknown"]
    stale_after: datetime
    valid_at_retrieval: bool


class GuideRecord(BaseModel):
    """A compact, curated guide record; never an archived article body."""

    model_config = ConfigDict(extra="forbid")

    guide_id: str
    title: str
    subject: str
    guide_type: GuideType
    aliases: list[str] = Field(default_factory=list)
    freshness: FreshnessClass
    summary: str
    associated_entities: list[AssociatedDestinyEntity] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    steps: list[dict[str, Any]] = Field(default_factory=list)
    reward: dict[str, Any] | None = None
    encounters: list[dict[str, Any]] = Field(default_factory=list)
    farming: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)
    factual_claims: list[str] = Field(default_factory=list)
    claim_evidence: list[dict[str, Any]] = Field(default_factory=list)
    sources: list[KnowledgeSource] = Field(default_factory=list)
