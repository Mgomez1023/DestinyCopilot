from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


# --- Core Destiny knowledge (Knowledge Engine v0.1) -------------------------------------
# Records are deliberately flat and id-keyed so they map one-to-one onto future SQLite
# tables: predicates, sources, entities, aliases, facts, ambiguity groups.

CoreIdPattern = r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)+$"
CoreTokenPattern = r"^[a-z][a-z0-9_]*$"
PredicateValueType = Literal["enum", "int", "bool", "text", "place", "class", "slot", "entity"]
PredicateOwner = Literal["curated", "manifest", "manifest_preferred"]
CoreSourceKind = Literal["curated", "manifest_text", "manifest_structure", "official_bungie"]
CoreEntityKind = Literal[
    "game_mode",
    "activity_type",
    "activity",
    "destination",
    "weapon_archetype",
    "armor_concept",
    "class",
    "subclass_element",
    "progression",
    "vendor",
    "concept",
    "rule",
]
CoreEntityStatus = Literal["current", "legacy", "seasonal"]
CoreAliasType = Literal["alias", "misspelling"]
CoreRelationType = Literal["variant_not_same_as", "related_to", "part_of"]
CoreClassValue = Literal["Titan", "Hunter", "Warlock", "Any"]
CORE_CLASS_VALUES: tuple[str, ...] = ("Titan", "Hunter", "Warlock", "Any")


class CorePredicate(BaseModel):
    """One controlled fact predicate; its owner decides which source class may assert it."""

    model_config = ConfigDict(extra="forbid")

    value: PredicateValueType
    owner: PredicateOwner
    curatable: bool = True
    values: list[str] = Field(default_factory=list, max_length=32)
    freshness: FreshnessClass = "stable"
    corroborate: Literal["manifest_text", "manifest_structure"] | None = None
    description: str = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def consistent_definition(self) -> "CorePredicate":
        if self.value == "enum" and not self.values:
            raise ValueError("Enum predicates must declare their allowed values.")
        if self.value != "enum" and self.values:
            raise ValueError("Only enum predicates may declare allowed values.")
        if self.owner == "manifest" and self.curatable:
            raise ValueError("Manifest-owned predicates cannot be curatable.")
        return self


# A Manifest text source must cite one definition as "<DefinitionType>:<hash>".
CoreManifestRefPattern = r"^Destiny[A-Za-z0-9]+Definition:[0-9]+$"


class CoreSource(BaseModel):
    """Where a fact or summary comes from. manifest_text quotes are checked by the anchor
    validator against the cited definition; official_bungie sources need a URL ref."""

    model_config = ConfigDict(extra="forbid")

    kind: CoreSourceKind
    ref: str | None = Field(default=None, max_length=300)
    quote: str | None = Field(default=None, max_length=300)
    note: str | None = Field(default=None, max_length=300)
    verified_at: date | None = None


class CoreAlias(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=80)
    type: CoreAliasType = "alias"


class CoreFact(BaseModel):
    """A subject-predicate-object triple; the subject is the owning entity."""

    model_config = ConfigDict(extra="forbid")

    p: str = Field(pattern=CoreTokenPattern)
    o: bool | int | str
    o_ref: str | None = Field(default=None, pattern=CoreIdPattern)
    src: list[str] = Field(min_length=1, max_length=4)
    confidence: Confidence = "high"
    manifest_absent_reason: str | None = Field(default=None, max_length=300)

    @field_validator("src", mode="before")
    @classmethod
    def single_source(cls, value: Any) -> Any:
        return [value] if isinstance(value, str) else value


class CoreManifestAnchor(BaseModel):
    """A pointer used to corroborate a curated entity against the Bungie Manifest."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(pattern=r"^Destiny[A-Za-z0-9]+Definition$")
    name: str | None = Field(default=None, max_length=200)
    hash: int | None = Field(default=None, gt=0)
    # A presentation-node ancestor name expected above this definition.
    expect_ancestor: str | None = Field(default=None, max_length=200)
    # A destination bubble (sub-area) name this destination definition must declare.
    expect_sub_area: str | None = Field(default=None, max_length=200)
    # Dotted definition paths and their expected values, e.g. {"activityModeCategory": 2}.
    expect_fields: dict[str, bool | int | str] = Field(default_factory=dict, max_length=6)

    @model_validator(mode="after")
    def identifies_something(self) -> "CoreManifestAnchor":
        if self.name is None and self.hash is None:
            raise ValueError("A Manifest anchor needs a name or a hash.")
        return self


class CoreRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rel: CoreRelationType
    id: str = Field(pattern=CoreIdPattern)


class CoreEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=CoreIdPattern)
    name: str = Field(min_length=1, max_length=120)
    kind: CoreEntityKind
    status: CoreEntityStatus = "current"
    aliases: list[CoreAlias] = Field(default_factory=list, max_length=16)
    plural: bool = False
    summary: str = Field(min_length=1, max_length=300)
    # Sources the summary paraphrases; required for production entries.
    summary_src: list[str] = Field(default_factory=list, max_length=4)
    facts: list[CoreFact] = Field(default_factory=list, max_length=12)
    unknown: list[str] = Field(default_factory=list, max_length=8)
    live_sensitive: list[str] = Field(default_factory=list, max_length=8)
    related: list[CoreRelation] = Field(default_factory=list, max_length=8)
    manifest_anchors: list[CoreManifestAnchor] = Field(default_factory=list, max_length=4)
    guide_ids: list[str] = Field(default_factory=list, max_length=8)
    freshness: FreshnessClass = "stable"
    last_verified: date
    review_after: date | None = None

    @field_validator("aliases", mode="before")
    @classmethod
    def plain_alias_strings(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [{"text": item} if isinstance(item, str) else item for item in value]
        return value

    @field_validator("unknown", "live_sensitive")
    @classmethod
    def topic_tokens(cls, value: list[str]) -> list[str]:
        for topic in value:
            if not topic or not topic.replace("_", "a").isalnum() or topic != topic.lower():
                raise ValueError(f"Topic '{topic}' must be a lowercase snake_case token.")
        return value

    @model_validator(mode="after")
    def review_after_verification(self) -> "CoreEntity":
        if self.review_after is not None and self.review_after < self.last_verified:
            raise ValueError("review_after cannot be earlier than last_verified.")
        return self


class CoreShorthand(BaseModel):
    """An explicitly registered shorthand token such as "hc"; never inferred."""

    model_config = ConfigDict(extra="forbid")

    entity: str = Field(pattern=CoreIdPattern)
    note: str | None = Field(default=None, max_length=200)


class CoreAmbiguityGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=CoreIdPattern)
    aliases: list[str] = Field(min_length=1, max_length=8)
    candidates: list[str] = Field(min_length=2, max_length=8)
    note: str | None = Field(default=None, max_length=300)


class CoreKnowledgeCorpus(BaseModel):
    """A versioned curated corpus. Cross-record rules are enforced by the loader."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    corpus_version: str = Field(min_length=1, max_length=40)
    # The Manifest version the anchors and Manifest sources were verified against.
    manifest_version: str | None = Field(default=None, max_length=80)
    predicates: dict[str, CorePredicate]
    sources: dict[str, CoreSource]
    entities: list[CoreEntity]
    shorthands: dict[str, CoreShorthand] = Field(default_factory=dict)
    ambiguity_groups: list[CoreAmbiguityGroup] = Field(default_factory=list)
