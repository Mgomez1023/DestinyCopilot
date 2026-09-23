import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models import CharacterSummary, GuardianContext, ObjectiveSummary

EvidenceState = Literal["KNOWN_TRUE", "KNOWN_FALSE", "UNKNOWN", "NOT_RETURNED"]
ContentStatus = Literal["completed", "in_progress", "not_started", "unknown"]
EvidenceType = Literal["quest", "milestone", "progression", "record"]
CompletionScope = Literal["character", "profile", "unknown"]

MAX_CONTENT_RESULTS = 4
MAX_SUPPORTED_CONTENT = 32
MAX_EVIDENCE_ITEMS = 8
MAX_LIMITATIONS = 4
MAX_OBJECTIVES = 3
DATA_PATH = Path(__file__).parent / "data" / "content_progression.json"


class TerminalEvidenceRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_type: Literal["DestinyRecordDefinition"]
    hash: int = Field(gt=0)
    resolved_name: str
    expected_description: str
    expected_scope: Literal["profile", "character"]
    semantic: Literal["campaign_completion"]
    source_reference: str


class ContentDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canonical_name: str
    aliases: list[str] = Field(min_length=1, max_length=8)
    match_names: list[str] = Field(min_length=1, max_length=12)
    completion_quest_names: list[str] = Field(default_factory=list, max_length=12)
    completion_milestone_names: list[str] = Field(default_factory=list, max_length=12)
    completion_progression_names: list[str] = Field(default_factory=list, max_length=12)
    terminal_evidence: list[TerminalEvidenceRule] = Field(default_factory=list, max_length=12)
    rule_reason: str


class ContentDefinitionSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[2]
    contents: list[ContentDefinition] = Field(min_length=1, max_length=MAX_SUPPORTED_CONTENT)

    @model_validator(mode="after")
    def unique_names(self) -> "ContentDefinitionSet":
        names = [_normalize(value.canonical_name) for value in self.contents]
        if len(names) != len(set(names)):
            raise ValueError("Content definitions must have unique canonical names.")
        terminal_keys = [
            (rule.entity_type, rule.hash)
            for definition in self.contents
            for rule in definition.terminal_evidence
        ]
        if len(terminal_keys) != len(set(terminal_keys)):
            raise ValueError("Terminal evidence mappings must be unique.")
        return self


class ContentProgressEvidence(BaseModel):
    type: EvidenceType
    name: str
    state: Literal["active", "completed", "in_progress", "returned", "conflicting"]
    scope: CompletionScope = "unknown"
    current_step: str | None = None
    objective_summary: str | None = None


class ContentProgression(BaseModel):
    canonical_name: str
    supported: bool = True
    status: ContentStatus
    evidence_state: EvidenceState
    active_quest_state: EvidenceState
    completion_state: EvidenceState
    profile_completion_state: EvidenceState
    completion_scope: CompletionScope
    current_step: str | None = None
    progress_summary: str | None = None
    evidence: list[ContentProgressEvidence] = Field(
        default_factory=list, max_length=MAX_EVIDENCE_ITEMS
    )
    limitations: list[str] = Field(default_factory=list, max_length=MAX_LIMITATIONS)


class ContentProgressionResponse(BaseModel):
    definition_version: int
    character_class: str
    requested_content: str | None = None
    content: ContentProgression | None = None
    contents: list[ContentProgression] = Field(default_factory=list, max_length=MAX_CONTENT_RESULTS)
    supported_content: list[str] = Field(default_factory=list, max_length=MAX_SUPPORTED_CONTENT)


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _display_name_matches(value: str | None, names: list[str]) -> bool:
    """Match structured display names, not arbitrary descriptive prose."""
    if not value:
        return False
    normalized = _normalize(value)
    for name in names:
        candidate = _normalize(name)
        if normalized == candidate:
            return True
        qualified = rf"{re.escape(candidate)} (?:campaign|quest|chapter|step)"
        if re.fullmatch(rf"{qualified} [a-z0-9 ]+", normalized):
            return True
        if re.fullmatch(qualified, normalized):
            return True
    return False


def _prompt_mentions(message: str, aliases: list[str]) -> bool:
    normalized = f" {_normalize(message)} "
    return any(f" {_normalize(alias)} " in normalized for alias in aliases)


def _objective_summary(objectives: list[ObjectiveSummary]) -> str | None:
    visible = [value for value in objectives if value.visible][:MAX_OBJECTIVES]
    if not visible:
        return None
    parts: list[str] = []
    for objective in visible:
        if objective.progress is not None and objective.completion_value:
            parts.append(f"{objective.name}: {objective.progress}/{objective.completion_value}")
        else:
            parts.append(f"{objective.name}: {'complete' if objective.complete else 'incomplete'}")
    return "; ".join(parts)


@lru_cache(maxsize=1)
def load_content_definitions() -> ContentDefinitionSet:
    return ContentDefinitionSet.model_validate_json(DATA_PATH.read_text(encoding="utf-8"))


def supported_content_aliases() -> tuple[str, ...]:
    return tuple(
        _normalize(alias)
        for definition in load_content_definitions().contents
        for alias in definition.aliases
    )


def mentioned_content_names(message: str) -> list[str]:
    return [
        value.canonical_name
        for value in load_content_definitions().contents
        if _prompt_mentions(message, value.aliases)
    ][:MAX_CONTENT_RESULTS]


def terminal_record_hashes() -> set[int]:
    return {
        rule.hash
        for definition in load_content_definitions().contents
        for rule in definition.terminal_evidence
        if rule.entity_type == "DestinyRecordDefinition"
    }


class ManifestRecordResolver(Protocol):
    async def resolve_many(
        self, entity_type: str, entity_hashes: set[int]
    ) -> dict[int, dict[str, Any]]: ...


async def validate_terminal_mappings(resolver: ManifestRecordResolver) -> list[str]:
    """Validate every maintained terminal mapping against Manifest definitions."""
    rules = [
        (definition.canonical_name, rule)
        for definition in load_content_definitions().contents
        for rule in definition.terminal_evidence
    ]
    hashes = {rule.hash for _, rule in rules}
    resolved = await resolver.resolve_many("DestinyRecordDefinition", hashes)
    errors: list[str] = []
    scope_values = {"profile": 0, "character": 1}
    for content_name, rule in rules:
        definition = resolved.get(rule.hash)
        label = f"{content_name}:{rule.hash}"
        if not definition:
            errors.append(f"{label}: Manifest definition unavailable")
            continue
        display = definition.get("displayProperties") or {}
        if int(definition.get("hash", 0)) != rule.hash:
            errors.append(f"{label}: Manifest hash mismatch")
        if _normalize(str(display.get("name", ""))) != _normalize(rule.resolved_name):
            errors.append(f"{label}: Manifest name mismatch")
        if _normalize(str(display.get("description", ""))) != _normalize(rule.expected_description):
            errors.append(f"{label}: Manifest description mismatch")
        if definition.get("scope") != scope_values[rule.expected_scope]:
            errors.append(f"{label}: Manifest scope mismatch")
        if definition.get("redacted"):
            errors.append(f"{label}: Manifest definition is redacted")
    return errors


class ContentProgressionResolver:
    """Resolve bounded major-content state from one normalized GuardianContext."""

    def __init__(
        self,
        context: GuardianContext,
        definitions: ContentDefinitionSet | None = None,
    ) -> None:
        self.context = context
        self.definitions = definitions or load_content_definitions()

    def supported_names(self) -> list[str]:
        return [value.canonical_name for value in self.definitions.contents[:MAX_SUPPORTED_CONTENT]]

    def mentioned_content(self, message: str) -> list[str]:
        if self.definitions is load_content_definitions():
            return mentioned_content_names(message)
        return [
            value.canonical_name
            for value in self.definitions.contents
            if _prompt_mentions(message, value.aliases)
        ][:MAX_CONTENT_RESULTS]

    def resolve(
        self, character: CharacterSummary, content_name: str | None
    ) -> ContentProgressionResponse:
        if content_name is None:
            results = [
                self._resolve_definition(character, definition)
                for definition in self.definitions.contents[:MAX_CONTENT_RESULTS]
            ]
            return self._response(character, None, contents=results)

        definition = self._find_definition(content_name)
        if definition is None:
            unsupported = ContentProgression(
                canonical_name=content_name.strip() or "Unknown content",
                supported=False,
                status="unknown",
                evidence_state="UNKNOWN",
                active_quest_state="NOT_RETURNED",
                completion_state="UNKNOWN",
                profile_completion_state="UNKNOWN",
                completion_scope="unknown",
                limitations=["This content name is not in the maintained progression definitions."],
            )
            return self._response(character, content_name, content=unsupported)
        return self._response(
            character,
            content_name,
            content=self._resolve_definition(character, definition),
        )

    def _response(
        self,
        character: CharacterSummary,
        requested_content: str | None,
        *,
        content: ContentProgression | None = None,
        contents: list[ContentProgression] | None = None,
    ) -> ContentProgressionResponse:
        return ContentProgressionResponse(
            definition_version=self.definitions.version,
            character_class=character.class_name,
            requested_content=requested_content,
            content=content,
            contents=contents or [],
            supported_content=self.supported_names(),
        )

    def _find_definition(self, content_name: str) -> ContentDefinition | None:
        normalized = _normalize(content_name)
        for definition in self.definitions.contents:
            if normalized in {_normalize(value) for value in definition.aliases}:
                return definition
        return None

    def _resolve_definition(
        self, character: CharacterSummary, definition: ContentDefinition
    ) -> ContentProgression:
        evidence: list[ContentProgressEvidence] = []
        active_quests = []
        character_terminal_evidence: list[str] = []
        profile_terminal_evidence: list[str] = []
        terminal_conflict = False
        resolver_limitations: list[str] = []

        for quest in character.quests:
            matched = _display_name_matches(
                quest.name, definition.match_names
            ) or _display_name_matches(quest.step_name, definition.match_names)
            terminal = (
                quest.completed
                and quest.redeemed
                and _display_name_matches(quest.name, definition.completion_quest_names)
            )
            if not matched and not terminal:
                continue
            if terminal:
                character_terminal_evidence.append(quest.name)
            active = not (quest.completed and quest.redeemed)
            if active:
                active_quests.append(quest)
            state: Literal["active", "completed"] = "active" if active else "completed"
            evidence.append(
                ContentProgressEvidence(
                    type="quest",
                    name=quest.name,
                    state=state,
                    scope="character",
                    current_step=quest.step_name,
                    objective_summary=_objective_summary(quest.objectives),
                )
            )

        for milestone in character.milestones:
            names = [milestone.name, *milestone.quest_names]
            matched = any(_display_name_matches(value, definition.match_names) for value in names)
            terminal = any(
                _display_name_matches(value, definition.completion_milestone_names)
                for value in names
            )
            if not matched and not terminal:
                continue
            if (
                terminal
                and milestone.objectives
                and all(value.complete for value in milestone.objectives)
            ):
                character_terminal_evidence.append(milestone.name)
            evidence.append(
                ContentProgressEvidence(
                    type="milestone",
                    name=milestone.name,
                    state=(
                        "completed"
                        if terminal
                        and milestone.objectives
                        and all(value.complete for value in milestone.objectives)
                        else "returned"
                    ),
                    scope="character",
                    objective_summary=_objective_summary(milestone.objectives),
                )
            )

        for progression in character.progressions:
            matched = _display_name_matches(progression.name, definition.match_names)
            terminal = _display_name_matches(
                progression.name, definition.completion_progression_names
            )
            if not matched and not terminal:
                continue
            if terminal and progression.level_cap and progression.level >= progression.level_cap:
                character_terminal_evidence.append(progression.name)
            summary = f"Level {progression.level}"
            if progression.level_cap:
                summary += f"/{progression.level_cap}"
            evidence.append(
                ContentProgressEvidence(
                    type="progression",
                    name=progression.name,
                    state=(
                        "completed"
                        if terminal
                        and progression.level_cap
                        and progression.level >= progression.level_cap
                        else "in_progress"
                    ),
                    scope="character",
                    objective_summary=summary,
                )
            )

        for rule in definition.terminal_evidence:
            same_hash = [
                record for record in self.context.records.records if record.record_hash == rule.hash
            ]
            if rule.expected_scope == "character":
                candidates = [
                    record
                    for record in same_hash
                    if record.scope == "character" and record.character_id == character.character_id
                ]
            else:
                candidates = [record for record in same_hash if record.scope == "profile"]
            for record in candidates:
                identity_matches = (
                    record.manifest_resolved
                    and _normalize(record.name) == _normalize(rule.resolved_name)
                    and _normalize(record.description or "")
                    == _normalize(rule.expected_description)
                )
                if not identity_matches:
                    resolver_limitations.append(
                        f"The maintained {definition.canonical_name} completion record could not "
                        "be validated against its current Manifest identity."
                    )
                    continue
                inconsistent_objectives = record.completed and record.objectives_complete is False
                if inconsistent_objectives:
                    terminal_conflict = True
                evidence.append(
                    ContentProgressEvidence(
                        type="record",
                        name=record.name,
                        state=(
                            "conflicting"
                            if inconsistent_objectives
                            else "completed"
                            if record.completed
                            else "returned"
                        ),
                        scope=record.scope,
                        objective_summary=_objective_summary(record.objectives),
                    )
                )
                if not record.completed or inconsistent_objectives:
                    continue
                if record.scope == "character":
                    character_terminal_evidence.append(record.name)
                elif record.scope == "profile":
                    profile_terminal_evidence.append(record.name)

            wrong_scope = [record for record in same_hash if record.scope != rule.expected_scope]
            if wrong_scope:
                resolver_limitations.append(
                    f"A {definition.canonical_name} record was returned at an unexpected scope "
                    "and was not used as completion evidence."
                )

        evidence = evidence[:MAX_EVIDENCE_ITEMS]
        limitations: list[str] = list(resolver_limitations)
        active_state: EvidenceState = "KNOWN_TRUE" if active_quests else "NOT_RETURNED"
        current_step = next((value.step_name for value in active_quests if value.step_name), None)
        progress_summary = next(
            (
                _objective_summary(value.objectives)
                for value in active_quests
                if _objective_summary(value.objectives)
            ),
            None,
        )

        has_character_terminal = bool(character_terminal_evidence)
        has_profile_terminal = bool(profile_terminal_evidence)
        completion_scope: CompletionScope = (
            "character"
            if has_character_terminal
            else "profile"
            if has_profile_terminal
            else "unknown"
        )
        profile_completion_state: EvidenceState = (
            "KNOWN_TRUE" if has_profile_terminal else "UNKNOWN"
        )

        if terminal_conflict or (has_character_terminal and active_quests):
            status: ContentStatus = "unknown"
            evidence_state: EvidenceState = "UNKNOWN"
            completion_state: EvidenceState = "UNKNOWN"
            limitations.append(
                "Terminal completion and current progression evidence conflict, so no character "
                "status is asserted."
            )
        elif has_character_terminal:
            status = "completed"
            evidence_state = "KNOWN_TRUE"
            completion_state = "KNOWN_TRUE"
        elif active_quests:
            status = "in_progress"
            evidence_state = "KNOWN_TRUE"
            completion_state = "UNKNOWN"
            if has_profile_terminal:
                limitations.append(
                    "The account has completion evidence, but it does not establish that this "
                    f"{character.class_name} completed the campaign."
                )
        elif has_profile_terminal:
            status = "unknown"
            evidence_state = "UNKNOWN"
            completion_state = "UNKNOWN"
            limitations.append(
                "The account has completion evidence, but the available Bungie data does not "
                f"identify whether this {character.class_name} completed the campaign."
            )
        else:
            status = "unknown"
            evidence_state = "UNKNOWN"
            completion_state = "UNKNOWN"
            limitations.append(
                "No matching active quest was returned. This does not establish completion, "
                "not-started state, availability, or ownership."
            )

        if (
            evidence
            and status == "unknown"
            and not (has_character_terminal or has_profile_terminal or terminal_conflict)
        ):
            limitations.append(
                "Related Bungie-backed evidence was returned, but no maintained rule makes it "
                "terminal campaign evidence."
            )
        missing = self.context.data_availability.unavailable_components
        if "CharacterProgressions" in missing or "Records" in missing:
            limitations.append(
                "Some Bungie progression components were not returned, so historical completion "
                "may be unavailable."
            )
        return ContentProgression(
            canonical_name=definition.canonical_name,
            status=status,
            evidence_state=evidence_state,
            active_quest_state=active_state,
            completion_state=completion_state,
            profile_completion_state=profile_completion_state,
            completion_scope=completion_scope,
            current_step=current_step,
            progress_summary=progress_summary,
            evidence=evidence,
            limitations=list(dict.fromkeys(limitations))[:MAX_LIMITATIONS],
        )
