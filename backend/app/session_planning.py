import re
from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field, model_serializer

from app.content_progression import (
    CompletionScope,
    ContentProgressionResolver,
    ContentStatus,
    EvidenceState,
)
from app.models import (
    CharacterSummary,
    GuardianContext,
    ObjectiveRequirement,
    ObjectiveSummary,
)
from app.objective_requirements import is_restrictive
from app.requirement_fit import (
    REQUIREMENT_BASIS,
    LoadoutEvidence,
    PlanningRequirement,
    loadout_evidence,
    planning_requirement,
)
from app.session_preferences import PrimaryGoal, SessionPreferenceContext

CandidateClassification = Literal[
    "meaningful_major_progression",
    "routine_or_grind_objective",
    "record_or_collectible_cleanup",
    "unknown_significance",
]

MAX_CANDIDATES = 12
MAX_OBJECTIVES_PER_CANDIDATE = 3
MAX_CANDIDATE_DESCRIPTION = 160
MAX_CONTEXT_PATH = 6
MAX_RECENT_ACTIVITIES = 6
MAX_PROGRESSIONS = 6
MAX_REQUIREMENTS_PER_CANDIDATE = 2
MAX_REQUIREMENT_CHARS = 2400
# Activity types whose grounded fireteam size is three or more (core knowledge: Strike,
# Nightfall, Raid, Dungeon). A solo player is not offered them as a first choice.
TEAM_ACTIVITY_TYPES = ("nightfall", "strike", "vanguard op", "raid", "dungeon", "gambit")

# Appended to every correction request: the revised answer replaces the draft, so it must not
# talk about the revision.
REWRITE_RULE = (
    "Rewrite the whole answer so it reads as if it had been right the first time: state the "
    "corrected fact directly, and do not announce, label, or apologize for a correction (no "
    "'quick correction', 'to clarify', 'don't mix them up'). Do not mention validation, tools, "
    "providers, or internal reasoning. Keep every source link, date, and sourced fact from the "
    "draft; change only what the correction names."
)


def _has_term(label: str, term: str) -> bool:
    """The term as whole words in a normalized label; a plural "s" is allowed."""

    return re.search(rf"(?:^| ){re.escape(_normalize(term))}s?(?: |$)", label) is not None


def _is_team_activity(value: str) -> bool:
    label = _normalize(value)
    return any(_has_term(label, term) for term in TEAM_ACTIVITY_TYPES)


# Words that identify each explicit session exclusion, in answers and in candidate names or
# activity types. A candidate that matches an exclusion is removed before the model sees it.
EXCLUSION_TERMS: dict[str, tuple[str, ...]] = {
    "pvp": ("pvp", "crucible", "trials", "iron banner"),
    "pve": ("pve",),
    "dungeons": ("dungeon",),
    "raids": ("raid",),
    "grinding": ("grind", "farm"),
    "group_play": ("fireteam", "matchmade", "with a group"),
    "solo_play": ("solo",),
}


class PlanningObjective(BaseModel):
    name: str
    completion_status: EvidenceState
    progress_percent: float | None = None


class ContentEvidence(BaseModel):
    content: str
    status: ContentStatus
    evidence_state: EvidenceState
    active_quest_returned: bool
    active_quest_status: EvidenceState
    completion_status: EvidenceState = "UNKNOWN"
    profile_completion_status: EvidenceState = "UNKNOWN"
    completion_scope: CompletionScope = "unknown"
    current_step: str | None = None
    progress_summary: str | None = None
    completion_evidence: list[str] = Field(default_factory=list, max_length=3)
    profile_completion_evidence: list[str] = Field(default_factory=list, max_length=3)
    limitations: list[str] = Field(default_factory=list, max_length=4)


class PlanningCandidate(BaseModel):
    name: str
    source_kinds: list[str] = Field(default_factory=list, max_length=4)
    classification: CandidateClassification
    active_quest_status: EvidenceState
    completion_status: EvidenceState
    available_to_character: EvidenceState
    content_status: ContentStatus | None = None
    content_evidence_state: EvidenceState | None = None
    profile_completion_status: EvidenceState = "UNKNOWN"
    completion_scope: CompletionScope = "unknown"
    current_step: str | None = None
    progress_summary: str | None = None
    activity_type: str | None = None
    destination: str | None = None
    place: str | None = None
    description: str | None = Field(default=None, max_length=MAX_CANDIDATE_DESCRIPTION)
    context_path: list[str] = Field(default_factory=list, max_length=MAX_CONTEXT_PATH)
    item_type_name: str | None = Field(default=None, max_length=80)
    # Manifest-declared reward names only; not a drop guarantee or a value judgement.
    reward_names: list[str] = Field(default_factory=list, max_length=4)
    objectives: list[PlanningObjective] = Field(default_factory=list, max_length=3)
    recently_repeated: bool = False
    recent_occurrences: int = Field(default=0, ge=0)
    overlap_signals: list[str] = Field(default_factory=list, max_length=3)
    external_significance_needed: bool = True
    # M4B: compact structured objective requirements with an equipped-loadout comparison.
    requirements: list[PlanningRequirement] = Field(
        default_factory=list, max_length=MAX_REQUIREMENTS_PER_CANDIDATE
    )
    requirements_consistency: Literal["differs_in_specificity", "conflicting"] | None = None

    @model_serializer(mode="wrap")
    def _omit_empty_requirements(self, handler):  # type: ignore[no-untyped-def]
        data = handler(self)
        for key in ("requirements", "requirements_consistency"):
            if not data.get(key):
                data.pop(key, None)
        return data


class SessionPlanningContext(BaseModel):
    version: Literal[6] = 6
    preferences: SessionPreferenceContext
    requested_character_scope: str | None
    requested_scope_explicit: bool
    character_class: str | None
    intent_category: str
    evidence_semantics: dict[EvidenceState, str]
    named_content: list[ContentEvidence] = Field(default_factory=list, max_length=4)
    candidates: list[PlanningCandidate] = Field(default_factory=list, max_length=12)
    notable_progression: list[dict[str, str | int]] = Field(default_factory=list, max_length=6)
    recent_activities: list[dict[str, str | int | bool | None]] = Field(
        default_factory=list, max_length=6
    )
    explicitly_known_completion_facts: list[str] = Field(default_factory=list, max_length=6)
    explicitly_unknown_completion_facts: list[str] = Field(default_factory=list, max_length=6)
    data_limitations: list[str] = Field(default_factory=list, max_length=8)
    # Set when the user rejected suggestions or asked for another option; see
    # planning_alternatives. Excluded candidates are already removed from `candidates`.
    excluded_by_user: list[str] = Field(default_factory=list, max_length=12)
    # Candidates removed because they match a stated exclusion ("no dungeons", "no Crucible").
    excluded_by_preference: list[str] = Field(default_factory=list, max_length=12)
    already_recommended: list[str] = Field(default_factory=list, max_length=3)
    alternative_guidance: str | None = None
    # M4B: how to read candidate `requirements`, present only when some candidate has any.
    requirement_basis: str | None = None
    requirements_omitted: int = Field(default=0, ge=0)

    @model_serializer(mode="wrap")
    def _omit_empty_requirement_fields(self, handler):  # type: ignore[no-untyped-def]
        data = handler(self)
        for key in ("requirement_basis", "requirements_omitted", "excluded_by_preference"):
            if not data.get(key):
                data.pop(key, None)
        return data


class PlanningViolation(BaseModel):
    code: str
    correction: str


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _character_for_scope(
    context: GuardianContext, requested_character: str | None
) -> CharacterSummary | None:
    if requested_character:
        for character in context.characters:
            if character.class_name.casefold() == requested_character.casefold():
                return character
        return None
    if not context.characters:
        return None
    return max(
        context.characters,
        key=lambda value: value.last_played.timestamp() if value.last_played else 0,
    )


def _bounded_description(value: str | None) -> str | None:
    text = " ".join((value or "").split())
    if not text:
        return None
    if len(text) <= MAX_CANDIDATE_DESCRIPTION:
        return text
    return text[: MAX_CANDIDATE_DESCRIPTION - 1].rstrip() + "…"


def _objectives(values: list[ObjectiveSummary]) -> list[PlanningObjective]:
    return [
        PlanningObjective(
            name=value.name,
            completion_status="KNOWN_TRUE" if value.complete else "KNOWN_FALSE",
            progress_percent=value.progress_percent,
        )
        for value in values[:MAX_OBJECTIVES_PER_CANDIDATE]
    ]


def _candidate_requirements(
    described: list[ObjectiveRequirement], objectives: list[ObjectiveSummary]
) -> list[ObjectiveRequirement]:
    """Requirements worth showing: restrictive ones, and any that could not be established.

    A clean generic "defeat" says nothing the description does not, so it is left out. Text
    from the description and from objective definitions stays separate; duplicates by content
    are dropped, never merged.
    """

    found: list[ObjectiveRequirement] = []
    seen: set[tuple[str, ...]] = set()
    for requirement in [*described, *(v.requirement for v in objectives if v.requirement)]:
        if requirement.parse_status == "verified" and not is_restrictive(requirement):
            continue
        key = (
            requirement.source_kind,
            requirement.element,
            requirement.completion_action,
            requirement.parse_status,
            *requirement.modifiers,
        )
        if key not in seen:
            seen.add(key)
            found.append(requirement)
    return found


def requirement_trace(context: SessionPlanningContext | None) -> dict[str, object]:
    """Counts only, for the turn trace: candidates with requirements and comparison outcomes."""

    if context is None:
        return {"candidates_with_requirements": 0}
    fits = Counter(
        value.fit.status for candidate in context.candidates for value in candidate.requirements
    )
    return {
        "candidates_with_requirements": sum(
            bool(value.requirements) for value in context.candidates
        ),
        "fit": dict(sorted(fits.items())),
        "omitted": context.requirements_omitted,
    }


def _attach_requirements(
    candidates: dict[str, PlanningCandidate],
    pending: dict[str, tuple[list[ObjectiveRequirement], str | None]],
    loadout: LoadoutEvidence | None,
) -> tuple[str | None, int]:
    """Attach bounded requirement evidence to candidates; return (basis, omitted count).

    Candidates are visited in order and the total serialized size is capped, so later
    candidates lose requirement detail first. Nothing here changes which candidates exist, their
    order, or any other field.
    """

    if loadout is None:
        return None, 0
    used = len(REQUIREMENT_BASIS)
    omitted = 0
    attached = False
    for key, candidate in candidates.items():
        requirements, consistency = pending.get(key, ([], None))
        if not requirements:
            continue
        kept: list[PlanningRequirement] = []
        for requirement in requirements:
            if len(kept) >= MAX_REQUIREMENTS_PER_CANDIDATE:
                omitted += 1
                continue
            entry = planning_requirement(requirement, loadout)
            size = len(entry.model_dump_json())
            if used + size > MAX_REQUIREMENT_CHARS:
                omitted += 1
                continue
            used += size
            kept.append(entry)
        if kept:
            candidate.requirements = kept
            candidate.requirements_consistency = consistency  # type: ignore[assignment]
            attached = True
    return (REQUIREMENT_BASIS if attached else None), omitted


def _classification(
    name: str, source_kind: str, named_content: set[str]
) -> CandidateClassification:
    normalized = _normalize(name)
    if normalized in named_content or "campaign" in normalized:
        return "meaningful_major_progression"
    if source_kind == "record":
        return "record_or_collectible_cleanup"
    if any(value in normalized for value in ("bounty", "weekly", "rank", "ritual")):
        return "routine_or_grind_objective"
    return "unknown_significance"


def build_session_planning_context(
    context: GuardianContext,
    message: str,
    intent_category: str,
    requested_character: str | None,
    preferences: SessionPreferenceContext | None = None,
) -> SessionPlanningContext:
    goal_by_intent: dict[str, PrimaryGoal] = {
        "story": "story_progression",
        "progression": "story_progression",
        "gear": "gear_rewards",
        "quest_completion": "quest_completion",
        "build": "build_improvement",
        "casual": "casual_chill",
        "challenge": "challenge",
        "exploration": "exploration",
    }
    effective_preferences = preferences or SessionPreferenceContext(
        character=requested_character or "unspecified",
        primary_goal=goal_by_intent.get(intent_category, "unspecified"),
    )
    requested_character = (
        effective_preferences.character
        if effective_preferences.character != "unspecified"
        else requested_character
    )
    character = _character_for_scope(context, requested_character)
    progression_resolver = ContentProgressionResolver(context)
    requested_content = progression_resolver.mentioned_content(message)
    requested_content_keys = {_normalize(value) for value in requested_content}
    limitations: list[str] = []
    if character is None:
        limitations.append(
            f"No {requested_character} character was returned."
            if requested_character
            else "No character was returned."
        )

    active_quests = (
        [value for value in character.quests if not (value.completed and value.redeemed)]
        if character
        else []
    )
    content_evidence: list[ContentEvidence] = []
    if character:
        for content in requested_content:
            resolved = progression_resolver.resolve(character, content).content
            if resolved is None:
                continue
            content_evidence.append(
                ContentEvidence(
                    content=resolved.canonical_name,
                    status=resolved.status,
                    evidence_state=resolved.evidence_state,
                    active_quest_returned=resolved.active_quest_state == "KNOWN_TRUE",
                    active_quest_status=resolved.active_quest_state,
                    completion_status=resolved.completion_state,
                    profile_completion_status=resolved.profile_completion_state,
                    completion_scope=resolved.completion_scope,
                    current_step=resolved.current_step,
                    progress_summary=resolved.progress_summary,
                    completion_evidence=(
                        [
                            value.name
                            for value in resolved.evidence
                            if value.state == "completed" and value.scope != "profile"
                        ][:3]
                        if resolved.completion_state == "KNOWN_TRUE"
                        else []
                    ),
                    profile_completion_evidence=(
                        [
                            value.name
                            for value in resolved.evidence
                            if value.type == "record"
                            and value.scope == "profile"
                            and value.state == "completed"
                        ][:3]
                        if resolved.profile_completion_state == "KNOWN_TRUE"
                        else []
                    ),
                    limitations=[
                        "An active quest is one candidate signal, not proof that this content "
                        "should be next.",
                        *resolved.limitations,
                    ][:4],
                )
            )
    else:
        content_evidence.extend(
            ContentEvidence(
                content=content,
                status="unknown",
                evidence_state="UNKNOWN",
                active_quest_returned=False,
                active_quest_status="NOT_RETURNED",
                completion_status="UNKNOWN",
                profile_completion_status="UNKNOWN",
                completion_scope="unknown",
                limitations=[
                    "The requested character was not returned, so content progress is unknown."
                ],
            )
            for content in requested_content
        )

    recent_counts = Counter(
        _normalize(value.name) for value in (character.recent_activities if character else [])
    )
    candidates: dict[str, PlanningCandidate] = {}
    # M4B: structured requirements by candidate key; compared after candidates are settled.
    pending_requirements: dict[str, tuple[list[ObjectiveRequirement], str | None]] = {}

    def add_candidate(candidate: PlanningCandidate) -> None:
        key = _normalize(candidate.name)
        existing = candidates.get(key)
        if existing is None:
            if len(candidates) < MAX_CANDIDATES:
                candidates[key] = candidate
            return
        for source in candidate.source_kinds:
            if source not in existing.source_kinds and len(existing.source_kinds) < 4:
                existing.source_kinds.append(source)
        if candidate.active_quest_status == "KNOWN_TRUE":
            existing.active_quest_status = "KNOWN_TRUE"
        if candidate.available_to_character == "KNOWN_TRUE":
            existing.available_to_character = "KNOWN_TRUE"
        if existing.classification == "unknown_significance":
            existing.classification = candidate.classification
        if candidate.content_status is not None:
            existing.content_status = candidate.content_status
            existing.content_evidence_state = candidate.content_evidence_state
            existing.profile_completion_status = candidate.profile_completion_status
            existing.completion_scope = candidate.completion_scope
            existing.current_step = candidate.current_step
            existing.progress_summary = candidate.progress_summary
        existing.overlap_signals = list(dict.fromkeys(existing.source_kinds))[:3]
        if candidate.objectives and not existing.objectives:
            existing.objectives = candidate.objectives
        for field_name in (
            "activity_type",
            "destination",
            "place",
            "description",
            "item_type_name",
        ):
            if getattr(existing, field_name) is None and getattr(candidate, field_name):
                setattr(existing, field_name, getattr(candidate, field_name))
        if candidate.context_path and not existing.context_path:
            existing.context_path = candidate.context_path
        if candidate.reward_names and not existing.reward_names:
            existing.reward_names = candidate.reward_names

    for content in content_evidence:
        key = _normalize(content.content)
        add_candidate(
            PlanningCandidate(
                name=content.content,
                source_kinds=["named_content"],
                classification="meaningful_major_progression",
                active_quest_status=content.active_quest_status,
                completion_status=content.completion_status,
                profile_completion_status=content.profile_completion_status,
                completion_scope=content.completion_scope,
                available_to_character="UNKNOWN",
                content_status=content.status,
                content_evidence_state=content.evidence_state,
                current_step=content.current_step,
                progress_summary=content.progress_summary,
                external_significance_needed=True,
            )
        )
        requested_content_keys.add(key)

    for quest in active_quests[:6]:
        key = _normalize(quest.name)
        is_named_content = key in requested_content_keys
        pending_requirements.setdefault(
            key,
            (
                _candidate_requirements(quest.requirements, quest.objectives),
                quest.requirements_consistency
                if quest.requirements_consistency in ("differs_in_specificity", "conflicting")
                else None,
            ),
        )
        add_candidate(
            PlanningCandidate(
                name=quest.name,
                source_kinds=["active_quest"],
                classification=_classification(quest.name, "quest", requested_content_keys),
                active_quest_status="KNOWN_TRUE",
                completion_status="UNKNOWN" if is_named_content else "KNOWN_FALSE",
                available_to_character="KNOWN_TRUE",
                description=_bounded_description(quest.description),
                item_type_name=quest.item_type_name,
                reward_names=list(quest.reward_names),
                objectives=_objectives(quest.objectives),
                recently_repeated=recent_counts[key] > 1,
                recent_occurrences=recent_counts[key],
                external_significance_needed=True,
            )
        )

    if character:
        for milestone in character.milestones[:4]:
            key = _normalize(milestone.name)
            pending_requirements.setdefault(
                key, (_candidate_requirements([], milestone.objectives), None)
            )
            add_candidate(
                PlanningCandidate(
                    name=milestone.name,
                    source_kinds=["milestone"],
                    classification=_classification(
                        milestone.name, "milestone", requested_content_keys
                    ),
                    active_quest_status="NOT_RETURNED",
                    completion_status="UNKNOWN",
                    available_to_character="KNOWN_TRUE",
                    objectives=_objectives(milestone.objectives),
                    recently_repeated=recent_counts[key] > 1,
                    recent_occurrences=recent_counts[key],
                    external_significance_needed=True,
                )
            )
        for activity in [value for value in character.available_activities if value.is_visible][:6]:
            key = _normalize(activity.name)
            pending_requirements.setdefault(
                key, (_candidate_requirements([], activity.objectives), None)
            )
            add_candidate(
                PlanningCandidate(
                    name=activity.name,
                    source_kinds=["available_activity"],
                    classification=_classification(
                        activity.name, "activity", requested_content_keys
                    ),
                    active_quest_status="NOT_RETURNED",
                    completion_status="KNOWN_TRUE" if activity.is_completed else "KNOWN_FALSE",
                    available_to_character="KNOWN_TRUE",
                    activity_type=activity.activity_type,
                    destination=activity.destination,
                    place=activity.place,
                    objectives=_objectives(activity.objectives),
                    recently_repeated=recent_counts[key] > 1,
                    recent_occurrences=recent_counts[key],
                    external_significance_needed=True,
                )
            )

    for objective in context.records.near_completion[:3]:
        add_candidate(
            PlanningCandidate(
                name=objective.name,
                source_kinds=["record"],
                classification="record_or_collectible_cleanup",
                active_quest_status="NOT_RETURNED",
                completion_status="KNOWN_TRUE" if objective.complete else "KNOWN_FALSE",
                available_to_character="UNKNOWN",
                # The Record's category path is passed through as-is. No destination is
                # inferred from it; destination stays reserved for resolved hashes.
                description=_bounded_description(objective.record_description),
                context_path=list(objective.context_path[:MAX_CONTEXT_PATH]),
                objectives=_objectives([objective]),
                external_significance_needed=True,
            )
        )

    if character:
        notable_progression = [
            {"name": value.name, "level": value.level, "scope": value.scope}
            for value in character.progressions[:MAX_PROGRESSIONS]
        ]
        recent_activities = [
            {
                "name": value.name,
                "completed": value.completed,
                "duration_seconds": value.duration_seconds,
            }
            for value in character.recent_activities[:MAX_RECENT_ACTIVITIES]
        ]
    else:
        notable_progression = []
        recent_activities = []

    requirement_basis, requirements_omitted = _attach_requirements(
        candidates, pending_requirements, loadout_evidence(character) if character else None
    )

    excluded_by_preference: list[str] = []
    solo = effective_preferences.fireteam == "solo"
    # Milestones whose activities are all fireteam activities (Weekly Vanguard -> Vanguard Ops).
    team_milestones = {
        _normalize(milestone.name)
        for milestone in (character.milestones if character else [])
        if milestone.activity_names
        and all(_is_team_activity(name) for name in milestone.activity_names)
    }
    team_set_aside: list[str] = []
    for key, candidate in list(candidates.items()):
        # The record's category path counts too: a Duality record sits under "Dungeons".
        label = _normalize(
            " ".join((candidate.name, candidate.activity_type or "", *candidate.context_path))
        )
        if any(
            _has_term(label, term)
            for exclusion in effective_preferences.exclusions
            for term in EXCLUSION_TERMS[exclusion]
        ):
            excluded_by_preference.append(candidate.name)
            del candidates[key]
        elif solo and (
            _is_team_activity(candidate.name)
            or _is_team_activity(candidate.activity_type or "")
            or key in team_milestones
        ):
            excluded_by_preference.append(candidate.name)
            team_set_aside.append(candidate.name)
            del candidates[key]
    if solo:
        limitations.insert(
            0,
            "Solo may mean strictly alone or queuing alone into matchmaking. The data does not "
            "say which an activity supports, so do not state that one is solo-playable.",
        )
    if team_set_aside:
        limitations.insert(
            0,
            f"Set aside for the solo preference (grounded fireteam size is three or more): "
            f"{', '.join(team_set_aside[:4])}.",
        )

    known_completion = [
        f"Campaign completed: {value.content}"
        for value in content_evidence
        if value.completion_status == "KNOWN_TRUE"
    ]
    known_completion.extend(
        f"Account completion evidence: {value.content}"
        for value in content_evidence
        if value.profile_completion_status == "KNOWN_TRUE"
        and value.completion_status != "KNOWN_TRUE"
    )
    known_completion = known_completion[:6]
    unknown_completion = [
        f"Campaign completion unknown: {value.content}"
        for value in content_evidence
        if value.completion_status == "UNKNOWN"
    ]
    if context.data_availability.unavailable_components:
        limitations.extend(
            f"Guardian component not returned: {name}"
            for name in list(context.data_availability.unavailable_components)[:4]
        )
    limitations.extend(context.data_availability.notes[:3])

    return SessionPlanningContext(
        preferences=effective_preferences,
        requested_character_scope=requested_character,
        requested_scope_explicit=requested_character is not None,
        character_class=character.class_name if character else None,
        intent_category=intent_category,
        evidence_semantics={
            "KNOWN_TRUE": "The normalized Guardian data explicitly establishes this as true.",
            "KNOWN_FALSE": "The normalized Guardian data explicitly establishes this as false.",
            "UNKNOWN": "The available Guardian data cannot establish either true or false.",
            "NOT_RETURNED": "No matching value was returned; no further state may be inferred.",
        },
        named_content=content_evidence,
        candidates=list(candidates.values()),
        notable_progression=notable_progression,
        recent_activities=recent_activities,
        explicitly_known_completion_facts=known_completion,
        explicitly_unknown_completion_facts=unknown_completion,
        data_limitations=limitations[:8],
        excluded_by_preference=excluded_by_preference[:12],
        requirement_basis=requirement_basis,
        requirements_omitted=requirements_omitted,
    )


# "Run X", "- Try X", "**Queue into X**": the usual way a recommendation is phrased. Questions
# ("Play solo or with friends?") are excluded where this is used.
_IMPERATIVE_RECOMMENDATION = re.compile(
    r"^\s*(?:[-*•]\s*|\d+[.)]\s*)?(?:\*\*)?(?:(?:primary|backup|alternate|option \d)"
    r"(?: pick)?\s*[:—-]\s*)?(?:run|try|go (?:with|for|do)|queue(?: up| into)?|"
    r"jump into|hop into|head (?:to|into)|load into|launch|farm|grind|play|finish|complete|"
    r"do|tackle|work on)\b"
)
_RECOMMENDATION_MARKERS = (
    "recommend",
    "should",
    "play ",
    "choose",
    "pick",
    "clear next",
    "clearly next",
    "best next",
    "next step",
    "continue",
    "start ",
    "do the ",
)


def _recommends(sentence: str) -> bool:
    """One lowercased sentence recommends something ("you should…", "Run X", not "Run X?")."""

    return any(marker in sentence for marker in _RECOMMENDATION_MARKERS) or bool(
        _IMPERATIVE_RECOMMENDATION.match(sentence) and not sentence.rstrip().endswith("?")
    )


def contains_recommendation(text: str) -> bool:
    return any(_recommends(value) for value in re.split(r"(?<=[.!?])\s+|[\r\n]+", text.casefold()))


_AVOID_SENTENCE_MARKERS = (
    "instead of",
    "other than",
    "besides",
    "rather than",
    "aside from",
    "apart from",
    "without",
    "skipping",
    "set aside",
    "already",
)


class PlanningAnswerValidator:
    _recommendation_markers = _RECOMMENDATION_MARKERS
    _meaningful_reason_markers = (
        "unlock",
        "advance",
        "progress the story",
        "prerequisite",
        "fits your",
        "within your",
        "reward",
        "overlap",
        "significance",
        "because you want",
    )
    _uncertainty_markers = (
        "unknown",
        "can't confirm",
        "cannot confirm",
        "not established",
        "does not establish",
        "don't know",
    )

    def validate(self, answer: str, context: SessionPlanningContext) -> list[PlanningViolation]:
        normalized = answer.casefold()
        violations: list[PlanningViolation] = []
        recommends = any(value in normalized for value in self._recommendation_markers)
        meaningful_reason = any(value in normalized for value in self._meaningful_reason_markers)

        if recommends and "active" in normalized and not meaningful_reason:
            violations.append(
                PlanningViolation(
                    code="active_quest_only",
                    correction=(
                        "An active quest is only one signal and cannot be the sole reason for "
                        "the recommendation."
                    ),
                )
            )

        if recommends and not meaningful_reason:
            near_complete = bool(
                re.search(r"\b(?:9[0-9]|100)\s*%", normalized)
                or "near completion" in normalized
                or "almost complete" in normalized
            )
            if near_complete:
                violations.append(
                    PlanningViolation(
                        code="near_completion_only",
                        correction=(
                            "Near-completion is one signal and cannot be the sole reason for a "
                            "planning recommendation."
                        ),
                    )
                )

        scope = context.requested_character_scope
        if context.requested_scope_explicit and scope:
            other_classes = {"titan", "hunter", "warlock"} - {scope.casefold()}
            for other_class in other_classes:
                cross_scope_phrases = (
                    f"switch to {other_class}",
                    f"switch to your {other_class}",
                    f"switching to {other_class}",
                    f"switching to your {other_class}",
                    f"play {other_class}",
                    f"play your {other_class}",
                    f"play on {other_class}",
                    f"play on your {other_class}",
                    f"try {other_class}",
                    f"try your {other_class}",
                    f"use {other_class}",
                    f"use your {other_class}",
                )
                if any(value in normalized for value in cross_scope_phrases):
                    violations.append(
                        PlanningViolation(
                            code="character_scope",
                            correction=(
                                f"Keep recommendations and backups scoped to the user's {scope}."
                            ),
                        )
                    )
                    break

        preferences = context.preferences
        if preferences.character != "unspecified" and re.search(
            r"\b(?:which|what) character\b|\b(?:titan|hunter) or (?:hunter|warlock)\b",
            normalized,
        ):
            violations.append(
                PlanningViolation(
                    code="reasked_known_character",
                    correction=(
                        f"The user already selected {preferences.character}; do not ask which "
                        "character to use."
                    ),
                )
            )
        if preferences.time_minutes is not None and re.search(
            r"\bhow much time\b|\bhow long (?:do you|have you) have\b|\btime do you have\b",
            normalized,
        ):
            violations.append(
                PlanningViolation(
                    code="reasked_known_time",
                    correction=(
                        f"The user already established a {preferences.time_minutes}-minute "
                        "session; do not ask for their available time again."
                    ),
                )
            )
        if preferences.primary_goal != "unspecified" and re.search(
            r"\bstory or (?:loot|gear)\b|\b(?:loot|gear) or story\b|"
            r"\bwhat (?:is your|do you want as your) (?:goal|priority)\b",
            normalized,
        ):
            violations.append(
                PlanningViolation(
                    code="reasked_known_goal",
                    correction=(
                        f"The user's current goal is {preferences.primary_goal}; do not ask them "
                        "to choose that goal again."
                    ),
                )
            )
        questions = re.findall(r"[^.!?\r\n]*\?", normalized)
        if preferences.fireteam != "unspecified" and any(
            re.search(
                r"\b(?:do|would) you (?:want|prefer)\b.{0,60}\b"
                r"(?:solo|matchmade|matchmaking|fireteam|group|premade)\b|"
                r"\bsolo\b.{0,40}\bor\b.{0,40}\b(?:matchmade|matchmaking|group|fireteam)\b|"
                r"\b(?:matchmade|matchmaking|group|fireteam)\b.{0,40}\bor\b.{0,40}\bsolo\b",
                question,
            )
            for question in questions
        ):
            violations.append(
                PlanningViolation(
                    code="reasked_known_fireteam",
                    correction=(
                        f"The user's current fireteam preference is {preferences.fireteam}; "
                        "apply it without asking again."
                    ),
                )
            )
        if preferences.activity_mode != "unspecified" and any(
            re.search(
                r"\b(?:pve|pvp)\b.{0,30}\bor\b.{0,30}\b(?:pve|pvp)\b|"
                r"\b(?:do|would) you (?:want|prefer)\b.{0,50}\b(?:pve|pvp)\b",
                question,
            )
            for question in questions
        ):
            violations.append(
                PlanningViolation(
                    code="reasked_known_activity_mode",
                    correction=(
                        f"The user's current activity preference is "
                        f"{preferences.activity_mode.upper()}; apply it without asking again."
                    ),
                )
            )
        if preferences.intensity != "unspecified" and any(
            re.search(
                r"\b(?:chill|casual|relaxed)\b.{0,40}\bor\b.{0,40}\b"
                r"(?:challenging|intense|hard)\b|"
                r"\b(?:do|would) you (?:want|prefer)\b.{0,50}\b"
                r"(?:chill|casual|relaxed|challenging|intense|hard)\b",
                question,
            )
            for question in questions
        ):
            violations.append(
                PlanningViolation(
                    code="reasked_known_intensity",
                    correction=(
                        f"The user's current intensity preference is {preferences.intensity}; "
                        "apply it without asking again."
                    ),
                )
            )
        if preferences.duration_preference != "unspecified" and any(
            re.search(
                r"\bhow (?:long|short)\b|\b(?:do|would) you (?:want|prefer)\b.{0,50}\b"
                r"(?:short|shorter|long|longer)\b",
                question,
            )
            for question in questions
        ):
            violations.append(
                PlanningViolation(
                    code="reasked_known_duration",
                    correction=(
                        f"The user's current duration preference is "
                        f"{preferences.duration_preference}; apply it without asking again."
                    ),
                )
            )
        excluded_question_terms: dict[str, tuple[str, ...]] = {
            "pvp": ("pvp", "crucible", "trials", "iron banner"),
            "pve": ("pve",),
            "dungeons": ("dungeon",),
            "raids": ("raid",),
            "grinding": ("grind", "farm"),
            "group_play": ("fireteam", "matchmade", "group"),
            "solo_play": ("solo",),
        }
        for exclusion in preferences.exclusions:
            if any(
                re.search(r"\b(?:can|could|would|do|are) you\b", question)
                and any(term in question for term in excluded_question_terms[exclusion])
                for question in questions
            ):
                violations.append(
                    PlanningViolation(
                        code=f"reasked_excluded_{exclusion}",
                        correction=(
                            f"The user already excluded {exclusion}; apply that constraint "
                            "without asking whether it is acceptable."
                        ),
                    )
                )
        if answer.count("?") > 1:
            violations.append(
                PlanningViolation(
                    code="too_many_followups",
                    correction="Ask at most one targeted follow-up question.",
                )
            )

        forbidden_terms = EXCLUSION_TERMS
        recommendation_sentences = [
            sentence
            for sentence in re.split(r"(?<=[.!?])\s+|[\r\n]+", normalized)
            if _recommends(sentence)
            and not any(
                marker in sentence
                for marker in ("avoid ", "skip ", "no ", "not ", "don't ", "do not ")
            )
        ]
        for exclusion in preferences.exclusions:
            terms = forbidden_terms[exclusion]
            if any(
                any(term in sentence for term in terms) for sentence in recommendation_sentences
            ):
                violations.append(
                    PlanningViolation(
                        code=f"excluded_{exclusion}",
                        correction=f"Honor the user's explicit exclusion of {exclusion}.",
                    )
                )

        def recommended(name: str) -> bool:
            alias = _normalize(re.sub(r"^(?:the|a|an)\s+", "", name, flags=re.I))
            return len(alias) >= 4 and any(
                alias in _normalize(sentence)
                and not any(marker in sentence for marker in _AVOID_SENTENCE_MARKERS)
                for sentence in recommendation_sentences
            )

        if any(
            recommended(name) for name in [*context.excluded_by_user, *context.already_recommended]
        ):
            violations.append(
                PlanningViolation(
                    code="rejected_candidate_recommended",
                    correction=(
                        "The user rejected or already received that option; recommend a "
                        "different candidate from the planning context instead."
                    ),
                )
            )
        for name in context.excluded_by_preference:
            if recommended(name):
                violations.append(
                    PlanningViolation(
                        code="excluded_candidate_recommended",
                        correction=(
                            f"{name} matches an activity the user excluded; recommend a "
                            "candidate from the planning context instead."
                        ),
                    )
                )
                break
        mode_conflicts = {
            "pve": ("pvp", "crucible", "trials", "iron banner"),
            "pvp": ("pve", "strike", "nightfall", "dungeon", "raid", "campaign"),
        }
        if preferences.activity_mode in mode_conflicts and any(
            any(term in sentence for term in mode_conflicts[preferences.activity_mode])
            for sentence in recommendation_sentences
        ):
            violations.append(
                PlanningViolation(
                    code="activity_mode_constraint",
                    correction=(
                        f"Keep the recommendation within the user's explicit "
                        f"{preferences.activity_mode.upper()} preference."
                    ),
                )
            )
        if preferences.fireteam == "solo" and any(
            any(term in sentence for term in ("fireteam", "with a group", "matchmade", "raid"))
            for sentence in recommendation_sentences
        ):
            violations.append(
                PlanningViolation(
                    code="solo_constraint",
                    correction="Keep the recommendation compatible with solo play.",
                )
            )
        if preferences.fireteam == "group" and any(
            "solo" in sentence for sentence in recommendation_sentences
        ):
            violations.append(
                PlanningViolation(
                    code="group_constraint",
                    correction="Keep the recommendation compatible with group play.",
                )
            )
        if preferences.intensity == "chill" and any(
            any(term in sentence for term in ("raid", "trials", "grandmaster", "challenging"))
            for sentence in recommendation_sentences
        ):
            violations.append(
                PlanningViolation(
                    code="chill_constraint",
                    correction="Keep the recommendation compatible with the user's chill intent.",
                )
            )
        if preferences.time_minutes is not None:
            for sentence in recommendation_sentences:
                durations = re.findall(
                    r"\b(\d{1,3})\s*[- ]?\s*(minutes?|mins?|hours?|hrs?)\b", sentence
                )
                exceeds_window = any(
                    int(amount) * (60 if unit.startswith(("hour", "hr")) else 1)
                    > preferences.time_minutes
                    for amount, unit in durations
                )
                if exceeds_window:
                    violations.append(
                        PlanningViolation(
                            code="known_time_window_exceeded",
                            correction=(
                                f"Keep the recommendation within the user's known "
                                f"{preferences.time_minutes}-minute window."
                            ),
                        )
                    )
                    break

        for content in context.named_content:
            if content.completion_status != "UNKNOWN":
                continue
            content_name = content.content.casefold()
            for sentence in re.split(r"(?<=[.!?])\s+|[\r\n]+", normalized):
                if content_name not in sentence:
                    continue
                claims_known_state = bool(
                    re.search(
                        r"\b(?:is|was|must be|you(?:'ve| have)?)\s+"
                        r"(?:already\s+)?(?:complete|completed|finished|not started|unavailable)",
                        sentence,
                    )
                )
                absence_causal_completion = (
                    "active" in sentence
                    and any(
                        value in sentence
                        for value in ("therefore", " so ", "means", "must be", "because", "since")
                    )
                    and any(
                        value in sentence for value in ("complete", "completed", "finished", "done")
                    )
                )
                if (claims_known_state or absence_causal_completion) and not any(
                    value in sentence for value in self._uncertainty_markers
                ):
                    violations.append(
                        PlanningViolation(
                            code="unknown_completion_as_known",
                            correction=(
                                f"{content.content} completion is UNKNOWN in the planning context."
                            ),
                        )
                    )
                    break

        return list({value.code: value for value in violations}.values())

    @staticmethod
    def correction_instruction(violations: list[PlanningViolation]) -> str:
        constraints = " ".join(value.correction for value in violations)
        return (
            "Revise the previous recommendation once. Preserve grounded facts and source support, "
            "but correct these factual constraints: "
            f"{constraints} {REWRITE_RULE}"
        )

    @staticmethod
    def safe_answer(context: SessionPlanningContext) -> str:
        scope = (
            f"your {context.requested_character_scope}"
            if context.requested_character_scope
            else "the selected character"
        )
        if context.alternative_guidance:
            set_aside = [*context.excluded_by_user, *context.already_recommended]
            aside = f"I've set aside {', '.join(set_aside[:3])}. " if set_aside else ""
            if context.candidates:
                first = context.candidates[0]
                progress = next(
                    (
                        f" ({value.name}: {value.progress_percent:.0f}% done)"
                        for value in first.objectives
                        if value.progress_percent is not None
                    ),
                    "",
                )
                more = ", ".join(value.name for value in context.candidates[1:3])
                return (
                    f"{aside}Within your constraints, {first.name} is available to you"
                    f"{progress}, so that's the pick I can stand behind. Your data doesn't show "
                    "how its loot compares or how long it takes, so I'm not ranking it on either."
                    + (f" Also available: {more}." if more else "")
                )
            return (
                f"{aside}I don't see another grounded option for {scope} in the account data "
                "I have. If you tell me what kind of session you want, I can look for a match."
            )
        if len(context.named_content) >= 2:
            choices = " and ".join(value.content for value in context.named_content[:2])
            return (
                f"I'd keep this comparison scoped to {scope}. I can't make a grounded choice "
                f"between {choices} yet: an active quest is only one signal, and missing active "
                "quest data does not establish completion. The relevant completion state remains "
                "unknown until explicit Guardian progression establishes it."
            )
        return (
            f"I can't make a grounded next-step recommendation for {scope} from the available "
            "evidence. Active or nearly complete objectives are only candidate signals, not enough "
            "to choose the best activity."
        )
