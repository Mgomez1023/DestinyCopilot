import re
from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field

from app.content_progression import (
    CompletionScope,
    ContentProgressionResolver,
    ContentStatus,
    EvidenceState,
)
from app.models import CharacterSummary, GuardianContext, ObjectiveSummary
from app.session_preferences import PrimaryGoal, SessionPreferenceContext

CandidateClassification = Literal[
    "meaningful_major_progression",
    "routine_or_grind_objective",
    "record_or_collectible_cleanup",
    "unknown_significance",
]

MAX_CANDIDATES = 12
MAX_OBJECTIVES_PER_CANDIDATE = 3
MAX_RECENT_ACTIVITIES = 6
MAX_PROGRESSIONS = 6


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
    objectives: list[PlanningObjective] = Field(default_factory=list, max_length=3)
    recently_repeated: bool = False
    recent_occurrences: int = Field(default=0, ge=0)
    overlap_signals: list[str] = Field(default_factory=list, max_length=3)
    external_significance_needed: bool = True


class SessionPlanningContext(BaseModel):
    version: Literal[4] = 4
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


def _objectives(values: list[ObjectiveSummary]) -> list[PlanningObjective]:
    return [
        PlanningObjective(
            name=value.name,
            completion_status="KNOWN_TRUE" if value.complete else "KNOWN_FALSE",
            progress_percent=value.progress_percent,
        )
        for value in values[:MAX_OBJECTIVES_PER_CANDIDATE]
    ]


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
        add_candidate(
            PlanningCandidate(
                name=quest.name,
                source_kinds=["active_quest"],
                classification=_classification(quest.name, "quest", requested_content_keys),
                active_quest_status="KNOWN_TRUE",
                completion_status="UNKNOWN" if is_named_content else "KNOWN_FALSE",
                available_to_character="KNOWN_TRUE",
                objectives=_objectives(quest.objectives),
                recently_repeated=recent_counts[key] > 1,
                recent_occurrences=recent_counts[key],
                external_significance_needed=True,
            )
        )

    if character:
        for milestone in character.milestones[:4]:
            key = _normalize(milestone.name)
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
    )


class PlanningAnswerValidator:
    _recommendation_markers = (
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

        forbidden_terms: dict[str, tuple[str, ...]] = {
            "pvp": ("pvp", "crucible", "trials", "iron banner"),
            "pve": ("pve",),
            "dungeons": ("dungeon",),
            "raids": ("raid",),
            "grinding": ("grind", "farm"),
            "group_play": ("fireteam", "matchmade", "with a group"),
            "solo_play": ("solo",),
        }
        recommendation_sentences = [
            sentence
            for sentence in re.split(r"(?<=[.!?])\s+|[\r\n]+", normalized)
            if any(marker in sentence for marker in self._recommendation_markers)
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
            f"{constraints} Do not mention validation, tools, providers, or internal reasoning."
        )

    @staticmethod
    def safe_answer(context: SessionPlanningContext) -> str:
        scope = (
            f"your {context.requested_character_scope}"
            if context.requested_character_scope
            else "the selected character"
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
