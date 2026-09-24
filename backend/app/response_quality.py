import re
from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field

from app.build_analysis import is_explicit_build_request
from app.session_planning import (
    PlanningAnswerValidator,
    PlanningViolation,
    SessionPlanningContext,
)

ResponseMode = Literal[
    "direct_fact",
    "recommendation",
    "comparison",
    "walkthrough",
    "build_advice",
    "account_summary",
    "troubleshooting",
]


class ResponseModeContext(BaseModel):
    mode: ResponseMode
    target_min_words: int | None
    target_max_words: int | None
    detail_level: Literal["concise", "moderate", "detailed", "as_needed"]
    source_limit: int
    detailed_requested: bool = False
    timed_itinerary_requested: bool = False
    continuation: bool = False
    focused_build: bool = False


class BuildResponseValidationContext(BaseModel):
    is_build_request: bool = False
    requires_current_build_analysis: bool = False
    requires_owned_inventory: bool = False
    current_build_analysis_used: bool = False
    guardian_build_data_used: bool = False
    inventory_ownership_checked: bool = False
    preserve_equipped_exotics: bool = False
    owned_item_names: list[str] = Field(default_factory=list)
    locked_item_names: list[str] = Field(default_factory=list)
    equipped_exotic_weapons: list[str] = Field(default_factory=list)
    equipped_exotic_armor: list[str] = Field(default_factory=list)
    owned_exotic_weapons: list[str] = Field(default_factory=list)
    owned_exotic_armor: list[str] = Field(default_factory=list)
    current_external_grounding: bool = False
    activity_mode: str = "unspecified"
    fireteam: str = "unspecified"
    normal_change_limit: int | None = 3


class ReadOnlyActionValidationContext(BaseModel):
    intent_resolved: bool = False
    lookup_required: bool = False


_DETAILED = re.compile(
    r"\b(?:detailed|in[- ]depth|comprehensive|thorough|step[- ]by[- ]step)\b",
    re.IGNORECASE,
)
_TIMED_PLAN = re.compile(
    r"\b(?:minute[- ]by[- ]minute|timed plan|timeboxed plan|schedule|itinerary)\b|"
    r"\bstep[- ]by[- ]step\b.{0,40}\b(?:minutes?|hours?)\b|"
    r"\b(?:minutes?|hours?)\b.{0,40}\bstep[- ]by[- ]step\b",
    re.IGNORECASE,
)
_COMPARISON = re.compile(
    r"\b(?:compare|versus|vs\.?)\b|\b(?:should|would) i\b.{0,180}\bor\b|"
    r"\bgiven\b.{0,180}\bor\b|\bwhich\b.{0,100}\b(?:better|next)\b",
    re.IGNORECASE,
)
_TROUBLESHOOTING = re.compile(
    r"\b(?:error|bug|broken|not working|won't work|can'?t connect|cannot connect|"
    r"troubleshoot|login issue|oauth issue|failed to)\b",
    re.IGNORECASE,
)
_WALKTHROUGH = re.compile(
    r"\b(?:walkthrough|guide|step[- ]by[- ]step|guide me through|how (?:do|can) i|"
    r"how to|quest steps?|encounter mechanics?)\b",
    re.IGNORECASE,
)
_DIRECT_ACCOUNT = re.compile(
    r"\b(?:equipped|currently using|subclass am i using|do i own|what do i own|"
    r"what subclass.{0,40}using|have i (?:started|completed|finished)|how far am i|"
    r"what quests? (?:are|is) active)\b",
    re.IGNORECASE,
)
_ACCOUNT_SUMMARY = re.compile(
    r"\b(?:summarize|summary|overview)\b.{0,40}\b(?:account|guardian|character|inventory)\b|"
    r"\bhow (?:is|does) my (?:account|guardian|character)\b|"
    r"\bwhat (?:quests?|activities) (?:do i have|are available)\b",
    re.IGNORECASE,
)
_EXTERNAL_KNOWLEDGE_REQUEST = re.compile(
    r"\b(?:how (?:do|can) i get|where (?:do|can) i get|what does\b|"
    r"unlocks?|rewards?|perks?|traits?|patch(?:es| notes)?|meta|drop source|"
    r"is (?:it|that|this) good|better|upgrade|replace)\b",
    re.IGNORECASE,
)


def classify_response_mode(
    message: str, *, session_planning: bool, build_request: bool | None = None
) -> ResponseMode:
    if _TROUBLESHOOTING.search(message):
        return "troubleshooting"
    if build_request is True or (build_request is None and is_explicit_build_request(message)):
        return "build_advice"
    if _COMPARISON.search(message):
        return "comparison"
    if _WALKTHROUGH.search(message):
        return "walkthrough"
    if _DIRECT_ACCOUNT.search(message):
        return "direct_fact"
    if _ACCOUNT_SUMMARY.search(message):
        return "account_summary"
    if session_planning:
        return "recommendation"
    return "direct_fact"


def is_account_fact_only(message: str) -> bool:
    """Return true only for personal facts fully answerable by Guardian tools."""

    return bool(
        (_DIRECT_ACCOUNT.search(message) or _ACCOUNT_SUMMARY.search(message))
        and not _EXTERNAL_KNOWLEDGE_REQUEST.search(message)
    )


def response_mode_context(
    message: str,
    *,
    session_planning: bool,
    continuation: bool,
    build_request: bool | None = None,
    focused_build: bool = False,
) -> ResponseModeContext:
    mode = classify_response_mode(
        message,
        session_planning=session_planning,
        build_request=build_request,
    )
    detailed = bool(_DETAILED.search(message))
    timed = bool(_TIMED_PLAN.search(message))
    targets: dict[ResponseMode, tuple[int | None, int | None, str, int]] = {
        "direct_fact": (20, 80, "concise", 4),
        "account_summary": (40, 120, "concise", 4),
        "recommendation": (80, 180, "moderate", 4),
        "comparison": (80, 180, "moderate", 4),
        "build_advice": (100, 220, "detailed", 6),
        "walkthrough": (None, None, "as_needed", 8),
        "troubleshooting": (None, None, "as_needed", 8),
    }
    minimum, maximum, detail_level, source_limit = targets[mode]
    if mode == "build_advice" and focused_build and not detailed:
        minimum, maximum, detail_level, source_limit = (50, 120, "concise", 4)
    if detailed:
        detail_level = "detailed"
        source_limit = 8
    return ResponseModeContext(
        mode=mode,
        target_min_words=minimum,
        target_max_words=maximum,
        detail_level=detail_level,  # type: ignore[arg-type]
        source_limit=source_limit,
        detailed_requested=detailed,
        timed_itinerary_requested=timed,
        continuation=continuation,
        focused_build=bool(mode == "build_advice" and focused_build and not detailed),
    )


def response_mode_instruction(context: ResponseModeContext) -> str:
    if context.target_min_words is None:
        range_text = "Use enough detail to solve the request; no fixed word target applies."
    else:
        range_text = (
            f"A typical useful answer is about {context.target_min_words}-"
            f"{context.target_max_words} words, but this is guidance, not a truncation rule."
        )
    shapes = {
        "direct_fact": (
            "Answer in the first sentence. Include only the requested fact and, if useful, one "
            "brief clarifier or follow-up. Avoid headings and unrelated account metadata."
        ),
        "account_summary": (
            "Lead with the useful account summary, then include only the few facts relevant to "
            "the request. Avoid raw metadata and exhaustive dumps."
        ),
        "recommendation": (
            "Lead with one clear primary route, give a few concrete actions and a brief reason, "
            "then optionally one compatible backup and one useful follow-up. Moderate actionable "
            "detail is welcome; do not compress it into only two or three sentences."
        ),
        "comparison": (
            "Lead with the grounded choice or conditional answer, compare both sides briefly, "
            "state important uncertainty naturally, and optionally give one next step."
        ),
        "build_advice": (
            "Lead with the item choice and one or two grounded reasons; keep the answer focused on "
            "the requested gear decision."
            if context.focused_build
            else "Lead with the main build change, then explain the relevant gear, subclass, "
            "stats, and tradeoffs. Headings or bullets are useful when they improve scanning."
        ),
        "walkthrough": (
            "Give ordered, actionable steps. Headings and bullets are appropriate, and the answer "
            "may be longer when the task genuinely needs it."
        ),
        "troubleshooting": (
            "Lead with the likely fix or first diagnostic, then give the minimum ordered checks "
            "needed to resolve the issue."
        ),
    }
    continuation = (
        "This is a conversational follow-up: adapt the prior route directly instead of restarting "
        "with a mechanical Goal/Primary plan template. "
        if context.continuation
        else ""
    )
    timed = (
        "The user explicitly requested a timed plan, so a grounded timed structure is allowed."
        if context.timed_itinerary_requested
        else (
            "Do not create an unrequested minute-by-minute itinerary. A stated time window is a "
            "fit constraint, not permission to invent prep/activity/cleanup durations."
        )
    )
    return (
        f"Response mode: {context.mode}. {range_text} {shapes[context.mode]} "
        f"{continuation}{timed} Answer first; do not open with 'Based on the available data,' "
        "'Primary recommendation,' 'Short answer,' or generic throat-clearing. Preserve useful "
        "uncertainty, steps, and citations even when that exceeds the typical range."
    )


class ResponseQualityValidator:
    _BAD_OPENERS = (
        "based on the available data",
        "based on the data",
        "primary recommendation",
        "short answer",
        "there are several things to consider",
    )
    _INTERNAL_JARGON = re.compile(
        r"\b(?:guardian|manifest|guide|live) tools?\b|\bapi (?:returned|response|says)\b|"
        r"\b(?:provider terminology|function_call|evidence_state|known_true|known_false|"
        r"not_returned|completion_state|profile_completion_state|active_quest_status|"
        r"planning context|resolver)\b",
        re.IGNORECASE,
    )
    _DEBUG_REQUEST = re.compile(r"\b(?:debug|api|tool|provider|trace)\b", re.IGNORECASE)
    _TIME_SEGMENT = re.compile(r"\b\d{1,3}\s*[- ]?\s*(?:minutes?|mins?|hours?|hrs?)\b", re.I)
    _READ_ONLY_PERMISSION_QUESTION = re.compile(
        r"\b(?:do you want me to|would you like me to|should i|shall i|can i|"
        r"is it (?:ok|okay)|okay if i|ok if i)\b[^?]{0,180}\b"
        r"(?:check|scan|fetch|retrieve|look up|inspect|review|analy[sz]e|compare)\b",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        self.planning = PlanningAnswerValidator()

    def validate(
        self,
        answer: str,
        mode: ResponseModeContext,
        user_message: str,
        planning_context: SessionPlanningContext | None,
        build_context: BuildResponseValidationContext | None = None,
        read_only_context: ReadOnlyActionValidationContext | None = None,
    ) -> list[PlanningViolation]:
        violations = (
            self.planning.validate(answer, planning_context) if planning_context is not None else []
        )
        normalized = answer.casefold().strip()
        if answer.count("?") > 1 and not any(
            value.code == "too_many_followups" for value in violations
        ):
            violations.append(
                PlanningViolation(
                    code="too_many_followups",
                    correction="Ask at most one targeted follow-up question.",
                )
            )
        if any(normalized.startswith(value) for value in self._BAD_OPENERS):
            violations.append(
                PlanningViolation(
                    code="answer_not_first",
                    correction="Put the useful answer or recommendation in the first sentence.",
                )
            )
        if not self._DEBUG_REQUEST.search(user_message) and self._INTERNAL_JARGON.search(answer):
            violations.append(
                PlanningViolation(
                    code="internal_jargon",
                    correction=(
                        "Remove API, tool, provider, and internal evidence-state terminology; "
                        "express uncertainty naturally."
                    ),
                )
            )
        time_segments = self._TIME_SEGMENT.findall(answer)
        schedule_language = bool(
            re.search(
                r"\b(?:prep|cleanup|then|first|next|finish with|schedule|itinerary)\b",
                answer,
                re.I,
            )
        )
        if not mode.timed_itinerary_requested and (
            len(time_segments) >= 3 or (len(time_segments) >= 2 and schedule_language)
        ):
            violations.append(
                PlanningViolation(
                    code="unrequested_timed_itinerary",
                    correction=(
                        "Use the known session length as a fit constraint; remove the unrequested "
                        "minute-by-minute schedule and any invented segment durations."
                    ),
                )
            )
        extreme_max = {
            "direct_fact": 160,
            "account_summary": 220,
            "recommendation": 320,
            "comparison": 320,
            "build_advice": 380,
            "walkthrough": 700,
            "troubleshooting": 700,
        }[mode.mode]
        if mode.focused_build:
            extreme_max = 220
        word_count = len(re.findall(r"\b[\w'-]+\b", answer))
        if not mode.detailed_requested and word_count > extreme_max:
            violations.append(
                PlanningViolation(
                    code="extreme_verbosity",
                    correction=(
                        f"Remove redundant material while preserving grounded detail; this "
                        f"{mode.mode} answer is far beyond its useful detail range."
                    ),
                )
            )
        sentences = [
            re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
            for value in re.split(r"(?<=[.!?])\s+|[\r\n]+", answer)
        ]
        substantive = [value for value in sentences if len(value.split()) >= 6]
        if len(substantive) != len(set(substantive)):
            violations.append(
                PlanningViolation(
                    code="repeated_content",
                    correction="Remove the repeated recommendation or reason.",
                )
            )
        heading_count = sum(
            bool(re.match(r"^\s*(?:#{1,3}\s+|\*\*[^*]+\*\*|[A-Z][^.!?]{1,30}:)\s*", line))
            for line in answer.splitlines()
            if line.strip()
        )
        compact_modes = {"direct_fact", "account_summary", "recommendation", "comparison"}
        if mode.mode in compact_modes and heading_count >= 4:
            violations.append(
                PlanningViolation(
                    code="mechanical_structure",
                    correction=(
                        "Replace repetitive sections with a natural paragraph or short list."
                    ),
                )
            )
        if mode.continuation and re.match(
            r"\s*(?:goal|primary plan|primary recommendation)\s*:", answer, re.I
        ):
            violations.append(
                PlanningViolation(
                    code="followup_restarts_plan",
                    correction=(
                        "Adapt the previous recommendation directly instead of restarting it."
                    ),
                )
            )
        if build_context is not None and build_context.is_build_request:
            violations.extend(
                self._validate_build_answer(answer, mode, user_message, build_context)
            )
        if (
            read_only_context is not None
            and read_only_context.intent_resolved
            and read_only_context.lookup_required
            and self._READ_ONLY_PERMISSION_QUESTION.search(answer)
        ):
            violations.append(
                PlanningViolation(
                    code="unnecessary_read_only_permission",
                    correction=(
                        "The user already requested this read-only lookup. Do not ask permission "
                        "or confirm a result format; use the retrieved facts and answer directly."
                    ),
                )
            )
        return list({value.code: value for value in violations}.values())

    @staticmethod
    def _validate_build_answer(
        answer: str,
        mode: ResponseModeContext,
        user_message: str,
        context: BuildResponseValidationContext,
    ) -> list[PlanningViolation]:
        normalized = answer.casefold()
        violations: list[PlanningViolation] = []
        missing_evidence_acknowledged = bool(
            re.search(
                r"\b(?:can(?:not|'t)|could(?: not|n'?t)|unable|need|would need)\b"
                r"[^.!?]{0,120}\b(?:verify|inspect|check|review|see)\b",
                answer,
                re.IGNORECASE,
            )
        )
        if (
            context.requires_current_build_analysis
            and not context.current_build_analysis_used
            and not missing_evidence_acknowledged
        ):
            violations.append(
                PlanningViolation(
                    code="missing_current_build_analysis",
                    correction=(
                        "Do not judge or recommend changes to the current setup because its "
                        "equipped build was not verified. State briefly that personalized advice "
                        "is unavailable until the current setup is checked."
                    ),
                )
            )
        if (
            context.requires_owned_inventory
            and not context.inventory_ownership_checked
            and not missing_evidence_acknowledged
        ):
            violations.append(
                PlanningViolation(
                    code="missing_owned_inventory_evidence",
                    correction=(
                        "Do not name or recommend an owned or vault option because no owned "
                        "candidates or instance perks were verified. State briefly that a "
                        "personalized choice is unavailable until owned gear is checked."
                    ),
                )
            )
        if re.search(
            r"\b(?:sign in(?: again)?|log in(?: again)?|connect|reconnect|link)\b"
            r"[^.!?]{0,50}\b(?:bungie|account|platform)\b|"
            r"\bprovide\b[^.!?]{0,40}\b(?:platform|account)\b",
            answer,
            re.IGNORECASE,
        ):
            violations.append(
                PlanningViolation(
                    code="false_authentication_guidance",
                    correction=(
                        "Guardian context is already authenticated. Remove sign-in, account-link, "
                        "and platform guidance; describe only the specific build or character "
                        "lookup limitation."
                    ),
                )
            )
        ownership_claim = re.search(
            r"\b(?:you|your (?:guardian|titan|hunter|warlock))\s+"
            r"(?:already\s+)?(?:own|have|possess)\b",
            answer,
            re.IGNORECASE,
        )
        if ownership_claim and not context.inventory_ownership_checked:
            violations.append(
                PlanningViolation(
                    code="unsupported_ownership_claim",
                    correction=(
                        "Remove ownership claims that are not backed by the owned-inventory "
                        "results. Separate verified owned options from aspirational items."
                    ),
                )
            )

        for item in context.locked_item_names:
            escaped = re.escape(item)
            if re.search(
                rf"\b(?:replace|remove|drop|swap out|change)\b[^.!?]{{0,60}}\b{escaped}\b|"
                rf"\b{escaped}\b[^.!?]{{0,60}}\b(?:replace|remove|drop|swap out|change)\b",
                answer,
                re.IGNORECASE,
            ):
                violations.append(
                    PlanningViolation(
                        code="locked_item_replaced",
                        correction=f"Keep {item}; the user explicitly locked that item.",
                    )
                )

        current_claim = re.search(
            r"\b(?:meta|best dps|currently strongest|current(?:ly)? best|"
            r"nerfed|buffed|current artifact|this season(?:'s)? best)\b",
            answer,
            re.IGNORECASE,
        )
        if current_claim and not context.current_external_grounding:
            violations.append(
                PlanningViolation(
                    code="ungrounded_current_build_claim",
                    correction=(
                        "Remove current-meta, patch, artifact, buff, or nerf claims unless Live "
                        "or web research grounded them."
                    ),
                )
            )

        if re.search(
            r"\bbuild score\s*:?\s*\d{1,3}(?:\s*/\s*100|\s*%)?|"
            r"\b[abcds][+-]?\s*[- ]?tier\b",
            answer,
            re.IGNORECASE,
        ):
            violations.append(
                PlanningViolation(
                    code="fake_build_score",
                    correction=(
                        "Remove universal numeric scores and tier ratings; discuss only "
                        "goal-specific dimensions supported by evidence."
                    ),
                )
            )

        if re.search(
            r"\b(?:i(?:'ve| have)?|we(?:'ve| have)?)\s+"
            r"(?:equipped|switched|changed|applied|infused|dismantled|spent|claimed)\b",
            answer,
            re.IGNORECASE,
        ):
            violations.append(
                PlanningViolation(
                    code="implied_bungie_write",
                    correction=(
                        "Use recommendation language only; do not imply that any in-game change "
                        "was performed."
                    ),
                )
            )

        if not re.search(r"\b(?:debug|identifier|id)\b", user_message, re.IGNORECASE) and (
            re.search(r"\b(?:character|item|instance)[ _-]?id\b", answer, re.IGNORECASE)
            or re.search(r"\b\d{16,}\b", answer)
        ):
            violations.append(
                PlanningViolation(
                    code="raw_build_identifier",
                    correction="Remove raw character, item, instance, and hash identifiers.",
                )
            )

        if (
            re.search(
                r"\b\d{1,3}\s+(?:resilience|recovery|mobility|discipline|intellect|strength)\b"
                r"[^.!?]{0,50}\b(?:bad|good|must|need|target|threshold|too low|enough)\b",
                answer,
                re.IGNORECASE,
            )
            and not context.current_external_grounding
        ):
            violations.append(
                PlanningViolation(
                    code="unsupported_stat_threshold",
                    correction=(
                        "Describe the observed stat distribution without asserting an exact "
                        "threshold unless current mechanics were grounded externally."
                    ),
                )
            )

        mentioned_owned = sum(
            item.casefold() in normalized for item in set(context.owned_item_names)
        )
        inventory_lines = sum(
            bool(re.match(r"\s*(?:[-*]|\d+[.)])\s+", line)) for line in answer.splitlines()
        )
        if not mode.detailed_requested and (mentioned_owned > 8 or inventory_lines > 12):
            violations.append(
                PlanningViolation(
                    code="inventory_dump",
                    correction=(
                        "Return only a bounded shortlist and the highest-impact changes, not a "
                        "large inventory dump."
                    ),
                )
            )

        if context.normal_change_limit is not None and not mode.detailed_requested:
            change_lines = [
                line
                for line in answer.splitlines()
                if re.match(r"\s*(?:[-*]|\d+[.)])\s+", line)
                and re.search(
                    r"\b(?:swap|replace|change|equip|switch|remove|add|use instead)\b",
                    line,
                    re.IGNORECASE,
                )
            ]
            if len(change_lines) > context.normal_change_limit:
                violations.append(
                    PlanningViolation(
                        code="too_many_build_changes",
                        correction=(
                            f"Prioritize at most {context.normal_change_limit} high-impact changes "
                            "unless the user explicitly requested a complete detailed rebuild."
                        ),
                    )
                )

        exotic_groups = (
            (
                context.equipped_exotic_weapons,
                context.owned_exotic_weapons,
                "Exotic weapon",
            ),
            (
                context.equipped_exotic_armor,
                context.owned_exotic_armor,
                "Exotic armor piece",
            ),
        )
        locked_normalized = {value.casefold() for value in context.locked_item_names}
        for equipped, owned, label in exotic_groups:
            preserved = [value for value in equipped if value.casefold() in locked_normalized]
            if not preserved:
                continue
            for candidate in owned:
                if candidate.casefold() in {value.casefold() for value in preserved}:
                    continue
                if candidate.casefold() in normalized and re.search(
                    rf"\b(?:equip|run|use|swap (?:to|in))\b[^.!?]{{0,60}}"
                    rf"\b{re.escape(candidate)}\b",
                    answer,
                    re.IGNORECASE,
                ):
                    violations.append(
                        PlanningViolation(
                            code="incompatible_exotic_recommendation",
                            correction=(
                                f"Do not recommend a second {label} while preserving the "
                                f"equipped one ({', '.join(preserved)})."
                            ),
                        )
                    )

        constraint_counts = Counter()
        if context.activity_mode == "pve" and re.search(
            r"\b(?:for|in|play|recommend)\s+(?:the\s+)?(?:pvp|crucible)\b",
            answer,
            re.IGNORECASE,
        ):
            constraint_counts["activity"] += 1
        if context.activity_mode == "pvp" and re.search(
            r"\b(?:for|in|play|recommend)\s+(?:the\s+)?(?:pve|campaign|dungeon)\b",
            answer,
            re.IGNORECASE,
        ):
            constraint_counts["activity"] += 1
        if context.fireteam == "solo" and re.search(
            r"\b(?:requires? a fireteam|group[- ]only|for group support)\b",
            answer,
            re.IGNORECASE,
        ):
            constraint_counts["fireteam"] += 1
        if constraint_counts:
            violations.append(
                PlanningViolation(
                    code="build_constraint_conflict",
                    correction=(
                        "Keep the advice consistent with the explicit PvE/PvP and solo/group "
                        "constraints."
                    ),
                )
            )
        return violations

    @staticmethod
    def safe_build_answer(context: BuildResponseValidationContext) -> str:
        """Return a build-specific evidence fallback without activity-planning copy."""

        if context.requires_owned_inventory:
            return (
                "I couldn't retrieve a grounded shortlist of your owned gear and its returned "
                "perk data, so I don't want to guess."
            )
        if context.requires_current_build_analysis:
            return (
                "I couldn't inspect the required build data, so I can't give a grounded "
                "personalized loadout recommendation yet."
            )
        return (
            "I couldn't retrieve the required build evidence, so I can't give a grounded "
            "personalized recommendation yet."
        )

    @staticmethod
    def correction_instruction(violations: list[PlanningViolation]) -> str:
        return PlanningAnswerValidator.correction_instruction(violations)
