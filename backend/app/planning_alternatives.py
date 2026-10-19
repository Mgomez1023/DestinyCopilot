"""Rejected and already-offered recommendations within one planning exchange.

The planning context lists every grounded candidate, so without this the model sees the
rejected objective again and has to infer from chat text what "that" or "that either" meant.
This module resolves that deterministically against canonical candidates:

* A rejection ("I don't want to do X", "not that") excludes the named candidate, every
  candidate of a named activity type, or, for "that"/"either", what the previous reply offered.
* An alternatives request ("what else?") sets aside the previous reply's primary choice.
* A reversal ("actually X is fine") reinstates a rejected candidate.

Candidates are matched by their canonical name, quoted objective names and activity type, never
by hardcoded item names. Character scope, ranking and ownership are untouched; this only
removes already-declined candidates from the list the model may choose from.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.models import ChatTurn
from app.planning_intent import (
    REJECTION_PATTERN,
    REVERSAL_PATTERN,
    is_alternatives_request,
)
from app.session_planning import PlanningCandidate, SessionPlanningContext, _normalize

MIN_ALIAS_LENGTH = 4
MAX_TRACED_NAMES = 6

_ALL_OFFERED = re.compile(r"\b(?:either|those|them|these|neither|any of|none of)\b", re.IGNORECASE)
_ONE_OFFERED = re.compile(r"\b(?:that|this|it|the first|that one|this one)\b", re.IGNORECASE)
_CLAUSE = re.compile(r"(?:(?!\s[\u2014\u2013-]\s|[.!?;:\r\n]).)+", re.DOTALL)
_NEGATED_SENTENCE = (
    "non-",
    "non\u2011",
    "avoid ",
    "skip",
    "no ",
    "not ",
    "don't ",
    "do not ",
    "instead of",
    "other than",
    "besides",
    "rather than",
    "aside from",
    "apart from",
    "without",
)


def _aliases(candidate: PlanningCandidate) -> list[str]:
    values = [candidate.name]
    values.extend(
        re.findall(
            r"[\"“']([^\"”']{4,60})[\"”']",
            " ".join(objective.name for objective in candidate.objectives),
        )
    )
    aliases: list[str] = []
    for value in values:
        for part in (value, value.split(":", 1)[0]):
            normalized = _normalize(part)
            normalized = re.sub(r"^(?:the|a|an)\s+", "", normalized)
            if len(normalized) >= MIN_ALIAS_LENGTH and normalized not in aliases:
                aliases.append(normalized)
    return aliases


def _first_mention(text: str, alias: str) -> int | None:
    match = re.search(rf"\b{re.escape(alias)}s?\b", text)
    return match.start() if match else None


def _type_alias(candidate: PlanningCandidate) -> str | None:
    if not candidate.activity_type:
        return None
    normalized = _normalize(candidate.activity_type)
    return normalized if len(normalized) >= MIN_ALIAS_LENGTH else None


def _named_candidates(text: str, candidates: Sequence[PlanningCandidate]) -> list[str]:
    """Candidates the text names directly, or whose activity type it names."""

    normalized = _normalize(text)
    found: list[str] = []
    for candidate in candidates:
        by_name = any(
            _first_mention(normalized, alias) is not None for alias in _aliases(candidate)
        )
        type_alias = _type_alias(candidate)
        by_type = type_alias is not None and _first_mention(normalized, type_alias) is not None
        if (by_name or by_type) and candidate.name not in found:
            found.append(candidate.name)
    return found


def offered_candidates(reply: str, candidates: Sequence[PlanningCandidate]) -> list[str]:
    """Candidates a reply recommended, in order of appearance; negated clauses do not count."""

    positioned: list[tuple[int, str]] = []
    for clause in _CLAUSE.finditer(reply):
        folded = clause.group(0).casefold()
        if any(marker in folded for marker in _NEGATED_SENTENCE):
            continue
        normalized = _normalize(clause.group(0))
        for candidate in candidates:
            if any(_first_mention(normalized, alias) is not None for alias in _aliases(candidate)):
                positioned.append((clause.start(), candidate.name))
    ordered: list[str] = []
    for _, name in sorted(positioned, key=lambda value: value[0]):
        if name not in ordered:
            ordered.append(name)
    return ordered


@dataclass
class AlternativeState:
    rejected: list[str] = field(default_factory=list)
    reinstated: list[str] = field(default_factory=list)
    already_recommended: list[str] = field(default_factory=list)
    alternative_requested: bool = False
    rejection_detected: bool = False
    unresolved_rejection: bool = False

    @property
    def active(self) -> bool:
        return bool(
            self.rejected
            or self.already_recommended
            or self.alternative_requested
            or self.rejection_detected
        )

    def trace(self, *, candidates_remaining: int) -> dict[str, object]:
        return {
            "alternative_requested": self.alternative_requested,
            "rejection_detected": self.rejection_detected,
            "unresolved_rejection": self.unresolved_rejection,
            "rejected": self.rejected[:MAX_TRACED_NAMES],
            "reinstated": self.reinstated[:MAX_TRACED_NAMES],
            "already_recommended": self.already_recommended[:MAX_TRACED_NAMES],
            "candidates_remaining": candidates_remaining,
        }


def derive_alternatives(
    segment_history: Sequence[ChatTurn],
    message: str,
    candidates: Sequence[PlanningCandidate],
    *,
    character_switched: bool = False,
) -> AlternativeState:
    """Replay the planning segment's rejections against the current candidates."""

    state = AlternativeState()
    rejected: list[str] = []
    last_reply: str | None = None
    for turn, current in [*((value, False) for value in segment_history), (None, True)]:
        if turn is not None and turn.role == "assistant":
            last_reply = turn.content
            continue
        text = message if current else (turn.content if turn else "")
        offered = offered_candidates(last_reply, candidates) if last_reply else []
        named = _named_candidates(text, candidates)
        rejecting = bool(REJECTION_PATTERN.search(text))
        if rejecting:
            targets = list(named)
            if not targets and offered:
                if _ALL_OFFERED.search(text):
                    targets = list(offered)
                elif _ONE_OFFERED.search(text):
                    targets = offered[:1]
            for name in targets:
                if name not in rejected:
                    rejected.append(name)
            if current:
                state.rejection_detected = True
                state.unresolved_rejection = not targets and bool(
                    _ALL_OFFERED.search(text) or _ONE_OFFERED.search(text)
                )
        elif named and REVERSAL_PATTERN.search(text):
            for name in named:
                if name in rejected:
                    rejected.remove(name)
                    if current:
                        state.reinstated.append(name)
        if current:
            asked_for_another = is_alternatives_request(text)
            state.alternative_requested = asked_for_another or rejecting
            # Only a plain "what else?" sets the last primary aside; a rejection already names
            # what to exclude, and the rest of the previous reply may still be acceptable.
            if asked_for_another and not rejecting and offered and not character_switched:
                primary = offered[0]
                if primary not in rejected:
                    state.already_recommended = [primary]
    state.rejected = rejected
    return state


def apply_alternatives(
    context: SessionPlanningContext, state: AlternativeState
) -> SessionPlanningContext:
    """Remove declined candidates and tell the model what is excluded and why."""

    if not state.active:
        return context
    avoid = set(state.rejected)
    kept = [value for value in context.candidates if value.name not in avoid]
    soft = [name for name in state.already_recommended if name not in avoid]
    without_soft = [value for value in kept if value.name not in soft]
    if without_soft:
        kept = without_soft
    else:
        soft = []  # Offering it again beats offering nothing; the answer must say so.
    guidance = (
        "The user rejected suggestions or asked for another option. Recommend a different "
        "candidate from `candidates` now (recommend only what appears there), keeping every "
        "effective preference (time, character, fireteam, avoided activities). Never recommend "
        "anything in "
        "`excluded_by_user`; do not lead with `already_recommended`. Do not ask a clarifying "
        "question before answering."
    )
    if not kept:
        guidance += (
            " No grounded candidate remains: say specifically that you have no other grounded "
            "option for this character after setting aside the excluded ones, and do not "
            "invent activities."
        )
    elif not soft and state.already_recommended:
        guidance += (
            " The only remaining grounded candidate was already recommended; say so plainly."
        )
    return context.model_copy(
        update={
            "candidates": kept,
            "excluded_by_user": list(state.rejected)[:12],
            "already_recommended": soft[:3],
            "alternative_guidance": guidance,
        }
    )
