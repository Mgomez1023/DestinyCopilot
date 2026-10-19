"""Deterministic planning-intent detection and planning conversation continuity.

One shared mechanism answers three questions for a chat turn:

1. Is this message a standalone planning request, made of composable signals (an open
   "what can I do" question, a stated time budget, a "give me something" ask, a preference
   about what to play) rather than a list of whole sentences?
2. Is it a follow-up that only means something inside an active planning exchange (a
   constraint-only reply, a request for alternatives, a character switch)?
3. Which user turns belong to the current planning segment, so constraints from before an
   unrelated topic are not carried into a new plan?

No model call is involved. Character scope itself stays with `character_selection`; this
module only decides whether a turn is planning and which stated constraints apply to it.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from app.models import ChatTurn
from app.session_preferences import (
    MAX_PREFERENCE_HISTORY_TURNS,
    PreferenceField,
    SessionPreferenceContext,
    message_updates,
    replay_preferences,
)

PlanningKind = Literal["new_request", "continuation", "character_switch", "none"]
TurnKind = Literal[
    "standalone", "alternatives", "feedback", "constraint", "character_only", "other"
]

_I = r"(?:i|we)"
_TIME_UNIT = r"(?:minutes?|mins?|hours?|hrs?)"

# Whole-message patterns that predate the shared signals; kept so established phrasings
# continue to classify identically.
LEGACY_PLANNING_PATTERNS = (
    re.compile(r"\bwhat should i (?:do|play|start)(?: next| on my \w+)?\b", re.IGNORECASE),
    re.compile(r"\bi (?:have|have got|got)\b.{0,24}\b" + _TIME_UNIT + r"\b", re.IGNORECASE),
    re.compile(r"\bi (?:just )?bought\b.{0,80}\b(?:dlcs?|expansions?|campaigns?)\b", re.IGNORECASE),
    re.compile(r"\bshould i\b.{0,160}\bor\b", re.IGNORECASE),
    re.compile(
        r"\bi want\b.{0,80}\b(?:better gear|loot|rewards?|something chill|challenge|challenging)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bgive me something\b", re.IGNORECASE),
    re.compile(r"\b(?:what(?:'s| is) )?worth doing\b", re.IGNORECASE),
    re.compile(r"\brecommend\b.{0,80}\b(?:activity|quest|campaign|play|do)\b", re.IGNORECASE),
)

_PLAN_VERBS = (
    r"(?:do|play|run|try|tackle|work|farm|grind|start|finish|complete|get into|jump into|"
    r"focus|progress|make|knock out)"
)
# Composable signals: a planning request is any one of these, absent a veto.
_SIGNALS = (
    # Open question about what the player can/should do: "what can I do for an hour",
    # "what else could I do", "where should I start", "what can I work toward".
    re.compile(
        r"\b(?:what|which|anything|something|anywhere|where)\b[^.?!]{0,40}?\b"
        rf"(?:(?:can|could|should|might|may|shall)\s+{_I}|{_I}\s+(?:can|could|should|might|"
        rf"may|want to|wanna|need to))\s+(?:also\s+|even\s+)?{_PLAN_VERBS}\b",
        re.IGNORECASE,
    ),
    # A stated time budget: "I've got an hour to play", "we have 20 min".
    re.compile(
        rf"\b{_I}(?:'ve|’ve|\s+have|\s+got|\s+only have|\s+only got|\s+just have)\b.{{0,30}}?"
        rf"\b{_TIME_UNIT}\b",
        re.IGNORECASE,
    ),
    # "I want to make progress", "I'd like to knock something out".
    re.compile(
        rf"\b{_I}\s*(?:want to|wanna|would like to|'d like to|need to)\s+(?:make\s+(?:some\s+)?"
        r"progress|get\s+(?:something|some)\s+done|work\s+toward|knock\s+out|"
        r"finish\s+something|do\s+something|play\s+something)\b",
        re.IGNORECASE,
    ),
    # Asking for a pick: "give me something quick", "find me something", "pick one for me".
    re.compile(
        r"\b(?:give|show|find|pick|suggest)\s+me\s+(?:something|anything|one|a\s+\w+\s+to)\b",
        re.IGNORECASE,
    ),
    # "anything worth doing solo", "anything good to play".
    re.compile(
        r"\bany(?:thing|where)\s+(?:worth|good|fun|worthwhile|useful|productive)\b",
        re.IGNORECASE,
    ),
    # Preference about what to play: "I don't feel like playing Crucible", "I'm in the mood".
    re.compile(
        rf"\b{_I}(?:'m|’m|\s+am)?\s*(?:(?:don'?t|do not|not)\s+)?"
        r"(?:feel like|in the mood for|want to|wanna|up for)\s+"
        r"(?:playing|doing|running|trying|play|do|something|anything|some|pvp|pve|crucible|"
        r"gambit|strikes?|raids?|dungeons?)\b",
        re.IGNORECASE,
    ),
)
_INFORMATIONAL_START = re.compile(
    r"^\s*(?:what|who|why|how)\s+(?:is|are|was|were|does|did|do|long|many|much)\b|"
    r"^\s*(?:explain|define|tell me about)\b",
    re.IGNORECASE,
)
# "I spent an hour…", "yesterday", "last night": a report of the past, not a plan.
_NARRATIVE = re.compile(
    r"\b(?:yesterday|last\s+(?:night|week|weekend|season)|ago|earlier today)\b|"
    rf"\b{_I}\s+(?:spent|played|ran|did|finished|beat|cleared|completed|farmed|grinded)\b",
    re.IGNORECASE,
)
# "do X with <thing>" is a session question only when the thing is a session resource: a time
# budget, the character being played, or company. Anything else (a perk, "this", a gear or
# ability name) is a question about that object, so no item names are listed here.
_SESSION_RESOURCE = (
    rf"(?:(?:about |around |only |just )?(?:an?|one|two|three|four|\d+|half an?|a few|some|"
    rf"another)\s+(?:more\s+)?{_TIME_UNIT}"
    r"|my\s+(?:titan|hunter|warlock|time|evening|friends?|fireteam|clan)"
    r"|friends?|fireteam|an?\s+(?:group|fireteam|friend)|no time|limited time"
    r"|the\s+(?:rest|whole|evening|night|day)|(?:this\s+)?(?:evening|tonight))"
)
_OBJECT_QUESTION = re.compile(rf"\bwith\s+(?!{_SESSION_RESOURCE}\b)\S", re.IGNORECASE)
_ALTERNATIVES = re.compile(
    r"\b(?:what|anything|something|anywhere)\s+else\b|"
    r"\b(?:other|another|different|more|alternative)\s+"
    r"(?:options?|ideas?|suggestions?|choices?|activit(?:y|ies)|things?)\b|"
    r"\bany\s+(?:other|alternatives)\b",
    re.IGNORECASE,
)
# Feedback on what was suggested: declining it, or taking a declined option back.
REJECTION_PATTERN = re.compile(
    r"\b(?:don'?t|do not|didn'?t|would rather not|rather not)\b[^.?!]{0,25}"
    r"\b(?:want|feel like|wanna|like|care for|care about)\b|"
    r"\b(?:not|no)\s+(?:that|this|it|those|them)\b|\bno thanks\b|\bpass on\b|"
    r"\bskip\s+(?:that|this|it|those|them)\b|\b(?:nah|nope)\b|"
    r"\bsounds?\s+(?:boring|bad|tedious)\b|\bnot interested\b|\b(?:sick|tired) of\b|"
    r"\bdone with\b|\banything but\b",
    re.IGNORECASE,
)
REVERSAL_PATTERN = re.compile(
    r"\b(?:fine|okay|ok|actually|after all|changed my mind|let'?s|i'?ll|i will|go with|"
    r"bring back|sure)\b",
    re.IGNORECASE,
)
MAX_FOLLOWUP_WORDS = 14
MAX_SWITCH_WORDS = 10
_NON_CHARACTER_FIELDS: frozenset[PreferenceField] = frozenset(
    {
        "time_minutes",
        "primary_goal",
        "fireteam",
        "activity_mode",
        "intensity",
        "duration_preference",
        "exclusions",
        "all",
    }
)
_TRACKED_FIELDS = (
    "time_minutes",
    "primary_goal",
    "fireteam",
    "activity_mode",
    "intensity",
    "duration_preference",
    "exclusions",
)


def _blocked(message: str) -> bool:
    return bool(
        _INFORMATIONAL_START.search(message)
        or _NARRATIVE.search(message)
        or _OBJECT_QUESTION.search(message)
    )


def is_standalone_planning(message: str) -> bool:
    """True when the message alone asks for a plan, with no conversation context needed."""

    if any(pattern.search(message) for pattern in LEGACY_PLANNING_PATTERNS):
        return not (_NARRATIVE.search(message) or _OBJECT_QUESTION.search(message))
    if _blocked(message):
        return False
    return any(pattern.search(message) for pattern in _SIGNALS)


def is_alternatives_request(message: str) -> bool:
    return bool(_ALTERNATIVES.search(message)) and not _INFORMATIONAL_START.search(message)


def _turn_kind(message: str) -> TurnKind:
    if is_standalone_planning(message):
        return "standalone"
    if _INFORMATIONAL_START.search(message) or _NARRATIVE.search(message):
        return "other"
    words = len(message.split())
    if is_alternatives_request(message) and words <= MAX_FOLLOWUP_WORDS:
        return "alternatives"
    updates = message_updates(message)
    if updates & _NON_CHARACTER_FIELDS and words <= MAX_FOLLOWUP_WORDS:
        return "constraint"
    if updates == {"character"} and words <= MAX_SWITCH_WORDS:
        return "character_only"
    if words <= MAX_FOLLOWUP_WORDS and (
        REJECTION_PATTERN.search(message) or REVERSAL_PATTERN.search(message)
    ):
        return "feedback"
    return "other"


@dataclass(frozen=True)
class PlanningIntent:
    is_planning: bool
    kind: PlanningKind
    signal: TurnKind
    preferences: SessionPreferenceContext
    current_updates: list[PreferenceField] = field(default_factory=list)
    inherited: list[str] = field(default_factory=list)
    replaced: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    cleared: list[str] = field(default_factory=list)
    stale_constraints_dropped: bool = False
    # Turns (both roles) of the current planning segment, oldest first, excluding this message.
    segment_history: list[ChatTurn] = field(default_factory=list)

    def trace(self, *, context_rebuilt: bool) -> dict[str, object]:
        return {
            "classification": self.signal,
            "kind": self.kind,
            "is_planning": self.is_planning,
            "constraints": {
                "inherited": self.inherited,
                "replaced": self.replaced,
                "added": self.added,
                "cleared": self.cleared,
            },
            "stale_constraints_dropped": self.stale_constraints_dropped,
            "context_rebuilt": context_rebuilt,
            "character_switch_continuation": self.kind == "character_switch",
        }


def _value(preferences: SessionPreferenceContext, name: str) -> object:
    value = getattr(preferences, name)
    return None if value in ("unspecified", [], None) else value


def classify_planning_intent(
    history: Sequence[ChatTurn],
    message: str,
    *,
    character_switched: bool,
    character: str = "unspecified",
) -> PlanningIntent:
    """Classify this turn and rebuild the constraints that belong to its planning segment.

    `character_switched` is the P0 conversation-scope verdict that the user explicitly moved
    to another character this turn; `character` is that scope's stated character, which is
    kept on the returned preferences untouched.
    """

    window = list(history[-MAX_PREFERENCE_HISTORY_TURNS:])
    users = [turn.content for turn in window if turn.role == "user"]
    state: Literal["idle", "pending", "active"] = "idle"
    segment_start = 0
    for index, text in enumerate(users):
        kind = _turn_kind(text)
        if kind == "standalone":
            if state == "idle":
                segment_start = index
            state = "active"  # constraints stated just before the ask join this plan
        elif kind == "feedback":
            continue  # reacting to a suggestion neither starts nor ends a planning exchange
        elif kind in ("alternatives", "constraint"):
            if state == "idle":
                segment_start, state = index, "pending"
        elif kind == "character_only":
            continue  # a bare switch neither starts nor ends a planning exchange
        else:
            state = "idle"

    signal = _turn_kind(message)
    kind: PlanningKind = "none"
    if signal == "standalone":
        kind = "continuation" if state == "active" else "new_request"
    elif (
        signal == "feedback"
        and state == "active"
        or signal in ("alternatives", "constraint")
        and state in ("active", "pending")
    ):
        kind = "continuation"
    elif signal == "character_only" and state == "active" and character_switched:
        kind = "character_switch"

    is_planning = kind != "none"
    prior_turns = users[segment_start:] if state != "idle" else []
    segment_history: list[ChatTurn] = []
    if state != "idle":
        seen = 0
        for position, turn in enumerate(window):
            if turn.role == "user":
                if seen == segment_start:
                    segment_history = window[position:]
                    break
                seen += 1
    before = replay_preferences(prior_turns)
    after = replay_preferences([*prior_turns, message])
    # Character is P0-owned: keep exactly what conversation scope derived.
    after.character = character  # type: ignore[assignment]
    updates = message_updates(message)

    inherited: list[str] = []
    replaced: list[str] = []
    added: list[str] = []
    cleared: list[str] = []
    for name in _TRACKED_FIELDS if is_planning else ():
        old, new = _value(before, name), _value(after, name)
        if old is None and new is None:
            continue
        if old is None:
            added.append(name)
        elif new is None:
            cleared.append(name)
        elif old != new:
            replaced.append(name)
        else:
            inherited.append(name)
    everything = replay_preferences(users)
    dropped = any(
        _value(everything, name) is not None and _value(before, name) is None
        for name in _TRACKED_FIELDS
    )
    return PlanningIntent(
        is_planning=is_planning,
        kind=kind,
        signal=signal,
        preferences=after,
        current_updates=sorted(updates),
        inherited=inherited,
        replaced=replaced,
        added=added,
        cleared=cleared,
        stale_constraints_dropped=is_planning and dropped,
        segment_history=segment_history if is_planning else [],
    )
