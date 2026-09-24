import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from app.models import ChatTurn

CharacterPreference = Literal["Titan", "Hunter", "Warlock", "unspecified"]
PrimaryGoal = Literal[
    "story_progression",
    "gear_rewards",
    "quest_completion",
    "build_improvement",
    "casual_chill",
    "challenge",
    "exploration",
    "unspecified",
]
FireteamPreference = Literal["solo", "group", "either", "unspecified"]
ActivityModePreference = Literal["pve", "pvp", "either", "unspecified"]
IntensityPreference = Literal["chill", "normal", "challenging", "unspecified"]
DurationPreference = Literal["shorter", "short", "long", "unspecified"]
SessionExclusion = Literal[
    "pvp",
    "pve",
    "dungeons",
    "raids",
    "grinding",
    "group_play",
    "solo_play",
]
PreferenceField = Literal[
    "character",
    "time_minutes",
    "primary_goal",
    "fireteam",
    "activity_mode",
    "intensity",
    "duration_preference",
    "exclusions",
    "all",
]

MAX_SESSION_MINUTES = 24 * 60
MAX_EXCLUSIONS = 7
MAX_PREFERENCE_HISTORY_TURNS = 12
_FIELD_ORDER: tuple[PreferenceField, ...] = (
    "all",
    "character",
    "time_minutes",
    "primary_goal",
    "fireteam",
    "activity_mode",
    "intensity",
    "duration_preference",
    "exclusions",
)


class SessionPreferenceContext(BaseModel):
    """Explicit, bounded preferences reconstructed from the current chat window only."""

    version: Literal[1] = 1
    character: CharacterPreference = "unspecified"
    time_minutes: int | None = Field(default=None, ge=1, le=MAX_SESSION_MINUTES)
    primary_goal: PrimaryGoal = "unspecified"
    fireteam: FireteamPreference = "unspecified"
    activity_mode: ActivityModePreference = "unspecified"
    intensity: IntensityPreference = "unspecified"
    duration_preference: DurationPreference = "unspecified"
    exclusions: list[SessionExclusion] = Field(default_factory=list, max_length=MAX_EXCLUSIONS)

    def has_constraints(self) -> bool:
        return any(
            (
                self.character != "unspecified",
                self.time_minutes is not None,
                self.primary_goal != "unspecified",
                self.fireteam != "unspecified",
                self.activity_mode != "unspecified",
                self.intensity != "unspecified",
                self.duration_preference != "unspecified",
                bool(self.exclusions),
            )
        )

    def active_constraint_count(self) -> int:
        return sum(
            (
                self.character != "unspecified",
                self.time_minutes is not None,
                self.primary_goal != "unspecified",
                self.fireteam != "unspecified",
                self.activity_mode != "unspecified",
                self.intensity != "unspecified",
                self.duration_preference != "unspecified",
                bool(self.exclusions),
            )
        )


class SessionPreferenceDerivation(BaseModel):
    preferences: SessionPreferenceContext
    current_turn_updates: list[PreferenceField] = Field(default_factory=list, max_length=9)
    prior_context_present: bool = False

    @property
    def is_planning_followup(self) -> bool:
        if not self.current_turn_updates:
            return False
        non_character_updates = {
            "all",
            "time_minutes",
            "primary_goal",
            "fireteam",
            "activity_mode",
            "intensity",
            "duration_preference",
            "exclusions",
        }
        if any(value in non_character_updates for value in self.current_turn_updates):
            return True
        return self.prior_context_present


_FULL_RESET = re.compile(
    r"\b(?:start over|ignore what i (?:said|asked) before|forget everything(?: i said)?)\b",
    re.IGNORECASE,
)
_CHARACTER_RESET = re.compile(
    r"\b(?:any|either) character(?: is fine| works?)?|\bi don'?t care which character\b",
    re.IGNORECASE,
)
_TIME_RESET = re.compile(
    r"\b(?:forget|ignore|drop|remove) (?:the )?(?:time limit|time constraint)|"
    r"\bno time limit\b",
    re.IGNORECASE,
)
_TIME = re.compile(
    r"\b(?P<amount>half(?: an)?|another|\d{1,4}|an?|one|two|three|four)\s*[- ]?"
    r"(?P<unit>minutes?|mins?|hours?|hrs?)\b",
    re.IGNORECASE,
)
_CLASS = re.compile(r"\b(titan|hunter|warlock)\b", re.IGNORECASE)

_GOAL_PATTERNS: dict[PrimaryGoal, tuple[re.Pattern[str], ...]] = {
    "story_progression": (
        re.compile(
            r"\b(?:want|wanna|would like|work on|focus on|what about)\b.{0,24}\bstory\b", re.I
        ),
        re.compile(r"^\s*story(?:\s*(?:please|instead))?[.!?]?\s*$", re.I),
    ),
    "gear_rewards": (
        re.compile(
            r"\b(?:want|need|focus on|what about)\b.{0,24}\b(?:better gear|gear|loot|rewards?)\b",
            re.I,
        ),
        re.compile(r"\b\d+\s*[- ]?minute\s+(?:gear|loot)\s+run\b", re.I),
        re.compile(r"^\s*(?:gear|loot|rewards?)(?:\s*(?:please|instead))?[.!?]?\s*$", re.I),
    ),
    "quest_completion": (
        re.compile(r"\b(?:want|need|work on|focus on)\b.{0,24}\b(?:quests?|objectives?)\b", re.I),
        re.compile(r"^\s*(?:quests?|quest completion)(?:\s*(?:please|instead))?[.!?]?\s*$", re.I),
    ),
    "build_improvement": (
        re.compile(
            r"\b(?:improve|fix|work on|focus on)\b.{0,24}\b(?:build|loadout|subclass)\b", re.I
        ),
        re.compile(
            r"\b(?:build around|boss dps|add clear|prep(?:are)? me for|"
            r"what should i run|do i own a better)\b",
            re.I,
        ),
        re.compile(r"^\s*(?:build|build improvement)(?:\s*(?:please|instead))?[.!?]?\s*$", re.I),
    ),
    "casual_chill": (
        re.compile(
            r"\b(?:something|keep it|make it|want|need|feeling)\b.{0,20}\b"
            r"(?:chill|casual|relaxing|low[- ]stress)\b",
            re.I,
        ),
        re.compile(
            r"^\s*(?:nah\s+)?(?:something\s+)?(?:chill|casual)(?:\s+instead)?[.!?]?\s*$", re.I
        ),
    ),
    "challenge": (
        re.compile(
            r"\b(?:want|give me|looking for|what about)\b.{0,20}\b"
            r"(?:a challenge|something challenging|hard content)\b",
            re.I,
        ),
        re.compile(r"^\s*(?:challenge|something challenging)(?:\s+instead)?[.!?]?\s*$", re.I),
    ),
    "exploration": (
        re.compile(
            r"\b(?:want|feel like|what about)\b.{0,20}\b(?:exploring|exploration|explore)\b", re.I
        ),
        re.compile(r"^\s*(?:exploration|explore)(?:\s+instead)?[.!?]?\s*$", re.I),
    ),
}

_GOAL_CLEAR: dict[PrimaryGoal, re.Pattern[str]] = {
    "story_progression": re.compile(
        r"\b(?:forget|ignore|drop) (?:the )?story(?: preference| goal)?\b", re.I
    ),
    "gear_rewards": re.compile(
        r"\b(?:forget|ignore|drop) (?:the )?(?:gear|loot|reward)"
        r"(?: preference| goal)?\b|\bi don'?t care about "
        r"(?:loot|gear|rewards?) anymore\b",
        re.I,
    ),
    "quest_completion": re.compile(
        r"\b(?:forget|ignore|drop) (?:the )?quest(?: preference| goal)?\b", re.I
    ),
    "build_improvement": re.compile(
        r"\b(?:forget|ignore|drop) (?:the )?build(?: preference| goal)?\b", re.I
    ),
    "casual_chill": re.compile(
        r"\b(?:forget|ignore|drop) (?:the )?(?:chill|casual)(?: preference| goal)?\b", re.I
    ),
    "challenge": re.compile(
        r"\b(?:forget|ignore|drop) (?:the )?challenge(?: preference| goal)?\b", re.I
    ),
    "exploration": re.compile(
        r"\b(?:forget|ignore|drop) (?:the )?exploration(?: preference| goal)?\b", re.I
    ),
}

_EXCLUSION_PATTERNS: dict[SessionExclusion, re.Pattern[str]] = {
    "pvp": re.compile(
        r"\b(?:no|avoid|don'?t want|not interested in)\s+(?:any\s+)?(?:pvp|crucible)\b", re.I
    ),
    "pve": re.compile(r"\b(?:no|avoid|don'?t want|not interested in)\s+(?:any\s+)?pve\b", re.I),
    "dungeons": re.compile(r"\b(?:no|avoid|don'?t want)\s+(?:any\s+)?dungeons?\b", re.I),
    "raids": re.compile(r"\b(?:no|avoid|don'?t want)(?:\s+to\s+do)?\s+(?:any\s+)?raids?\b", re.I),
    "grinding": re.compile(
        r"\b(?:no|avoid|don'?t want)(?:\s+to)?\s+(?:grind|grinding|farm|farming)\b", re.I
    ),
    "group_play": re.compile(
        r"\b(?:no|avoid|don'?t want)\s+(?:group|fireteam|matchmade)\s+"
        r"(?:play|activities|content)\b",
        re.I,
    ),
    "solo_play": re.compile(
        r"\b(?:no|avoid|don'?t want)\s+solo\s+(?:play|activities|content)\b", re.I
    ),
}


def _minutes(message: str) -> int | None:
    matches = list(_TIME.finditer(message))
    if not matches:
        return None
    normalized = message.casefold()
    constraint_language = bool(
        re.search(
            r"\b(?:i (?:only )?(?:have|got)|i(?:'ve|'d) got|another|available|"
            r"time|session|give me|what if|play for|spend)\b",
            normalized,
        )
    )
    stripped = re.sub(r"[.!?]", "", normalized).strip()
    if not constraint_language and not _TIME.fullmatch(stripped):
        return None
    match = matches[-1]
    amount = match.group("amount").casefold()
    unit = match.group("unit").casefold()
    if amount.startswith("half"):
        value = 30 if unit.startswith("hour") or unit.startswith("hr") else None
    else:
        numbers = {"a": 1, "an": 1, "one": 1, "another": 1, "two": 2, "three": 3, "four": 4}
        value = int(amount) if amount.isdigit() else numbers.get(amount)
        if value is not None and (unit.startswith("hour") or unit.startswith("hr")):
            value *= 60
    return value if value is not None and 1 <= value <= MAX_SESSION_MINUTES else None


def _character(message: str) -> CharacterPreference | None:
    matches = list(_CLASS.finditer(message))
    if not matches:
        return None
    unique = {match.group(1).casefold() for match in matches}
    normalized = message.casefold()
    if len(unique) > 1 and not re.search(r"\b(?:actually|instead|switch to|use)\b", normalized):
        return None
    if len(unique) == 1 or re.search(r"\b(?:actually|instead|switch to|use)\b", normalized):
        return matches[-1].group(1).title()  # type: ignore[return-value]
    return None


def _goal(message: str, suppressed: set[PrimaryGoal]) -> PrimaryGoal | None:
    matches: list[tuple[int, PrimaryGoal]] = []
    for goal, patterns in _GOAL_PATTERNS.items():
        if goal in suppressed:
            continue
        for pattern in patterns:
            match = pattern.search(message)
            if match:
                matches.append((match.start(), goal))
                break
    unique = {goal for _, goal in matches}
    if len(unique) > 1 and not re.search(
        r"\b(?:actually|instead|not .{0,20}(?:want|give me))\b", message, re.I
    ):
        return None
    return max(matches, default=(0, None), key=lambda value: value[0])[1]


def _add_exclusion(context: SessionPreferenceContext, value: SessionExclusion) -> None:
    if value not in context.exclusions and len(context.exclusions) < MAX_EXCLUSIONS:
        context.exclusions.append(value)


def _apply_message(
    context: SessionPreferenceContext, message: str
) -> tuple[SessionPreferenceContext, set[PreferenceField]]:
    updates: set[PreferenceField] = set()
    normalized = message.casefold()
    if _FULL_RESET.search(message):
        context = SessionPreferenceContext()
        updates.add("all")

    if _CHARACTER_RESET.search(message):
        context.character = "unspecified"
        updates.add("character")
    character = _character(message)
    if character is not None:
        context.character = character
        updates.add("character")

    if _TIME_RESET.search(message):
        context.time_minutes = None
        context.duration_preference = "unspecified"
        updates.update(("time_minutes", "duration_preference"))
    minutes = _minutes(message)
    if minutes is not None:
        context.time_minutes = minutes
        context.duration_preference = "unspecified"
        updates.update(("time_minutes", "duration_preference"))
    elif re.search(r"\b(?:something|make it|anything) shorter\b", message, re.I):
        context.duration_preference = "shorter"
        updates.add("duration_preference")
    elif re.search(r"\b(?:a |something )?(?:short|quick) session\b", message, re.I):
        context.duration_preference = "short"
        updates.add("duration_preference")
    elif re.search(r"\b(?:a |something )?(?:long|extended) session\b", message, re.I):
        context.duration_preference = "long"
        updates.add("duration_preference")

    suppressed_goals = {goal for goal, pattern in _GOAL_CLEAR.items() if pattern.search(message)}
    for cleared_goal in suppressed_goals:
        if context.primary_goal == cleared_goal:
            context.primary_goal = "unspecified"
        updates.add("primary_goal")
    goal = _goal(message, suppressed_goals)
    if goal is not None:
        context.primary_goal = goal
        updates.add("primary_goal")
        if goal == "casual_chill":
            context.intensity = "chill"
            updates.add("intensity")
        elif goal == "challenge":
            context.intensity = "challenging"
            updates.add("intensity")

    if re.search(
        r"\bsolo\b.{0,32}\b(?:or|and)\b.{0,16}\bmatchmak(?:ing|ed)\b|"
        r"\bmatchmak(?:ing|ed)\b.{0,32}\b(?:or|and)\b.{0,16}\bsolo\b",
        message,
        re.I,
    ):
        context.fireteam = "either"
        updates.add("fireteam")
    elif re.search(r"\b(?:solo|by myself|on my own)(?:\s+though|\s+instead)?\b", message, re.I):
        context.fireteam = "solo"
        context.exclusions = [value for value in context.exclusions if value != "solo_play"]
        updates.add("fireteam")
    elif re.search(
        r"\b(?:with (?:a )?(?:group|fireteam|friends)|group play|fireteam)(?:\s+instead)?\b",
        message,
        re.I,
    ):
        context.fireteam = "group"
        context.exclusions = [value for value in context.exclusions if value != "group_play"]
        updates.add("fireteam")
    elif re.search(
        r"\b(?:solo or group|either fireteam|any fireteam)(?: is fine)?\b", message, re.I
    ):
        context.fireteam = "either"
        updates.add("fireteam")

    pvp_excluded = bool(_EXCLUSION_PATTERNS["pvp"].search(message))
    pve_excluded = bool(_EXCLUSION_PATTERNS["pve"].search(message))
    if not pvp_excluded and re.search(
        r"(?:^|\b(?:want|play|something|prefer)\s+)pvp\b|"
        r"\b(?:crucible|iron banner)\b",
        message,
        re.I,
    ):
        context.activity_mode = "pvp"
        context.exclusions = [value for value in context.exclusions if value != "pvp"]
        updates.add("activity_mode")
    elif not pve_excluded and re.search(
        r"(?:^|\b(?:want|play|something|prefer)\s+)pve\b", message, re.I
    ):
        context.activity_mode = "pve"
        context.exclusions = [value for value in context.exclusions if value != "pve"]
        updates.add("activity_mode")
    elif re.search(
        r"\b(?:pvp or pve|pve or pvp|either mode|any mode)(?: is fine)?\b", message, re.I
    ):
        context.activity_mode = "either"
        context.exclusions = [value for value in context.exclusions if value not in {"pvp", "pve"}]
        updates.add("activity_mode")

    if re.search(r"\b(?:chill|casual|relaxing|low[- ]stress)\b", message, re.I):
        context.intensity = "chill"
        updates.add("intensity")
    elif re.search(r"\b(?:challenging|high[- ]intensity|hard content)\b", message, re.I):
        context.intensity = "challenging"
        updates.add("intensity")
    elif re.search(r"\bnormal intensity\b", message, re.I):
        context.intensity = "normal"
        updates.add("intensity")
    elif re.search(r"\b(?:any intensity|difficulty doesn'?t matter)\b", message, re.I):
        context.intensity = "unspecified"
        updates.add("intensity")

    if re.search(r"\b(?:no exclusions|anything is fine)\b", message, re.I):
        context.exclusions = []
        updates.add("exclusions")
    for exclusion, pattern in _EXCLUSION_PATTERNS.items():
        if pattern.search(message):
            _add_exclusion(context, exclusion)
            updates.add("exclusions")
    for exclusion, label in (
        ("pvp", "pvp|crucible"),
        ("pve", "pve"),
        ("dungeons", "dungeons?"),
        ("raids", "raids?"),
        ("grinding", "grind(?:ing)?|farm(?:ing)?"),
    ):
        if re.search(rf"\b(?:{label}) (?:is|are) fine now\b", normalized):
            context.exclusions = [value for value in context.exclusions if value != exclusion]
            updates.add("exclusions")

    return context, updates


def derive_session_preferences(
    history: Sequence[ChatTurn], current_message: str
) -> SessionPreferenceDerivation:
    """Rebuild effective preferences from user-authored turns in the bounded request window."""

    context = SessionPreferenceContext()
    for turn in history[-MAX_PREFERENCE_HISTORY_TURNS:]:
        if turn.role == "user":
            context, _ = _apply_message(context, turn.content)
    prior_context_present = context.has_constraints()
    context, current_updates = _apply_message(context, current_message)
    ordered_updates = [value for value in _FIELD_ORDER if value in current_updates]
    return SessionPreferenceDerivation(
        preferences=context,
        current_turn_updates=ordered_updates,
        prior_context_present=prior_context_present,
    )
