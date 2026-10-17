"""Deterministic character selection over an authenticated GuardianContext."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from app.models import CharacterSummary, GuardianContext

CharacterClass = Literal["Titan", "Hunter", "Warlock"]

CurrentCharacterReason = Literal[
    "only_character",
    "most_recently_played",
    "last_played_tie_playtime",
    "last_played_tie_character_id",
    "last_played_unavailable_playtime",
    "last_played_unavailable_character_id",
]
SelectorSource = Literal["explicit_id", "explicit_class", "unspecified"]


@dataclass(frozen=True, slots=True)
class CurrentCharacter:
    """The deterministically selected current character and why it was chosen."""

    character_id: str
    class_name: str
    reason: CurrentCharacterReason
    last_played: datetime | None
    candidates: int


def _id_rank(character_id: str) -> tuple[int, int, str]:
    # Bungie character IDs are numeric strings that grow over time; compare them as numbers.
    digits = character_id.isdigit()
    return (1 if digits else 0, int(character_id) if digits else 0, character_id)


def resolve_current_character(context: GuardianContext) -> CurrentCharacter | None:
    """Select the player's current character from Bungie data alone; the model never guesses.

    Rule, in order:
    1. The character with the latest `dateLastPlayed` (a missing timestamp ranks last).
    2. If several share the latest timestamp (or none has one), the one with the most total
       minutes played.
    3. If that also ties, the highest character ID (numeric when possible), i.e. the most
       recently created character.
    The result depends only on character data, never on list order.
    """

    characters = context.characters
    if not characters:
        return None

    def key(character: CharacterSummary) -> tuple[Any, ...]:
        return (
            character.last_played is not None,
            character.last_played.timestamp() if character.last_played else 0.0,
            character.minutes_played_total,
            _id_rank(character.character_id),
        )

    ranked = sorted(characters, key=key, reverse=True)
    top = ranked[0]
    if len(ranked) == 1:
        reason: CurrentCharacterReason = "only_character"
    else:
        runner_up = ranked[1]
        dated = top.last_played is not None
        if dated and (runner_up.last_played is None or top.last_played > runner_up.last_played):
            reason = "most_recently_played"
        else:
            by_playtime = top.minutes_played_total != runner_up.minutes_played_total
            prefix = "last_played_tie" if dated else "last_played_unavailable"
            reason = f"{prefix}_{'playtime' if by_playtime else 'character_id'}"  # type: ignore[assignment]
    return CurrentCharacter(
        character_id=top.character_id,
        class_name=top.class_name,
        reason=reason,
        last_played=top.last_played,
        candidates=len(ranked),
    )


ScopeSource = Literal["current_default", "explicit_prompt", "persisted_conversation_scope"]
_CLASS_NAMES = ("Titan", "Hunter", "Warlock")


@dataclass(frozen=True, slots=True)
class ConversationScope:
    """Which character this conversation is about, and how that was decided.

    `bungie_current` is the account's most recently played character (never changed by what
    the user says). `active` is the character the conversation is scoped to: the class the
    user explicitly chose in this prompt or earlier in the conversation, otherwise the Bungie
    current character. Nothing is stored server-side: the explicit choice is re-derived from
    the user's own turns in each request, so a new conversation starts at the Bungie default.
    """

    bungie_current: str | None
    active: str | None
    source: ScopeSource
    previous_active: str | None
    switched: bool

    @property
    def explicit_class(self) -> str | None:
        """The user-chosen class, or None when the Bungie default applies."""

        return None if self.source == "current_default" else self.active


def resolve_conversation_scope(
    context: GuardianContext,
    previous_preference: str,
    new_preference: str,
    *,
    chosen_this_turn: bool,
) -> ConversationScope:
    """Resolve the active conversation character.

    `previous_preference` and `new_preference` are the user-stated character ("Titan",
    "Hunter", "Warlock", or "unspecified") before and after this prompt. An explicit class in
    the prompt wins and becomes the conversation's scope until the user states another one
    (or resets), so an explicit question about another character is a switch.
    """

    current = resolve_current_character(context)
    bungie = current.class_name if current else None
    new = new_preference if new_preference in _CLASS_NAMES else None
    previous = previous_preference if previous_preference in _CLASS_NAMES else None
    if new is None:
        source: ScopeSource = "current_default"
    elif chosen_this_turn:
        source = "explicit_prompt"
    else:
        source = "persisted_conversation_scope"
    active = new or bungie
    previous_active = previous or bungie
    return ConversationScope(
        bungie_current=bungie,
        active=active,
        source=source,
        previous_active=previous_active,
        switched=active != previous_active,
    )


def selector_source(character_id: object, character_class: object) -> SelectorSource:
    """How a tool call picked its character: an explicit selector, or none."""

    if character_id:
        return "explicit_id"
    if character_class:
        return "explicit_class"
    return "unspecified"


class GuardianToolError(ValueError):
    """Safe, user-correctable failure while executing a Guardian tool."""

    code = "guardian_tool_error"


class CharacterResolutionError(GuardianToolError):
    """A character selector could not resolve to exactly one character."""

    code = "character_resolution_failed"


class CharacterSelectionError(CharacterResolutionError):
    code = "invalid_character_selector"


class CharacterNotFoundError(CharacterResolutionError):
    code = "character_not_found"


class GuardianCharacterResolver:
    """Resolve an opaque ID or public class name without exposing account identifiers."""

    def __init__(
        self, context: GuardianContext, default_class: CharacterClass | None = None
    ) -> None:
        self.context = context
        # The conversation's explicit character, used when a call names no selector. Without
        # one, the Bungie current character is the default.
        self.default_class = default_class

    def resolve_current(self) -> CharacterSummary:
        if self.default_class is not None:
            return self.resolve_one(character_class=self.default_class)
        current = resolve_current_character(self.context)
        if current is None:
            raise CharacterNotFoundError("No characters were found in Guardian data.")
        return next(
            value for value in self.context.characters if value.character_id == current.character_id
        )

    def resolve_one(
        self,
        *,
        character_id: str | None = None,
        character_class: CharacterClass | None = None,
        default_to_current: bool = False,
    ) -> CharacterSummary:
        """Resolve exactly one character.

        An explicit selector always wins. With no selector, the player's current
        (most recently played) character is used only when `default_to_current` is set;
        otherwise a selector is required.
        """

        if character_id is not None and character_class is not None:
            raise CharacterSelectionError(
                "Provide either character_id or character_class, not both."
            )
        if character_id is None and character_class is None:
            if default_to_current:
                return self.resolve_current()
            raise CharacterSelectionError(
                "A character_id or character_class is required for this Guardian lookup."
            )

        if character_id is not None:
            matches = [
                character
                for character in self.context.characters
                if character.character_id == character_id
            ]
            label = "The requested character"
        else:
            matches = [
                character
                for character in self.context.characters
                if character.class_name.casefold() == character_class.casefold()
            ]
            label = f"A {character_class}"

        if not matches:
            raise CharacterNotFoundError(f"{label} was not found in Guardian data.")
        if len(matches) > 1:
            raise CharacterSelectionError(
                f"{label} matched more than one character; selection is ambiguous."
            )
        return matches[0]

    def resolve_many(
        self,
        *,
        character_id: str | None = None,
        character_class: CharacterClass | None = None,
    ) -> list[CharacterSummary]:
        if character_id is None and character_class is None:
            return self.context.characters
        return [self.resolve_one(character_id=character_id, character_class=character_class)]
