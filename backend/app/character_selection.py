"""Deterministic character selection over an authenticated GuardianContext."""

from typing import Literal

from app.models import CharacterSummary, GuardianContext

CharacterClass = Literal["Titan", "Hunter", "Warlock"]


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

    def __init__(self, context: GuardianContext) -> None:
        self.context = context

    def resolve_one(
        self,
        *,
        character_id: str | None = None,
        character_class: CharacterClass | None = None,
    ) -> CharacterSummary:
        if character_id is not None and character_class is not None:
            raise CharacterSelectionError(
                "Provide either character_id or character_class, not both."
            )
        if character_id is None and character_class is None:
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
