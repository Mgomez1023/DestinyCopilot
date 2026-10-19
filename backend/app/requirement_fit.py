"""Compare a structured objective requirement with one character's equipped loadout.

Knowledge Engine v0.2, M4B. M4A turned Manifest objective text into structured requirements;
this module checks them against evidence the Guardian profile already holds, for one
character, with no lookups and no network.

What a result means
-------------------
The comparison is about the **currently equipped loadout**. It says whether equipped weapons
or equipped subclass abilities have the damage source and element the requirement states.
It never says the objective will complete, that an unequipped item could not, or that the
player has an item they did not equip:

* `verified_match`: equipped evidence explicitly has the stated source and element.
* `potential_match`: some of the requirement lines up, but the evidence cannot establish it
  (partial requirement, subclass element without ability data, an ability whose element was
  not recorded, or a condition the loadout cannot show such as precision hits).
* `verified_mismatch`: the equipped evidence is complete enough to show it lacks the stated
  source or element. This describes the loadout, not what the player could equip.
* `unknown`: ambiguous or unparsed requirement, or the needed evidence was not retrieved.

Weapon damage is never ability damage. An Arc weapon does not satisfy an Arc ability
requirement, and a weapon perk that triggers after an ability kill is not an ability kill;
neither perks nor bonus-progress text are read here at all.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_serializer

from app.build_analysis import classify_plug
from app.models import CharacterSummary, ItemSummary, ObjectiveRequirement, SocketedPlugSummary
from app.objective_requirements import compact_requirement, is_restrictive

FitStatus = Literal["verified_match", "potential_match", "verified_mismatch", "unknown"]

ELEMENTS = ("Arc", "Solar", "Void", "Stasis", "Strand")
MAX_NAMES = 3
REQUIREMENT_BASIS = (
    "requirements are parsed from Manifest text (see source); verified means unambiguous "
    "wording, not a guarantee. fit compares only this character's EQUIPPED weapons and subclass "
    "abilities. verified_match = the equipped loadout has the stated source and element; it "
    "does NOT mean the other conditions are met (activity, location, fireteam, bonus text: "
    "see unassessed) or that completion is guaranteed. potential_match = partly lines up, not "
    "established. verified_mismatch = the equipped loadout lacks it; vaulted, carried or "
    "unowned alternatives are not considered. unknown = not established. Weapon damage is "
    "not ability damage. bonus_conditions are raw text, not mechanics."
)
# Damage-dealing ability kinds. A class ability (a rift, a barricade, a dodge) and movement
# do not by themselves deal damage, so they never satisfy a final-blow requirement.
_DAMAGING_KINDS = frozenset({"super", "grenade", "melee"})
_KIND_SEGMENTS = {
    "supers": "super",
    "grenades": "grenade",
    "prism_grenade": "grenade",
    "melee": "melee",
    "class_abilities": "class_ability",
    "movement": "movement",
}
_REQUIREMENT_MODIFIER_KINDS = {
    "super": "super",
    "grenade": "grenade",
    "melee": "melee",
    "class_ability": "class_ability",
}
# Modifiers a loadout cannot establish: they describe how a kill is made, not what is equipped.
_UNASSESSABLE_MODIFIERS = frozenset({"precision"})


class RequirementFit(BaseModel):
    """The compact comparison result carried with a requirement."""

    status: FitStatus
    reason: str
    # Equipped items that explicitly support the match, as "Name (element kind)". At most 3.
    equipped: list[str] = Field(default_factory=list, max_length=MAX_NAMES)
    # Equipped items that look related but do not count, with why.
    not_counted: list[str] = Field(default_factory=list, max_length=2)
    # Stated parts of the requirement this comparison could not assess.
    unassessed: list[str] = Field(default_factory=list, max_length=3)

    @model_serializer(mode="wrap")
    def _omit_empty(self, handler: Any) -> dict[str, Any]:
        data = handler(self)
        # The equipped names already say why a verified match holds.
        if data.get("status") == "verified_match":
            data.pop("reason", None)
        return {key: value for key, value in data.items() if value not in ([], None)}


class AbilityEvidence(BaseModel):
    name: str
    kind: str
    element: str | None  # None when the plug category does not name one (Prismatic)

    def label(self) -> str:
        return f"{self.name} ({self.element + ' ' if self.element else ''}{self.kind})"


class LoadoutEvidence(BaseModel):
    """Equipped weapons and subclass abilities of one character, as already normalized."""

    weapons: list[ItemSummary] = Field(default_factory=list)
    abilities: list[AbilityEvidence] = Field(default_factory=list)
    ability_data: bool = False
    subclass_name: str | None = None
    subclass_element: str | None = None


def _plug_element(plug: SocketedPlugSummary) -> str | None:
    segments = (plug.category_identifier or "").casefold().split(".")
    for element in ELEMENTS:
        if element.casefold() in segments:
            return element
    # Prismatic plugs share one category family; Bungie's own type label names the element
    # ("Solar Grenade | Light Ability"). Anything else stays unnamed, not guessed.
    label = (plug.plug_type or "").split(" ", 1)[0]
    return label if label in ELEMENTS else None


def _plug_kind(plug: SocketedPlugSummary) -> str | None:
    for segment in (plug.category_identifier or "").casefold().split("."):
        if segment in _KIND_SEGMENTS:
            return _KIND_SEGMENTS[segment]
    return None


def loadout_evidence(character: CharacterSummary) -> LoadoutEvidence:
    """Collect only what is **equipped** on this character. Inventory and vault are excluded."""

    weapons = [
        value
        for value in character.equipped_gear
        if value.item_type == "Weapon" and value.location == "equipped"
    ]
    abilities: list[AbilityEvidence] = []
    subclass = character.subclass
    if subclass is not None:
        for plug in subclass.socketed_plug_details:
            if plug.is_enabled is False:
                continue
            if classify_plug(plug.category_identifier, plug.item_type) != "ability":
                continue
            kind = _plug_kind(plug)
            if kind is not None:
                abilities.append(
                    AbilityEvidence(name=plug.name, kind=kind, element=_plug_element(plug))
                )
    element = (subclass.damage_type or "").title() if subclass else ""
    return LoadoutEvidence(
        weapons=weapons,
        abilities=abilities,
        ability_data=bool(abilities),
        subclass_name=subclass.name if subclass else None,
        subclass_element=element if element in ELEMENTS else None,
    )


def _weapon_label(item: ItemSummary) -> str:
    return f"{item.name} ({item.damage_type + ' ' if item.damage_type else ''}weapon)"


def _cap(status: FitStatus, requirement: ObjectiveRequirement, unassessed: list[str]) -> FitStatus:
    """A match can only be as certain as the requirement and what the loadout can show."""

    if status != "verified_match":
        return status
    if requirement.parse_status != "verified":
        unassessed.append("partial_parse")
        return "potential_match"
    modifiers = [value for value in requirement.modifiers if value in _UNASSESSABLE_MODIFIERS]
    if modifiers:
        unassessed.extend(f"modifier:{value}" for value in modifiers)
        return "potential_match"
    return status


def _weapon_fit(
    requirement: ObjectiveRequirement, loadout: LoadoutEvidence
) -> tuple[FitStatus, str, list[str]]:
    weapons = loadout.weapons
    if not weapons:
        return "unknown", "no_equipped_weapons_returned", []
    wanted = requirement.element if requirement.element != "unknown" else None
    if wanted is None:
        return "verified_match", "weapon_equipped", [_weapon_label(value) for value in weapons]
    matching = [value for value in weapons if (value.damage_type or "") == wanted]
    if matching:
        return "verified_match", f"{wanted}_weapon_equipped", [_weapon_label(v) for v in matching]
    if any(not value.damage_type for value in weapons):
        return "unknown", "weapon_damage_type_not_returned", []
    return "verified_mismatch", f"no_equipped_{wanted}_weapon", []


def _ability_fit(
    requirement: ObjectiveRequirement, loadout: LoadoutEvidence
) -> tuple[FitStatus, str, list[str]]:
    wanted = requirement.element if requirement.element != "unknown" else None
    if not loadout.ability_data:
        # An element on the subclass alone does not show which ability is equipped.
        if wanted and loadout.subclass_element == wanted:
            return "potential_match", "subclass_element_only_no_ability_data", []
        return "unknown", "no_equipped_ability_data", []
    allowed = {
        kind
        for modifier, kind in _REQUIREMENT_MODIFIER_KINDS.items()
        if modifier in requirement.modifiers
    }
    kinds = allowed or _DAMAGING_KINDS
    candidates = [value for value in loadout.abilities if value.kind in kinds]
    if not candidates:
        return "verified_mismatch", "no_equipped_damaging_ability_of_that_kind", []
    if wanted is None:
        return "verified_match", "ability_equipped", [value.label() for value in candidates]
    matching = [value for value in candidates if value.element == wanted]
    if matching:
        return "verified_match", f"{wanted}_ability_equipped", [v.label() for v in matching]
    if any(value.element is None for value in candidates):
        return "potential_match", "ability_element_not_recorded", []
    return "verified_mismatch", f"no_equipped_{wanted}_ability", []


def _any_source_fit(
    requirement: ObjectiveRequirement, loadout: LoadoutEvidence
) -> tuple[FitStatus, str, list[str]]:
    weapon = _weapon_fit(requirement, loadout)
    ability = _ability_fit(requirement, loadout)
    for status in ("verified_match", "potential_match"):
        found = [value for value in (weapon, ability) if value[0] == status]
        if found:
            names = [name for value in found for name in value[2]]
            return status, found[0][1], names  # type: ignore[return-value]
    if weapon[0] == ability[0] == "verified_mismatch":
        return "verified_mismatch", "no_equipped_source_with_that_element", []
    return "unknown", "equipped_evidence_incomplete", []


def _not_counted(requirement: ObjectiveRequirement, loadout: LoadoutEvidence) -> list[str]:
    """Equipped items of the right element but the wrong damage source, named so they are not
    mistaken for a match."""

    wanted = requirement.element if requirement.element != "unknown" else None
    notes: list[str] = []
    if requirement.source_kind == "ability":
        weapons = [
            value for value in loadout.weapons if wanted is None or value.damage_type == wanted
        ]
        if weapons:
            label = f"{wanted} " if wanted else ""
            notes.append(f"{label}weapons: not ability damage")
    elif requirement.source_kind == "weapon":
        abilities = [
            value for value in loadout.abilities if wanted is None or value.element == wanted
        ]
        if abilities:
            label = f"{wanted} " if wanted else ""
            notes.append(f"{label}abilities: not weapon damage")
    return notes[:2]


def compare_requirement(
    requirement: ObjectiveRequirement, loadout: LoadoutEvidence
) -> RequirementFit:
    """Deterministically compare one requirement with one character's equipped loadout."""

    unassessed: list[str] = []
    if requirement.conditions:
        unassessed.append("conditions")
    if requirement.bonus_conditions:
        unassessed.append("bonus_text")
    if requirement.parse_status in ("ambiguous", "unknown"):
        return RequirementFit(
            status="unknown", reason="requirement_not_established", unassessed=unassessed[:3]
        )
    if not is_restrictive(requirement):
        return RequirementFit(
            status="unknown", reason="requirement_names_no_source_or_element", unassessed=unassessed
        )

    if requirement.source_kind == "ability":
        status, reason, names = _ability_fit(requirement, loadout)
    elif requirement.source_kind == "weapon":
        status, reason, names = _weapon_fit(requirement, loadout)
    elif requirement.source_kind == "any" or requirement.element != "unknown":
        # "Any" source, or damage of an element from no stated source: either source counts.
        status, reason, names = _any_source_fit(requirement, loadout)
    else:
        # Only an action or modifier was stated; the loadout has nothing to match against.
        return RequirementFit(
            status="unknown", reason="requirement_names_no_source_or_element", unassessed=unassessed
        )
    status = _cap(status, requirement, unassessed)
    return RequirementFit(
        status=status,
        reason=reason,
        equipped=names[:MAX_NAMES],
        not_counted=_not_counted(requirement, loadout),
        unassessed=list(dict.fromkeys(unassessed))[:3],
    )


# --- Planning-facing composition ----------------------------------------------------------------

MAX_BONUS_CHARS = 160
MAX_RAW_CHARS = 160


class PlanningRequirement(BaseModel):
    """A compact requirement plus its equipped-loadout comparison, for planning candidates."""

    requirement: dict[str, Any]
    fit: RequirementFit

    @model_serializer(mode="wrap")
    def _flatten(self, handler: Any) -> dict[str, Any]:
        data = handler(self)
        return {**data["requirement"], "fit": data["fit"]}


def planning_requirement(
    requirement: ObjectiveRequirement, loadout: LoadoutEvidence
) -> PlanningRequirement:
    """Compact form: fields, provenance, and raw text only where the structure is not enough.

    The raw sentence is kept for anything but a clean verified requirement (the candidate's
    own description already carries it). Bonus and condition text is kept, bounded, so it is
    never silently dropped or paraphrased.
    """

    data = compact_requirement(requirement)
    if requirement.parse_status == "verified":
        data.pop("raw_requirement", None)
    elif "raw_requirement" in data:
        data["raw_requirement"] = data["raw_requirement"][:MAX_RAW_CHARS]
    if bonus := data.get("bonus_conditions"):
        data["bonus_conditions"] = [value[:MAX_BONUS_CHARS] for value in bonus[:1]]
    return PlanningRequirement(requirement=data, fit=compare_requirement(requirement, loadout))
