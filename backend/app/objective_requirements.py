"""Conservative requirement extraction from Bungie Manifest objective text.

Knowledge Engine v0.2, M4A. Quest, bounty and objective definitions state what counts toward
completion in plain text ("Defeat combatants or Guardians with Arc ability final blows.").
That text already reaches the model; this module adds a structured, provenance-carrying view
of the parts the wording states explicitly, so "ability" is never confused with "weapon" and a
generic "defeat" is never read as "final blows".

Rules, in order of importance:

* Parse only what the words say. A field the text does not state stays `unknown`; `unknown`
  never means "no restriction".
* An element counts only when it directly modifies damage, an ability, a weapon, a Super or
  final blows ("Arc ability final blows"). "Combatants affected by Arc debuffs" and "your Solar
  subclass" are not element requirements.
* Elemental damage is not a source: "Arc damage" does not say weapon or ability.
* Bonus-progress sentences are kept as raw text and never interpreted.
* Two sources (quest description, step description, objective label) are parsed separately and
  never merged; disagreement is reported, not resolved.
* Anything the grammar cannot place is kept in `conditions` or makes the parse `partial` or
  `ambiguous`. The original sentence is always preserved.

It is lexical and deterministic: no model call, no network, no item or activity names.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, Literal

from app.models import (
    ObjectiveRequirement,
    RequirementAction,
    RequirementConsistency,
    RequirementElement,
    RequirementModifier,
    RequirementParseStatus,
    RequirementSource,
    RequirementSourceKind,
    RequirementTarget,
)

TextKind = Literal["description", "objective_label"]

MAX_REQUIREMENTS_PER_TEXT = 3
MAX_CONDITIONS = 3
MAX_CONDITION_LENGTH = 100
MAX_BONUS = 3
MAX_BONUS_LENGTH = 200

ELEMENT_WORDS = "Arc|Solar|Void|Stasis|Strand|Kinetic"
_GLYPH = re.compile(r"\[[^\]]{1,32}\]\s*")
_SENTENCES = re.compile(r"(?<=[.!?])\s+|\n+")
_BONUS = re.compile(
    r"\b(?:bonus|additional|greater|extra|more|substantially more)\s+(?:progress|points)\b",
    re.I,
)
_DESCRIPTIVE = re.compile(
    r"\b(?:tracks?|displays?|generates?|is caused|are caused|deals? elemental)\b", re.I
)
_GATE = re.compile(r"\b(?:defeat|kill|land|deal)\b|\bfinal blows?\b", re.I)
_PREFIX = re.compile(r"^((?:In|On|As|During|Within|At)\b[^,]{1,60}),\s*", re.I)

_FINAL_BLOW = re.compile(r"\bfinal blows?\b", re.I)
_DAMAGE_WORD = re.compile(r"\bdamage\b", re.I)
_DEFEAT = re.compile(r"\b(?:defeat(?:s|ed|ing)?|kills?|killing)\b", re.I)
_DEAL = re.compile(r"\b(?:deal|dealing)\b", re.I)
_ABILITY = re.compile(r"\babilit(?:y|ies)\b", re.I)
_SUPER = re.compile(r"\bsupers?\b", re.I)
_GRENADE = re.compile(r"\bgrenades?\b(?!\s+launchers?)", re.I)
_MELEE = re.compile(r"\bmelee\b", re.I)
_CLASS_ABILITY = re.compile(r"\bclass abilit(?:y|ies)\b", re.I)
_WEAPON = re.compile(r"\bweapons?\b", re.I)
_ARCHETYPE = re.compile(
    r"\b(?:hand cannons?|auto rifles?|pulse rifles?|scout rifles?|sniper rifles?|"
    r"(?:linear )?fusion rifles?|trace rifles?|shotguns?|sidearms?|submachine guns?|"
    r"machine guns?|rocket launchers?|grenade launchers?|swords?|glaives?|bows?)\b",
    re.I,
)
_PRECISION = re.compile(r"\bprecision\b", re.I)
_ANY_SOURCE = re.compile(r"\bany (?:final blows?|source|damage)\b", re.I)
_ELEMENT_DAMAGE = re.compile(
    rf"\b({ELEMENT_WORDS})\s+(?:damage|abilit(?:y|ies)|weapons?|final blows?|supers?|"
    r"grenades?|melee|kills?)\b",
    re.I,
)
_ALTERNATIVE_BLOWS = re.compile(r"\bfinal blows?\b[^.]*\bor\b[^.]*\bfinal blows?\b", re.I)
_OTHER_OBJECTIVE = re.compile(
    r"\b(?:complete|collect|win|finish|open|earn|gain|generate|activate|survive|capture)\b", re.I
)
_RESTRICTING = re.compile(
    r"\b(?:with|using|by|wearing)\s+([^,.;]+?)"
    r"(?=\s+(?:while|in|on|as|to|without|during|with|using|by|wearing)\b|[,.;]|$)",
    re.I,
)
_REQUIREMENT_VERB = re.compile(r"\b(?:defeat|kill|land|deal)\b", re.I)
# Element words that do not describe damage, even when they appear in the same sentence.
_ELEMENT_ANYWHERE = re.compile(
    rf"\b({ELEMENT_WORDS})\b(?!\s+(?:subclass|debuffs?|buffs?|effects?|light)\b)", re.I
)
_EXPLAINED = (
    _FINAL_BLOW, _DAMAGE_WORD, _ABILITY, _SUPER, _GRENADE, _MELEE, _CLASS_ABILITY, _WEAPON,
    _ARCHETYPE, _PRECISION, _ANY_SOURCE,
)  # fmt: skip
_CONDITION_PATTERNS = (
    re.compile(r"\b(?:while|without|during|within)\b[^.;]*", re.I),
    re.compile(r"\bas a (?:team|fireteam)\b", re.I),
    re.compile(r"\b(?:in|on) [^,.;]*?(?=\s+(?:with|using|while|without|as a|to)\b|[,.;]|$)", re.I),
)

_TARGETS: tuple[tuple[re.Pattern[str], RequirementTarget], ...] = (
    (
        re.compile(r"\bcombatants?\b[^.]*?\bor\b[^.]*?\bguardians?\b", re.I),
        "combatants_or_guardians",
    ),
    (re.compile(r"\bcombatants?\b", re.I), "combatants"),
    (re.compile(r"\bguardians?\b", re.I), "guardians"),
    (re.compile(r"\bopponents?\b", re.I), "opponents"),
    (re.compile(r"\benem(?:y|ies)\b", re.I), "enemies"),
    (re.compile(r"\btargets?\b", re.I), "targets"),
)  # fmt: skip


def clean(text: str) -> str:
    """Drop icon glyph markers such as "[Stasis]" and collapse whitespace."""

    return " ".join(_GLYPH.sub("", text).split())


def _first(patterns: Iterable[tuple[re.Pattern[str], str]], text: str) -> str | None:
    for pattern, value in patterns:
        if pattern.search(text):
            return value
    return None


def _source_kind(text: str) -> tuple[RequirementSourceKind, bool, bool]:
    """(source kind, ambiguous, archetype named). Abilities and weapons together are ambiguous."""

    ability = any(
        pattern.search(text) for pattern in (_ABILITY, _SUPER, _GRENADE, _MELEE, _CLASS_ABILITY)
    )
    weapon_words = bool(_WEAPON.search(text) or _ARCHETYPE.search(text))
    archetype = bool(_ARCHETYPE.search(text))
    if ability and weapon_words:
        return "unknown", True, archetype
    if ability:
        return "ability", False, False
    if weapon_words:
        return "weapon", False, archetype
    if _ANY_SOURCE.search(text):
        return "any", False, False
    return "unknown", False, False


def _elements(text: str) -> tuple[set[str], bool]:
    """(elements attached to damage/ability/weapon words, whether other elements are listed).

    "Solar, Arc, Void, or Stasis final blows" attaches only the last element; the others are
    listed alternatives, so the caller must treat the element as ambiguous.
    """

    attached = {value.capitalize() for value in _ELEMENT_DAMAGE.findall(text)}
    everywhere = {value.capitalize() for value in _ELEMENT_ANYWHERE.findall(text)}
    return attached, bool(everywhere - attached)


def _modifiers(text: str) -> list[RequirementModifier]:
    found: list[RequirementModifier] = []
    for pattern, name in (
        (_PRECISION, "precision"),
        (_MELEE, "melee"),
        (_GRENADE, "grenade"),
        (_CLASS_ABILITY, "class_ability"),
        (_SUPER, "super"),
    ):
        if pattern.search(text) and name not in found:
            found.append(name)  # type: ignore[arg-type]
    return found


def _action(text: str) -> RequirementAction:
    if _FINAL_BLOW.search(text):
        return "final_blow"
    if _DEAL.search(text) and _DAMAGE_WORD.search(text) and not _DEFEAT.search(text):
        return "damage"
    if _DEFEAT.search(text):
        return "defeat"
    return "unknown"


def _conditions(sentence: str) -> list[str]:
    found: list[str] = []
    prefix = _PREFIX.match(sentence)
    if prefix:
        found.append(prefix.group(1).strip())
    body = sentence[prefix.end() :] if prefix else sentence
    for pattern in _CONDITION_PATTERNS:
        for match in pattern.finditer(body):
            text = match.group(0).strip(" ,;")
            if text and text not in found:
                found.append(text)
    return [value[:MAX_CONDITION_LENGTH] for value in found][:MAX_CONDITIONS]


def _unexplained_restrictions(sentence: str) -> list[str]:
    """with/using/by/wearing phrases no modeled token explains (a subclass, a cosmetic)."""

    leftovers: list[str] = []
    for match in _RESTRICTING.finditer(sentence):
        phrase = match.group(1)
        if re.match(r"(?:defeat|kill|land|deal)\w*\b", phrase, re.I):
            continue  # "by defeating targets" introduces the action; it restricts nothing
        if _ELEMENT_DAMAGE.search(phrase) or any(pattern.search(phrase) for pattern in _EXPLAINED):
            continue
        if re.search(r"\bthis weapon\b", phrase, re.I):
            continue
        leftovers.append(match.group(0).strip())
    return leftovers


def _parse_sentence(
    raw_sentence: str, source: RequirementSource, bonus: list[str]
) -> ObjectiveRequirement | None:
    sentence = clean(raw_sentence)
    elements, extra_elements = _elements(sentence)
    source_kind, mixed_sources, archetype = _source_kind(sentence)
    action = _action(sentence)
    if action == "unknown":
        return None
    if len(_REQUIREMENT_VERB.findall(sentence)) > 1:
        # Several requirement clauses in one sentence: not combined, not guessed.
        return ObjectiveRequirement(
            raw_requirement=raw_sentence.strip(),
            source=source,
            conditions=_conditions(sentence),
            bonus_conditions=[value[:MAX_BONUS_LENGTH] for value in bonus[:MAX_BONUS]],
            parse_status="ambiguous",
        )
    ambiguous = (
        mixed_sources
        or extra_elements
        or len(elements) > 1
        or bool(_ALTERNATIVE_BLOWS.search(sentence))
    )
    leftovers = _unexplained_restrictions(sentence)
    other_objective = bool(
        _OTHER_OBJECTIVE.search(sentence) and re.search(r"\band\b", sentence, re.I)
    )
    conditions = _conditions(sentence)
    extras = [value[:MAX_CONDITION_LENGTH] for value in leftovers]
    if archetype:
        # A named weapon type restricts the weapon; the type itself is kept as raw text.
        extras += [match.group(0)[:MAX_CONDITION_LENGTH] for match in _ARCHETYPE.finditer(sentence)]
    for extra in extras:
        if not any(extra.casefold() in value.casefold() for value in conditions):
            conditions.append(extra)
    status: RequirementParseStatus
    if ambiguous:
        status = "ambiguous"
    elif leftovers or other_objective or archetype:
        status = "partial"
    else:
        status = "verified"
    element: RequirementElement = (
        next(iter(elements))  # type: ignore[assignment]
        if len(elements) == 1 and not extra_elements
        else "unknown"
    )
    target = _first(
        ((pattern, value) for pattern, value in _TARGETS),
        sentence,
    )
    return ObjectiveRequirement(
        raw_requirement=raw_sentence.strip(),
        source=source,
        source_kind="unknown" if ambiguous and mixed_sources else source_kind,
        element=element,
        completion_action=action,
        target_kind=(target or "unknown"),  # type: ignore[arg-type]
        modifiers=_modifiers(sentence),
        conditions=list(dict.fromkeys(conditions))[:MAX_CONDITIONS],
        bonus_conditions=[value[:MAX_BONUS_LENGTH] for value in bonus[:MAX_BONUS]],
        parse_status=status,
    )


def _parse_label(text: str, source: RequirementSource) -> ObjectiveRequirement | None:
    """A short objective label ("Grenade kills", "Arc weapon final blows")."""

    sentence = clean(text)
    elements, extra_elements = _elements(sentence)
    source_kind, mixed_sources, archetype = _source_kind(sentence)
    action = _action(sentence)
    modifiers = _modifiers(sentence)
    known = bool(
        elements or source_kind != "unknown" or modifiers or action in ("final_blow", "damage")
    )
    if not known and action == "unknown":
        return None
    ambiguous = mixed_sources or extra_elements or len(elements) > 1
    status: RequirementParseStatus
    if ambiguous:
        status = "ambiguous"
    elif action != "unknown" and (elements or source_kind != "unknown") and not archetype:
        status = "verified"
    elif known:
        status = "partial"
    else:
        status = "unknown"
    return ObjectiveRequirement(
        raw_requirement=text.strip(),
        source=source,
        source_kind="unknown" if mixed_sources else source_kind,
        element=(
            next(iter(elements))  # type: ignore[arg-type]
            if len(elements) == 1 and not extra_elements
            else "unknown"
        ),
        completion_action=action,
        modifiers=modifiers,
        parse_status=status,
    )


def parse_requirements(
    text: str | None, *, kind: TextKind, source: RequirementSource
) -> list[ObjectiveRequirement]:
    """Requirements stated in one Manifest text, each carrying the text's provenance.

    A description yields one entry per requirement sentence (bonus-progress sentences attach
    to the first as raw text). A label yields at most one entry. Text with no recognizable
    requirement yields none.
    """

    if not text or not text.strip():
        return []
    if kind == "objective_label":
        parsed = _parse_label(text, source)
        return [parsed] if parsed else []
    sentences = [value.strip() for value in _SENTENCES.split(text) if value.strip()]
    bonus = [value for value in sentences if _BONUS.search(value)]
    found: list[ObjectiveRequirement] = []
    for sentence in sentences:
        cleaned = clean(sentence)
        if sentence in bonus or _DESCRIPTIVE.search(cleaned) or not _GATE.search(cleaned):
            continue
        parsed = _parse_sentence(sentence, source, bonus if not found else [])
        if parsed is not None:
            found.append(parsed)
        if len(found) >= MAX_REQUIREMENTS_PER_TEXT:
            break
    if len(found) > 1:
        # Several requirement sentences in one text: reported separately, never combined.
        pass
    return found


def objective_definition_requirement(
    objective_hash: int, definition: dict[str, Any]
) -> ObjectiveRequirement | None:
    """A restrictive requirement from an objective definition's own text, else None.

    The label (name or progressDescription) is tried first, then the description. Generic text
    ("Targets defeated") is not returned: absence means nothing restrictive was stated.
    """

    if not definition:
        return None
    display = definition.get("displayProperties") or {}
    label_field = "displayProperties.name" if display.get("name") else "progressDescription"
    label = str(display.get("name") or definition.get("progressDescription") or "")
    for field, text, kind in (
        (label_field, label, "objective_label"),
        ("displayProperties.description", display.get("description"), "description"),
    ):
        source = RequirementSource(
            definition="DestinyObjectiveDefinition", hash=objective_hash, field=field
        )
        for requirement in parse_requirements(text, kind=kind, source=source):  # type: ignore[arg-type]
            if is_restrictive(requirement):
                return requirement
    return None


def item_description_requirements(
    item_hash: int, definition: dict[str, Any]
) -> list[ObjectiveRequirement]:
    """Requirements stated in a quest, step, bounty or order item's own description."""

    if not item_hash or not definition:
        return []
    source = RequirementSource(
        definition="DestinyInventoryItemDefinition",
        hash=item_hash,
        field="displayProperties.description",
    )
    text = (definition.get("displayProperties") or {}).get("description")
    return parse_requirements(text, kind="description", source=source)


def is_restrictive(requirement: ObjectiveRequirement) -> bool:
    """True when the requirement names a source, element, modifier or a non-generic action."""

    return bool(
        requirement.source_kind != "unknown"
        or requirement.element != "unknown"
        or requirement.modifiers
        or requirement.completion_action in ("final_blow", "damage")
    )


_OBJECTIVE_DEFINITION = "DestinyObjectiveDefinition"


def _compare(left: ObjectiveRequirement, right: ObjectiveRequirement) -> RequirementConsistency:
    for attribute in ("source_kind", "element"):
        first, second = getattr(left, attribute), getattr(right, attribute)
        if first != "unknown" and second != "unknown" and first != second:
            return "conflicting"
    actions = {left.completion_action, right.completion_action} - {"unknown"}
    if actions == {"defeat", "final_blow"}:
        return "differs_in_specificity"
    if len(actions) > 1:
        return "conflicting"
    return "consistent"


def requirements_consistency(
    requirements: list[ObjectiveRequirement],
) -> RequirementConsistency | None:
    """Whether a quest's description and its objectives' own text agree.

    Only an item description and an objective definition are compared, never objectives with
    each other: a quest with an Arc objective and a Solar objective has two objectives, not a
    contradiction. An objective conflicts only if it conflicts with every usable description
    requirement.

    * `conflicting`: stated sources or elements differ, or the actions are incompatible;
    * `differs_in_specificity`: one says "defeat" where the other says "final blows". Both
      are kept and neither is promoted;
    * `consistent`: whatever both state matches (a source silent on a field never disagrees).

    None when there is nothing to compare. Sources are never merged.
    """

    usable = [value for value in requirements if value.parse_status != "ambiguous"]
    descriptions = [value for value in usable if value.source.definition != _OBJECTIVE_DEFINITION]
    objectives = [value for value in usable if value.source.definition == _OBJECTIVE_DEFINITION]
    if not descriptions or not objectives:
        return None
    outcomes: list[RequirementConsistency] = []
    for objective in objectives:
        compared = [_compare(description, objective) for description in descriptions]
        if "consistent" in compared:
            outcomes.append("consistent")
        elif "differs_in_specificity" in compared:
            outcomes.append("differs_in_specificity")
        else:
            outcomes.append("conflicting")
    if "conflicting" in outcomes:
        return "conflicting"
    if "differs_in_specificity" in outcomes:
        return "differs_in_specificity"
    return "consistent"


def compact_requirement(
    requirement: ObjectiveRequirement, *, include_raw: bool = True
) -> dict[str, Any]:
    """Tool-facing form: stated fields, provenance and (optionally) the original text.

    `include_raw=False` is for results that already carry the source text (a quest's own
    description): the sentence and bonus clauses are then not repeated, and only the number of
    bonus clauses is noted. Provenance is always kept, as "<Definition>:<hash>:<field>".
    """

    data = requirement.model_dump(mode="json")
    source = requirement.source
    data["source"] = f"{source.definition}:{source.hash}:{source.field}"
    if not include_raw:
        data.pop("raw_requirement", None)
        if bonus := data.pop("bonus_conditions", None):
            data["bonus_clauses"] = len(bonus)
    return data
