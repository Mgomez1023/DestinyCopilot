"""Bounded, factual build views over normalized Guardian inventory data."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models import CharacterSummary, ChatTurn, GuardianContext, ItemSummary, SocketedPlugSummary
from app.session_preferences import SessionPreferenceContext

MAX_LOCKED_ITEMS = 8
MAX_REQUIRED_PERKS = 6
MAX_ALTERNATIVES = 8
MAX_BUILD_ITEMS = 12
MAX_SOCKETED_PLUGS = 16

InventoryLocation = Literal["equipped", "character", "vault", "profile"]


def _nullable_string(description: str) -> dict[str, Any]:
    return {"type": ["string", "null"], "description": description}


def _strict_tool(name: str, description: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    }


BUILD_TOOL_DEFINITIONS = [
    _strict_tool(
        "analyze_current_build",
        (
            "Inspect one character's normalized equipped build, explicit constraints, factual "
            "coverage observations, data gaps, and external knowledge needs. Produces no score."
        ),
        {
            "character_id": {"type": "string", "description": "Required character ID."},
            "goal": _nullable_string("Build goal such as solo PvE or boss DPS, or null."),
            "activity": _nullable_string("Named activity or encounter, or null."),
            "locked_items": {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "maxItems": MAX_LOCKED_ITEMS,
                "description": "Items the user explicitly said to preserve, or null.",
            },
            "preserve_exotics": {
                "type": "boolean",
                "description": "Preserve currently equipped Exotic items.",
            },
        },
    ),
    _strict_tool(
        "find_build_alternatives",
        (
            "Find a bounded factual shortlist of owned weapon or armor copies for a build. "
            "Filters actual normalized instance rolls and enforces preserved Exotic constraints."
        ),
        {
            "character_id": {"type": "string", "description": "Required character ID."},
            "slot": _nullable_string("Equipment bucket/slot filter, or null."),
            "item_type": _nullable_string("Weapon or Armor, or null."),
            "subtype": _nullable_string("Weapon or armor subtype, or null."),
            "damage_type": _nullable_string("Damage type such as Solar, or null."),
            "rarity": _nullable_string("Rarity such as Exotic or Legendary, or null."),
            "required_perks": {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "maxItems": MAX_REQUIRED_PERKS,
                "description": "Perk/socket names every returned owned roll must contain.",
            },
            "exotic": {
                "type": ["boolean", "null"],
                "description": "True for Exotic only, false for non-Exotic, or null.",
            },
            "equipped": {
                "type": ["boolean", "null"],
                "description": "Equipped-state filter, or null.",
            },
            "locations": {
                "type": ["array", "null"],
                "items": {
                    "type": "string",
                    "enum": ["equipped", "character", "vault", "profile"],
                },
                "maxItems": 4,
                "description": "Allowed owned-item locations, or null for all in-scope locations.",
            },
            "preserve_exotics": {
                "type": "boolean",
                "description": "Exclude candidates incompatible with equipped Exotics.",
            },
            "locked_items": {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "maxItems": MAX_LOCKED_ITEMS,
                "description": "Equipped items the user explicitly said not to replace.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_ALTERNATIVES,
                "description": "Maximum owned copies returned.",
            },
        },
    ),
]


class BuildRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnalyzeCurrentBuildRequest(BuildRequest):
    character_id: str = Field(min_length=1, max_length=128)
    goal: str | None = Field(default=None, max_length=160)
    activity: str | None = Field(default=None, max_length=160)
    locked_items: list[str] | None = Field(default=None, max_length=MAX_LOCKED_ITEMS)
    preserve_exotics: bool = False

    @field_validator("locked_items")
    @classmethod
    def normalize_locked_items(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        return list(dict.fromkeys(item.strip() for item in value if item.strip()))


class FindBuildAlternativesRequest(BuildRequest):
    character_id: str = Field(min_length=1, max_length=128)
    slot: str | None = Field(default=None, max_length=80)
    item_type: str | None = Field(default=None, max_length=80)
    subtype: str | None = Field(default=None, max_length=80)
    damage_type: str | None = Field(default=None, max_length=80)
    rarity: str | None = Field(default=None, max_length=80)
    required_perks: list[str] | None = Field(default=None, max_length=MAX_REQUIRED_PERKS)
    exotic: bool | None = None
    equipped: bool | None = None
    locations: list[InventoryLocation] | None = Field(default=None, max_length=4)
    preserve_exotics: bool = True
    locked_items: list[str] | None = Field(default=None, max_length=MAX_LOCKED_ITEMS)
    limit: int = Field(default=5, ge=1, le=MAX_ALTERNATIVES)

    @field_validator("required_perks", "locked_items")
    @classmethod
    def normalize_string_lists(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        return list(dict.fromkeys(item.strip() for item in value if item.strip()))


class BuildRequestContext(BaseModel):
    """Small application-derived constraint view included in build turns."""

    is_build_request: bool
    goal: str | None = None
    activity: str | None = None
    locked_items: list[str] = Field(default_factory=list, max_length=MAX_LOCKED_ITEMS)
    preserve_equipped_exotics: bool = False
    activity_mode: str = "unspecified"
    fireteam: str = "unspecified"
    intensity: str = "unspecified"
    normal_change_limit: int | None = Field(default=3, ge=1, le=12)


_BUILD_REQUEST = re.compile(
    r"\b(?:build|loadout|boss dps|add clear|survivability|solo pve|"
    r"what should i replace|do i own a better|prep(?:are)? me for|"
    r"what should i run for|don'?t change my exotic|use what i own|"
    r"compare my\b.{0,80}\brolls?)\b",
    re.IGNORECASE,
)
_DETAILED_REBUILD = re.compile(
    r"\b(?:complete|full|detailed|in[- ]depth)\b.{0,30}\b(?:rebuild|build|loadout)\b|"
    r"\b(?:rebuild|build|loadout)\b.{0,30}\b(?:complete|full|detailed|in[- ]depth)\b",
    re.IGNORECASE,
)
_NAMED_LOCK = re.compile(
    r"\b(?:build around|keep|preserve|don'?t change|do not change)\s+(?:my\s+)?"
    r"(?P<name>[A-Z][A-Za-z0-9' -]{1,60}?)(?=[,.!?]|\s+(?:and|but|for|while)\b|$)",
    re.IGNORECASE,
)
_ACTIVITY = re.compile(
    r"\b(?:prep(?:are)? me for|what should i run for|make this (?:good|better) for)\s+"
    r"(?P<activity>[^?.!]{2,100})",
    re.IGNORECASE,
)
_PRESERVE_EXOTIC = re.compile(
    r"\b(?:don'?t|do not) change my exotic\b|\bwithout changing my exotic\b|"
    r"\bkeep my exotic\b|\bpreserve my exotic\b",
    re.IGNORECASE,
)
_RELEASE_EXOTIC = re.compile(
    r"\b(?:you can|feel free to) (?:change|replace|remove) my exotic\b|"
    r"\bmy exotic (?:can|may) change\b|\bdon'?t (?:need to )?keep my exotic\b",
    re.IGNORECASE,
)
_RELEASE_NAMED = re.compile(
    r"\b(?:you can|feel free to) (?:change|replace|remove)\s+"
    r"(?P<name>[A-Z][A-Za-z0-9' -]{1,60}?)(?=[,.!?]|\s+now\b|$)",
    re.IGNORECASE,
)


def derive_build_request_context(
    message: str,
    preferences: SessionPreferenceContext,
    history: Sequence[ChatTurn] = (),
) -> BuildRequestContext:
    normalized = message.casefold()
    conversation = [turn.content for turn in history if turn.role == "user"] + [message]
    active_locks: dict[str, str] = {}
    preserve_exotics = False
    for value in conversation:
        for match in _NAMED_LOCK.finditer(value):
            name = match.group("name").strip()
            if name.casefold() not in {"exotic", "my exotic"}:
                active_locks[_normalized(name)] = name
        for match in _RELEASE_NAMED.finditer(value):
            active_locks.pop(_normalized(match.group("name")), None)
        if _PRESERVE_EXOTIC.search(value):
            preserve_exotics = True
        if _RELEASE_EXOTIC.search(value):
            preserve_exotics = False
    is_build = bool(
        _BUILD_REQUEST.search(message)
        or preferences.primary_goal == "build_improvement"
        or active_locks
        or preserve_exotics
    )
    goal: str | None = None
    goal_markers = (
        ("boss_dps", ("boss dps", "single-target", "single target")),
        ("solo_pve", ("solo pve", "solo pve", "solo")),
        ("pvp", ("pvp", "crucible")),
        ("add_clear", ("add clear", "ad clear")),
        ("survivability", ("survivability", "survive", "tank")),
        ("group_support", ("group support", "team support", "support build")),
        ("casual", ("easy to use", "chill", "casual")),
    )
    for candidate, markers in goal_markers:
        if any(marker in normalized for marker in markers):
            goal = candidate
            break
    if goal is None and preferences.primary_goal == "build_improvement":
        goal = "build_improvement"

    activity_match = _ACTIVITY.search(message)
    activity = activity_match.group("activity").strip() if activity_match else None
    if _normalized(activity) in {
        "solo pve",
        "boss dps",
        "add clear",
        "survivability",
        "pvp",
    }:
        activity = None
    locks = list(active_locks.values())
    return BuildRequestContext(
        is_build_request=is_build,
        goal=goal,
        activity=activity,
        locked_items=list(dict.fromkeys(locks))[:MAX_LOCKED_ITEMS],
        preserve_equipped_exotics=preserve_exotics,
        activity_mode=preferences.activity_mode,
        fireteam=preferences.fireteam,
        intensity=preferences.intensity,
        normal_change_limit=None if _DETAILED_REBUILD.search(message) else 3,
    )


def _normalized(value: str | None) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (value or "").casefold()))


def _matches(value: str | None, expected: str | None) -> bool:
    return expected is None or _normalized(expected) in _normalized(value)


def _plug(value: SocketedPlugSummary) -> dict[str, Any]:
    return {
        "name": value.name,
        "description": value.description[:240] if value.description else None,
        "type": value.item_type,
        "category": value.category_identifier,
    }


def _plug_groups(plugs: list[SocketedPlugSummary]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {
        "abilities": [],
        "aspects": [],
        "fragments": [],
        "mods": [],
        "other": [],
    }
    for value in plugs[:MAX_SOCKETED_PLUGS]:
        category = (value.category_identifier or "").casefold()
        item_type = (value.item_type or "").casefold()
        if "aspect" in category:
            key = "aspects"
        elif "fragment" in category:
            key = "fragments"
        elif any(marker in category for marker in ("abilit", "grenade", "melee", "super", "jump")):
            key = "abilities"
        elif item_type == "mod" or "mod" in category:
            key = "mods"
        else:
            key = "other"
        groups[key].append(_plug(value))
    return groups


def _data_state(explicit: bool | None, values_present: bool) -> str:
    if explicit is True:
        return "available"
    if explicit is False:
        return "not_returned"
    return "available" if values_present else "unknown"


def _item(value: ItemSummary, *, copy_number: int | None = None) -> dict[str, Any]:
    groups = _plug_groups(value.socketed_plug_details)
    result: dict[str, Any] = {
        "name": value.name,
        "item_type": value.item_type,
        "subtype": value.item_subtype,
        "slot": value.bucket_name,
        "rarity": value.tier,
        "damage_type": value.damage_type,
        "location": value.location,
        "equipped": value.is_equipped,
        "exotic": value.tier == "Exotic",
        "locked_in_game": value.is_locked,
        "crafted": value.is_crafted,
        "stats": {stat.name: stat.value for stat in value.stats},
        "socketed_plugs": [
            _plug(plug) for plug in value.socketed_plug_details[:MAX_SOCKETED_PLUGS]
        ],
        "socketed_plug_names": value.socketed_plugs[:MAX_SOCKETED_PLUGS],
        "relevant_perks": [
            plug
            for plug in groups["other"]
            if any(
                marker in (plug.get("category") or "").casefold()
                for marker in ("perk", "intrinsic", "trait")
            )
            or (plug.get("type") or "").casefold() in {"perk", "trait"}
        ],
        "data_quality": {
            "manifest": (
                "unresolved"
                if value.name == "Unknown" or value.item_type == "Unknown"
                else "resolved"
            ),
            "instance": _data_state(value.instance_data_available, bool(value.instance_id)),
            "stats": _data_state(value.stat_data_available, bool(value.stats)),
            "sockets": _data_state(
                value.socket_data_available,
                bool(value.socketed_plug_details or value.socketed_plugs),
            ),
            "socket_count": value.socket_count,
            "empty_socket_count": value.empty_socket_count,
            "roll_or_perk_data": _data_state(
                value.socket_data_available,
                bool(value.socketed_plug_details or value.socketed_plugs),
            ),
        },
    }
    if copy_number is not None:
        result["copy_number"] = copy_number
    return result


def _equipped_exotics(character: CharacterSummary) -> list[ItemSummary]:
    return [
        item
        for item in character.equipped_gear
        if item.tier == "Exotic" and item.item_type in {"Weapon", "Armor"}
    ]


class BuildAnalysisError(ValueError):
    pass


class BuildAnalysisService:
    """Build facts and owned alternatives without a universal quality score."""

    def __init__(self, context: GuardianContext) -> None:
        self.context = context

    def _character(self, character_id: str) -> CharacterSummary:
        for character in self.context.characters:
            if character.character_id == character_id:
                return character
        raise BuildAnalysisError("The requested character was not found in Guardian data.")

    def analyze_current_build(self, request: AnalyzeCurrentBuildRequest) -> dict[str, Any]:
        character = self._character(request.character_id)
        items = [item for item in character.equipped_gear if item.item_type in {"Weapon", "Armor"}][
            :MAX_BUILD_ITEMS
        ]
        weapons = [item for item in items if item.item_type == "Weapon"]
        armor = [item for item in items if item.item_type == "Armor"]
        exotics = _equipped_exotics(character)
        locked_names = list(request.locked_items or [])
        if request.preserve_exotics:
            locked_names.extend(item.name for item in exotics)
        locked_names = list(dict.fromkeys(locked_names))[:MAX_LOCKED_ITEMS]
        scoped_owned_items = [
            *character.equipped_gear,
            *[
                item
                for item in self.context.inventory.items
                if item.location in {"vault", "profile"}
                or item.character_id == request.character_id
            ],
        ]
        locked_items = []
        for name in locked_names:
            matches = [
                item for item in scoped_owned_items if _normalized(item.name) == _normalized(name)
            ]
            locked_items.append(
                {
                    "name": name,
                    "ownership": "verified_owned" if matches else "not_found_in_returned_inventory",
                    "equipped": any(item.is_equipped for item in matches),
                }
            )

        armor_stats: Counter[str] = Counter()
        for item in armor:
            armor_stats.update({stat.name: stat.value for stat in item.stats})
        subtype_counts = Counter(item.item_subtype for item in weapons if item.item_subtype)
        duplicate_roles = [
            {"subtype": subtype, "count": count}
            for subtype, count in subtype_counts.items()
            if count > 1
        ]
        empty_sockets = sum(
            item.empty_socket_count or 0 for item in [*items, character.subclass] if item
        )
        missing_roll_data = [
            item.name
            for item in weapons
            if _data_state(
                item.socket_data_available,
                bool(item.socketed_plug_details or item.socketed_plugs),
            )
            != "available"
        ]
        weapon_exotics = [item.name for item in exotics if item.item_type == "Weapon"]
        armor_exotics = [item.name for item in exotics if item.item_type == "Armor"]
        violations = []
        if len(weapon_exotics) > 1:
            violations.append("more_than_one_exotic_weapon_equipped")
        if len(armor_exotics) > 1:
            violations.append("more_than_one_exotic_armor_piece_equipped")

        requested_text = " ".join(value for value in (request.goal, request.activity) if value)
        volatile = bool(
            re.search(
                r"\b(?:meta|best|strongest|nerf|buff|artifact|this season|current)\b",
                requested_text,
                re.IGNORECASE,
            )
        )
        limitations = list(self.context.data_availability.notes[:4])
        if missing_roll_data:
            limitations.append(
                "Actual socket/perk data was not returned for some equipped weapons; their roll "
                "quality cannot be assessed."
            )
        if any(item.empty_socket_count is None for item in items):
            limitations.append(
                "Empty sockets cannot be determined for every equipped item from the returned "
                "components."
            )
        if any(item.name == "Unknown" or item.item_type == "Unknown" for item in items):
            limitations.append(
                "Manifest identity or type resolution failed for at least one equipped item; "
                "that item is not interpreted further."
            )

        subclass_groups = (
            _plug_groups(character.subclass.socketed_plug_details) if character.subclass else None
        )
        armor_mods = [
            plug for item in armor for plug in _plug_groups(item.socketed_plug_details)["mods"]
        ][:MAX_SOCKETED_PLUGS]

        return {
            "character_class": character.class_name,
            "goal": request.goal,
            "activity": request.activity,
            "subclass": (
                {
                    "name": character.subclass.name,
                    "abilities": subclass_groups["abilities"],
                    "aspects": subclass_groups["aspects"],
                    "fragments": subclass_groups["fragments"],
                    "other_socketed_configuration": subclass_groups["other"],
                    "data_quality": _item(character.subclass)["data_quality"],
                }
                if character.subclass
                else None
            ),
            "equipped_weapons": [_item(item) for item in weapons],
            "equipped_armor": [_item(item) for item in armor],
            "equipped_exotics": {
                "weapon": weapon_exotics,
                "armor": armor_exotics,
                "constraint_violations": violations,
            },
            "armor_stat_totals": dict(sorted(armor_stats.items())),
            "armor_mods": armor_mods,
            "locked_items": locked_items,
            "observations": {
                "duplicate_weapon_subtypes": duplicate_roles,
                "observable_empty_sockets": empty_sockets,
                "weapons_missing_roll_data": missing_roll_data,
                "damage_types": list(
                    dict.fromkeys(item.damage_type for item in weapons if item.damage_type)
                ),
                "weapon_slots": list(
                    dict.fromkeys(item.bucket_name for item in weapons if item.bucket_name)
                ),
            },
            "supported_dimensions": [
                value
                for value, supported in (
                    ("armor_stats", bool(armor_stats)),
                    ("weapon_slot_coverage", any(item.bucket_name for item in weapons)),
                    ("elemental_coverage", any(item.damage_type for item in weapons)),
                    ("owned_rolls", len(missing_roll_data) < len(weapons)),
                    ("exotic_compatibility", True),
                    ("socket_completeness", any(item.socket_count is not None for item in items)),
                )
                if supported
            ],
            "knowledge_needed": {
                "manifest": any(
                    item.name == "Unknown" or item.item_type == "Unknown" for item in items
                ),
                "guide": bool(request.activity or request.goal),
                "web_or_live_for_current_claims": volatile,
                "reason": (
                    "Use Guide/Web for exact interactions or encounter needs; use Live/Web for "
                    "current meta, patch, artifact, buff, or nerf claims."
                ),
            },
            "limitations": limitations[:8],
            "scoring": "No universal numeric or tier score is produced.",
        }

    def find_build_alternatives(self, request: FindBuildAlternativesRequest) -> dict[str, Any]:
        character = self._character(request.character_id)
        equipped_exotics = _equipped_exotics(character)
        locked = {_normalized(value) for value in request.locked_items or []}
        if request.preserve_exotics:
            locked.update(_normalized(item.name) for item in equipped_exotics)

        all_items = list(self.context.inventory.items)
        all_items.extend(character.equipped_gear)
        scoped = [
            item
            for item in all_items
            if item.location in {"vault", "profile"} or item.character_id == request.character_id
        ]
        locked_equipped = [
            item
            for item in character.equipped_gear
            if _normalized(item.name) in locked
            and _matches(item.bucket_name, request.slot)
            and _matches(item.item_type, request.item_type)
        ]

        def perk_matches(item: ItemSummary) -> bool:
            if not request.required_perks:
                return True
            names = [_normalized(value) for value in item.socketed_plugs]
            return all(
                any(_normalized(required) in name for name in names)
                for required in request.required_perks
            )

        def exotic_compatible(item: ItemSummary) -> bool:
            if not request.preserve_exotics or item.tier != "Exotic":
                return True
            same_item = any(item.name == equipped.name for equipped in equipped_exotics)
            if same_item:
                return True
            return not any(equipped.item_type == item.item_type for equipped in equipped_exotics)

        filtered = (
            []
            if locked_equipped and request.slot
            else [
                item
                for item in scoped
                if item.item_type in {"Weapon", "Armor"}
                and _matches(item.bucket_name, request.slot)
                and _matches(item.item_type, request.item_type)
                and _matches(item.item_subtype, request.subtype)
                and _matches(item.damage_type, request.damage_type)
                and _matches(item.tier, request.rarity)
                and (request.exotic is None or (item.tier == "Exotic") is request.exotic)
                and (request.equipped is None or item.is_equipped is request.equipped)
                and (request.locations is None or item.location in request.locations)
                and perk_matches(item)
                and exotic_compatible(item)
            ]
        )
        filtered.sort(
            key=lambda item: (
                _normalized(item.name),
                {"equipped": 0, "character": 1, "vault": 2, "profile": 3}.get(item.location, 4),
                tuple(_normalized(value) for value in item.socketed_plugs),
            )
        )
        copy_counts: Counter[str] = Counter()
        candidates = []
        for item in filtered[: request.limit]:
            copy_counts[item.name] += 1
            candidate = _item(item, copy_number=copy_counts[item.name])
            candidate["preserves_locked_items"] = (
                _normalized(item.name) not in locked or item.is_equipped
            )
            candidates.append(candidate)

        missing_perk_data = sum(
            _data_state(
                item.socket_data_available,
                bool(item.socketed_plug_details or item.socketed_plugs),
            )
            != "available"
            for item in filtered
        )
        limitations = []
        unresolved_items = [
            item for item in scoped if item.name == "Unknown" or item.item_type == "Unknown"
        ]
        if unresolved_items:
            limitations.append(
                "Owned entries with unresolved Manifest identity or type were excluded from the "
                "candidate shortlist."
            )
        if locked_equipped and request.slot:
            limitations.append(
                "The requested slot contains an explicitly preserved equipped item, so no "
                "replacement candidates were returned."
            )
        if request.required_perks and not filtered:
            limitations.append(
                "No owned copy with returned socket data matched every requested perk. Missing "
                "socket data is not treated as a match."
            )
        if missing_perk_data:
            limitations.append(
                "Some matching owned copies lack returned socket/perk data and cannot be judged "
                "by roll quality."
            )
        limitations.extend(self.context.data_availability.notes[:3])
        limitations.extend(
            f"Guardian component unavailable: {name}"
            for name in list(self.context.data_availability.unavailable_components)[:3]
        )
        return {
            "character_class": character.class_name,
            "filters": {
                "slot": request.slot,
                "item_type": request.item_type,
                "subtype": request.subtype,
                "damage_type": request.damage_type,
                "rarity": request.rarity,
                "required_perks": request.required_perks or [],
                "exotic": request.exotic,
                "equipped": request.equipped,
                "locations": request.locations,
            },
            "preservation": {
                "preserve_exotics": request.preserve_exotics,
                "locked_items": [
                    item.name
                    for item in character.equipped_gear
                    if _normalized(item.name) in locked
                ],
                "equipped_exotic_weapon": [
                    item.name for item in equipped_exotics if item.item_type == "Weapon"
                ],
                "equipped_exotic_armor": [
                    item.name for item in equipped_exotics if item.item_type == "Armor"
                ],
            },
            "candidates": candidates,
            "total_matching_owned_copies": len(filtered),
            "returned": len(candidates),
            "limit": request.limit,
            "truncated": len(filtered) > request.limit,
            "ranking": (
                "No universal quality score was used; candidates are ordered by name and location."
            ),
            "limitations": limitations[:8],
        }
