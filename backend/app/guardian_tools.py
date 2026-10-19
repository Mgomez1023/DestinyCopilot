from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.build_analysis import (
    BUILD_TOOL_DEFINITIONS,
    AnalyzeCurrentBuildRequest,
    BuildAnalysisService,
    FindBuildAlternativesRequest,
    class_compatibility,
    equipped_plug_effects,
    plug_groups,
)
from app.character_selection import (
    CharacterClass,
    CharacterNotFoundError,
    GuardianCharacterResolver,
    GuardianToolError,
    resolve_current_character,
)
from app.content_progression import ContentProgressionResolver
from app.mechanics_evidence import equipped_mechanics_section
from app.models import (
    CharacterSummary,
    GuardianContext,
    ItemSummary,
    ProgressionSummary,
    SocketedPlugSummary,
)
from app.objective_requirements import compact_requirement, is_restrictive
from app.requirement_fit import (
    REQUIREMENT_BASIS,
    LoadoutEvidence,
    compare_requirement,
    loadout_evidence,
)

__all__ = ["CharacterNotFoundError", "GuardianToolError", "GuardianToolService"]


class ToolRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OptionalCharacterRequest(ToolRequest):
    character_id: str | None = None
    character_class: CharacterClass | None = None


class CharacterRequest(OptionalCharacterRequest):
    pass


class ContentProgressionRequest(CharacterRequest):
    content_name: str | None


class RecentActivitiesRequest(OptionalCharacterRequest):
    limit: int = Field(default=10, ge=1, le=25)


class InventorySearchRequest(OptionalCharacterRequest):
    query: str | None = None
    item_type: str | None = None
    subtype: str | None = None
    bucket: str | None = None
    equipped_only: bool = False
    limit: int = Field(default=25, ge=1, le=100)


class TitleProgressRequest(ToolRequest):
    title_name: str | None = Field(default=None, min_length=1, max_length=100)
    limit: int = Field(default=5, ge=1, le=10)


class ToolInvocationResponse(BaseModel):
    tool_name: str
    result: dict[str, Any]


class UnknownGuardianToolError(GuardianToolError):
    pass


def _nullable_string(description: str) -> dict[str, Any]:
    return {"type": ["string", "null"], "description": description}


CURRENT_CHARACTER_NOTE = (
    "When both selectors are null, the player's current (most recently played) character is "
    "used; pass a selector only when the user names a character."
)
# Character-scoped tools that fall back to the current character when no selector is given.
CURRENT_DEFAULT_TOOLS = frozenset(
    {
        "get_equipped_loadout",
        "get_build_details",
        "analyze_current_build",
        "find_build_alternatives",
    }
)


# Per-character views whose unspecified selector means the conversation's explicit character.
CONVERSATION_SCOPED_TOOLS = frozenset(
    {"get_active_quests", "get_available_activities", "get_recent_activities"}
)


def _character_selector(*, optional: bool, current_default: bool = False) -> dict[str, Any]:
    scope = "or null for every character" if optional else "or null when character_class is used"
    if current_default:
        scope = "or null to use the current character"
    class_note = (
        "Character class selector, or null. Do not provide this together with character_id."
    )
    if current_default:
        class_note = f"{class_note} {CURRENT_CHARACTER_NOTE}"
    return {
        "character_id": _nullable_string(f"Opaque character ID, {scope}."),
        "character_class": {
            "type": ["string", "null"],
            "enum": ["Titan", "Hunter", "Warlock", None],
            "description": class_note,
        },
    }


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


GUARDIAN_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    _strict_tool(
        "get_character_summary",
        "Get compact character identity, equipped gear names, subclass, and progression.",
        _character_selector(optional=True),
    ),
    _strict_tool(
        "get_equipped_loadout",
        "Get one character's equipped weapons, armor, subclass, stats, and socketed perks. "
        "Defaults to the current character.",
        _character_selector(optional=False, current_default=True),
    ),
    _strict_tool(
        "get_active_quests",
        "Get active quests and their current, resolved objective progress.",
        _character_selector(optional=True),
    ),
    _strict_tool(
        "get_content_progression",
        (
            "Resolve conservative major campaign/content progress for one character using "
            "maintained Bungie-backed evidence rules. Prefer this over reconstructing campaign "
            "completion from generic quest data."
        ),
        {
            **_character_selector(optional=False),
            "content_name": _nullable_string(
                "Supported campaign/content name, alias, or null for every supported line."
            ),
        },
    ),
    _strict_tool(
        "get_available_activities",
        "Get compact, deduplicated activities currently visible to the character(s).",
        _character_selector(optional=True),
    ),
    _strict_tool(
        "get_recent_activities",
        "Get recent activity history, sorted newest first.",
        {
            **_character_selector(optional=True),
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 25,
                "description": "Maximum results to return.",
            },
        },
    ),
    _strict_tool(
        "get_progression",
        "Get season, reputation, milestone, quest, record, collection, and crafting progress.",
        _character_selector(optional=True),
    ),
    _strict_tool(
        "get_title_progress",
        (
            "Get bounded account-wide Destiny Title/Seal Triumph progress, ranked closest first "
            "by fewest remaining required records and then observed objective progress. Use for "
            "personal title completion questions. This does not estimate time or effort."
        ),
        {
            "title_name": _nullable_string(
                "Optional case-insensitive Title/Seal name filter, or null for closest titles."
            ),
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "description": "Maximum Title/Seal summaries to return.",
            },
        },
    ),
    _strict_tool(
        "search_inventory",
        "Search normalized owned and equipped items using name and structured filters.",
        {
            "query": _nullable_string("Case-insensitive item-name substring, or null."),
            **_character_selector(optional=True),
            "item_type": _nullable_string("Item type filter such as Weapon or Armor, or null."),
            "subtype": _nullable_string("Subtype filter such as Auto Rifle, or null."),
            "bucket": _nullable_string(
                "Equipment-slot filter such as Helmet or Energy Weapons, or null. Vaulted items "
                "are matched by their definition slot, not by the vault."
            ),
            "equipped_only": {
                "type": "boolean",
                "description": "Whether to return only equipped items.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "description": "Maximum results to return.",
            },
        },
    ),
    _strict_tool(
        "get_build_details",
        "Get a build-focused subclass, exotic, weapon perk, armor stat, and mod view. "
        "Defaults to the current character.",
        _character_selector(optional=False, current_default=True),
    ),
]


REQUEST_MODELS: dict[str, type[ToolRequest]] = {
    "get_character_summary": OptionalCharacterRequest,
    "get_equipped_loadout": CharacterRequest,
    "get_active_quests": OptionalCharacterRequest,
    "get_content_progression": ContentProgressionRequest,
    "get_available_activities": OptionalCharacterRequest,
    "get_recent_activities": RecentActivitiesRequest,
    "get_progression": OptionalCharacterRequest,
    "get_title_progress": TitleProgressRequest,
    "search_inventory": InventorySearchRequest,
    "get_build_details": CharacterRequest,
}


class GuardianToolService:
    """Read-only, bounded queries over one already-normalized GuardianContext."""

    def __init__(
        self, context: GuardianContext, active_character: CharacterClass | None = None
    ) -> None:
        self.context = context
        # The character the user explicitly chose for this conversation, if any. Calls that
        # name no selector use it instead of reverting to the most recently played character.
        self.active_character = active_character
        self.characters = GuardianCharacterResolver(context, active_character)

    @staticmethod
    def definitions() -> list[dict[str, Any]]:
        return [*GUARDIAN_TOOL_DEFINITIONS, *BUILD_TOOL_DEFINITIONS]

    def execute(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        if name == "analyze_current_build":
            request = AnalyzeCurrentBuildRequest.model_validate(arguments or {})
            return BuildAnalysisService(self.context, self.active_character).analyze_current_build(
                request
            )
        if name == "find_build_alternatives":
            request = FindBuildAlternativesRequest.model_validate(arguments or {})
            return BuildAnalysisService(
                self.context, self.active_character
            ).find_build_alternatives(request)
        request_model = REQUEST_MODELS.get(name)
        if request_model is None:
            raise UnknownGuardianToolError(f"Unknown Guardian tool: {name}")
        request = request_model.model_validate(arguments or {})
        if name == "get_character_summary":
            return self.get_character_summary(  # type: ignore[attr-defined]
                request.character_id, request.character_class
            )
        if name == "get_equipped_loadout":
            return self.get_equipped_loadout(  # type: ignore[attr-defined]
                request.character_id, request.character_class
            )
        if name == "get_active_quests":
            return self.get_active_quests(  # type: ignore[attr-defined]
                request.character_id, request.character_class
            )
        if name == "get_content_progression":
            progression = ContentProgressionRequest.model_validate(request.model_dump())
            return self.get_content_progression(
                progression.character_id,
                progression.content_name,
                progression.character_class,
            )
        if name == "get_available_activities":
            return self.get_available_activities(  # type: ignore[attr-defined]
                request.character_id, request.character_class
            )
        if name == "get_recent_activities":
            recent = RecentActivitiesRequest.model_validate(request.model_dump())
            return self.get_recent_activities(
                recent.character_id, recent.limit, recent.character_class
            )
        if name == "get_progression":
            return self.get_progression(  # type: ignore[attr-defined]
                request.character_id, request.character_class
            )
        if name == "get_title_progress":
            title_request = TitleProgressRequest.model_validate(request.model_dump())
            return self.get_title_progress(title_request)
        if name == "search_inventory":
            search = InventorySearchRequest.model_validate(request.model_dump())
            return self.search_inventory(search)
        build = CharacterRequest.model_validate(request.model_dump())
        return self.get_build_details(build.character_id, build.character_class)

    def _characters(
        self,
        character_id: str | None,
        character_class: CharacterClass | None = None,
        *,
        scoped_default: bool = False,
    ) -> list[CharacterSummary]:
        # Quest, activity, and recent-activity views are per character. Inside a conversation
        # scoped to one character, an unspecified selector means that character rather than
        # every character, so another character's quests cannot leak into the answer.
        if scoped_default and character_id is None and character_class is None:
            character_class = self.active_character
        return self.characters.resolve_many(
            character_id=character_id,
            character_class=character_class,
        )

    @staticmethod
    def _progression(value: ProgressionSummary) -> dict[str, Any]:
        return {
            "name": value.name,
            "scope": value.scope,
            "level": value.level,
            "level_cap": value.level_cap or None,
            "progress_to_next_level": value.progress_to_next_level,
            "next_level_at": value.next_level_at,
            "weekly_progress": value.weekly_progress,
            "weekly_limit": value.weekly_limit or None,
            "resets": value.current_reset_count,
        }

    @staticmethod
    def _item(value: ItemSummary, *, details: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": value.name,
            "item_type": value.item_type,
            "subtype": value.item_subtype,
            # The equipment slot from the item's definition, not where this copy is stored.
            "bucket": value.gear_slot(),
            "location": value.location,
            "character_id": value.character_id,
            "equipped": value.is_equipped,
            "tier": value.tier,
            "class_restriction": value.class_restriction,
            "damage_type": value.damage_type,
            "item_hash": value.item_hash,
            "instance_id": value.instance_id,
        }
        if value.item_type in {"Weapon", "Armor"} and value.slot_status() != "known":
            result["slot_status"] = value.slot_status()
        if details:
            result["stats"] = {stat.name: stat.value for stat in value.stats}
            result["perks_and_sockets"] = value.socketed_plugs[:16]
        return result

    @staticmethod
    def _plug_groups(
        plugs: list[SocketedPlugSummary],
    ) -> dict[str, list[str]]:
        groups = plug_groups(plugs)
        return {
            key: list(dict.fromkeys(plug.name for plug in groups[key]))
            for key in ("abilities", "aspects", "fragments", "mods")
        }

    def get_character_summary(
        self,
        character_id: str | None = None,
        character_class: CharacterClass | None = None,
    ) -> dict[str, Any]:
        characters = []
        current = resolve_current_character(self.context)
        current_id = current.character_id if current else None
        for character in self._characters(character_id, character_class):
            characters.append(
                {
                    "character_id": character.character_id,
                    "class": character.class_name,
                    "race": character.race_name,
                    "last_played": (
                        character.last_played.isoformat() if character.last_played else None
                    ),
                    "is_current": character.character_id == current_id,
                    "subclass": character.subclass.name if character.subclass else None,
                    "equipped_weapons": [
                        item.name for item in character.equipped_gear if item.item_type == "Weapon"
                    ],
                    "equipped_armor": [
                        item.name for item in character.equipped_gear if item.item_type == "Armor"
                    ],
                    "progression": [
                        self._progression(value) for value in character.progressions[:10]
                    ],
                }
            )
        return {
            "characters": characters,
            "current_character": current.class_name if current else None,
        }

    def get_equipped_loadout(
        self,
        character_id: str | None = None,
        character_class: CharacterClass | None = None,
    ) -> dict[str, Any]:
        character = self.characters.resolve_one(
            character_id=character_id,
            character_class=character_class,
            default_to_current=True,
        )
        weapons = [item for item in character.equipped_gear if item.item_type == "Weapon"]
        armor = [item for item in character.equipped_gear if item.item_type == "Armor"]
        subclass_groups = self._plug_groups(
            character.subclass.socketed_plug_details if character.subclass else []
        )
        armor_stats: dict[str, int] = {}
        for item in armor:
            for stat in item.stats:
                armor_stats[stat.name] = armor_stats.get(stat.name, 0) + stat.value
        return {
            "character": {
                "character_id": character.character_id,
                "class": character.class_name,
                "power_eligibility": "unknown_not_compared",
            },
            "subclass": {
                "name": character.subclass.name if character.subclass else None,
                **subclass_groups,
            },
            "weapons": [self._item(item) for item in weapons],
            "armor": [self._item(item) for item in armor],
            "armor_stats": armor_stats,
        }

    @staticmethod
    def _requirement_fields(quest: Any, loadout: LoadoutEvidence) -> dict[str, Any]:
        """Structured requirements parsed from Manifest text, only when there are any.

        The original sentence is repeated only where it is not already in this result: a
        verified requirement from the quest's own description is that description. Each
        restrictive or unestablished requirement is compared with this quest's character's
        equipped loadout (M4B); a clean generic requirement has nothing to compare.
        """

        if not quest.requirements:
            return {}
        entries: list[dict[str, Any]] = []
        for value in quest.requirements:
            entry = compact_requirement(
                value,
                include_raw=value.parse_status != "verified"
                or value.source.hash != quest.quest_hash,
            )
            if value.parse_status != "verified" or is_restrictive(value):
                entry["fit"] = compare_requirement(value, loadout).model_dump(mode="json")
            entries.append(entry)
        fields: dict[str, Any] = {"requirements": entries}
        if quest.requirements_consistency:
            fields["requirements_consistency"] = quest.requirements_consistency
        return fields

    def get_active_quests(
        self,
        character_id: str | None = None,
        character_class: CharacterClass | None = None,
    ) -> dict[str, Any]:
        quests: list[dict[str, Any]] = []
        for character in self._characters(character_id, character_class, scoped_default=True):
            loadout = loadout_evidence(character)
            for quest in character.quests:
                if quest.completed and quest.redeemed:
                    continue
                quests.append(
                    {
                        "character_id": character.character_id,
                        "character_class": character.class_name,
                        "quest_hash": quest.quest_hash,
                        "step_hash": quest.step_hash,
                        "name": quest.name,
                        "step_name": quest.step_name,
                        "description": quest.description,
                        "tracked": quest.tracked,
                        "completed": quest.completed,
                        "redeemed": quest.redeemed,
                        "objectives": [value.model_dump(mode="json") for value in quest.objectives],
                        "item_type_name": quest.item_type_name,
                        "reward_names": list(quest.reward_names),
                        "reward_total": quest.reward_total,
                        "rewards_unresolved": quest.rewards_unresolved,
                        **self._requirement_fields(quest, loadout),
                    }
                )
        quests.sort(
            key=lambda value: (
                value["tracked"],
                bool(value["objectives"]),
                not value["completed"],
                value["name"],
            ),
            reverse=True,
        )
        limit = 50
        result: dict[str, Any] = {
            "quests": quests[:limit],
            "count": len(quests),
            "truncated": len(quests) > limit,
        }
        if any(
            "fit" in requirement
            for value in quests[:limit]
            for requirement in value.get("requirements", [])
        ):
            result["requirement_fit_basis"] = REQUIREMENT_BASIS
        if any(value["reward_total"] for value in quests[:limit]):
            result["limitations"] = [
                "reward_names are labels declared in the Manifest quest or bounty definition "
                "(at most 4); they do not describe quantities, odds, or current reward value. "
                "rewards_unresolved counts declared rewards whose names were unavailable."
            ]
        return result

    def get_content_progression(
        self,
        character_id: str | None,
        content_name: str | None,
        character_class: CharacterClass | None = None,
    ) -> dict[str, Any]:
        character = self.characters.resolve_one(
            character_id=character_id,
            character_class=character_class,
        )
        result = ContentProgressionResolver(self.context).resolve(character, content_name)
        return result.model_dump(mode="json")

    def get_available_activities(
        self,
        character_id: str | None = None,
        character_class: CharacterClass | None = None,
    ) -> dict[str, Any]:
        merged: dict[int, dict[str, Any]] = {}
        for character in self._characters(character_id, character_class, scoped_default=True):
            for activity in character.available_activities:
                if not activity.is_visible:
                    continue
                current = merged.setdefault(
                    activity.activity_hash,
                    {
                        "activity_hash": activity.activity_hash,
                        "name": activity.name,
                        "description": activity.description,
                        "activity_type": activity.activity_type,
                        "destination": activity.destination,
                        "place": activity.place,
                        "difficulty": activity.difficulty,
                        "display_level": activity.display_level,
                        "power_eligibility": "unknown_not_compared",
                        "character_ids": [],
                        "new": False,
                        "complete": True,
                        "can_lead": False,
                        "can_join": False,
                        "launchable": False,
                        "objectives": [],
                    },
                )
                current["character_ids"].append(character.character_id)
                current["new"] = current["new"] or activity.is_new
                current["complete"] = current["complete"] and activity.is_completed
                current["can_lead"] = current["can_lead"] or activity.can_lead
                current["can_join"] = current["can_join"] or activity.can_join
                current["launchable"] = current["can_lead"]
                objectives = [value.model_dump(mode="json") for value in activity.objectives]
                if len(objectives) > len(current["objectives"]):
                    current["objectives"] = objectives
        activities = list(merged.values())
        activities.sort(
            key=lambda value: (
                bool(value["objectives"]),
                value["new"],
                not value["complete"],
                value["name"],
            ),
            reverse=True,
        )
        limit = 25
        return {
            "activities": activities[:limit],
            "total_matching": len(activities),
            "truncated": len(activities) > limit,
            "availability_scope": "guardian_character_activities",
            "current_rotation_authoritative": False,
            "limitations": [
                "These are activities Bungie returned for this Guardian. This does not establish "
                "the current weekly featured rotation, Nightfall, modifiers, or loot rotation."
            ],
        }

    def get_recent_activities(
        self,
        character_id: str | None = None,
        limit: int = 10,
        character_class: CharacterClass | None = None,
    ) -> dict[str, Any]:
        activities = [
            value
            for character in self._characters(character_id, character_class, scoped_default=True)
            for value in character.recent_activities
        ]
        activities.sort(
            key=lambda value: value.period or datetime.min.replace(tzinfo=UTC),
            reverse=True,
        )
        values = [
            {
                **value.model_dump(mode="json", exclude={"recommended_power"}),
                "power_eligibility": "unknown_not_compared",
            }
            for value in activities[:limit]
        ]
        return {
            "activities": values,
            "total_matching": len(activities),
            "limit": limit,
        }

    def get_progression(
        self,
        character_id: str | None = None,
        character_class: CharacterClass | None = None,
    ) -> dict[str, Any]:
        characters: list[dict[str, Any]] = []
        for character in self._characters(character_id, character_class):
            milestones = []
            for milestone in character.milestones[:12]:
                milestones.append(
                    {
                        "name": milestone.name,
                        "description": milestone.description,
                        "start_date": (
                            milestone.start_date.isoformat() if milestone.start_date else None
                        ),
                        "end_date": milestone.end_date.isoformat() if milestone.end_date else None,
                        "activities": milestone.activity_names,
                        "quests": milestone.quest_names,
                        "objectives": [
                            value.model_dump(mode="json") for value in milestone.objectives
                        ],
                    }
                )
            characters.append(
                {
                    "character_id": character.character_id,
                    "class": character.class_name,
                    "progressions": [
                        self._progression(value) for value in character.progressions[:20]
                    ],
                    "milestones": milestones,
                    "active_quest_count": sum(
                        not (value.completed and value.redeemed) for value in character.quests
                    ),
                }
            )
        return {
            "current_season": {
                "hash": self.context.current_season_hash,
                "name": self.context.current_season_name,
            },
            "profile_progressions": [
                self._progression(value) for value in self.context.profile_progressions
            ],
            "characters": characters,
            "collectibles": self.context.collectibles.model_dump(mode="json"),
            "records": self.context.records.model_dump(mode="json", exclude={"records", "titles"}),
            "crafting": self.context.crafting.model_dump(mode="json"),
        }

    def get_title_progress(self, request: TitleProgressRequest) -> dict[str, Any]:
        records = self.context.records
        if not records.title_data_available or not records.titles:
            return {
                "availability": "insufficient_data",
                "titles": [],
                "returned": 0,
                "limit": request.limit,
                "ranking_basis": (
                    "fewest remaining required records, then greatest observed remaining "
                    "objective progress"
                ),
                "limitations": [
                    "Usable Title/Seal presentation data was not available in the normalized "
                    "Guardian Records snapshot. No completion ranking can be established."
                ],
            }

        title_filter = request.title_name.casefold() if request.title_name else None
        matches = [
            value
            for value in records.titles
            if title_filter is None or title_filter in value.name.casefold()
        ]
        incomplete = [value for value in matches if value.completed is not True]
        complete = [value for value in matches if value.completed is True]

        def rank(value: Any) -> tuple[Any, ...]:
            return (
                value.remaining_records,
                -(value.remaining_objective_progress_percent or 0.0),
                -(value.completion_percent or 0.0),
                not value.data_complete,
                value.name.casefold(),
            )

        ranked = sorted(incomplete, key=rank) + sorted(complete, key=rank)
        selected = ranked[: request.limit]
        public_titles: list[dict[str, Any]] = []
        for title in selected:
            remaining = []
            for record in title.remaining[:8]:
                remaining.append(
                    {
                        "name": record.name,
                        "completion_state": (
                            "complete"
                            if record.completed is True
                            else "incomplete"
                            if record.completed is False
                            else "unknown"
                        ),
                        "objectives": [
                            {
                                "name": objective.name,
                                "progress": objective.progress,
                                "completion_value": objective.completion_value,
                                "progress_percent": objective.progress_percent,
                                "complete": objective.complete,
                            }
                            for objective in record.objectives
                        ],
                    }
                )
            public_titles.append(
                {
                    "title": title.name,
                    "completion_state": (
                        "complete"
                        if title.completed is True
                        else "incomplete"
                        if title.completed is False
                        else "unknown"
                    ),
                    "completed_records": title.completed_records,
                    "total_records": title.total_records,
                    "remaining_count": title.remaining_records,
                    "completion_percent": title.completion_percent,
                    "remaining_objective_progress_percent": (
                        title.remaining_objective_progress_percent
                    ),
                    "remaining_objectives": remaining,
                    "remaining_objectives_truncated": bool(
                        title.remaining_truncated or len(title.remaining) > len(remaining)
                    ),
                    "data_complete": title.data_complete,
                }
            )

        limitations: list[str] = [
            "Ranking measures observed completion proximity, not expected time or difficulty."
        ]
        if records.titles_truncated:
            limitations.append(
                "The normalized Title/Seal catalog was bounded, so titles outside it were not "
                "ranked."
            )
        if any(not value.data_complete for value in selected):
            limitations.append(
                "Some returned titles have missing or bounded Record state; preserve that "
                "uncertainty."
            )
        availability = "available" if public_titles else "no_match"
        if not public_titles and request.title_name:
            limitations.append("No normalized Title/Seal matched the requested name.")
        return {
            "availability": availability,
            "titles": public_titles,
            "returned": len(public_titles),
            "total_matching": len(ranked),
            "limit": request.limit,
            "truncated": len(ranked) > request.limit or records.titles_truncated,
            "ranking_basis": (
                "fewest remaining required records, then greatest observed remaining "
                "objective progress"
            ),
            "interpretation": (
                "Closest to completion based on current Triumph progress; actual time and effort "
                "may differ."
            ),
            "limitations": limitations,
        }

    def search_inventory(self, request: InventorySearchRequest) -> dict[str, Any]:
        selected: CharacterSummary | None = None
        if request.character_id is not None or request.character_class is not None:
            selected = self.characters.resolve_one(
                character_id=request.character_id,
                character_class=request.character_class,
            )
        items = list(self.context.inventory.items)
        items.extend(
            item for character in self.context.characters for item in character.equipped_gear
        )

        def matches(value: str | None, expected: str | None) -> bool:
            return expected is None or expected.casefold() in (value or "").casefold()

        def is_shared(item: ItemSummary) -> bool:
            # Vault and profile items belong to no single character, so a character-scoped
            # search includes the shared items that character's class can actually use.
            return item.location in {"vault", "profile"} and item.character_id is None

        candidates = [
            item
            for item in items
            if matches(item.name, request.query)
            and matches(item.item_type, request.item_type)
            and matches(item.item_subtype, request.subtype)
            and matches(item.gear_slot(), request.bucket)
            and (not request.equipped_only or item.is_equipped)
            and (selected is None or item.character_id == selected.character_id or is_shared(item))
        ]
        filtered = [
            item
            for item in candidates
            if selected is None
            or item.character_id == selected.character_id
            or class_compatibility(item, selected.class_name) != "incompatible"
        ]
        tier_order = {"Exotic": 4, "Legendary": 3, "Rare": 2, "Uncommon": 1}
        filtered.sort(
            key=lambda item: (
                item.is_equipped,
                tier_order.get(item.tier or "", 0),
                item.power or 0,
                item.name,
            ),
            reverse=True,
        )
        return {
            "items": [self._item(item) for item in filtered[: request.limit]],
            "total_matching": len(filtered),
            "excluded_other_class_count": len(candidates) - len(filtered),
            "limit": request.limit,
            "truncated": len(filtered) > request.limit,
        }

    def get_build_details(
        self,
        character_id: str | None = None,
        character_class: CharacterClass | None = None,
    ) -> dict[str, Any]:
        character = self.characters.resolve_one(
            character_id=character_id,
            character_class=character_class,
            default_to_current=True,
        )
        loadout = self.get_equipped_loadout(character.character_id)
        exotics = [
            self._item(item)
            for item in character.equipped_gear
            if item.tier == "Exotic" and item.item_type in {"Weapon", "Armor"}
        ]
        all_plugs = [
            plug for item in character.equipped_gear for plug in item.socketed_plug_details
        ]
        groups = self._plug_groups(all_plugs)
        role_groups = plug_groups(all_plugs)
        return {
            "character": loadout["character"],
            "subclass": loadout["subclass"],
            "equipped_exotics": exotics,
            "weapons": loadout["weapons"],
            "armor": loadout["armor"],
            "armor_stats": loadout["armor_stats"],
            "mods": groups["mods"],
            # Retrieved Manifest effect text for build-affecting equipped plugs, bounded.
            "equipped_plug_effects": equipped_plug_effects(character),
            # Manifest socket structure joined with the equipped plugs, ranked and capped.
            "equipped_mechanics": equipped_mechanics_section(character),
            # Weapon/armor perks and unclassified sockets; cosmetics are not build-affecting.
            "other_build_affecting_sockets": list(
                dict.fromkeys(plug.name for plug in [*role_groups["perks"], *role_groups["other"]])
            )[:30],
        }
