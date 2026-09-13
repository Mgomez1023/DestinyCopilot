import asyncio
import hashlib
import json
import logging
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.bungie.manifest import DefinitionResolver
from app.config import Settings

logger = logging.getLogger(__name__)

BUNGIE_ROOT_URL = "https://www.bungie.net"
KNOWLEDGE_INDEX_SCHEMA_VERSION = 4

INDEXED_DEFINITION_TYPES: tuple[str, ...] = (
    "DestinyInventoryItemDefinition",
    "DestinyActivityDefinition",
    "DestinyActivityTypeDefinition",
    "DestinyDestinationDefinition",
    "DestinyPlaceDefinition",
    "DestinyObjectiveDefinition",
    "DestinyRecordDefinition",
    "DestinyProgressionDefinition",
    "DestinyFactionDefinition",
    "DestinySeasonDefinition",
    "DestinyCollectibleDefinition",
    "DestinySandboxPerkDefinition",
    "DestinySocketTypeDefinition",
    "DestinyStatDefinition",
    "DestinyTraitDefinition",
    "DestinyActivityModifierDefinition",
    "DestinyRewardSourceDefinition",
    "DestinyDamageTypeDefinition",
    "DestinyItemCategoryDefinition",
    "DestinyClassDefinition",
)

ENTITY_LABELS = {
    "DestinyInventoryItemDefinition": "item",
    "DestinyActivityDefinition": "activity",
    "DestinyActivityTypeDefinition": "activity_type",
    "DestinyDestinationDefinition": "destination",
    "DestinyPlaceDefinition": "place",
    "DestinyObjectiveDefinition": "objective",
    "DestinyRecordDefinition": "record",
    "DestinyProgressionDefinition": "progression",
    "DestinyFactionDefinition": "faction",
    "DestinySeasonDefinition": "season",
    "DestinyCollectibleDefinition": "collectible",
    "DestinySandboxPerkDefinition": "perk",
    "DestinySocketTypeDefinition": "socket_type",
    "DestinyStatDefinition": "stat",
    "DestinyTraitDefinition": "trait",
    "DestinyActivityModifierDefinition": "activity_modifier",
    "DestinyRewardSourceDefinition": "reward_source",
    "DestinyDamageTypeDefinition": "damage_type",
    "DestinyItemCategoryDefinition": "item_category",
    "DestinyClassDefinition": "class",
}

ENTITY_TYPE_ALIASES = {
    **{key.casefold(): key for key in INDEXED_DEFINITION_TYPES},
    **{value: key for key, value in ENTITY_LABELS.items()},
    "items": "DestinyInventoryItemDefinition",
    "activities": "DestinyActivityDefinition",
    "quests": "DestinyInventoryItemDefinition",
    "perks": "DestinySandboxPerkDefinition",
    "destinations": "DestinyDestinationDefinition",
}

ITEM_TYPE_NAMES = {
    1: "Currency",
    2: "Armor",
    3: "Weapon",
    8: "Engram",
    9: "Consumable",
    12: "Quest Step",
    14: "Emblem",
    15: "Quest",
    16: "Subclass",
    19: "Mod",
    20: "Dummy",
    21: "Ship",
    22: "Vehicle",
    23: "Emote",
    24: "Ghost",
    25: "Package",
    26: "Bounty",
    28: "Artifact",
    29: "Finisher",
    30: "Pattern",
}

ACTIVITY_DIFFICULTIES = {
    0: "Trivial",
    1: "Easy",
    2: "Normal",
    3: "Challenging",
    4: "Hard",
    5: "Brave",
    6: "Almost Impossible",
    7: "Impossible",
}

CLASS_TYPE_NAMES = {0: "Titan", 1: "Hunter", 2: "Warlock", 3: "Any class"}


def _normalized(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return " ".join(re.findall(r"[a-z0-9]+", ascii_value.casefold()))


def _display(definition: dict[str, Any]) -> tuple[str, str, str | None]:
    value = definition.get("displayProperties") or {}
    icon = value.get("icon")
    return (
        str(value.get("name") or ""),
        str(value.get("description") or ""),
        f"{BUNGIE_ROOT_URL}{icon}" if isinstance(icon, str) and icon.startswith("/") else icon,
    )


def _unique_hashes(values: list[Any]) -> list[int]:
    found: list[int] = []
    for value in values:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed and parsed not in found:
            found.append(parsed)
    return found


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


DESTINY_KNOWLEDGE_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    _strict_tool(
        "search_destiny_entities",
        "Fuzzy search official Destiny Manifest entities by name and description.",
        {
            "query": {"type": "string", "description": "Destiny entity name or phrase."},
            "entity_types": {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "description": "Optional entity categories such as item, activity, or quest.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 25,
                "description": "Maximum results.",
            },
        },
    ),
    _strict_tool(
        "get_item_details",
        "Get normalized static Manifest details for one Destiny item.",
        {"item_name_or_hash": {"type": "string", "description": "Exact/fuzzy name or hash."}},
    ),
    _strict_tool(
        "get_activity_details",
        "Get normalized static Manifest details for one Destiny activity.",
        {
            "activity_name_or_hash": {
                "type": "string",
                "description": "Exact/fuzzy activity name or hash.",
            }
        },
    ),
    _strict_tool(
        "get_quest_details",
        "Get Manifest-backed quest steps, objectives, and related activities.",
        {"quest_name_or_hash": {"type": "string", "description": "Quest name or hash."}},
    ),
    _strict_tool(
        "find_item_source",
        "Find only Manifest-supported acquisition hints and sources for an item.",
        {"item_name_or_hash": {"type": "string", "description": "Item name or hash."}},
    ),
]


class KnowledgeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchEntitiesRequest(KnowledgeRequest):
    query: str = Field(min_length=1, max_length=160)
    entity_types: list[str] | None = None
    limit: int = Field(default=10, ge=1, le=25)


class ItemDetailsRequest(KnowledgeRequest):
    item_name_or_hash: str = Field(min_length=1, max_length=160)


class ActivityDetailsRequest(KnowledgeRequest):
    activity_name_or_hash: str = Field(min_length=1, max_length=160)


class QuestDetailsRequest(KnowledgeRequest):
    quest_name_or_hash: str = Field(min_length=1, max_length=160)


class KnowledgeToolError(ValueError):
    pass


class UnknownKnowledgeToolError(KnowledgeToolError):
    pass


class DestinyKnowledgeProvider(Protocol):
    source_name: str
    knowledge_category: str
    tool_names: frozenset[str]

    def definitions(self) -> list[dict[str, Any]]: ...

    def handles(self, name: str) -> bool: ...

    async def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...

    async def status(self) -> dict[str, Any]: ...


@dataclass(slots=True)
class KnowledgeIndexEntry:
    definition_type: str
    entity_type: str
    entity_hash: int
    name: str
    description: str = ""
    icon_url: str | None = None
    subtype: str | None = None
    tier: str | None = None
    item_type_code: int | None = None
    has_collectible: bool = False
    is_quest: bool = False
    related_item_hashes: list[int] = field(default_factory=list)
    completion_value: int | float | None = None
    search_text: str = ""

    def public(self, score: float | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "definition_type": self.definition_type,
            "entity_type": self.entity_type,
            "hash": self.entity_hash,
            "name": self.name,
            "description": self.description or None,
            "icon_url": self.icon_url,
            "subtype": self.subtype,
            "tier": self.tier,
        }
        if score is not None:
            result["match_score"] = round(score, 3)
        return result


class ManifestKnowledgeProvider:
    source_name = "bungie_manifest"
    knowledge_category = "manifest"
    tool_names = frozenset(value["name"] for value in DESTINY_KNOWLEDGE_TOOL_DEFINITIONS)

    def __init__(self, resolver: DefinitionResolver, settings: Settings) -> None:
        self.resolver = resolver
        self.settings = settings
        self._entries: list[KnowledgeIndexEntry] | None = None
        self._hash_lookup: dict[tuple[str, int], KnowledgeIndexEntry] = {}
        self._trigram_lookup: dict[str, set[int]] = {}
        self._index_lock = asyncio.Lock()
        self._indexed_types: list[str] = []
        self._empty_types: list[str] = []
        self._unavailable_types: list[str] = []

    @staticmethod
    def definitions() -> list[dict[str, Any]]:
        return DESTINY_KNOWLEDGE_TOOL_DEFINITIONS

    def handles(self, name: str) -> bool:
        return name in self.tool_names

    async def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "search_destiny_entities":
            request = SearchEntitiesRequest.model_validate(arguments)
            return await self.search_entities(request.query, request.entity_types, request.limit)
        if name == "get_item_details":
            request = ItemDetailsRequest.model_validate(arguments)
            return await self.get_item_details(request.item_name_or_hash)
        if name == "get_activity_details":
            request = ActivityDetailsRequest.model_validate(arguments)
            return await self.get_activity_details(request.activity_name_or_hash)
        if name == "get_quest_details":
            request = QuestDetailsRequest.model_validate(arguments)
            return await self.get_quest_details(request.quest_name_or_hash)
        if name == "find_item_source":
            request = ItemDetailsRequest.model_validate(arguments)
            return await self.find_item_source(request.item_name_or_hash)
        raise UnknownKnowledgeToolError(f"Unknown Destiny knowledge tool: {name}")

    async def _ensure_index(self) -> None:
        if self._entries is not None:
            return
        async with self._index_lock:
            if self._entries is not None:
                return
            version = await self.resolver.manifest_version()
            version_key = hashlib.sha256(version.encode()).hexdigest()[:12]
            root = self.settings.resolve_local_path(self.settings.manifest_cache_dir)
            cache_path = root / f"destiny-knowledge-index-{version_key}.json"
            entries = await asyncio.to_thread(self._read_index, cache_path)
            if entries is None:
                entries = await self._build_index()
                await asyncio.to_thread(self._write_index, cache_path, entries)
            self._install_index(entries)
            logger.info(
                "Destiny knowledge index ready entries=%d definition_types=%s unavailable=%s",
                len(entries),
                self._indexed_types,
                self._unavailable_types,
            )

    @staticmethod
    def _read_index(path: Path) -> list[KnowledgeIndexEntry] | None:
        if not path.is_file():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("schema_version") != KNOWLEDGE_INDEX_SCHEMA_VERSION:
                return None
            return [KnowledgeIndexEntry(**value) for value in payload.get("entries", [])]
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            logger.warning("Ignoring unreadable Destiny knowledge index %s", path)
            return None

    @staticmethod
    def _write_index(path: Path, entries: list[KnowledgeIndexEntry]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "schema_version": KNOWLEDGE_INDEX_SCHEMA_VERSION,
                        "entries": [asdict(value) for value in entries],
                    },
                    handle,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            temporary.replace(path)
        except OSError:
            logger.warning("Could not persist Destiny knowledge index %s", path)

    async def _build_index(self) -> list[KnowledgeIndexEntry]:
        entries: list[KnowledgeIndexEntry] = []
        self._indexed_types = []
        self._unavailable_types = []
        for definition_type in INDEXED_DEFINITION_TYPES:
            try:
                table = await self.resolver.load_table(definition_type)
            except Exception:
                logger.warning("Manifest definition type unavailable: %s", definition_type)
                self._unavailable_types.append(definition_type)
                continue
            self._indexed_types.append(definition_type)
            for raw in table.values():
                if not isinstance(raw, dict) or raw.get("redacted"):
                    continue
                entry = self._index_entry(definition_type, raw)
                if entry is not None:
                    entries.append(entry)
            self.resolver.release_table(definition_type)
        return entries

    @staticmethod
    def _index_entry(
        definition_type: str, definition: dict[str, Any]
    ) -> KnowledgeIndexEntry | None:
        name, description, icon = _display(definition)
        if definition_type == "DestinyObjectiveDefinition" and not name:
            name = str(definition.get("progressDescription") or "")
        if not name:
            return None
        subtype = definition.get("itemTypeDisplayName")
        if definition_type == "DestinyRecordDefinition":
            subtype = definition.get("recordTypeName") or subtype
        tier = (definition.get("inventory") or {}).get("tierTypeName")
        item_type_code = definition.get("itemType")
        extras = [
            str(subtype or ""),
            str(tier or ""),
            str(definition.get("displaySource") or ""),
            str(definition.get("sourceString") or ""),
            str((definition.get("setData") or {}).get("questLineName") or ""),
        ]
        related_items: list[int] = []
        if definition_type == "DestinyActivityDefinition":
            for reward in definition.get("rewards", []) or []:
                related_items.extend(
                    _unique_hashes(
                        [item.get("itemHash") for item in reward.get("rewardItems", []) or []]
                    )
                )
        elif definition_type == "DestinyRecordDefinition":
            related_items.extend(
                _unique_hashes(
                    [item.get("itemHash") for item in definition.get("rewardItems", []) or []]
                )
            )
            for challenge in definition.get("challenges", []) or []:
                for reward in challenge.get("displayRewards", []) or []:
                    related_items.extend(
                        _unique_hashes([(reward.get("itemQuantity") or {}).get("itemHash")])
                    )
        try:
            entity_hash = int(definition.get("hash"))
        except (TypeError, ValueError):
            return None
        description = description[:500]
        search_text = _normalized(" ".join([name, description, *extras]))
        return KnowledgeIndexEntry(
            definition_type=definition_type,
            entity_type=ENTITY_LABELS[definition_type],
            entity_hash=entity_hash,
            name=name,
            description=description,
            icon_url=icon,
            subtype=str(subtype) if subtype else None,
            tier=str(tier) if tier else None,
            item_type_code=int(item_type_code) if item_type_code is not None else None,
            has_collectible=bool(definition.get("collectibleHash")),
            is_quest=(
                definition_type == "DestinyInventoryItemDefinition"
                and (
                    int(item_type_code or 0) in {12, 15, 26}
                    or "quest" in str(subtype or "").casefold()
                )
            ),
            related_item_hashes=list(dict.fromkeys(related_items)),
            completion_value=(
                definition.get("completionValue")
                if definition_type == "DestinyObjectiveDefinition"
                else None
            ),
            search_text=search_text,
        )

    def _install_index(self, entries: list[KnowledgeIndexEntry]) -> None:
        self._entries = entries
        self._indexed_types = sorted({value.definition_type for value in entries})
        self._empty_types = sorted(
            set(INDEXED_DEFINITION_TYPES) - set(self._indexed_types) - set(self._unavailable_types)
        )
        self._hash_lookup = {(value.definition_type, value.entity_hash): value for value in entries}
        trigrams: dict[str, set[int]] = {}
        for index, entry in enumerate(entries):
            for trigram in self._trigrams(_normalized(entry.name)):
                trigrams.setdefault(trigram, set()).add(index)
        self._trigram_lookup = trigrams

    @staticmethod
    def _trigrams(value: str) -> set[str]:
        padded = f"  {value}  "
        return {padded[index : index + 3] for index in range(max(1, len(padded) - 2))}

    @staticmethod
    def _allowed_types(entity_types: list[str] | None) -> set[str] | None:
        if not entity_types:
            return None
        return {
            resolved
            for value in entity_types
            if (resolved := ENTITY_TYPE_ALIASES.get(value.casefold())) is not None
        }

    async def _matching_entries(
        self, query: str, entity_types: list[str] | None, limit: int
    ) -> list[tuple[KnowledgeIndexEntry, float]]:
        await self._ensure_index()
        assert self._entries is not None
        allowed = self._allowed_types(entity_types)
        quests_only = bool(
            entity_types and any(value.casefold() in {"quest", "quests"} for value in entity_types)
        )
        query_normalized = _normalized(query)
        if not query_normalized:
            return []

        if query_normalized.isdigit():
            entity_hash = int(query_normalized)
            exact = [
                value
                for value in self._entries
                if value.entity_hash == entity_hash
                and (allowed is None or value.definition_type in allowed)
                and (not quests_only or value.is_quest)
            ]
            return [(value, 1.0) for value in exact[:limit]]

        candidate_indexes: set[int] = set()
        for trigram in self._trigrams(query_normalized):
            candidate_indexes.update(self._trigram_lookup.get(trigram, set()))
        if len(query_normalized) < 3 or not candidate_indexes:
            candidate_indexes = set(range(len(self._entries)))

        query_tokens = set(query_normalized.split())
        scored: list[tuple[KnowledgeIndexEntry, float]] = []
        for index in candidate_indexes:
            entry = self._entries[index]
            if allowed is not None and entry.definition_type not in allowed:
                continue
            if quests_only and not entry.is_quest:
                continue
            name = _normalized(entry.name)
            name_ratio = SequenceMatcher(None, query_normalized, name).ratio()
            overlap = len(query_tokens & set(entry.search_text.split())) / max(1, len(query_tokens))
            score = name_ratio * 0.55 + overlap * 0.2
            if query_normalized == name:
                score += 1.0
            elif name.startswith(query_normalized):
                score += 0.45
            elif query_normalized in entry.search_text:
                score += 0.3
            if score >= 0.28:
                scored.append((entry, score))

        def search_preference(value: tuple[KnowledgeIndexEntry, float]) -> tuple[float, int, str]:
            entry, score = value
            preference = 0
            if entry.definition_type == "DestinyInventoryItemDefinition":
                if quests_only:
                    preference = 2 if entry.item_type_code == 15 else 1
                else:
                    preference += 3 if entry.item_type_code in {2, 3} else 0
                    preference += 1 if entry.has_collectible else 0
                    preference -= 2 if entry.is_quest or entry.item_type_code == 25 else 0
            return score, preference, entry.name

        scored.sort(key=search_preference, reverse=True)
        return scored[:limit]

    async def search_entities(
        self, query: str, entity_types: list[str] | None = None, limit: int = 10
    ) -> dict[str, Any]:
        matches = await self._matching_entries(query, entity_types, limit)
        return {
            "query": query,
            "results": [entry.public(score) for entry, score in matches],
            "count": len(matches),
            "source": self.source_name,
            "limitations": (
                [] if matches else ["No matching visible English Manifest entity was found."]
            ),
        }

    async def _resolve_entry(
        self, value: str, definition_type: str, *, quest: bool = False
    ) -> KnowledgeIndexEntry | None:
        matches = await self._matching_entries(value, [definition_type], 25)
        if not matches:
            return None
        normalized_value = _normalized(value)

        def preference(candidate: tuple[KnowledgeIndexEntry, float]) -> tuple[int, float]:
            entry, score = candidate
            exact = int(_normalized(entry.name) == normalized_value)
            bonus = 0
            if definition_type == "DestinyInventoryItemDefinition":
                if quest:
                    bonus += 80 if entry.item_type_code == 15 else 40 if entry.is_quest else -80
                else:
                    bonus += 80 if entry.item_type_code in {2, 3} else 0
                    bonus += 30 if entry.has_collectible else 0
                    bonus -= 60 if entry.is_quest or entry.item_type_code == 25 else 0
            return exact * 100 + bonus, score

        return max(matches, key=preference)[0]

    async def _definition(self, entry: KnowledgeIndexEntry) -> dict[str, Any] | None:
        values = await self.resolver.resolve_many(entry.definition_type, {entry.entity_hash})
        return values.get(entry.entity_hash)

    async def _briefs(
        self, definition_type: str, hashes: list[int], limit: int = 16
    ) -> list[dict[str, Any]]:
        selected = _unique_hashes(hashes)[:limit]
        await self._ensure_index()
        definitions: dict[int, dict[str, Any]] = {}
        unresolved: set[int] = set()
        for entity_hash in selected:
            indexed = self._hash_lookup.get((definition_type, entity_hash))
            if indexed is None:
                unresolved.add(entity_hash)
                continue
            definitions[entity_hash] = {
                "displayProperties": {
                    "name": indexed.name,
                    "description": indexed.description,
                }
            }
        if unresolved:
            definitions.update(await self.resolver.resolve_many(definition_type, unresolved))
        values: list[dict[str, Any]] = []
        for entity_hash in selected:
            definition = definitions.get(entity_hash)
            if not definition:
                continue
            name, description, _ = _display(definition)
            brief: dict[str, Any] = {
                "hash": entity_hash,
                "name": name or None,
                "description": description or None,
                "definition_available": bool(name),
            }
            indexed = self._hash_lookup.get((definition_type, entity_hash))
            if indexed is not None and indexed.completion_value is not None:
                brief["completion_value"] = indexed.completion_value
            values.append(brief)
        return values

    async def _collectible_source(self, definition: dict[str, Any]) -> dict[str, Any] | None:
        collectible_hash = definition.get("collectibleHash")
        if not collectible_hash:
            return None
        values = await self.resolver.resolve_many(
            "DestinyCollectibleDefinition", {int(collectible_hash)}
        )
        collectible = values.get(int(collectible_hash))
        if not collectible:
            return None
        name, description, _ = _display(collectible)
        return {
            "collectible_hash": int(collectible_hash),
            "name": name or None,
            "description": description or None,
            "source_string": collectible.get("sourceString") or None,
            "source_hash": collectible.get("sourceHash") or None,
        }

    async def get_item_details(self, value: str) -> dict[str, Any]:
        entry = await self._resolve_entry(value, "DestinyInventoryItemDefinition")
        if entry is None:
            return self._not_found(value, "item")
        definition = await self._definition(entry)
        if not definition:
            return self._not_found(value, "item")
        name, description, icon = _display(definition)
        item_type_code = int(definition.get("itemType", 0) or 0)
        objective_hashes = _unique_hashes(
            list((definition.get("objectives") or {}).get("objectiveHashes", []) or [])
        )
        perk_hashes = _unique_hashes(
            [value.get("perkHash") for value in definition.get("perks", []) or []]
        )
        socket_hashes = _unique_hashes(
            [
                value.get("singleInitialItemHash")
                for value in (definition.get("sockets") or {}).get("socketEntries", []) or []
            ]
        )
        trait_hashes = _unique_hashes(list(definition.get("traitHashes", []) or []))
        stats_raw = (definition.get("stats") or {}).get("stats", {}) or {}
        stat_hashes = _unique_hashes([value.get("statHash") for value in stats_raw.values()])
        stat_names = {
            value["hash"]: value["name"]
            for value in await self._briefs("DestinyStatDefinition", stat_hashes)
        }
        stats = [
            {
                "name": stat_names.get(int(raw.get("statHash", 0)), "Unknown stat"),
                "value": raw.get("value"),
            }
            for raw in stats_raw.values()
            if raw.get("value") is not None
        ][:20]
        collectible = await self._collectible_source(definition)
        class_type = int(definition.get("classType", 3) or 0)
        damage_type_hashes = _unique_hashes(list(definition.get("damageTypeHashes", []) or []))
        category_hashes = _unique_hashes(list(definition.get("itemCategoryHashes", []) or []))
        return {
            "found": True,
            "source": self.source_name,
            "item": {
                "hash": entry.entity_hash,
                "name": name,
                "description": description or None,
                "flavor_text": definition.get("flavorText") or None,
                "icon_url": icon,
                "item_type": ITEM_TYPE_NAMES.get(item_type_code, "Other"),
                "subtype": definition.get("itemTypeDisplayName") or None,
                "rarity": (definition.get("inventory") or {}).get("tierTypeName") or None,
                "equippable": bool(definition.get("equippable")),
                "class_restriction": {
                    "code": class_type,
                    "name": CLASS_TYPE_NAMES.get(class_type, "Unknown class restriction"),
                },
                "damage_types": await self._briefs(
                    "DestinyDamageTypeDefinition", damage_type_hashes
                ),
                "display_source": definition.get("displaySource") or None,
                "collectible": collectible,
                "intrinsic_perks": await self._briefs("DestinySandboxPerkDefinition", perk_hashes),
                "default_socket_plugs": await self._briefs(
                    "DestinyInventoryItemDefinition", socket_hashes
                ),
                "traits": await self._briefs("DestinyTraitDefinition", trait_hashes),
                "objectives": await self._briefs("DestinyObjectiveDefinition", objective_hashes),
                "base_definition_stats": stats,
                "related_hashes": {
                    "collectible_hash": definition.get("collectibleHash") or None,
                    "lore_hash": definition.get("loreHash") or None,
                    "item_category_hashes": _unique_hashes(category_hashes),
                },
                "item_categories": await self._briefs(
                    "DestinyItemCategoryDefinition", category_hashes
                ),
            },
            "limitations": [
                "Manifest item stats and sockets are generic definition data; random rolls and "
                "owned-instance state require Guardian inventory data."
            ],
        }

    async def get_activity_details(self, value: str) -> dict[str, Any]:
        entry = await self._resolve_entry(value, "DestinyActivityDefinition")
        if entry is None:
            return self._not_found(value, "activity")
        definition = await self._definition(entry)
        if not definition:
            return self._not_found(value, "activity")
        name, description, icon = _display(definition)
        activity_type = await self._briefs(
            "DestinyActivityTypeDefinition", [definition.get("activityTypeHash")]
        )
        destination = await self._briefs(
            "DestinyDestinationDefinition", [definition.get("destinationHash")]
        )
        place = await self._briefs("DestinyPlaceDefinition", [definition.get("placeHash")])
        modifier_hashes = _unique_hashes(
            [value.get("activityModifierHash") for value in definition.get("modifiers", []) or []]
        )
        objective_hashes = _unique_hashes(
            [value.get("objectiveHash") for value in definition.get("challenges", []) or []]
        )
        reward_items: list[int] = []
        reward_text: list[str] = []
        for reward in definition.get("rewards", []) or []:
            if reward.get("rewardText"):
                reward_text.append(str(reward["rewardText"]))
            reward_items.extend(
                _unique_hashes(
                    [item.get("itemHash") for item in reward.get("rewardItems", []) or []]
                )
            )
        tier = definition.get("tier")
        matchmaking = definition.get("matchmaking") or {}
        return {
            "found": True,
            "source": self.source_name,
            "activity": {
                "hash": entry.entity_hash,
                "name": name,
                "description": description or None,
                "icon_url": icon,
                "activity_type": activity_type[0] if activity_type else None,
                "destination": destination[0] if destination else None,
                "place": place[0] if place else None,
                "difficulty": ACTIVITY_DIFFICULTIES.get(int(tier)) if tier is not None else None,
                "power_eligibility": "unknown_not_compared",
                "is_playlist": bool(definition.get("isPlaylist")),
                "matchmaking": {
                    "enabled": bool(matchmaking.get("isMatchmade")),
                    "minimum_party": matchmaking.get("minParty"),
                    "maximum_party": matchmaking.get("maxParty"),
                    "maximum_players": matchmaking.get("maxPlayers"),
                },
                "possible_modifiers": await self._briefs(
                    "DestinyActivityModifierDefinition", modifier_hashes
                ),
                "challenge_objectives": await self._briefs(
                    "DestinyObjectiveDefinition", objective_hashes
                ),
                "possible_reward_text": list(dict.fromkeys(reward_text)),
                "possible_reward_items": await self._briefs(
                    "DestinyInventoryItemDefinition", reward_items
                ),
            },
            "limitations": [
                "Manifest modifiers, challenges, and rewards are possible static metadata, not a "
                "guarantee that they are currently active or that a specific item will drop.",
                "Manifest activity Power is not exposed for player eligibility comparisons because "
                "its semantics may be obsolete or incompatible with current Guardian Power.",
            ],
        }

    async def get_quest_details(self, value: str) -> dict[str, Any]:
        entry = await self._resolve_entry(value, "DestinyInventoryItemDefinition", quest=True)
        if entry is None or not entry.is_quest:
            return self._not_found(value, "quest")
        definition = await self._definition(entry)
        if not definition:
            return self._not_found(value, "quest")
        name, description, icon = _display(definition)
        set_data = definition.get("setData") or {}
        step_hashes = _unique_hashes(
            [item.get("itemHash") for item in set_data.get("itemList", []) or []]
        )
        if not step_hashes:
            step_hashes = [entry.entity_hash]
        step_definitions = await self.resolver.resolve_many(
            "DestinyInventoryItemDefinition", set(step_hashes[:25])
        )
        steps: list[dict[str, Any]] = []
        related_activity_hashes: list[int] = []
        for step_hash in step_hashes[:25]:
            step = step_definitions.get(step_hash)
            if not step:
                continue
            step_name, step_description, _ = _display(step)
            objectives = step.get("objectives") or {}
            objective_hashes = _unique_hashes(list(objectives.get("objectiveHashes", []) or []))
            activity_hashes = _unique_hashes(
                list(objectives.get("displayActivityHashes", []) or [])
            )
            related_activity_hashes.extend(activity_hashes)
            activity_by_objective = {
                objective_hash: activity_hashes[index]
                for index, objective_hash in enumerate(objective_hashes)
                if index < len(activity_hashes) and activity_hashes[index]
            }
            objective_details = await self._briefs("DestinyObjectiveDefinition", objective_hashes)
            for objective in objective_details:
                objective["activity_hash"] = activity_by_objective.get(objective["hash"])
            steps.append(
                {
                    "hash": step_hash,
                    "name": step_name,
                    "description": step_description or None,
                    "step_summary": (step.get("setData") or {}).get("questStepSummary") or None,
                    "objectives": objective_details,
                }
            )
        related_activity_hashes = _unique_hashes(related_activity_hashes)
        activity_definitions = await self.resolver.resolve_many(
            "DestinyActivityDefinition", set(related_activity_hashes)
        )
        destination_hashes = _unique_hashes(
            [activity.get("destinationHash") for activity in activity_definitions.values()]
        )
        return {
            "found": True,
            "source": self.source_name,
            "quest": {
                "hash": entry.entity_hash,
                "name": name,
                "description": description or set_data.get("questLineDescription") or None,
                "icon_url": icon,
                "quest_line_name": set_data.get("questLineName") or name,
                "featured": bool(set_data.get("setIsFeatured")),
                "steps": steps,
                "related_activities": await self._briefs(
                    "DestinyActivityDefinition", related_activity_hashes
                ),
                "related_destinations": await self._briefs(
                    "DestinyDestinationDefinition", destination_hashes
                ),
                "display_source": definition.get("displaySource") or None,
            },
            "limitations": [
                "The Manifest can describe quest steps and objectives, but it does not reliably "
                "provide a complete walkthrough, encounter strategy, or every acquisition "
                "prerequisite."
            ],
        }

    async def find_item_source(self, value: str) -> dict[str, Any]:
        entry = await self._resolve_entry(value, "DestinyInventoryItemDefinition")
        if entry is None:
            return self._not_found(value, "item")
        definition = await self._definition(entry)
        if not definition:
            return self._not_found(value, "item")
        name, _, _ = _display(definition)
        evidence: list[dict[str, Any]] = []
        display_source = str(definition.get("displaySource") or "").strip()
        if display_source:
            evidence.append(
                {
                    "kind": "item_display_source",
                    "text": display_source,
                    "confidence": "manifest_text",
                }
            )
        collectible = await self._collectible_source(definition)
        if collectible and collectible.get("source_string"):
            evidence.append(
                {
                    "kind": "collectible_source_string",
                    "text": collectible["source_string"],
                    "confidence": "manifest_hint",
                    "collectible_hash": collectible["collectible_hash"],
                }
            )
        source_hashes: list[int] = []
        source_data = definition.get("sourceData") or {}
        if isinstance(source_data, dict):
            source_hashes.extend(_unique_hashes(list(source_data.get("sourceHashes", []) or [])))
        elif isinstance(source_data, list):
            for source in source_data:
                if isinstance(source, dict):
                    source_hashes.extend(_unique_hashes(list(source.get("sourceHashes", []) or [])))
        for source in await self._briefs("DestinyRewardSourceDefinition", source_hashes):
            evidence.append(
                {
                    "kind": "heuristic_reward_source",
                    "text": source["name"],
                    "description": source["description"],
                    "confidence": "bungie_heuristic",
                    "source_hash": source["hash"],
                }
            )

        await self._ensure_index()
        assert self._entries is not None
        activity_matches = [
            item.public()
            for item in self._entries
            if item.definition_type == "DestinyActivityDefinition"
            and entry.entity_hash in item.related_item_hashes
        ][:10]
        if activity_matches:
            evidence.append(
                {
                    "kind": "possible_activity_reward",
                    "activities": activity_matches,
                    "confidence": "possible_static_reward",
                }
            )

        record_matches = [
            item.public()
            for item in self._entries
            if item.definition_type == "DestinyRecordDefinition"
            and entry.entity_hash in item.related_item_hashes
        ][:10]
        if record_matches:
            evidence.append(
                {
                    "kind": "record_reward",
                    "records": record_matches,
                    "confidence": "manifest_reward",
                }
            )

        limitations = [
            "Manifest source strings are acquisition hints, not step-by-step guides.",
            "Reward-source mappings are heuristic and activity reward lists may contain "
            "placeholders.",
        ]
        if not evidence:
            limitations.insert(
                0,
                "The current Manifest contains no reliable acquisition source for this item.",
            )
        return {
            "found": True,
            "source": self.source_name,
            "item": {"hash": entry.entity_hash, "name": name},
            "source_known": bool(evidence),
            "evidence": evidence,
            "limitations": limitations,
        }

    @staticmethod
    def _not_found(query: str, kind: str) -> dict[str, Any]:
        return {
            "found": False,
            "query": query,
            "entity_type": kind,
            "source": "bungie_manifest",
            "limitations": [f"No matching visible English Manifest {kind} was found."],
        }

    async def index_status(self) -> dict[str, Any]:
        await self._ensure_index()
        return {
            "source": self.source_name,
            "entries": len(self._entries or []),
            "indexed_definition_types": self._indexed_types,
            "empty_definition_types": self._empty_types,
            "unavailable_definition_types": self._unavailable_types,
        }

    async def status(self) -> dict[str, Any]:
        return await self.index_status()


class DestinyKnowledgeService:
    """Tool-facing game knowledge service with replaceable normalized providers."""

    def __init__(self, providers: list[DestinyKnowledgeProvider]) -> None:
        if not providers:
            raise ValueError("DestinyKnowledgeService requires at least one provider.")
        self.providers = providers

    def definitions(self) -> list[dict[str, Any]]:
        definitions: list[dict[str, Any]] = []
        seen: set[str] = set()
        for provider in self.providers:
            for definition in provider.definitions():
                name = str(definition["name"])
                if name not in seen:
                    definitions.append(definition)
                    seen.add(name)
        return definitions

    def provider_for_tool(self, name: str) -> DestinyKnowledgeProvider | None:
        return next((value for value in self.providers if value.handles(name)), None)

    def handles(self, name: str) -> bool:
        return self.provider_for_tool(name) is not None

    def category_for_tool(self, name: str) -> str | None:
        provider = self.provider_for_tool(name)
        return provider.knowledge_category if provider else None

    def has_category(self, category: str) -> bool:
        return any(provider.knowledge_category == category for provider in self.providers)

    async def execute(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        provider = self.provider_for_tool(name)
        if provider is None:
            raise UnknownKnowledgeToolError(f"Unknown Destiny knowledge tool: {name}")
        return await provider.execute(name, arguments or {})

    async def index_status(self) -> dict[str, Any]:
        statuses = await asyncio.gather(*(provider.status() for provider in self.providers))
        manifest = next(
            (value for value in statuses if value.get("source") == "bungie_manifest"), {}
        )
        return {
            **manifest,
            "source": "multiple" if len(statuses) > 1 else statuses[0].get("source"),
            "provider_count": len(statuses),
            "providers": statuses,
        }


BungieManifestProvider = ManifestKnowledgeProvider
