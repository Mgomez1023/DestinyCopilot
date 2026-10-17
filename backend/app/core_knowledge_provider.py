"""Knowledge-provider adapter for the curated core Destiny knowledge engine.

Exposes one model-facing tool, get_destiny_concept, with exactly the engine's exact,
alias, shorthand, and ambiguity semantics: no fuzzy matching, unknown stays unknown, and
an ambiguous term returns every candidate without choosing one.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core_knowledge import DestinyKnowledgeEngine
from app.destiny_knowledge import UnknownKnowledgeToolError

CORE_PROVIDER_CATEGORY = "core"
CORE_SOURCE_NAME = "core_knowledge"
CORE_CONCEPT_TOOL = "get_destiny_concept"

CORE_KNOWLEDGE_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": CORE_CONCEPT_TOOL,
        "description": (
            "Exact lookup of curated, Bungie-sourced core Destiny concepts: activity types and "
            "modes, classes, weapon archetypes, Exotic equip rules, selected named activities, "
            "and registered shorthand such as HC. Exact names and aliases only, no fuzzy "
            "matching. Returns grounded facts, unknown fields, live-sensitive topics, or an "
            "explicit ambiguity."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Concept name, alias, or shorthand."}
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    }
]

UNKNOWN_LIMITATION = (
    "No curated core concept has this exact name, alias, or registered shorthand. This tool "
    "does not fuzzy-match; use Manifest search for other Destiny entities."
)
UNKNOWN_FIELDS_LIMITATION = (
    "Fields listed in unknown are not established by the curated corpus; do not fill them "
    "from memory."
)
LIVE_SENSITIVE_LIMITATION = (
    "Topics listed in live_sensitive are current-state questions that this record does not "
    "answer; they need live or sourced current information."
)
AMBIGUOUS_LIMITATION = (
    "This term names several distinct entities. Do not choose one without the user's context."
)


class ConceptRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)


def _facts(card: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"predicate": predicate, "value": value, "sources": sources.split("+")}
        for predicate, value, sources in card.get("facts", [])
    ]


class CoreKnowledgeProvider:
    source_name = CORE_SOURCE_NAME
    knowledge_category = CORE_PROVIDER_CATEGORY
    tool_names = frozenset({CORE_CONCEPT_TOOL})

    def __init__(self, engine: DestinyKnowledgeEngine) -> None:
        self.engine = engine

    @staticmethod
    def definitions() -> list[dict[str, Any]]:
        return CORE_KNOWLEDGE_TOOL_DEFINITIONS

    def handles(self, name: str) -> bool:
        return name in self.tool_names

    async def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name != CORE_CONCEPT_TOOL:
            raise UnknownKnowledgeToolError(f"Unknown Destiny knowledge tool: {name}")
        return self.concept(ConceptRequest.model_validate(arguments).name)

    def concept(self, name: str) -> dict[str, Any]:
        """Pure, synchronous, in-memory lookup of one exact term."""

        found = self.engine.lookup(name)
        base: dict[str, Any] = {
            "source": self.source_name,
            "corpus_version": self.engine.corpus_version,
            "query": name,
        }
        if found["status"] == "ambiguous":
            return {
                **base,
                "found": False,
                "status": "ambiguous",
                "matched_text": found["alias"],
                "candidates": found["candidates"],
                "limitations": [AMBIGUOUS_LIMITATION],
            }
        if found["status"] != "entity":
            return {
                **base,
                "found": False,
                "status": "unknown",
                "limitations": [UNKNOWN_LIMITATION],
            }
        card = found["entity"]
        entity: dict[str, Any] = {
            key: card[key] for key in ("name", "kind", "status", "summary") if key in card
        }
        entity["facts"] = _facts(card)
        entity["unknown"] = list(card.get("unknown", []))
        entity["live_sensitive"] = list(card.get("live_sensitive", []))
        if card.get("not_same_as"):
            entity["not_same_as"] = list(card["not_same_as"])
        limitations: list[str] = []
        if entity["unknown"]:
            limitations.append(UNKNOWN_FIELDS_LIMITATION)
        if entity["live_sensitive"]:
            limitations.append(LIVE_SENSITIVE_LIMITATION)
        return {
            **base,
            "found": True,
            "status": "entity",
            "matched_via": found["matched_via"],
            "entity": entity,
            "limitations": limitations,
        }

    async def status(self) -> dict[str, Any]:
        return {
            "source": self.source_name,
            "category": self.knowledge_category,
            "corpus_version": self.engine.corpus_version,
            "entities": self.engine.entity_count,
        }
