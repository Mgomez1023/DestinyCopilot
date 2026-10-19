"""Trace-only detection of Destiny mechanics claims that lack retrieved evidence.

Knowledge Engine v0.2, checkpoint M0. This module measures; it never decides. It is read by
the trace and by the evaluation harness, and nothing in the answer path depends on it.

What it does, and does not do
-----------------------------
* A `MechanicsEvidenceLedger` records what the tools returned during one turn: entity names,
  verbatim effect text, armor slot and class, whether a plug is equipped, and any evidence
  that a configuration change is legal.
* `detect_mechanics_claims` scans a finished answer, sentence by sentence, for a narrow set of
  claims (an effect of a named entity, an armor slot or class, a damage-source requirement, a
  configuration change) and labels each with the evidence that was retrieved for it.
* Statuses describe **retrieval**, not truth. "No evidence retrieved" means the turn did not
  fetch it; it is never proof that the mechanic does not exist or is wrong. Only a structured
  disagreement between an answer and retrieved evidence is reported as a contradiction.

It is a lexical heuristic, not language understanding. See `KNOWN_LIMITATIONS`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from app.core_knowledge import normalize_text

ClaimCategory = Literal["item_effect", "identity", "damage_source", "configuration"]
ClaimStatus = Literal[
    # The entity's effect text was retrieved and the claim shares its content words.
    "supported",
    # Text was retrieved for the entity but the claim shares almost nothing with it. This is
    # an extrapolation or a paraphrase; it is not a detected contradiction.
    "evidence_retrieved_no_overlap",
    # The entity is known to the turn (or resolvable) but no effect/identity text was fetched.
    "mentioned_without_evidence",
    # The entity could not be resolved by a retrieval that ran, or by the Manifest resolver.
    "unresolved_entity",
    # A structured fact (slot, class, damage source) disagrees with retrieved evidence.
    "contradicted_by_evidence",
    # A proposed change to a socket, perk or mod, with no retrieved legality evidence.
    "config_change_without_legality",
    "config_change_with_legality",
    # A statement about what is equipped now, backed by Guardian socket data.
    "equipped_configuration",
    "equipped_claim_without_evidence",
]
MissingEvidence = Literal[
    "effect_text", "slot", "class", "objective_text", "socket_options", "entity_resolution"
]

# Statuses counted as "needs attention" in the summary. This is a measurement, not a verdict.
UNSUPPORTED_STATUSES: frozenset[str] = frozenset(
    {
        "mentioned_without_evidence",
        "unresolved_entity",
        "contradicted_by_evidence",
        "config_change_without_legality",
        "equipped_claim_without_evidence",
    }
)
KNOWN_LIMITATIONS = (
    "Lexical heuristic: it sees only sentences that pair a mechanic word with an effect verb.",
    "Entities are found by name from this turn's tool results, or by a capitalized phrase "
    "directly before an effect verb; other phrasings are missed.",
    "'supported' means evidence was retrieved and shares content words, not that the claim is "
    "correct.",
    "Absence of retrieved evidence is a retrieval gap, never proof a mechanic is absent.",
    "Only slot, class and ability/weapon damage source are checked for contradiction.",
)

MAX_ENTITIES = 400
MAX_TEXTS_PER_ENTITY = 8
MAX_TEXT_LENGTH = 400
MAX_TRACED_CLAIMS = 16
MIN_ENTITY_KEY_LENGTH = 4
OVERLAP_MIN_SHARED_WORDS = 2
MAX_WALK_DEPTH = 7

ARMOR_SLOTS: dict[str, str] = {
    "helmet": "helmet",
    "helmets": "helmet",
    "gauntlets": "gauntlets",
    "arms": "gauntlets",
    "chest armor": "chest",
    "chest": "chest",
    "leg armor": "legs",
    "legs": "legs",
    "boots": "legs",
    "class item": "class item",
    "class armor": "class item",
    "bond": "class item",
    "mark": "class item",
    "cloak": "class item",
}
CLASS_NAMES = ("titan", "hunter", "warlock")

# A sentence is a mechanic claim only when it pairs a mechanic noun with an effect verb or a
# conditional trigger. Both lists are deliberately small and generic; they name no items.
_MECHANIC_NOUNS = (
    r"damage|reload|handling|stability|range|recoil|accuracy|cooldown|recharge|regenerat\w*|"
    r"energy|super|grenade|melee|class ability|ability|abilities|resilience|recovery|mobility|"
    r"discipline|intellect|strength|armor charge|final blows?|precision|ammo|ammunition|health|"
    r"overshield|shield|orbs?|stacks?|buff|debuff|jolt\w*|scorch\w*|sever\w*|suppress\w*|"
    r"volatile|weaken\w*|slow\w*|freeze|frozen|ignit\w*|radiant|restoration|woven mail"
)
_EFFECT_VERBS = (
    r"grants?|gives?|increases?|decreases?|reduces?|boosts?|extends?|restores?|refunds?|"
    r"causes?|triggers?|applies|apply|creates?|spawns?|makes?|allows?|lets?|activates?|"
    r"improves?|enhances?|amplif\w+|counts? toward|stacks?|generates?|returns?|unlocks?|"
    r"empowers?|highlights?|detonates?|explodes?"
)
_CONDITIONAL = r"\b(?:when|while|after|whenever|each time|every time)\b"
_MECHANIC_NOUN_RE = re.compile(rf"\b(?:{_MECHANIC_NOUNS})\b", re.I)
_EFFECT_VERB_RE = re.compile(rf"\b(?:{_EFFECT_VERBS})\b", re.I)
_CONDITIONAL_RE = re.compile(_CONDITIONAL, re.I)
# A capitalized phrase directly before an effect verb: "Foo Bar grants ...". Used only to find
# an entity the turn did not retrieve; ledger names are matched directly instead.
_NAMED_ACTOR_RE = re.compile(
    rf"((?:[A-Z][\w'’\-]*)(?:\s+(?:of|the|and|for|to|in)\s+[A-Z][\w'’\-]*|\s+[A-Z][\w'’\-]*){{0,3}})"
    rf"\s+(?:{_EFFECT_VERBS})\b"
)
_NOT_ENTITY = frozenset(
    {
        "you", "your", "this", "that", "these", "those", "it", "they", "the", "a", "an", "if",
        "when", "while", "also", "then", "arc", "solar", "void", "stasis", "strand", "kinetic",
        "titan", "hunter", "warlock", "primary", "backup", "why", "how", "what", "note",
        "tip", "weapon", "armor", "ability", "super", "grenade", "melee", "mod", "perk",
    }
)  # fmt: skip
_CHANGE_VERBS = re.compile(
    r"\b(?:swap(?:ping)?|switch(?:ing)?|change|replace|reroll|re-roll|roll|slot(?:ting)?|"
    r"socket(?:ing)?|put|add|try|upgrade|craft|enhance|attach|install|insert|apply)\b"
    r"(?!\s+(?:your|the)\s+(?:class|super|grenade|melee)\b)",
    re.I,
)
_KEEP_VERBS = re.compile(
    r"\b(?:keep|leave|stay|stick with|retain|already|currently|you(?:'re| are) running|"
    r"you have|has|have equipped|equipped|running)\b",
    re.I,
)
_EQUIPPED_NOW = re.compile(
    r"\b(?:currently|already|you(?:'re| are) (?:running|using)|you have (?:it )?equipped|"
    r"equipped (?:right now|on))\b",
    re.I,
)
_SOCKET_NOUNS = re.compile(
    r"\b(?:barrel|magazine|mag|grip|stock|scope|battery|bowstring|arrow|haft|blade|guard|"
    r"perk|trait|frame|origin|mod|masterwork|catalyst|fragment|aspect|socket|column)s?\b",
    re.I,
)
_HEDGE = re.compile(
    r"\b(?:unknown|not sure|can'?t confirm|cannot confirm|unable to confirm|isn'?t established|"
    r"not established|don'?t know|can'?t verify|cannot verify|couldn'?t (?:confirm|verify)|"
    r"not (?:retrieved|returned|listed)|unclear|I don'?t have)\b",
    re.I,
)
_NEGATED_ENTITY = re.compile(r"\b(?:no|not|isn'?t|doesn'?t|don'?t|cannot|can'?t|without)\b", re.I)
_SPLIT = re.compile(r"(?<=[.!?])\s+|[\r\n]+")
_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "them",
        "then",
        "this",
        "to",
        "up",
        "with",
        "you",
        "your",
        "when",
        "while",
        "after",
        "each",
        "every",
        "than",
        "can",
        "will",
        "may",
        "also",
        "more",
        "most",
        "about",
        "over",
        "under",
        "out",
        "all",
        "any",
        "not",
        "no",
        "but",
        "if",
        "so",
    }
)
_IDENTITY_CLASS_CUE = re.compile(
    r"\b(?:(?:is|are)\s+(?:an?\s+)?(?:\w+\s+){0,2}(?P<a>titan|hunter|warlock)\b|"
    r"(?P<b>titan|hunter|warlock)[- ](?:only|exclusive)|"
    r"(?P<c>titan|hunter|warlock)\s+(?:exotic|helmet|gauntlets|chest|legs|armor|piece)|"
    r"(?:restricted|locked|exclusive) to (?:the )?(?P<d>titan|hunter|warlock))",
    re.I,
)
_SLOT_WORD = re.compile(
    r"\b(?:helmets?|gauntlets|chest armor|leg armor|class items?|class armor|boots|"
    r"chest piece|arms|legs)\b",
    re.I,
)
_ABILITY_SOURCE = re.compile(r"\bability (?:final blows?|kills?)\b", re.I)
_WEAPON_SOURCE = re.compile(r"\bweapon (?:final blows?|kills?)\b", re.I)
_WEAPON_CLAIM = re.compile(
    r"\b(?:weapon|gun|primary|special|heavy)\b[^.!?]{0,30}\b(?:final blows?|kills?|damage)\b|"
    r"\b(?:final blows?|kills?|damage)\b[^.!?]{0,30}\bwith (?:a |your )?(?:weapon|gun)s?\b",
    re.I,
)
_ABILITY_CLAIM = re.compile(
    r"\b(?:grenade|melee|ability|abilities|super|rift|barricade|class ability)\b"
    r"[^.!?]{0,30}\b(?:final blows?|kills?|damage)\b",
    re.I,
)


def _key(value: str) -> str:
    return normalize_text(value)


def _words(text: str) -> set[str]:
    found: set[str] = set()
    for word in _WORD.findall(text.casefold()):
        if word in _STOP or len(word) < 3:
            continue
        found.add(word[:-1] if word.endswith("s") and len(word) > 4 else word)
    return found


@dataclass
class EntityEvidence:
    """Everything one turn retrieved about one named entity."""

    name: str
    texts: list[str] = field(default_factory=list)
    sources: set[str] = field(default_factory=set)
    slot: str | None = None
    class_restriction: str | None = None
    equipped: bool = False
    # Manifest socket structure by column ("barrels", "frames"): the configurability the build
    # tools reported for this item's sockets. Observation only.
    socket_classes: dict[str, str] = field(default_factory=dict)

    @property
    def has_text(self) -> bool:
        return bool(self.texts)

    def add_text(self, text: str, source: str) -> None:
        cleaned = " ".join(text.split())[:MAX_TEXT_LENGTH]
        if cleaned and cleaned not in self.texts and len(self.texts) < MAX_TEXTS_PER_ENTITY:
            self.texts.append(cleaned)
        self.sources.add(source)


_EQUIPPED_KEYS = frozenset(
    {"equipped_weapons", "equipped_armor", "equipped_exotics", "subclass", "armor_mods",
     "weapons", "armor", "mods", "equipped_plug_effects", "equipped_mechanics"}
)  # fmt: skip
_NAME_LIST_KEYS = frozenset(
    {"perks_and_sockets", "socketed_plug_names", "mods", "fragments", "aspects", "abilities",
     "other_build_affecting_sockets"}
)  # fmt: skip
_TEXT_KEYS = ("description", "progress_description")
# Lists of short effect strings (M1 plug evidence: Bungie sandbox-perk summaries).
_TEXT_LIST_KEYS = ("sandbox_effects",)
# An item's effect often lives in a plug it carries (an Exotic's intrinsic trait), so the
# plug's text also counts as retrieved text for the item.
_CARRIED_KEYS = ("default_socket_plugs", "intrinsic_perks", "socketed_plugs")


class MechanicsEvidenceLedger:
    """What the tools returned in one turn, reduced to what claim checking needs.

    `record` only reads tool results. It keeps bounded verbatim effect text; it stores no
    identifiers and never modifies anything it is given.
    """

    def __init__(self) -> None:
        self.entities: dict[str, EntityEvidence] = {}
        self.not_found: set[str] = set()
        self.legality: dict[str, set[str]] = {}
        self.equipped_plugs: set[str] = set()
        self.tools: list[str] = []
        self.web_research = False

    def note_web_research(self) -> None:
        self.web_research = True

    def entity(self, name: str) -> EntityEvidence | None:
        return self.entities.get(_key(name))

    def _touch(self, name: str) -> EntityEvidence | None:
        key = _key(name)
        if len(key) < MIN_ENTITY_KEY_LENGTH:
            return None
        existing = self.entities.get(key)
        if existing is not None:
            return existing
        if len(self.entities) >= MAX_ENTITIES:
            return None
        created = EntityEvidence(name=name.strip())
        self.entities[key] = created
        return created

    def add_legality(self, name: str, evidence: str = "socket_options") -> None:
        """Register retrieved evidence that a configuration of `name` is legal (M3+)."""

        self.legality.setdefault(_key(name), set()).add(evidence)

    def record(self, tool_name: str, category: str, result: Any) -> None:
        if tool_name not in self.tools:
            self.tools.append(tool_name)
        source = f"{category}:{tool_name}"
        if isinstance(result, dict):
            if result.get("found") is False and isinstance(result.get("query"), str):
                self.not_found.add(_key(result["query"]))
            socket_options = result.get("socket_options")
            if isinstance(socket_options, dict):
                for name in socket_options:
                    self.add_legality(str(name))
        self._walk(result, source, category == "guardian", None, 0)

    def _walk(
        self, node: Any, source: str, guardian: bool, equipped: bool | None, depth: int
    ) -> None:
        if depth > MAX_WALK_DEPTH:
            return
        if isinstance(node, list):
            for value in node:
                self._walk(value, source, guardian, equipped, depth + 1)
            return
        if not isinstance(node, dict):
            return
        is_equipped = bool(equipped) or node.get("equipped") is True
        is_equipped = is_equipped or node.get("location") == "equipped"
        name = node.get("name")
        if isinstance(name, str) and name.strip():
            self._record_entity(node, name, source, is_equipped)
        plug_name = node.get("equipped_plug")
        if isinstance(plug_name, str) and plug_name.strip():
            # A plug on the owned copy (build tools' socket evidence), not a Manifest sample.
            self._record_name_only(plug_name, source, True)
        for key, value in node.items():
            child_equipped = is_equipped or (guardian and key in _EQUIPPED_KEYS)
            if key in _NAME_LIST_KEYS and isinstance(value, list):
                for entry in value:
                    if isinstance(entry, str):
                        self._record_name_only(entry, source, child_equipped)
                    else:
                        self._walk(entry, source, guardian, child_equipped, depth + 1)
            elif isinstance(value, (dict, list)):
                self._walk(value, source, guardian, child_equipped, depth + 1)

    def _record_name_only(self, name: str, source: str, equipped: bool) -> None:
        entity = self._touch(name)
        if entity is None:
            return
        entity.sources.add(source)
        if equipped:
            entity.equipped = True
            self.equipped_plugs.add(_key(name))

    def _record_entity(self, node: dict[str, Any], name: str, source: str, equipped: bool) -> None:
        entity = self._touch(name)
        if entity is None:
            return
        entity.sources.add(source)
        for field_name in _TEXT_KEYS:
            text = node.get(field_name)
            if isinstance(text, str) and text.strip():
                entity.add_text(text, source)
        item_type = node.get("item_type")
        if item_type == "Armor":
            slot = (
                node.get("equipment_slot")
                or node.get("slot")
                or node.get("bucket")
                or node.get("subtype")
            )
            if isinstance(slot, str) and slot.casefold() in ARMOR_SLOTS:
                entity.slot = ARMOR_SLOTS[slot.casefold()]
        restriction = node.get("class_restriction")
        if isinstance(restriction, dict):
            restriction = restriction.get("name")
        if isinstance(restriction, str) and restriction.casefold() in CLASS_NAMES:
            entity.class_restriction = restriction.casefold()
        sockets = node.get("sockets")
        for socket in sockets if isinstance(sockets, list) else []:
            if isinstance(socket, dict) and isinstance(socket.get("configurability"), str):
                column = socket.get("column")
                if isinstance(column, str) and column.strip():
                    entity.socket_classes[column.strip().casefold()] = socket["configurability"]
        for list_key in _TEXT_LIST_KEYS:
            values = node.get(list_key)
            for text in values if isinstance(values, list) else []:
                if isinstance(text, str) and text.strip():
                    entity.add_text(text, source)
        for carried_key in _CARRIED_KEYS:
            carried = node.get(carried_key)
            for plug in carried if isinstance(carried, list) else []:
                text = plug.get("description") if isinstance(plug, dict) else None
                if isinstance(text, str) and text.strip():
                    entity.add_text(text, source)
        if equipped:
            entity.equipped = True
            if item_type not in {"Armor", "Weapon"}:
                self.equipped_plugs.add(_key(name))

    def summary(self) -> dict[str, Any]:
        with_text = sum(value.has_text for value in self.entities.values())
        return {
            "tools": list(self.tools),
            "entities": len(self.entities),
            "entities_with_effect_text": with_text,
            "entities_names_only": len(self.entities) - with_text,
            "not_found_queries": len(self.not_found),
            "legality_evidence_entities": len(self.legality),
            "web_research": self.web_research,
        }


@dataclass(frozen=True)
class MechanicsClaim:
    category: ClaimCategory
    entity: str
    status: ClaimStatus
    evidence_sources: tuple[str, ...] = ()
    suspected_missing: tuple[MissingEvidence, ...] = ()
    sentence_index: int = 0
    # What the Manifest says about the socket a proposed change targets (for example
    # "fixed_per_manifest"). Context for the reader; the status is unchanged by it.
    manifest_structure: str | None = None

    def trace(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "category": self.category,
            "entity": self.entity,
            "status": self.status,
            "evidence_sources": list(self.evidence_sources),
            "suspected_missing": list(self.suspected_missing),
            "sentence": self.sentence_index,
        }
        if self.manifest_structure:
            result["manifest_structure"] = self.manifest_structure
        return result


@dataclass(frozen=True)
class MechanicsClaimReport:
    claims: tuple[MechanicsClaim, ...]
    sentences_checked: int
    hedged_sentences: int
    retrieval: dict[str, Any]

    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for claim in self.claims:
            counts[claim.status] = counts.get(claim.status, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def needs_attention(self) -> int:
        return sum(claim.status in UNSUPPORTED_STATUSES for claim in self.claims)

    def to_trace(self) -> dict[str, Any]:
        return {
            "mode": "trace_only",
            "basis": "retrieved_evidence_only",
            "claims_total": len(self.claims),
            "needs_attention": self.needs_attention,
            "by_status": self.counts(),
            "claims": [claim.trace() for claim in self.claims[:MAX_TRACED_CLAIMS]],
            "sentences_checked": self.sentences_checked,
            "hedged_sentences": self.hedged_sentences,
            "retrieval": self.retrieval,
        }


def _entity_mentions(sentence_key: str, ledger: MechanicsEvidenceLedger) -> list[EntityEvidence]:
    padded = f" {sentence_key} "
    found: list[EntityEvidence] = []
    for key, entity in ledger.entities.items():
        if key in _NOT_ENTITY or (f" {key} " not in padded and f" {key}s " not in padded):
            continue
        # A name inside a longer matched name is the same mention ("Chambered Compensator").
        if any(key != other and key in _key(other.name) for other in found):
            continue
        found.append(entity)
    return found


def _is_mechanic_sentence(sentence: str) -> bool:
    return bool(
        _MECHANIC_NOUN_RE.search(sentence)
        and (_EFFECT_VERB_RE.search(sentence) or _CONDITIONAL_RE.search(sentence))
    )


def _slot_in(sentence: str) -> set[str]:
    return {ARMOR_SLOTS[value.casefold()] for value in _SLOT_WORD.findall(sentence)}


def _classes_in(sentence: str) -> set[str]:
    found: set[str] = set()
    for match in _IDENTITY_CLASS_CUE.finditer(sentence):
        value = next((group for group in match.groups() if group), None)
        if value:
            found.add(value.casefold())
    return found


_SOCKET_NOUN_COLUMNS = {
    "barrel": "barrels",
    "magazine": "magazines",
    "mag": "magazines",
    "grip": "grips",
    "stock": "stocks",
    "scope": "scopes",
    "battery": "batteries",
    "blade": "blades",
    "guard": "guards",
    "haft": "hafts",
    "bowstring": "bowstrings",
    "arrow": "arrows",
    "frame": "frames",
}


def _manifest_structure(sentence: str, subject: EntityEvidence | None) -> str | None:
    """The reported configurability of the one socket column a sentence names, if known."""

    if subject is None or not subject.socket_classes:
        return None
    columns = {
        column
        for noun, column in _SOCKET_NOUN_COLUMNS.items()
        if re.search(rf"\b{noun}s?\b", sentence, re.I)
    }
    known = {subject.socket_classes[value] for value in columns if value in subject.socket_classes}
    return next(iter(known)) if len(known) == 1 else None


def _classify_configuration(
    sentence: str, mentions: list[EntityEvidence], ledger: MechanicsEvidenceLedger, index: int
) -> list[MechanicsClaim]:
    """Proposed change versus statement about what is equipped now."""

    plug_names = [value for value in mentions if _key(value.name) in ledger.equipped_plugs]
    non_equipped = [value for value in mentions if value not in plug_names]
    proposes = bool(_CHANGE_VERBS.search(sentence) and _SOCKET_NOUNS.search(sentence))
    states_equipped = bool(_KEEP_VERBS.search(sentence)) and not _CHANGE_VERBS.search(sentence)
    says_equipped_now = bool(_EQUIPPED_NOW.search(sentence)) and not _CHANGE_VERBS.search(sentence)
    if proposes:
        subject = next((value for value in mentions if value.socket_classes), None)
        subject = subject or next(
            (value for value in mentions if value.slot or value.class_restriction), None
        )
        subject = subject or (mentions[0] if mentions else None)
        name = subject.name if subject else "unspecified"
        legal = any(_key(value.name) in ledger.legality for value in mentions)
        structure = _manifest_structure(sentence, subject)
        return [
            MechanicsClaim(
                "configuration",
                name,
                "config_change_with_legality" if legal else "config_change_without_legality",
                tuple(sorted(subject.sources)) if subject else (),
                () if legal else ("socket_options",),
                index,
                structure,
            )
        ]
    if states_equipped and plug_names:
        return [
            MechanicsClaim(
                "configuration",
                value.name,
                "equipped_configuration",
                tuple(sorted(value.sources)),
                (),
                index,
            )
            for value in plug_names
        ]
    if says_equipped_now and non_equipped and _SOCKET_NOUNS.search(sentence):
        # Described as equipped, yet this turn only retrieved it from knowledge, not Guardian
        # socket data. Items are excluded: ownership has its own validator.
        return [
            MechanicsClaim(
                "configuration",
                value.name,
                "equipped_claim_without_evidence",
                tuple(sorted(value.sources)),
                ("socket_options",),
                index,
            )
            for value in non_equipped
            if value.slot is None and value.class_restriction is None
        ]
    return []


def detect_mechanics_claims(
    answer: str,
    ledger: MechanicsEvidenceLedger,
    *,
    entity_resolver: Callable[[str], bool] | None = None,
) -> MechanicsClaimReport:
    """Label mechanics claims in `answer` by the evidence this turn retrieved.

    `entity_resolver(name)` may report whether the Manifest knows a name; without it, an
    unretrieved capitalized actor is reported as `mentioned_without_evidence`, never as
    unresolved. Nothing here changes `answer`.
    """

    claims: list[MechanicsClaim] = []
    hedged = 0
    sentences = [value for value in _SPLIT.split(answer) if value.strip()]
    seen: set[tuple[str, str, str]] = set()

    def add(claim: MechanicsClaim) -> None:
        marker = (claim.category, _key(claim.entity), claim.status)
        if marker not in seen:
            seen.add(marker)
            claims.append(claim)

    for index, raw in enumerate(sentences):
        sentence = raw.strip(" \t-*•#>")
        if _HEDGE.search(sentence):
            hedged += 1
            continue
        key = _key(sentence)
        mentions = _entity_mentions(key, ledger)
        for claim in _classify_configuration(sentence, mentions, ledger, index):
            add(claim)
        proposes_change = bool(_CHANGE_VERBS.search(sentence) and _SOCKET_NOUNS.search(sentence))

        for entity in mentions:
            for claim in _identity_claims(sentence, entity, index):
                add(claim)
            damage = _damage_source_claim(sentence, entity, index)
            if damage is not None:
                add(damage)

        if proposes_change or not _is_mechanic_sentence(sentence):
            continue

        covered: set[str] = set()
        for entity in mentions:
            if _key(entity.name) in ledger.not_found:
                add(
                    MechanicsClaim(
                        "item_effect",
                        entity.name,
                        "unresolved_entity",
                        (),
                        ("entity_resolution",),
                        index,
                    )
                )
                covered.add(_key(entity.name))
                continue
            located = re.search(re.escape(entity.name), sentence, re.I)
            if located and _NEGATED_ENTITY.search(sentence[: located.start()][-24:]):
                continue
            covered.add(_key(entity.name))
            sources = tuple(sorted(entity.sources))
            if not entity.has_text:
                add(
                    MechanicsClaim(
                        "item_effect",
                        entity.name,
                        "mentioned_without_evidence",
                        sources,
                        ("effect_text",),
                        index,
                    )
                )
                continue
            claim_words = _words(sentence) - _words(entity.name)
            evidence_words = set().union(*(_words(text) for text in entity.texts))
            shared = claim_words & evidence_words
            status: ClaimStatus = (
                "supported" if len(shared) >= OVERLAP_MIN_SHARED_WORDS
                else "evidence_retrieved_no_overlap"
            )  # fmt: skip
            add(MechanicsClaim("item_effect", entity.name, status, sources, (), index))

        for match in _NAMED_ACTOR_RE.finditer(sentence):
            actor = match.group(1).strip()
            actor_key = _key(actor)
            if (
                actor_key in _NOT_ENTITY
                or len(actor_key) < MIN_ENTITY_KEY_LENGTH
                or actor_key in covered
                or actor_key in ledger.entities
            ):
                continue
            if actor.split()[0].casefold() in _NOT_ENTITY and len(actor.split()) == 1:
                continue
            if actor_key in ledger.not_found or (
                entity_resolver is not None and not entity_resolver(actor)
            ):
                status = "unresolved_entity"
                missing: tuple[MissingEvidence, ...] = ("entity_resolution",)
            else:
                status = "mentioned_without_evidence"
                missing = ("effect_text",)
            add(MechanicsClaim("item_effect", actor, status, (), missing, index))

    return MechanicsClaimReport(
        claims=tuple(claims),
        sentences_checked=len(sentences),
        hedged_sentences=hedged,
        retrieval=ledger.summary(),
    )


def _identity_claims(sentence: str, entity: EntityEvidence, index: int) -> Iterable[MechanicsClaim]:
    slots = _slot_in(sentence)
    classes = _classes_in(sentence)
    sources = tuple(sorted(entity.sources))
    if len(slots) == 1:
        slot = next(iter(slots))
        if entity.slot is None:
            yield MechanicsClaim(
                "identity", entity.name, "mentioned_without_evidence", sources, ("slot",), index
            )
        elif entity.slot != slot:
            yield MechanicsClaim(
                "identity", entity.name, "contradicted_by_evidence", sources, (), index
            )
        else:
            yield MechanicsClaim("identity", entity.name, "supported", sources, (), index)
    if len(classes) == 1:
        stated = next(iter(classes))
        if entity.class_restriction is None:
            yield MechanicsClaim(
                "identity", entity.name, "mentioned_without_evidence", sources, ("class",), index
            )
        elif entity.class_restriction != stated:
            yield MechanicsClaim(
                "identity", entity.name, "contradicted_by_evidence", sources, (), index
            )
        else:
            yield MechanicsClaim("identity", entity.name, "supported", sources, (), index)


def _damage_source_claim(
    sentence: str, entity: EntityEvidence, index: int
) -> MechanicsClaim | None:
    """Ability-versus-weapon final-blow requirement stated in retrieved objective text."""

    text = " ".join(entity.texts)
    required_ability = bool(_ABILITY_SOURCE.search(text))
    required_weapon = bool(_WEAPON_SOURCE.search(text))
    if required_ability == required_weapon:
        return None
    sources = tuple(sorted(entity.sources))
    says_weapon = bool(_WEAPON_CLAIM.search(sentence)) and not _ABILITY_CLAIM.search(sentence)
    says_ability = bool(_ABILITY_CLAIM.search(sentence)) and not _WEAPON_CLAIM.search(sentence)
    if required_ability and says_weapon or required_weapon and says_ability:
        return MechanicsClaim(
            "damage_source", entity.name, "contradicted_by_evidence", sources, (), index
        )
    if required_ability and says_ability or required_weapon and says_weapon:
        return MechanicsClaim("damage_source", entity.name, "supported", sources, (), index)
    return None
