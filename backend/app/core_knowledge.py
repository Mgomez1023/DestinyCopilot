"""Destiny Knowledge Engine v0.1: curated core concepts, rules, aliases, and facts.

The engine is pure, in-memory, synchronous, and deterministic. It reads its corpus once at
construction and afterwards performs only exact, bounded dictionary lookups: no network,
no Manifest fuzzy search, no disk access, and no model calls. It knows; the model reasons.
"""

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from app.knowledge_models import (
    CORE_CLASS_VALUES,
    CoreEntity,
    CoreFact,
    CoreKnowledgeCorpus,
    CoreManifestRefPattern,
    CorePredicate,
)

MAX_SPAN_TOKENS = 6
MAX_BUNDLE_ENTITIES = 5
MAX_BUNDLE_FACTS_PER_ENTITY = 6
MAX_BUNDLE_RULES = 3
MAX_BUNDLE_AMBIGUITIES = 2
# About 600 tokens at the usual ~4 characters per token.
MAX_BUNDLE_CHARS = 2400
# Any alias this short and single-token must be registered as explicit shorthand.
SHORT_ALIAS_MAX_CHARS = 4
SHORTHAND_PATTERN = re.compile(r"^[a-z0-9]{2,8}$")
MANIFEST_REF_PATTERN = re.compile(CoreManifestRefPattern)

_APOSTROPHES = re.compile(r"['‘’ʼ`]")
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")

ResolutionStatus = Literal["entity", "ambiguous", "unknown"]
MatchVia = Literal["canonical", "alias", "misspelling", "plural", "shorthand", "ambiguity"]
MentionOrigin = Literal["message", "candidate", "rules_question"]


# A message that asks whether gear may be worn, slotted, or combined is a rules question even
# when it names no concept the corpus has an alias for (it may name only a specific item).
_EQUIP_VERB = re.compile(
    r"\b(?:wear|wearing|equip|equipped|equipping|slot|slotted|stack|stacking|combine|combining|"
    r"pair|put on|keep\b[^?.!]{0,30}\bon)\b",
    re.I,
)
_RULE_FRAME = re.compile(
    r"\b(?:can|could|may|allowed|able|possible|limit|limits|restriction|restrictions|rule|"
    r"rules|only one|how many|at the same time|at once|both|two|more than one|"
    r"same time|simultaneously)\b",
    re.I,
)


def asks_equip_rule(text: str) -> bool:
    """Whether the text asks what gear may be equipped or combined (a stable-rules question)."""

    return bool(_EQUIP_VERB.search(text) and _RULE_FRAME.search(text))


def normalize_text(value: str) -> str:
    """NFKD, ASCII-fold, casefold, drop apostrophes, and collapse punctuation to spaces."""

    decomposed = _APOSTROPHES.sub("", unicodedata.normalize("NFKD", value))
    # Non-ASCII punctuation, symbols, and separators (such as an em dash) become spaces
    # before ASCII folding, so folding cannot glue neighboring words together.
    spaced = "".join(
        " " if unicodedata.category(character)[0] in "PSZ" else character
        for character in decomposed
    )
    folded = spaced.encode("ascii", "ignore").decode("ascii").casefold()
    return " ".join(_NON_ALPHANUMERIC.sub(" ", folded).split())


class CoreKnowledgeError(ValueError):
    """Raised when a corpus is malformed; lists every problem found."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("Invalid core knowledge corpus: " + "; ".join(errors))


@dataclass(frozen=True, slots=True)
class _AliasTarget:
    kind: Literal["entity", "ambiguity"]
    id: str
    via: MatchVia


@dataclass(frozen=True, slots=True)
class Resolution:
    """The exact-match outcome for one normalized phrase."""

    status: ResolutionStatus
    query: str
    normalized: str
    via: MatchVia | None = None
    entity_id: str | None = None
    candidate_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Mention:
    """An exact span match found in text; spans never overlap."""

    origin: MentionOrigin
    start_token: int
    end_token: int
    text: str
    resolution: Resolution


@dataclass(frozen=True, slots=True)
class GroundedFact:
    predicate: str
    value: bool | int | str
    source_kinds: tuple[str, ...]
    confidence: str


@dataclass(frozen=True, slots=True)
class BundleMatch:
    """How one bundled entity, rule, or ambiguity group was matched (for traces)."""

    id: str
    via: MatchVia
    origin: MentionOrigin


@dataclass(frozen=True, slots=True)
class FactBundle:
    """A bounded, deterministic evidence payload. It carries facts, never instructions."""

    payload: dict[str, Any]
    text: str
    truncated: bool
    # What the rendered payload contains, in payload order; ids never reach the model.
    matches: tuple[BundleMatch, ...] = ()

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass(slots=True)
class _Selection:
    entities: list[tuple[str, Mention]] = field(default_factory=list)
    rules: list[tuple[str, Mention]] = field(default_factory=list)
    ambiguities: list[tuple[str, Mention]] = field(default_factory=list)


def _alias_keys(entity: CoreEntity) -> list[tuple[str, MatchVia]]:
    keys: list[tuple[str, MatchVia]] = [(normalize_text(entity.name), "canonical")]
    keys.extend((normalize_text(alias.text), alias.type) for alias in entity.aliases)
    if entity.plural:
        keys.extend((f"{key}s", "plural") for key, _ in list(keys) if key and not key.endswith("s"))
    return [(key, via) for key, via in keys if key]


def _value_error(predicate_name: str, predicate: CorePredicate, fact: CoreFact) -> str | None:
    value = fact.o
    kind = predicate.value
    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            return f"predicate '{predicate_name}' requires an integer value"
    elif kind == "bool":
        if not isinstance(value, bool):
            return f"predicate '{predicate_name}' requires a boolean value"
    elif not isinstance(value, str) or not value.strip():
        return f"predicate '{predicate_name}' requires a non-empty text value"
    elif kind == "enum" and value not in predicate.values:
        return f"'{value}' is not an allowed value for predicate '{predicate_name}'"
    elif kind == "class" and value not in CORE_CLASS_VALUES:
        return f"'{value}' is not a valid class for predicate '{predicate_name}'"
    elif kind == "entity" and fact.o_ref is None:
        return f"predicate '{predicate_name}' requires an o_ref entity reference"
    if kind != "entity" and fact.o_ref is not None:
        return f"predicate '{predicate_name}' does not accept an o_ref"
    return None


def validate_corpus(corpus: CoreKnowledgeCorpus) -> None:
    """Enforce every cross-record rule, collecting all problems before raising."""

    errors: list[str] = []
    for source_id, source in corpus.sources.items():
        where = f"source '{source_id}'"
        if source.kind == "manifest_text" and (
            not source.ref or not MANIFEST_REF_PATTERN.fullmatch(source.ref) or not source.quote
        ):
            errors.append(f"{where}: manifest_text needs a '<Type>:<hash>' ref and a quote")
        elif source.kind == "manifest_structure" and not source.ref:
            errors.append(f"{where}: manifest_structure needs a ref describing the structure")
        elif source.kind == "official_bungie" and not (source.ref or "").startswith("https://"):
            errors.append(f"{where}: official_bungie needs an https ref")

    entity_ids: set[str] = set()
    for entity in corpus.entities:
        if entity.id in entity_ids:
            errors.append(f"duplicate entity id '{entity.id}'")
        entity_ids.add(entity.id)

    for entity in corpus.entities:
        where = f"entity '{entity.id}'"
        for source in entity.summary_src:
            if source not in corpus.sources:
                errors.append(f"{where}: summary cites unknown source '{source}'")
        volatile_predicates = [
            fact.p
            for fact in entity.facts
            if fact.p in corpus.predicates and corpus.predicates[fact.p].freshness != "stable"
        ]
        if (entity.freshness != "stable" or volatile_predicates) and entity.review_after is None:
            errors.append(
                f"{where}: non-stable knowledge needs review_after "
                f"(freshness={entity.freshness}, predicates={volatile_predicates})"
            )
        for fact in entity.facts:
            predicate = corpus.predicates.get(fact.p)
            if predicate is None:
                errors.append(f"{where}: unknown predicate '{fact.p}'")
                continue
            if not predicate.curatable:
                errors.append(
                    f"{where}: predicate '{fact.p}' is owned by the Manifest and cannot be curated"
                )
            elif predicate.owner == "manifest_preferred" and not fact.manifest_absent_reason:
                errors.append(
                    f"{where}: curated '{fact.p}' needs a manifest_absent_reason because the "
                    "Manifest is the preferred source"
                )
            if (problem := _value_error(fact.p, predicate, fact)) is not None:
                errors.append(f"{where}: {problem}")
            if fact.o_ref is not None and fact.o_ref not in entity_ids:
                errors.append(f"{where}: fact '{fact.p}' references unknown entity '{fact.o_ref}'")
            for source in fact.src:
                if source not in corpus.sources:
                    errors.append(f"{where}: fact '{fact.p}' cites unknown source '{source}'")
        for relation in entity.related:
            if relation.id not in entity_ids:
                errors.append(f"{where}: related entity '{relation.id}' does not exist")
            elif relation.id == entity.id:
                errors.append(f"{where}: an entity cannot relate to itself")
        for alias in entity.aliases:
            key = normalize_text(alias.text)
            if not key:
                errors.append(f"{where}: alias '{alias.text}' is empty after normalization")
            elif " " not in key and len(key) <= SHORT_ALIAS_MAX_CHARS:
                errors.append(
                    f"{where}: short alias '{alias.text}' must be registered as explicit shorthand"
                )

    for token, shorthand in corpus.shorthands.items():
        if not SHORTHAND_PATTERN.fullmatch(token):
            errors.append(f"shorthand '{token}' must be one lowercase alphanumeric token")
        if shorthand.entity not in entity_ids:
            errors.append(f"shorthand '{token}' targets unknown entity '{shorthand.entity}'")

    group_ids: set[str] = set()
    for group in corpus.ambiguity_groups:
        where = f"ambiguity group '{group.id}'"
        if group.id in group_ids or group.id in entity_ids:
            errors.append(f"{where}: duplicate id")
        group_ids.add(group.id)
        if len(set(group.candidates)) != len(group.candidates):
            errors.append(f"{where}: candidates must be unique")
        if len(set(group.candidates)) < 2:
            errors.append(f"{where}: needs at least two distinct candidates")
        for candidate in group.candidates:
            if candidate not in entity_ids:
                errors.append(f"{where}: candidate '{candidate}' does not exist")

    # Every normalized key may belong to exactly one owner, including generated plurals.
    owners: dict[str, str] = {}

    def claim(key: str, owner: str) -> None:
        if not key:
            return
        previous = owners.setdefault(key, owner)
        if previous != owner:
            errors.append(f"alias collision: '{key}' is claimed by '{previous}' and '{owner}'")

    for entity in corpus.entities:
        for key, _ in _alias_keys(entity):
            claim(key, entity.id)
    for token in corpus.shorthands:
        claim(normalize_text(token), f"shorthand:{token}")
    for group in corpus.ambiguity_groups:
        for alias in group.aliases:
            claim(normalize_text(alias), group.id)

    if errors:
        raise CoreKnowledgeError(errors)


def overdue_reviews(corpus: CoreKnowledgeCorpus, today: date) -> list[str]:
    """Entity ids whose review_after date has passed.

    Never raised by the loader: an overdue review must not take knowledge offline at
    runtime. The production-corpus test fails on it instead, forcing re-verification.
    """

    return [
        entity.id
        for entity in corpus.entities
        if entity.review_after is not None and entity.review_after < today
    ]


def corpus_audit(corpus: CoreKnowledgeCorpus) -> dict[str, Any]:
    """Counts and alias exposure, so corpus growth and risky aliases stay visible."""

    single_token: dict[str, str] = {}
    for entity in corpus.entities:
        for key, _ in _alias_keys(entity):
            if " " not in key:
                single_token[key] = entity.id
    for token, shorthand in corpus.shorthands.items():
        single_token[normalize_text(token)] = shorthand.entity
    for group in corpus.ambiguity_groups:
        for alias in group.aliases:
            key = normalize_text(alias)
            if " " not in key:
                single_token[key] = group.id
    return {
        "entities": len(corpus.entities),
        "by_kind": dict(sorted(Counter(entity.kind for entity in corpus.entities).items())),
        "facts": sum(len(entity.facts) for entity in corpus.entities),
        "shorthands": sorted(corpus.shorthands),
        "ambiguity_groups": sorted(group.id for group in corpus.ambiguity_groups),
        "single_token_keys": dict(sorted(single_token.items())),
        "live_sensitive": sorted(entity.id for entity in corpus.entities if entity.live_sensitive),
        "non_stable": sorted(
            entity.id for entity in corpus.entities if entity.review_after is not None
        ),
    }


def load_core_knowledge(path: Path) -> CoreKnowledgeCorpus:
    """Read, parse, and fully validate a corpus file. The only disk read in this module."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise CoreKnowledgeError([f"cannot read corpus {path.name}: {exc}"]) from exc
    return parse_core_knowledge(payload)


def parse_core_knowledge(payload: Any) -> CoreKnowledgeCorpus:
    try:
        corpus = CoreKnowledgeCorpus.model_validate(payload)
    except ValidationError as exc:
        raise CoreKnowledgeError(
            [
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in exc.errors()
            ]
        ) from exc
    validate_corpus(corpus)
    return corpus


class DestinyKnowledgeEngine:
    """The single public interface to curated core Destiny knowledge."""

    def __init__(self, corpus: CoreKnowledgeCorpus) -> None:
        validate_corpus(corpus)
        # A private deep copy keeps the engine deterministic even if the caller later
        # mutates the corpus objects it passed in.
        corpus = corpus.model_copy(deep=True)
        self.corpus_version = corpus.corpus_version
        self._predicates = dict(corpus.predicates)
        self._sources = dict(corpus.sources)
        self._entities = {entity.id: entity for entity in corpus.entities}
        self._groups = {group.id: group for group in corpus.ambiguity_groups}
        index: dict[str, _AliasTarget] = {}
        for entity in corpus.entities:
            for key, via in _alias_keys(entity):
                index.setdefault(key, _AliasTarget("entity", entity.id, via))
        for token, shorthand in corpus.shorthands.items():
            index[normalize_text(token)] = _AliasTarget("entity", shorthand.entity, "shorthand")
        for group in corpus.ambiguity_groups:
            for alias in group.aliases:
                index[normalize_text(alias)] = _AliasTarget("ambiguity", group.id, "ambiguity")
        self._index = index

    @property
    def entity_count(self) -> int:
        return len(self._entities)

    @classmethod
    def from_file(cls, path: Path) -> "DestinyKnowledgeEngine":
        return cls(load_core_knowledge(path))

    def resolve(self, text: str) -> Resolution:
        """Resolve one whole phrase exactly. Never guesses and never picks among candidates."""

        normalized = normalize_text(text)
        target = self._index.get(normalized)
        if target is None:
            return Resolution(status="unknown", query=text, normalized=normalized)
        if target.kind == "ambiguity":
            return Resolution(
                status="ambiguous",
                query=text,
                normalized=normalized,
                via="ambiguity",
                candidate_ids=tuple(self._groups[target.id].candidates),
            )
        return Resolution(
            status="entity",
            query=text,
            normalized=normalized,
            via=target.via,
            entity_id=target.id,
        )

    def mentions(self, text: str, *, origin: MentionOrigin = "message") -> list[Mention]:
        """Find exact whole-token spans, left to right, longest span first, without overlap."""

        tokens = normalize_text(text).split()
        found: list[Mention] = []
        position = 0
        while position < len(tokens):
            for length in range(min(MAX_SPAN_TOKENS, len(tokens) - position), 0, -1):
                span = " ".join(tokens[position : position + length])
                if span in self._index:
                    found.append(
                        Mention(
                            origin=origin,
                            start_token=position,
                            end_token=position + length,
                            text=span,
                            resolution=self.resolve(span),
                        )
                    )
                    position += length
                    break
            else:
                position += 1
        return found

    def facts_for(self, entity_id: str) -> list[GroundedFact]:
        entity = self._entities[entity_id]
        return [
            GroundedFact(
                predicate=fact.p,
                value=fact.o,
                source_kinds=tuple(dict.fromkeys(self._sources[value].kind for value in fact.src)),
                confidence=fact.confidence,
            )
            for fact in entity.facts
        ]

    def lookup(self, name: str) -> dict[str, Any]:
        """A JSON-safe card for one name: an entity, an explicit ambiguity, or unknown."""

        resolution = self.resolve(name)
        if resolution.status == "entity":
            assert resolution.entity_id is not None
            return {
                "status": "entity",
                "matched_via": resolution.via,
                "entity": self._card(resolution.entity_id, fact_limit=None),
            }
        if resolution.status == "ambiguous":
            return {
                "status": "ambiguous",
                "alias": resolution.normalized,
                "candidates": [self._brief(value) for value in resolution.candidate_ids],
            }
        return {"status": "unknown", "query": name}

    def bundle(self, message: str, candidate_names: tuple[str, ...] = ()) -> FactBundle | None:
        """Build the bounded grounded-facts payload, or None when nothing matched.

        A message asking an equip/combination rules question also carries every curated rule,
        because such a question often names only items, never the rule's own aliases.
        """

        mentions = self.mentions(message, origin="message")
        for name in candidate_names:
            mentions.extend(self.mentions(name, origin="candidate"))
        selection = self._select(mentions)
        if asks_equip_rule(message):
            known = {entity_id for entity_id, _ in selection.rules}
            for entity_id, entity in self._entities.items():
                if entity.kind == "rule" and entity_id not in known:
                    selection.rules.append((entity_id, self._rule_mention(entity_id)))
        if not (selection.entities or selection.rules or selection.ambiguities):
            return None

        omitted = {"entities": 0, "rules": 0, "ambiguity_groups": 0}
        # Ambiguity groups are never dropped for size. Beyond the count cap they are
        # reported as omitted so the omission is explicit, never silent.
        ambiguities = selection.ambiguities[:MAX_BUNDLE_AMBIGUITIES]
        omitted["ambiguity_groups"] = len(selection.ambiguities) - len(ambiguities)
        entities = selection.entities[:MAX_BUNDLE_ENTITIES]
        omitted["entities"] = len(selection.entities) - len(entities)
        rules = selection.rules[:MAX_BUNDLE_RULES]
        omitted["rules"] = len(selection.rules) - len(rules)

        while True:
            payload = self._payload(entities, rules, ambiguities, omitted)
            text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            if len(text) <= MAX_BUNDLE_CHARS:
                break
            # Lowest priority first: rules, then candidate entities, then message entities.
            if rules:
                rules = rules[:-1]
                omitted["rules"] += 1
            elif any(mention.origin == "candidate" for _, mention in entities):
                last = max(
                    index
                    for index, (_, mention) in enumerate(entities)
                    if mention.origin == "candidate"
                )
                entities = entities[:last] + entities[last + 1 :]
                omitted["entities"] += 1
            elif entities:
                entities = entities[:-1]
                omitted["entities"] += 1
            else:
                break
        matches = tuple(
            BundleMatch(item_id, mention.resolution.via or "canonical", mention.origin)
            for item_id, mention in (*ambiguities, *entities, *rules)
        )
        return FactBundle(
            payload=payload, text=text, truncated=any(omitted.values()), matches=matches
        )

    def entity_brief(self, entity_id: str) -> dict[str, Any]:
        """Name, kind, and non-current status of one entity."""

        return self._brief(entity_id)

    def located_on(self, entity_id: str) -> str | None:
        """The grounded destination of an entity, when the corpus establishes one."""

        return next(
            (str(fact.o) for fact in self._entities[entity_id].facts if fact.p == "located_on"),
            None,
        )

    def _rule_mention(self, entity_id: str) -> Mention:
        return Mention(
            origin="rules_question",
            start_token=0,
            end_token=0,
            text="",
            resolution=Resolution(
                status="entity",
                query="",
                normalized="",
                via="canonical",
                entity_id=entity_id,
            ),
        )

    def _select(self, mentions: list[Mention]) -> _Selection:
        selection = _Selection()
        seen: set[str] = set()
        # Message mentions come first, so an entity named by the user keeps message priority
        # even when a planning candidate also refers to it.
        ordered = [value for value in mentions if value.origin == "message"]
        ordered.extend(value for value in mentions if value.origin == "candidate")
        for mention in ordered:
            resolution = mention.resolution
            if resolution.status == "ambiguous":
                group_id = self._index[mention.text].id
                if group_id not in seen:
                    seen.add(group_id)
                    selection.ambiguities.append((group_id, mention))
            elif resolution.status == "entity" and resolution.entity_id not in seen:
                entity_id = resolution.entity_id
                assert entity_id is not None
                seen.add(entity_id)
                target = (
                    selection.rules
                    if self._entities[entity_id].kind == "rule"
                    else (selection.entities)
                )
                target.append((entity_id, mention))
        return selection

    def _payload(
        self,
        entities: list[tuple[str, Mention]],
        rules: list[tuple[str, Mention]],
        ambiguities: list[tuple[str, Mention]],
        omitted: dict[str, int],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"corpus_version": self.corpus_version}
        if ambiguities:
            payload["ambiguous"] = [
                {
                    "matched_text": mention.text,
                    "candidates": [
                        self._brief(value) for value in self._groups[group_id].candidates
                    ],
                }
                for group_id, mention in ambiguities
            ]
        if entities:
            payload["entities"] = [
                self._card(entity_id, fact_limit=MAX_BUNDLE_FACTS_PER_ENTITY, mention=mention)
                for entity_id, mention in entities
            ]
        if rules:
            payload["rules"] = [
                self._card(entity_id, fact_limit=MAX_BUNDLE_FACTS_PER_ENTITY, mention=mention)
                for entity_id, mention in rules
            ]
        if any(omitted.values()):
            payload["omitted"] = {key: value for key, value in omitted.items() if value}
        return payload

    def _brief(self, entity_id: str) -> dict[str, Any]:
        entity = self._entities[entity_id]
        brief: dict[str, Any] = {"name": entity.name, "kind": entity.kind}
        if entity.status != "current":
            brief["status"] = entity.status
        return brief

    def _card(
        self, entity_id: str, *, fact_limit: int | None, mention: Mention | None = None
    ) -> dict[str, Any]:
        entity = self._entities[entity_id]
        card = self._brief(entity_id)
        card["summary"] = entity.summary
        if mention is not None and mention.resolution.via not in {None, "canonical"}:
            card["matched"] = {"text": mention.text, "via": mention.resolution.via}
        facts = self.facts_for(entity_id)
        if facts:
            shown = facts if fact_limit is None else facts[:fact_limit]
            card["facts"] = [
                [value.predicate, value.value, "+".join(value.source_kinds)] for value in shown
            ]
            if len(shown) < len(facts):
                card["facts_omitted"] = len(facts) - len(shown)
        if entity.unknown:
            card["unknown"] = list(entity.unknown)
        if entity.live_sensitive:
            card["live_sensitive"] = list(entity.live_sensitive)
        not_same = [
            self._entities[relation.id].name
            for relation in entity.related
            if relation.rel == "variant_not_same_as"
        ]
        if not_same:
            card["not_same_as"] = not_same
        return card
