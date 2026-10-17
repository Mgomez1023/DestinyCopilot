"""Offline Manifest anchor validation for the curated core knowledge corpus.

Checks that every Manifest anchor and every manifest_text source in the corpus still
matches the Bungie Manifest: definitions exist, names match, expected ancestors, sub-areas
and field values hold, and quoted text is still present. It never uses the network and is
never on the chat hot path; it runs in tests (against a committed snapshot) and as a
maintenance command against the local Manifest SQLite database:

    python -m app.core_knowledge_anchors --manifest <world-content.sqlite3> [--write-snapshot P]
"""

import argparse
import json
import sqlite3
import sys
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from app.core_knowledge import load_core_knowledge, normalize_text
from app.knowledge_models import CoreKnowledgeCorpus, CoreManifestAnchor

MAX_ANCESTOR_DEPTH = 8
USUAL_AMMO_MIN_SHARE = 0.95
AMMO_TYPES = {1: "primary", 2: "special", 3: "heavy"}
DEFAULT_CORPUS_PATH = Path(__file__).parent / "data" / "core_knowledge.json"

IssueSeverity = Literal["error", "warning"]


class AnchorDefinitions(Protocol):
    def get(self, definition_type: str, entity_hash: int) -> dict[str, Any] | None: ...


@dataclass(frozen=True, slots=True)
class AnchorIssue:
    severity: IssueSeverity
    subject: str
    message: str

    def __str__(self) -> str:
        return f"{self.severity}: {self.subject}: {self.message}"


def _name(definition: dict[str, Any]) -> str:
    return str((definition.get("displayProperties") or {}).get("name") or "")


def _texts(definition: dict[str, Any]) -> str:
    display = definition.get("displayProperties") or {}
    return " ".join(
        str(value or "")
        for value in (
            display.get("name"),
            display.get("description"),
            definition.get("progressDescription"),
        )
    )


def _path_value(definition: dict[str, Any], path: str) -> Any:
    value: Any = definition
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return _MISSING
        value = value[part]
    return value


_MISSING = object()


def _parse_ref(ref: str) -> tuple[str, int]:
    definition_type, _, entity_hash = ref.partition(":")
    return definition_type, int(entity_hash)


def _ancestor_names(
    definitions: AnchorDefinitions, definition: dict[str, Any]
) -> tuple[list[str], list[int]]:
    names: list[str] = []
    visited: list[int] = []
    parents = list(definition.get("parentNodeHashes") or [])
    while parents and len(visited) < MAX_ANCESTOR_DEPTH:
        parent_hash = int(parents[0])
        if parent_hash in visited:
            break
        visited.append(parent_hash)
        parent = definitions.get("DestinyPresentationNodeDefinition", parent_hash)
        if parent is None:
            break
        names.append(_name(parent))
        parents = list(parent.get("parentNodeHashes") or [])
    return names, visited


def _check_anchor(
    definitions: AnchorDefinitions, subject: str, anchor: CoreManifestAnchor
) -> list[AnchorIssue]:
    if anchor.hash is None:
        return [
            AnchorIssue(
                "error", subject, f"{anchor.type} anchor '{anchor.name}' needs a hash to verify"
            )
        ]
    definition = definitions.get(anchor.type, anchor.hash)
    label = f"{anchor.type}:{anchor.hash}"
    if definition is None:
        return [AnchorIssue("error", subject, f"anchor drift: {label} no longer exists")]
    if definition.get("redacted"):
        return [AnchorIssue("error", subject, f"anchor drift: {label} is redacted")]
    issues: list[AnchorIssue] = []
    if anchor.name is not None and normalize_text(_name(definition)) != normalize_text(anchor.name):
        issues.append(
            AnchorIssue(
                "error",
                subject,
                f"anchor drift: {label} is named '{_name(definition)}', expected '{anchor.name}'",
            )
        )
    if anchor.expect_ancestor is not None:
        names, _ = _ancestor_names(definitions, definition)
        if normalize_text(anchor.expect_ancestor) not in {normalize_text(v) for v in names}:
            issues.append(
                AnchorIssue(
                    "error",
                    subject,
                    f"anchor drift: {label} ancestors {names} lack '{anchor.expect_ancestor}'",
                )
            )
    if anchor.expect_sub_area is not None:
        bubbles = {normalize_text(_name(value)) for value in definition.get("bubbles", []) or []}
        if normalize_text(anchor.expect_sub_area) not in bubbles:
            issues.append(
                AnchorIssue(
                    "error",
                    subject,
                    f"anchor drift: {label} no longer declares sub-area '{anchor.expect_sub_area}'",
                )
            )
    for path, expected in anchor.expect_fields.items():
        actual = _path_value(definition, path)
        if actual is _MISSING or actual != expected or type(actual) is not type(expected):
            shown = "missing" if actual is _MISSING else repr(actual)
            issues.append(
                AnchorIssue(
                    "error",
                    subject,
                    f"anchor drift: {label}.{path} is {shown}, expected {expected!r}",
                )
            )
    return issues


def validate_anchors(
    corpus: CoreKnowledgeCorpus, definitions: AnchorDefinitions
) -> list[AnchorIssue]:
    """Return every anchor and manifest_text problem; an empty list means verified."""

    issues: list[AnchorIssue] = []
    for entity in corpus.entities:
        for anchor in entity.manifest_anchors:
            issues.extend(_check_anchor(definitions, f"entity '{entity.id}'", anchor))
    for source_id, source in corpus.sources.items():
        if source.kind != "manifest_text" or not source.ref or not source.quote:
            continue
        subject = f"source '{source_id}'"
        definition = definitions.get(*_parse_ref(source.ref))
        if definition is None:
            issues.append(AnchorIssue("error", subject, f"{source.ref} no longer exists"))
        elif normalize_text(source.quote) not in normalize_text(_texts(definition)):
            issues.append(
                AnchorIssue("error", subject, f"quoted text is no longer present in {source.ref}")
            )
    return issues


def structural_claim_issues(
    corpus: CoreKnowledgeCorpus, items: Iterable[dict[str, Any]]
) -> list[AnchorIssue]:
    """Recount the aggregate Manifest claims behind the Exotic and ammo facts.

    `items` is every DestinyInventoryItemDefinition. Checks that Exotic weapons and armor
    carry their shared uniqueLabels (and nothing else does), that Exotic armor is always
    class-restricted, and that each curated usual_ammo_type still covers at least 95% of
    the archetype's collectible weapons (membership by the anchored item category).
    """

    categories: dict[int, tuple[str, str]] = {}
    for entity in corpus.entities:
        usual = next((f.o for f in entity.facts if f.p == "usual_ammo_type"), None)
        for anchor in entity.manifest_anchors:
            if anchor.type == "DestinyItemCategoryDefinition" and anchor.hash and usual:
                categories[anchor.hash] = (entity.id, str(usual))
    ammo: dict[int, Counter[str]] = {value: Counter() for value in categories}
    issues: list[AnchorIssue] = []
    for item in items:
        if item.get("redacted") or not item.get("equippable"):
            continue
        item_type = item.get("itemType")
        if item_type not in (2, 3):
            continue
        exotic = (item.get("inventory") or {}).get("tierTypeName") == "Exotic"
        label = (item.get("equippingBlock") or {}).get("uniqueLabel")
        expected = ("exotic_armor" if item_type == 2 else "exotic_weapon") if exotic else None
        if label != expected and (exotic or label in {"exotic_armor", "exotic_weapon"}):
            issues.append(
                AnchorIssue(
                    "error",
                    f"item {item.get('hash')}",
                    f"uniqueLabel {label!r} contradicts the Exotic equip-limit sources",
                )
            )
        if exotic and item_type == 2 and item.get("classType") not in (0, 1, 2):
            issues.append(
                AnchorIssue(
                    "error",
                    f"item {item.get('hash')}",
                    "Exotic armor is not class-restricted, contradicting "
                    "each_piece_restricted_to_one_class",
                )
            )
        if item_type == 3 and item.get("collectibleHash"):
            ammo_type = AMMO_TYPES.get((item.get("equippingBlock") or {}).get("ammoType"))
            for category in item.get("itemCategoryHashes") or []:
                if category in ammo and ammo_type:
                    ammo[category][ammo_type] += 1
    for category, (entity_id, usual) in categories.items():
        total = sum(ammo[category].values())
        share = ammo[category][usual] / total if total else 0.0
        if share < USUAL_AMMO_MIN_SHARE:
            issues.append(
                AnchorIssue(
                    "error",
                    f"entity '{entity_id}'",
                    f"usual_ammo_type '{usual}' covers {share:.1%} of {total} weapons "
                    f"({dict(ammo[category])}); below {USUAL_AMMO_MIN_SHARE:.0%}",
                )
            )
    return issues


def _anchor_references(corpus: CoreKnowledgeCorpus) -> list[tuple[str, int, list[str]]]:
    references: list[tuple[str, int, list[str]]] = []
    for entity in corpus.entities:
        for anchor in entity.manifest_anchors:
            if anchor.hash is not None:
                paths = [path.split(".")[0] for path in anchor.expect_fields]
                if anchor.expect_sub_area is not None:
                    paths.append("bubbles")
                references.append((anchor.type, anchor.hash, paths))
    for source in corpus.sources.values():
        if source.kind == "manifest_text" and source.ref:
            definition_type, entity_hash = _parse_ref(source.ref)
            references.append((definition_type, entity_hash, ["progressDescription"]))
    return references


def build_snapshot(corpus: CoreKnowledgeCorpus, definitions: AnchorDefinitions) -> dict[str, Any]:
    """The minimal Manifest subset needed to re-verify the corpus offline."""

    tables: dict[str, dict[str, Any]] = {}

    def keep(definition_type: str, entity_hash: int, paths: list[str]) -> dict[str, Any] | None:
        definition = definitions.get(definition_type, entity_hash)
        if definition is None:
            return None
        display = definition.get("displayProperties") or {}
        trimmed: dict[str, Any] = {
            "hash": entity_hash,
            "displayProperties": {
                "name": display.get("name", ""),
                "description": display.get("description", ""),
            },
        }
        for key in [*paths, "parentNodeHashes", "redacted"]:
            if key in definition:
                trimmed[key] = definition[key]
        if "bubbles" in trimmed:
            trimmed["bubbles"] = [
                {"hash": value.get("hash"), "displayProperties": {"name": _name(value)}}
                for value in trimmed["bubbles"]
            ]
        # A definition can be both anchored and quoted; keep the union of needed fields.
        table = tables.setdefault(definition_type, {})
        table[str(entity_hash)] = {**table.get(str(entity_hash), {}), **trimmed}
        return definition

    for definition_type, entity_hash, paths in _anchor_references(corpus):
        definition = keep(definition_type, entity_hash, paths)
        if definition is not None and definition_type == "DestinyPresentationNodeDefinition":
            _, ancestors = _ancestor_names(definitions, definition)
            for ancestor in ancestors:
                keep("DestinyPresentationNodeDefinition", ancestor, [])
    return {
        "manifest_version": corpus.manifest_version,
        "definitions": {key: dict(sorted(value.items())) for key, value in sorted(tables.items())},
    }


class SnapshotDefinitions:
    """Definitions from a committed snapshot produced by build_snapshot."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.manifest_version = payload.get("manifest_version")
        self._tables = payload.get("definitions", {})

    @classmethod
    def from_file(cls, path: Path) -> "SnapshotDefinitions":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def get(self, definition_type: str, entity_hash: int) -> dict[str, Any] | None:
        return self._tables.get(definition_type, {}).get(str(entity_hash))


class SqliteManifestDefinitions:
    """Read-only lookups against a local Bungie mobile Manifest SQLite database."""

    def __init__(self, path: Path) -> None:
        self._connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)

    def iter_table(self, definition_type: str) -> Iterable[dict[str, Any]]:
        for (raw,) in self._connection.execute(f'SELECT json FROM "{definition_type}"'):
            yield json.loads(raw)

    def get(self, definition_type: str, entity_hash: int) -> dict[str, Any] | None:
        signed = entity_hash - 2**32 if entity_hash >= 2**31 else entity_hash
        try:
            row = self._connection.execute(
                f'SELECT json FROM "{definition_type}" WHERE id = ?', (signed,)
            ).fetchone()
        except sqlite3.Error:
            return None
        return json.loads(row[0]) if row else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--write-snapshot", type=Path)
    parser.add_argument(
        "--structure",
        action="store_true",
        help="Also recount aggregate item claims (scans every item definition; slow).",
    )
    arguments = parser.parse_args(argv)
    corpus = load_core_knowledge(arguments.corpus)
    definitions = SqliteManifestDefinitions(arguments.manifest)
    issues = validate_anchors(corpus, definitions)
    if arguments.structure:
        issues += structural_claim_issues(
            corpus, definitions.iter_table("DestinyInventoryItemDefinition")
        )
    for issue in issues:
        print(issue)
    print(f"{len(issues)} anchor issue(s) for corpus {corpus.corpus_version}")
    if arguments.write_snapshot is not None and not issues:
        snapshot = build_snapshot(corpus, definitions)
        arguments.write_snapshot.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
        print(f"wrote {arguments.write_snapshot}")
    return 1 if any(issue.severity == "error" for issue in issues) else 0


if __name__ == "__main__":
    sys.exit(main())
