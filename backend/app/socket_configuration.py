"""Manifest socket configuration: what an item definition declares about each socket.

Knowledge Engine v0.2, M3A. This reads only the item *definition* and the plug-set, socket-type
and socket-category definitions it references. It answers "what does the Manifest say about
this socket's structure?" and nothing about a particular player's copy:

* A **randomized roll pool** (`randomizedPlugSetHash`) lists perks a copy of the item *can roll
  with*. A given copy holds a few of them; the pool is not a menu the player picks from.
* A **reusable pool** (a real `reusablePlugSetHash`, or inline `reusablePlugItems`) lists plugs
  the socket can hold. Whether this player has them, or can swap to them now, depends on the
  owned instance and account unlocks, which are not read here.
* A **fake plug set** exists for display. One option means the socket is fixed; several
  options are shown but never implied to be selectable.

Classification is deliberately conservative, uses no item names, and reports `unresolved`
rather than guessing when a referenced definition is missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from app.models import COSMETIC_PLUG_MARKERS

SocketClassification = Literal[
    "fixed_by_manifest",
    "randomized_roll_pool",
    "reusable_plug_pool",
    "display_only_options",
    "inventory_sourced",
    "insufficient_information",
]
PoolKind = Literal["randomized", "reusable", "inline"]

# DestinyPlugSources bit flags on a socket entry.
SOURCE_INVENTORY = 1
SOURCE_REUSABLE = 2
SOURCE_PROFILE_PLUG_SET = 4
SOURCE_CHARACTER_PLUG_SET = 8
SOURCE_NAMES = {
    SOURCE_INVENTORY: "inventory",
    SOURCE_REUSABLE: "reusable_plug_items",
    SOURCE_PROFILE_PLUG_SET: "profile_plug_set",
    SOURCE_CHARACTER_PLUG_SET: "character_plug_set",
}

MAX_SOCKETS_IN_OUTPUT = 12
MAX_SAMPLE_NAMES = 5
MAX_ACCEPTS = 2
# Counting distinct names needs every option's name; skip it for very large sets.
MAX_POOL_FOR_NAME_COUNT = 60

NOTE = (
    "Manifest structure only: a randomized pool lists possible rolls, not choices; a reusable "
    "pool does not show what this player owns or can use. Copy and account state are unknown."
)


@dataclass
class Pool:
    kind: PoolKind
    set_hash: int | None
    fake: bool | None
    hashes: list[int]
    rollable: int
    resolved: bool = True

    @property
    def options(self) -> int:
        return len(self.hashes)


@dataclass
class SocketSpec:
    index: int
    socket_type_hash: int
    category_hash: int | None
    initial_hash: int | None
    plug_sources: int
    pools: list[Pool] = field(default_factory=list)
    # Cosmetic sockets are never classified, and their (often huge) plug sets are not read.
    cosmetic: bool = False
    unresolved_sets: list[int] = field(default_factory=list)


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value else None


def socket_entries(definition: dict[str, Any]) -> list[dict[str, Any]]:
    entries = (definition.get("sockets") or {}).get("socketEntries")
    return (
        [value for value in entries if isinstance(value, dict)] if isinstance(entries, list) else []
    )


def category_by_socket_index(definition: dict[str, Any]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    categories = (definition.get("sockets") or {}).get("socketCategories")
    for category in categories if isinstance(categories, list) else []:
        category_hash = _int_or_none(category.get("socketCategoryHash"))
        for index in category.get("socketIndexes") or []:
            if category_hash and isinstance(index, int):
                mapping[index] = category_hash
    return mapping


def referenced_definition_hashes(definition: dict[str, Any]) -> dict[str, set[int]]:
    """Hashes phase one must resolve: plug sets, socket types and socket categories."""

    plug_sets: set[int] = set()
    types: set[int] = set()
    for entry in socket_entries(definition):
        for key in ("randomizedPlugSetHash", "reusablePlugSetHash"):
            if value := _int_or_none(entry.get(key)):
                plug_sets.add(value)
        if value := _int_or_none(entry.get("socketTypeHash")):
            types.add(value)
    categories = set(category_by_socket_index(definition).values())
    return {"plug_sets": plug_sets, "socket_types": types, "categories": categories}


def cosmetic_socket_indexes(
    definition: dict[str, Any],
    socket_types: dict[int, dict[str, Any]],
    category_definitions: dict[int, dict[str, Any]],
) -> set[int]:
    """Indexes of cosmetic sockets (shaders, ornaments, trackers), judged before any pool read."""

    categories = category_by_socket_index(definition)
    found: set[int] = set()
    for index, entry in enumerate(socket_entries(definition)):
        socket_type = socket_types.get(_int_or_none(entry.get("socketTypeHash")) or 0) or {}
        accepts = _informative_accepts(
            [
                value["categoryIdentifier"]
                for value in socket_type.get("plugWhitelist") or []
                if isinstance(value, dict) and isinstance(value.get("categoryIdentifier"), str)
            ]
        )
        if _is_cosmetic(_name(category_definitions, categories.get(index) or 0), accepts):
            found.add(index)
    return found


def plug_set_hashes(
    definition: dict[str, Any], skip: set[int] | frozenset[int] = frozenset()
) -> set[int]:
    """Plug sets to read: every referenced set except those of skipped (cosmetic) sockets."""

    found: set[int] = set()
    for index, entry in enumerate(socket_entries(definition)):
        if index in skip:
            continue
        for key in ("randomizedPlugSetHash", "reusablePlugSetHash"):
            if value := _int_or_none(entry.get(key)):
                found.add(value)
    return found


def _pool_from_set(
    kind: PoolKind, set_hash: int, plug_set: dict[str, Any] | None
) -> tuple[Pool | None, int | None]:
    if plug_set is None:
        return None, set_hash
    hashes: list[int] = []
    rollable = 0
    seen: set[int] = set()
    for option in plug_set.get("reusablePlugItems") or []:
        plug_hash = _int_or_none(option.get("plugItemHash")) if isinstance(option, dict) else None
        if plug_hash is None or plug_hash in seen:
            continue
        seen.add(plug_hash)
        hashes.append(plug_hash)
        rollable += option.get("currentlyCanRoll", True) is not False
    return Pool(kind, set_hash, bool(plug_set.get("isFakePlugSet")), hashes, rollable), None


def build_specs(
    definition: dict[str, Any],
    plug_sets: dict[int, dict[str, Any]],
    cosmetic: set[int] | frozenset[int] = frozenset(),
) -> list[SocketSpec]:
    """Socket specs with their pools, from the definition and its resolved plug sets."""

    categories = category_by_socket_index(definition)
    specs: list[SocketSpec] = []
    for index, entry in enumerate(socket_entries(definition)):
        spec = SocketSpec(
            index=index,
            socket_type_hash=_int_or_none(entry.get("socketTypeHash")) or 0,
            category_hash=categories.get(index),
            initial_hash=_int_or_none(entry.get("singleInitialItemHash")),
            plug_sources=entry.get("plugSources")
            if isinstance(entry.get("plugSources"), int)
            else 0,
        )
        if index in cosmetic:
            spec.cosmetic = True
            specs.append(spec)
            continue
        for key, kind in (
            ("randomizedPlugSetHash", "randomized"),
            ("reusablePlugSetHash", "reusable"),
        ):
            if set_hash := _int_or_none(entry.get(key)):
                pool, missing = _pool_from_set(kind, set_hash, plug_sets.get(set_hash))  # type: ignore[arg-type]
                if pool is not None:
                    spec.pools.append(pool)
                if missing is not None:
                    spec.unresolved_sets.append(missing)
        inline = entry.get("reusablePlugItems")
        if (
            isinstance(inline, list)
            and inline
            and not _int_or_none(entry.get("reusablePlugSetHash"))
        ):
            hashes = list(
                dict.fromkeys(
                    value
                    for option in inline
                    if isinstance(option, dict)
                    and (value := _int_or_none(option.get("plugItemHash"))) is not None
                )
            )
            spec.pools.append(Pool("inline", None, None, hashes, len(hashes)))
        specs.append(spec)
    return specs


def classify_socket(spec: SocketSpec) -> SocketClassification | None:
    """The conservative classification for one socket, or None for an unused socket.

    First match wins. Nothing here says a player can use an option: even a reusable pool is
    only a declared pool.
    """

    if not spec.socket_type_hash and not spec.initial_hash and not spec.pools:
        return None if not spec.unresolved_sets else "insufficient_information"
    if spec.unresolved_sets:
        return "insufficient_information"
    randomized = next((pool for pool in spec.pools if pool.kind == "randomized"), None)
    if randomized is not None and randomized.options:
        return "randomized_roll_pool"
    real = [pool for pool in spec.pools if pool.fake is not True and pool.options]
    fake = [pool for pool in spec.pools if pool.fake is True and pool.options]
    if any(pool.options >= 2 for pool in real):
        return "reusable_plug_pool"
    if any(pool.options >= 2 for pool in fake):
        return "display_only_options"
    if spec.plug_sources & SOURCE_INVENTORY:
        # Options come from the owner's inventory matching the socket type; the Manifest does
        # not enumerate them, so this is not "fixed" even when a single plug is declared.
        return "inventory_sourced"
    if real or fake:
        return "fixed_by_manifest"  # a declared pool with exactly one option
    if spec.initial_hash and spec.plug_sources == 0:
        # An initial plug alone is not proof: it counts only when the socket declares no
        # source of alternatives at all. Any declared source without a readable pool is
        # insufficient evidence, not fixedness.
        return "fixed_by_manifest"
    return "insufficient_information"


def sample_hashes(pool: Pool) -> list[int]:
    return pool.hashes[:MAX_SAMPLE_NAMES]


def name_hashes_needed(
    specs: list[SocketSpec], category_definitions: dict[int, dict[str, Any]]
) -> set[int]:
    """Plug hashes whose names phase two resolves.

    Perk-like sockets need their initial plug, a sample and (for small pools) every name to
    count distinct names. Other sockets are described by class and size only, so they need
    just a fixed plug's name.
    """

    needed: set[int] = set()
    for spec in specs:
        if spec.cosmetic:
            continue
        label = _name(category_definitions, spec.category_hash or 0) or ""
        perk = any(marker in label.casefold() for marker in _PERK_CATEGORY_MARKERS)
        if spec.initial_hash and (perk or classify_socket(spec) == "fixed_by_manifest"):
            needed.add(spec.initial_hash)
        if not perk:
            continue
        for pool in spec.pools:
            needed.update(sample_hashes(pool))
            if 0 < pool.options <= MAX_POOL_FOR_NAME_COUNT:
                needed.update(pool.hashes)
    return needed


def _name(definitions: dict[int, dict[str, Any]], plug_hash: int) -> str | None:
    value = ((definitions.get(plug_hash) or {}).get("displayProperties") or {}).get("name")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _is_cosmetic(category_name: str | None, accepts: list[str]) -> bool:
    if category_name and "cosmetic" in category_name.casefold():
        return True
    return bool(accepts) and all(
        any(marker in value.casefold() for marker in COSMETIC_PLUG_MARKERS) for value in accepts
    )


# Identifiers that appear on almost every socket type and say nothing about its role.
_UNINFORMATIVE_ACCEPTS = ("crafting.recipes.", ".mod_empty", "template_interface")
_PERK_CATEGORY_MARKERS = ("perk", "trait", "intrinsic")


def _informative_accepts(accepts: list[str]) -> list[str]:
    return [
        value for value in accepts if not any(marker in value for marker in _UNINFORMATIVE_ACCEPTS)
    ]


def _pool_summary(
    pool: Pool, names: dict[int, dict[str, Any]], *, perk_socket: bool
) -> dict[str, Any]:
    limit = MAX_SAMPLE_NAMES
    sample = list(
        dict.fromkeys(
            value for plug_hash in pool.hashes[:limit] if (value := _name(names, plug_hash))
        )
    )
    summary: dict[str, Any] = {"kind": pool.kind, "options": pool.options}
    if pool.rollable != pool.options:
        summary["rollable"] = pool.rollable
    if pool.set_hash:
        summary["set_hash"] = pool.set_hash
    if pool.fake is True:
        summary["display_only"] = True
    if 0 < pool.options <= MAX_POOL_FOR_NAME_COUNT:
        resolved = [_name(names, plug_hash) for plug_hash in pool.hashes]
        if all(resolved):
            summary["distinct_names"] = len(set(resolved))
    # Mod and masterwork pools are large and uniform; only perk pools get named samples.
    if sample and perk_socket:
        summary["sample"] = sample
    return summary


def _primary_pool(spec: SocketSpec) -> Pool | None:
    randomized = [pool for pool in spec.pools if pool.kind == "randomized" and pool.options]
    if randomized:
        return randomized[0]
    return max((pool for pool in spec.pools if pool.options), key=lambda p: p.options, default=None)


def describe_sockets(
    definition: dict[str, Any],
    specs: list[SocketSpec],
    socket_types: dict[int, dict[str, Any]],
    category_definitions: dict[int, dict[str, Any]],
    plug_names: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """The bounded, model-facing socket configuration summary for one item definition.

    Fixed sockets are one short line. Sockets with a pool carry the pool's size, how many
    distinct names it holds, and a small sample: counts and samples, never the whole pool.
    """

    summary: dict[str, int] = {}
    omitted = {"unused": 0, "cosmetic": 0, "uncategorized": 0, "over_limit": 0}
    sockets: list[dict[str, Any]] = []
    unresolved = False
    for spec in specs:
        if spec.cosmetic:
            omitted["cosmetic"] += 1
            continue
        classification = classify_socket(spec)
        if classification is None:
            omitted["unused"] += 1
            continue
        unresolved = unresolved or classification == "insufficient_information"
        socket_type = socket_types.get(spec.socket_type_hash) or {}
        accepts = [
            value["categoryIdentifier"]
            for value in socket_type.get("plugWhitelist") or []
            if isinstance(value, dict) and isinstance(value.get("categoryIdentifier"), str)
        ]
        accepts = _informative_accepts(accepts)
        category = _name(category_definitions, spec.category_hash or 0)
        summary[classification] = summary.get(classification, 0) + 1
        if _is_cosmetic(category, accepts):
            omitted["cosmetic"] += 1
            continue
        if category is None:
            # No socket category: a hidden or crafting-internal socket, not a build choice.
            omitted["uncategorized"] += 1
            continue
        if len(sockets) >= MAX_SOCKETS_IN_OUTPUT:
            omitted["over_limit"] += 1
            continue
        perk_socket = any(marker in category.casefold() for marker in _PERK_CATEGORY_MARKERS)
        entry: dict[str, Any] = {
            "index": spec.index,
            "category": category,
            "classification": classification,
        }
        if accepts:
            entry["accepts"] = accepts[: MAX_ACCEPTS if perk_socket else 1]
        initial = _name(plug_names, spec.initial_hash) if spec.initial_hash else None
        primary = _primary_pool(spec)
        if not perk_socket and classification != "fixed_by_manifest":
            # Mod, ability and similar sockets stay brief: the class and the pool size only.
            if primary is not None:
                entry["options"] = primary.options
            sockets.append(entry)
            continue
        if classification == "fixed_by_manifest":
            fixed = initial or (
                _name(plug_names, primary.hashes[0]) if primary and primary.hashes else None
            )
            if fixed:
                entry["plug"] = fixed
            if primary and primary.set_hash:
                entry["set_hash"] = primary.set_hash
        else:
            if initial:
                # The definition's default/sample plug, not a guaranteed roll on any copy.
                entry["initial_plug"] = initial
            if primary is not None:
                entry["pool"] = _pool_summary(primary, plug_names, perk_socket=perk_socket)
            others = [
                {"kind": pool.kind, "options": pool.options}
                for pool in spec.pools
                if pool is not primary and pool.kind != "inline" and pool.options
            ]
            if others:
                entry["other_pools"] = others
            if spec.unresolved_sets:
                entry["unresolved_plug_sets"] = spec.unresolved_sets
            # Reusable plug items are implied by a pool; inventory and account plug sets are
            # the sources that make availability depend on the owner.
            flags = [
                name
                for bit, name in SOURCE_NAMES.items()
                if spec.plug_sources & bit and bit != SOURCE_REUSABLE
            ]
            if flags:
                entry["plug_sources"] = flags
        sockets.append(entry)
    return {
        "status": "partial" if unresolved else "verified",
        "basis": "manifest_definition",
        "sockets": sockets,
        "classification_counts": dict(sorted(summary.items())),
        "omitted": {key: value for key, value in omitted.items() if value},
        "note": NOTE,
    }


def summary_entries(described: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten `describe_sockets` output into the fields stored on a normalized item."""

    entries: list[dict[str, Any]] = []
    for socket in described.get("sockets", []):
        pool = socket.get("pool") or {}
        accepts = socket.get("accepts") or []
        entries.append(
            {
                "index": socket["index"],
                "category": socket.get("category"),
                "classification": socket["classification"],
                "accepts": accepts[0] if accepts else None,
                "plug": socket.get("plug"),
                "initial_plug": socket.get("initial_plug"),
                "pool_kind": pool.get("kind"),
                "options": pool.get("options", socket.get("options")),
                "rollable": pool.get("rollable"),
                "distinct_names": pool.get("distinct_names"),
                "set_hash": pool.get("set_hash", socket.get("set_hash")),
                "display_only": bool(pool.get("display_only")),
                "sample": list(pool.get("sample", []))[:3],
                "plug_sources": list(socket.get("plug_sources", [])),
            }
        )
    return entries


def not_applicable(reason: str) -> dict[str, Any]:
    return {"status": "not_applicable", "reason": reason, "basis": "manifest_definition"}
