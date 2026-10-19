"""Shared mechanics evidence for equipped gear, joined by exact item hash and socket index.

Knowledge Engine v0.2, M3B. Three already-collected facts meet here and stay labelled apart:

* M2 identity: the exact definition hash, verified equipment slot, class and tier;
* M1 plug evidence: the plug actually **equipped** in a socket of the owned copy, with its
  effect-text status;
* M3A configuration: what the item *definition* declares for that socket (fixed, a pool of
  possible rolls, a declared pool of options, display-only options, inventory-sourced).

What this module never claims is player-specific availability. The Manifest describes the
definition, not this copy and not this account, so nothing here says a change can be made now.
Everything is read from data already on the normalized item: no lookups, no network.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.models import CharacterSummary, ItemSummary, SocketConfigSummary, SocketedPlugSummary

Configurability = Literal[
    "fixed_per_manifest",
    "possible_rolls_only",
    "declared_pool_availability_unverified",
    "display_only",
    "inventory_sourced_unverified",
    "unknown",
]
POOL_ROLE: dict[str, str] = {
    "possible_rolls_only": "possible_rolls",
    "declared_pool_availability_unverified": "declared_options",
    "display_only": "display_only_options",
}
CONFIGURABILITY: dict[str, Configurability] = {
    "fixed_by_manifest": "fixed_per_manifest",
    "randomized_roll_pool": "possible_rolls_only",
    "reusable_plug_pool": "declared_pool_availability_unverified",
    "display_only_options": "display_only",
    "inventory_sourced": "inventory_sourced_unverified",
    "insufficient_information": "unknown",
}

MAX_ITEMS = 5
MAX_SOCKETS_PER_ITEM = 8
MAX_SECTION_CHARS = 3600
MAX_SAMPLE = 3
_PERK_MARKERS = ("perk", "trait", "intrinsic")
_SLOT_ORDER = {"Kinetic Weapons": 0, "Energy Weapons": 1, "Power Weapons": 2}

BASIS = (
    "equipped_plug is on the owned copy. possible_rolls_only: rolls a copy can have, not "
    "choices. declared_pool_availability_unverified: listed options; whether this player has "
    "or can swap to them is unknown. fixed_per_manifest: one declared option."
)
COPY_STATE = (
    "Owned-copy alternatives, other rolls and account unlocks were not retrieved: do not "
    "call any change available."
)
SCOPE = "Equipped weapons and Exotic armor only; subclass and other armor are not classified."


def effect_status(value: SocketedPlugSummary) -> str | None:
    """Whether effect text exists, is verifiably absent, or was not retrieved.

    None means the plug was normalized without equipped-plug evidence (a cosmetic, an empty
    socket placeholder, or a context cached before this evidence existed).
    """

    evidence = value.evidence
    if evidence is None:
        return None
    if value.description or any(perk.description for perk in value.sandbox_perks):
        return "present"
    if evidence.description == "definition_unavailable" or evidence.sandbox_perks in {
        "unresolved",
        "definition_unavailable",
    }:
        return "not_retrieved"
    return "none_in_manifest"


class PoolEvidence(BaseModel):
    """A Manifest pool, with its role stated so it is never read as a menu of choices."""

    role: str
    options: int | None = None
    distinct_names: int | None = None
    sample: list[str] = Field(default_factory=list)
    set_hash: int | None = None


class SocketMechanics(BaseModel):
    index: int
    column: str | None = None
    equipped_plug: str | None = None
    effect_status: str | None = None
    classification: str
    configurability: Configurability
    pool: PoolEvidence | None = None

    def to_tool(self) -> dict[str, Any]:
        result: dict[str, Any] = {"index": self.index}
        if self.column:
            result["column"] = self.column
        if self.equipped_plug:
            result["equipped_plug"] = self.equipped_plug
        if self.effect_status and self.effect_status != "present":
            result["effect_status"] = self.effect_status
        result["configurability"] = self.configurability
        if self.pool is not None:
            pool: dict[str, Any] = {"options": self.pool.options}
            if self.pool.distinct_names not in (None, self.pool.options):
                pool["distinct_names"] = self.pool.distinct_names
            if self.pool.sample:
                pool["sample"] = self.pool.sample
            result[self.pool.role] = pool
        return result


class ItemMechanics(BaseModel):
    name: str
    item_hash: int
    item_type: str
    slot: str | None = None
    slot_status: str | None = None
    class_restriction: str | None = None
    tier: str | None = None
    config_status: str
    instance_sockets: str | None = None
    sockets: list[SocketMechanics] = Field(default_factory=list)
    omitted_sockets: int = 0
    other_sockets: dict[str, int] = Field(default_factory=dict)

    def to_tool(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "item_hash": self.item_hash,
            "item_type": self.item_type,
            "slot": self.slot,
        }
        for key in ("slot_status", "class_restriction", "tier", "instance_sockets"):
            if getattr(self, key):
                result[key] = getattr(self, key)
        if self.config_status != "verified":
            result["config_status"] = self.config_status
        result["sockets"] = [value.to_tool() for value in self.sockets]
        if self.omitted_sockets:
            result["omitted_sockets"] = self.omitted_sockets
        if self.other_sockets:
            result["other_sockets"] = dict(sorted(self.other_sockets.items()))
        return result


def _is_perk_socket(config: SocketConfigSummary) -> bool:
    label = (config.category or "").casefold()
    return any(marker in label for marker in _PERK_MARKERS)


def _pool(config: SocketConfigSummary, configurability: Configurability) -> PoolEvidence | None:
    role = POOL_ROLE.get(configurability)
    if role is None or config.options is None:
        return None
    return PoolEvidence(
        role=role,
        options=config.options,
        distinct_names=config.distinct_names,
        sample=config.sample[:MAX_SAMPLE],
        set_hash=config.set_hash,
    )


def item_mechanics(item: ItemSummary) -> ItemMechanics | None:
    """Join one equipped item's identity, equipped plugs and Manifest socket configuration.

    None when no socket configuration was collected for the item (so its configuration stays
    unknown rather than being invented).
    """

    if item.socket_configuration_status is None:
        return None
    plugs: dict[int, SocketedPlugSummary] = {
        value.socket_index: value
        for value in item.socketed_plug_details
        if value.socket_index is not None
    }
    sockets: list[SocketMechanics] = []
    other: dict[str, int] = {}
    omitted = 0
    for config in item.socket_configuration:
        configurability = CONFIGURABILITY.get(config.classification, "unknown")
        if not _is_perk_socket(config):
            other[configurability] = other.get(configurability, 0) + 1
            continue
        if len(sockets) >= MAX_SOCKETS_PER_ITEM:
            omitted += 1
            continue
        plug = plugs.get(config.index)
        sockets.append(
            SocketMechanics(
                index=config.index,
                column=config.accepts or config.category,
                equipped_plug=plug.name if plug else None,
                effect_status=effect_status(plug) if plug else None,
                classification=config.classification,
                configurability=configurability,
                pool=_pool(config, configurability),
            )
        )
    return ItemMechanics(
        name=item.name,
        item_hash=item.item_hash,
        item_type=item.item_type,
        slot=item.gear_slot(),
        slot_status=item.slot_status() if item.slot_status() != "known" else None,
        class_restriction=item.class_restriction,
        tier=item.tier,
        config_status=item.socket_configuration_status,
        instance_sockets=(
            "not_returned"
            if item.socket_data_available is False
            or (item.socket_data_available is None and not item.socketed_plug_details)
            else None
        ),
        sockets=sockets,
        omitted_sockets=omitted,
        other_sockets=other,
    )


def _rank(item: ItemSummary) -> tuple[int, int, str]:
    exotic = item.tier == "Exotic"
    group = 0 if exotic and item.item_type == "Weapon" else 1 if exotic else 2
    return group, _SLOT_ORDER.get(item.gear_slot() or "", 9), item.name.casefold()


def equipped_mechanics_section(
    character: CharacterSummary,
    *,
    max_items: int = MAX_ITEMS,
    max_chars: int = MAX_SECTION_CHARS,
) -> dict[str, Any]:
    """The bounded build-tool section: ranked equipped items, once each, with omissions counted.

    Ranking is deterministic: Exotic weapon, Exotic armor, then other weapons by slot. The
    section stops at the item or character cap, and says how many items it left out.
    """

    gear = [value for value in character.equipped_gear if value.item_type in {"Weapon", "Armor"}]
    configured = sorted(
        (value for value in gear if value.socket_configuration_status is not None), key=_rank
    )
    entries: list[dict[str, Any]] = []
    used = len(json.dumps({"basis": BASIS, "copy_state": COPY_STATE, "scope": SCOPE}))
    omitted = 0
    for item in configured:
        record = item_mechanics(item)
        if record is None:
            continue
        entry = record.to_tool()
        size = len(json.dumps(entry, ensure_ascii=False))
        if len(entries) >= max_items or used + size > max_chars:
            omitted += 1
            continue
        used += size
        entries.append(entry)
    section: dict[str, Any] = {"items": entries}
    if omitted:
        section["omitted_items"] = omitted
    uncovered = len(gear) - len(configured)
    if uncovered:
        section["items_without_configuration"] = uncovered
    section.update({"scope": SCOPE, "basis": BASIS, "copy_state": COPY_STATE})
    return section
