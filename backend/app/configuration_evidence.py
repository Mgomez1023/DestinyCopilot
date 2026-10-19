"""Configuration evidence: what Bungie reports for sockets fed by profile/character plug sets.

Knowledge Engine v0.2, M5B. This joins four facts that stay labelled apart:

* `manifest_possible`: the socket's declared plug set (exact `reusablePlugSetHash`) holds N options;
* `equipped_now`: the plug in that socket on the equipped item (M1 socket evidence);
* `account_listed` / `character_listed`: Bungie returned the plug in the profile- or
  character-scoped set, with the reported `canInsert` / `enabled` flags;
* `insertion_blocked_reported`: Bungie reported `canInsert=false` in that scope.

`instance_listed` is reserved for later, instance-specific evidence and is never produced here.
`unknown` means the evidence was not returned or the join is incomplete; it is never an empty
set. `actionable` is always `unknown` in M5B: no fully verified configuration change is
established, because material cost, armor energy, fragment slot capacity, activity restrictions
and instance state are not read.

Status of the underlying data: the shape of `profilePlugSets` / `characterPlugSets` follows
Bungie's published API definitions. It has not been verified against a real Bungie profile
response, so any missing, malformed or truncated data yields `unknown`, never an availability
or absence claim.

This is separate from `equipped_mechanics` and from M4's requirement fit, which describe the
equipped loadout. Nothing here changes either.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.models import (
    CharacterSummary,
    ItemSummary,
    PlugSetCollection,
    PlugSetEntry,
    PlugSetEvidence,
    PlugSetSocketRef,
)

ConfigState = Literal[
    "manifest_possible",
    "equipped_now",
    # The plug is socketed on an instance the profile says is not equipped (e.g. in the vault).
    "socketed_not_equipped",
    "account_listed",
    "character_listed",
    "instance_listed",
    "insertion_blocked_reported",
    "unknown",
]

MAX_GROUPS = 8
MAX_NOTABLE = 3
MAX_PLUGS_PER_GROUP = 4
MAX_SECTION_CHARS = 3600

BASIS = (
    "Reported by Bungie for the account (profile) and this character, joined to sockets by "
    "exact plug-set hash and the socket's declared plug sources. listed = Bungie returned the "
    "plug in that set; it does not prove a change can be made now. insertion_blocked_reported = "
    "canInsert false in that scope. not_enabled_reported = enabled false (the plug is not "
    "providing its benefits), which is not an insertion restriction. unknown = not returned or "
    "not joinable; missing data is never an empty set. actionable is always unknown here."
)
SCOPE = "Equipped subclass and armor sockets only; weapon instances and vault items are not read."


class ConfigSource(BaseModel):
    kind: Literal[
        "manifest_plug_set",
        "socket_component",
        "profile_plug_set",
        "character_plug_set",
        "instance_reusable_plugs",
    ]
    plug_set_hash: int | None = None
    item_hash: int | None = None
    socket_index: int | None = None
    component: str | None = None
    minted_at: str | None = None


class ConfigStateRecord(BaseModel):
    state: ConfigState
    source: ConfigSource
    # For manifest_possible: the pool's role ("randomized_roll_pool", "reusable_plug_pool", ...).
    pool: str | None = None
    can_insert: bool | None = None
    enabled: bool | None = None
    # Manifest messages for the reported failure indexes; None keeps an index unresolved.
    reasons: list[str | None] = Field(default_factory=list)


class ConfigurationOption(BaseModel):
    """One plug at one socket, with every observed state and where it came from."""

    item_hash: int
    socket_index: int
    plug_item_hash: int | None = None
    plug_name: str | None = None
    category: str | None = None
    states: list[ConfigStateRecord] = Field(default_factory=list)
    actionable: Literal["unknown"] = "unknown"
    limitations: list[str] = Field(default_factory=list)
    # Internal handle only; never serialized for the model.
    instance_id: str | None = Field(default=None, exclude=True)

    def state_names(self) -> list[str]:
        return list(dict.fromkeys(value.state for value in self.states))


# --- joins ------------------------------------------------------------------------------------


def _limitations(item: ItemSummary, category: str | None, sources: list[str]) -> list[str]:
    label = (category or "").casefold()
    if item.item_type == "Armor":
        notes = ["armor energy and mod stacking not assessed"]
    elif "fragment" in label:
        notes = ["fragment slot capacity not assessed"]
    elif "aspect" in label:
        notes = ["aspect compatibility and fragment slots not assessed"]
    else:
        notes = ["activity and equip restrictions not assessed"]
    if "inventory" in sources:
        notes.append("inventory-sourced plugs not read")
    if "reusable_plug_items" in sources:
        notes.append("instance reusable plugs not retrieved")
    return notes


def _blocked(entry: PlugSetEntry) -> bool:
    return entry.can_insert is False or bool(entry.insert_fail_indexes)


def _not_enabled(entry: PlugSetEntry) -> bool:
    return entry.enabled is False or bool(entry.enable_fail_indexes)


def _label(entry: PlugSetEntry) -> str:
    return entry.name or f"plug {entry.plug_item_hash}"


def scope_report(
    collection: PlugSetCollection, set_hash: int, scope: Literal["profile", "character"]
) -> dict[str, Any]:
    """Compact, bounded report for one declared scope of one plug set."""

    if collection.status != "returned":
        return {"status": collection.status}
    evidence: PlugSetEvidence | None = collection.get(set_hash)
    if evidence is None:
        return {"status": "set_not_returned"}
    blocked = [value for value in evidence.entries if _blocked(value)]
    disabled = [value for value in evidence.entries if _not_enabled(value) and not _blocked(value)]
    report: dict[str, Any] = {
        "status": "returned",
        "listed": evidence.listed,
        "insertable_reported": len(evidence.clear_plug_hashes)
        + sum(value.can_insert is True for value in evidence.entries),
    }
    if evidence.unmatched_entries:
        report["not_in_manifest_set"] = evidence.unmatched_entries
    if evidence.omitted_entries:
        # Truncated: the counts and lists below are lower bounds, and a plug that is not listed
        # here may simply not have been retained. That is unknown, never verified absence.
        report["entries_not_retained"] = evidence.omitted_entries
        report["counts_are_lower_bounds"] = True
    unreported = sum(value.can_insert is None for value in evidence.entries)
    if unreported:
        report["insertion_status_not_reported"] = unreported
    if blocked:
        report["insertion_blocked_reported"] = [
            {
                "plug": _label(value),
                "reasons": value.insert_fail_reasons or None,
                **({"reasons_unresolved": True} if None in value.insert_fail_reasons else {}),
            }
            for value in blocked[:MAX_NOTABLE]
        ]
        if len(blocked) > MAX_NOTABLE:
            report["insertion_blocked_omitted"] = len(blocked) - MAX_NOTABLE
    if disabled:
        report["not_enabled_reported"] = [_label(value) for value in disabled[:MAX_NOTABLE]]
    if evidence.sample and not blocked:
        report["sample"] = evidence.sample
    return {key: value for key, value in report.items() if value is not None}


def _equipped_plugs(item: ItemSummary, indexes: set[int]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for plug in item.socketed_plug_details:
        if plug.socket_index not in indexes:
            continue
        record: dict[str, Any] = {"plug": plug.name, "socket": plug.socket_index}
        if plug.is_enabled is False:
            record["not_enabled_reported"] = True
            if plug.enable_fail_indexes:
                record["reasons"] = plug.enable_fail_reasons
        found.append(record)
    return found


def _group_key(item: ItemSummary, ref: PlugSetSocketRef) -> tuple[str, int, str]:
    return item.item_type, ref.plug_set_hash, ref.category or ""


def configuration_groups(
    character: CharacterSummary, profile_sets: PlugSetCollection
) -> list[dict[str, Any]]:
    """Per plug set (and category): equipped plugs plus what each declared scope reported.

    Subclass groups come first. Sockets that share a set (two aspects, six fragments, an armor
    slot's mod sockets) are one group, and armor pieces that share a set are one group.
    """

    ordered: list[ItemSummary] = []
    if character.subclass is not None:
        ordered.append(character.subclass)
    ordered.extend(
        value
        for value in character.equipped_gear
        if value.item_type == "Armor" and value is not character.subclass
    )
    groups: dict[tuple[str, int, str], dict[str, Any]] = {}
    for item in ordered:
        for ref in item.plug_set_sockets:
            key = _group_key(item, ref)
            group = groups.get(key)
            if group is None:
                profile_declared = "profile_plug_set" in ref.sources
                character_declared = "character_plug_set" in ref.sources
                group = {
                    "kind": "subclass" if item.item_type == "Subclass" else "armor",
                    "category": ref.category,
                    "plug_set_hash": ref.plug_set_hash,
                    "items": [],
                    "equipped": [],
                    "sources": ref.sources,
                    "_indexes": {},
                    "_item": item,
                    "_ref": ref,
                    "_scopes": (profile_declared, character_declared),
                }
                groups[key] = group
            if item.item_type == "Armor" and item.name not in group["items"]:
                group["items"].append(item.name)
            group["_indexes"].setdefault(item.item_hash, set()).add(ref.socket_index)
    results: list[dict[str, Any]] = []
    for group in groups.values():
        item, ref = group.pop("_item"), group.pop("_ref")
        indexes = group.pop("_indexes")
        profile_declared, character_declared = group.pop("_scopes")
        set_hash = group["plug_set_hash"]
        equipped: list[dict[str, Any]] = []
        for armor_or_subclass in ordered:
            wanted = indexes.get(armor_or_subclass.item_hash)
            if wanted:
                equipped.extend(_equipped_plugs(armor_or_subclass, wanted))
        group["sockets"] = sum(len(value) for value in indexes.values())
        group["equipped"] = equipped[: MAX_PLUGS_PER_GROUP * 2]
        if profile_declared:
            group["profile"] = scope_report(profile_sets, set_hash, "profile")
        if character_declared:
            group["character"] = scope_report(character.plug_sets, set_hash, "character")
        reports = [group[name] for name in ("profile", "character") if name in group]
        manifest_options = ref.manifest_options
        for evidence in (profile_sets.get(set_hash), character.plug_sets.get(set_hash)):
            if manifest_options is None and evidence is not None:
                manifest_options = evidence.manifest_options
        if manifest_options is not None:
            group["manifest_options"] = manifest_options
        group["states"] = _states(group, reports)
        group["limitations"] = _limitations(item, ref.category, ref.sources)
        results.append(group)
    return results


def _states(group: dict[str, Any], reports: list[dict[str, Any]]) -> list[str]:
    states: list[str] = []
    if group.get("manifest_options") is not None:
        states.append("manifest_possible")
    if group["equipped"]:
        states.append("equipped_now")
    if "profile" in group and group["profile"].get("listed"):
        states.append("account_listed")
    if "character" in group and group["character"].get("listed"):
        states.append("character_listed")
    if any(report.get("insertion_blocked_reported") for report in reports):
        states.append("insertion_blocked_reported")
    if not reports or any(report.get("status") != "returned" for report in reports):
        states.append("unknown")
    return states


def configuration_options(
    character: CharacterSummary, profile_sets: PlugSetCollection
) -> list[ConfigurationOption]:
    """One option per equipped plug in a plug-set socket, with provenance for each state.

    Only evidence that was actually returned produces a state; nothing is manufactured for a
    scope that was not returned (that produces `unknown`).
    """

    options: list[ConfigurationOption] = []
    items = [value for value in [character.subclass, *character.equipped_gear] if value]
    seen: set[int] = set()
    for item in items:
        if item.item_hash in seen:
            continue
        seen.add(item.item_hash)
        plugs = {value.socket_index: value for value in item.socketed_plug_details}
        for ref in item.plug_set_sockets:
            plug = plugs.get(ref.socket_index)
            states: list[ConfigStateRecord] = []
            if ref.manifest_options is not None:
                states.append(
                    ConfigStateRecord(
                        state="manifest_possible",
                        source=ConfigSource(
                            kind="manifest_plug_set", plug_set_hash=ref.plug_set_hash
                        ),
                    )
                )
            if plug is not None:
                states.append(
                    ConfigStateRecord(
                        state="equipped_now",
                        source=ConfigSource(
                            kind="socket_component",
                            item_hash=item.item_hash,
                            socket_index=ref.socket_index,
                            component="ItemSockets",
                        ),
                        enabled=plug.is_enabled,
                        reasons=list(plug.enable_fail_reasons),
                    )
                )
            for scope, collection, flag in (
                ("profile", profile_sets, "profile_plug_set"),
                ("character", character.plug_sets, "character_plug_set"),
            ):
                if flag not in ref.sources:
                    continue
                states.extend(
                    _scope_states(scope, collection, ref, plug.item_hash if plug else None)
                )
            options.append(
                ConfigurationOption(
                    item_hash=item.item_hash,
                    socket_index=ref.socket_index,
                    plug_item_hash=plug.item_hash if plug else None,
                    plug_name=plug.name if plug else None,
                    category=ref.category,
                    states=states,
                    limitations=_limitations(item, ref.category, ref.sources),
                    instance_id=item.instance_id,
                )
            )
    return options


def _scope_states(
    scope: str, collection: PlugSetCollection, ref: PlugSetSocketRef, plug_hash: int | None
) -> list[ConfigStateRecord]:
    kind: Literal["profile_plug_set", "character_plug_set"] = (
        "profile_plug_set" if scope == "profile" else "character_plug_set"
    )
    source = ConfigSource(
        kind=kind,
        plug_set_hash=ref.plug_set_hash,
        component="ItemSockets",
        minted_at=collection.minted_at,
    )
    evidence = collection.get(ref.plug_set_hash) if collection.status == "returned" else None
    if evidence is None:
        return [ConfigStateRecord(state="unknown", source=source)]
    entry = next((value for value in evidence.entries if value.plug_item_hash == plug_hash), None)
    if entry is None and plug_hash in evidence.clear_plug_hashes:
        entry = PlugSetEntry(plug_item_hash=plug_hash, can_insert=True, enabled=True)
    if entry is None:
        # A plug missing from the retained entries is not evidence of absence when entries were
        # capped; the join is unknown rather than negative.
        return [ConfigStateRecord(state="unknown", source=source)]
    states = [
        ConfigStateRecord(
            state="account_listed" if scope == "profile" else "character_listed",
            source=source,
            can_insert=entry.can_insert,
            enabled=entry.enabled,
        )
    ]
    if _blocked(entry):
        states.append(
            ConfigStateRecord(
                state="insertion_blocked_reported",
                source=source,
                can_insert=entry.can_insert,
                enabled=entry.enabled,
                reasons=list(entry.insert_fail_reasons),
            )
        )
    return states


# --- tool section -----------------------------------------------------------------------------


def configuration_section(
    character: CharacterSummary,
    profile_sets: PlugSetCollection,
    *,
    max_groups: int = MAX_GROUPS,
    max_chars: int = MAX_SECTION_CHARS,
) -> dict[str, Any]:
    """The bounded build-tool section; omissions are counted, never silent."""

    groups = configuration_groups(character, profile_sets)
    if not groups:
        return {
            "groups": [],
            "note": "No equipped subclass or armor sockets drawing on plug sets were collected.",
        }
    section: dict[str, Any] = {
        "groups": [],
        "actionable": "unknown",
        "scope": SCOPE,
        "basis": BASIS,
    }
    used = len(json.dumps({"actionable": "unknown", "scope": SCOPE, "basis": BASIS}))
    omitted = 0
    for group in groups:
        size = len(json.dumps(group, ensure_ascii=False, default=str))
        if len(section["groups"]) >= max_groups or used + size > max_chars:
            omitted += 1
            continue
        used += size
        section["groups"].append(group)
    if omitted:
        section["omitted_groups"] = omitted
    return section
