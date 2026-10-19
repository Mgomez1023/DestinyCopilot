"""Join item-instance plug evidence to Manifest socket structure and account evidence (offline).

Knowledge Engine v0.2, M5C1. Inputs are already-parsed pieces: instance evidence from a GetItem
response (`instance_plug_evidence`), the item the profile holds for that instance, the M3
Manifest socket configuration for its definition, and optionally M5B plug-set evidence. Nothing
here fetches, caches or attaches data to a profile refresh.

The join key is the exact item definition hash, the internal instance identity and the socket
index. If the hash or the instance differs, nothing is joined. These stay separate states and
none is derived from another:

* `manifest_possible`: the socket's Manifest pool (a randomized pool lists possible rolls, never
  choices; a reusable pool lists plugs the socket can hold);
* `equipped_now` / `socketed_not_equipped`: the plug in the socket per component 305. The second
  is used when the profile says the item is not equipped (for example a vaulted weapon);
* `instance_listed`: the plug appears in this instance's reusable-plug list (component 310);
* `account_listed` / `character_listed`: M5B plug-set evidence, only where it was retained;
* `insertion_blocked_reported`: `canInsert=false` or insertion failure indexes in the instance
  list. `enabled=false` is reported on its own and is not an insertion restriction;
* `unknown`: not returned or not joinable.

Material cost, armor energy, fragment capacity and activity restrictions are not read, so
`actionable` is always `unknown` here.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.configuration_evidence import (
    ConfigSource,
    ConfigStateRecord,
    ConfigurationOption,
    _scope_states,
)
from app.instance_plug_evidence import InstancePlugEvidence, InstanceSocketEvidence
from app.models import (
    ItemSummary,
    PlugSetCollection,
    PlugSetSocketRef,
    SocketConfigSummary,
)

POOL_ROLE = {
    "randomized_roll_pool": "randomized_roll_pool",
    "reusable_plug_pool": "reusable_plug_pool",
    "fixed_by_manifest": "fixed",
    "display_only_options": "display_only",
    "inventory_sourced": "inventory_sourced",
    "insufficient_information": "unknown",
}
MAX_ROWS = 8
MAX_NOTABLE = 3
MAX_SAMPLE = 4
MAX_SECTION_CHARS = 3000

BASIS = (
    "From one item instance's GetItem response, joined by exact item hash, instance and socket "
    "index. manifest_possible = the socket's Manifest pool (a randomized pool lists possible "
    "rolls, not choices). instance_listed = this instance listed the plug; it does not prove "
    "a change can be made now. insertion_blocked_reported = canInsert false or insertion "
    "failure indexes; not_enabled_reported = enabled false, which is not an insertion "
    "restriction. no_entry = the response had no list for that socket, not that it has no "
    "options. unknown = not returned or not joinable. actionable is always unknown."
)
LIMITATIONS = (
    "Material cost, armor energy, fragment capacity and activity restrictions were not assessed."
)


class InstanceSocketJoin(BaseModel):
    socket_index: int
    category: str | None = None
    manifest_pool: dict[str, Any] | None = None
    reusable_listing: str
    equipped_plug_hash: int | None = None
    equipped_differs_from_profile: bool = False
    listed: int = 0
    omitted_options: int = 0
    listed_outside_manifest_pool: int | None = None
    account_listing: str | None = None


class InstanceJoin(BaseModel):
    """The joined result for one instance; `options` carry per-state provenance."""

    join: Literal["joined", "identity_mismatch", "not_returned"]
    evidence_status: str
    scope: Literal["equipped", "not_equipped"] = "equipped"
    sockets: list[InstanceSocketJoin] = Field(default_factory=list)
    options: list[ConfigurationOption] = Field(default_factory=list)
    actionable: Literal["unknown"] = "unknown"


def _source(evidence: InstancePlugEvidence, socket: int, component: str) -> ConfigSource:
    return ConfigSource(
        kind="instance_reusable_plugs" if component == "ItemReusablePlugs" else "socket_component",
        item_hash=evidence.item_hash,
        socket_index=socket,
        component=component,
    )


def _pool_description(
    config: SocketConfigSummary | None,
    pool_hashes: set[int] | None,
    listed: list[int],
) -> tuple[dict[str, Any] | None, int | None]:
    if config is None:
        return None, None
    pool: dict[str, Any] = {"role": POOL_ROLE.get(config.classification, "unknown")}
    if config.options is not None:
        pool["options"] = config.options
    if config.pool_kind:
        pool["kind"] = config.pool_kind
    outside = None
    if pool_hashes is not None:
        outside = sum(plug not in pool_hashes for plug in listed)
    return pool, outside


def join_instance_evidence(
    evidence: InstancePlugEvidence,
    item: ItemSummary | None,
    config: list[SocketConfigSummary] | None,
    *,
    profile_sets: PlugSetCollection | None = None,
    character_sets: PlugSetCollection | None = None,
    manifest_pool_hashes: dict[int, set[int]] | None = None,
) -> InstanceJoin:
    """Join one instance's evidence to the profile item and Manifest structure."""

    if evidence.status == "identity_mismatch":
        return InstanceJoin(join="identity_mismatch", evidence_status=evidence.status)
    if evidence.status == "not_returned":
        return InstanceJoin(join="not_returned", evidence_status=evidence.status)
    if (
        item is None
        or item.item_hash != evidence.item_hash
        or item.instance_id is None
        or item.instance_id != evidence.instance_id
    ):
        return InstanceJoin(join="identity_mismatch", evidence_status=evidence.status)

    equipped = item.location == "equipped"
    by_config = {value.index: value for value in config or []}
    refs_by_index = {value.socket_index: value for value in item.plug_set_sockets}
    profile_plugs = {
        value.socket_index: value.item_hash
        for value in item.socketed_plug_details
        if value.socket_index is not None
    }
    result = InstanceJoin(
        join="joined",
        evidence_status=evidence.status,
        scope="equipped" if equipped else "not_equipped",
    )
    for socket in evidence.sockets:
        cfg = by_config.get(socket.socket_index)
        listed_hashes = [value.plug_item_hash for value in socket.options]
        pool_hashes = (
            manifest_pool_hashes.get(cfg.set_hash)
            if cfg is not None and cfg.set_hash is not None and manifest_pool_hashes
            else None
        )
        pool, outside = _pool_description(cfg, pool_hashes, listed_hashes)
        row = InstanceSocketJoin(
            socket_index=socket.socket_index,
            category=cfg.category if cfg else None,
            manifest_pool=pool,
            reusable_listing=socket.reusable_listing,
            equipped_plug_hash=socket.equipped_plug_hash,
            equipped_differs_from_profile=(
                socket.equipped_plug_hash is not None
                and profile_plugs.get(socket.socket_index) not in (None, socket.equipped_plug_hash)
            ),
            listed=socket.listed,
            omitted_options=socket.omitted_options,
            listed_outside_manifest_pool=outside,
        )
        scopes = _account_scopes(
            cfg, refs_by_index.get(socket.socket_index), profile_sets, character_sets
        )
        if scopes == "not_retained":
            row.account_listing = "not_retained"
        result.sockets.append(row)
        result.options.extend(
            _socket_options(
                evidence,
                item,
                socket,
                cfg,
                pool_hashes,
                equipped,
                scopes if isinstance(scopes, list) else [],
            )
        )
    return result


def _account_scopes(
    cfg: SocketConfigSummary | None,
    ref: PlugSetSocketRef | None,
    profile_sets: PlugSetCollection | None,
    character_sets: PlugSetCollection | None,
) -> list[tuple[str, PlugSetCollection, PlugSetSocketRef]] | str | None:
    """Plug-set scopes the socket declares and for which M5B retained the set (else a note).

    The declaration comes from the profile item's M5B socket refs (exact set hash and source
    flags) when it has them, else from the M3 configuration. M3 keeps source flags only for
    some socket kinds, so an absent declaration means "not known to use plug sets".
    """

    if ref is None and cfg is not None and cfg.set_hash is not None and cfg.plug_sources:
        ref = PlugSetSocketRef(
            socket_index=cfg.index, sources=cfg.plug_sources, plug_set_hash=cfg.set_hash
        )
    if ref is None:
        return None
    declared: list[tuple[str, PlugSetCollection | None]] = []
    if "profile_plug_set" in ref.sources:
        declared.append(("profile", profile_sets))
    if "character_plug_set" in ref.sources:
        declared.append(("character", character_sets))
    if not declared:
        return None
    usable = [
        (scope, collection, ref)
        for scope, collection in declared
        if collection is not None
        and collection.status == "returned"
        and collection.get(ref.plug_set_hash) is not None
    ]
    return usable if len(usable) == len(declared) else "not_retained"


def _socket_options(
    evidence: InstancePlugEvidence,
    item: ItemSummary,
    socket: InstanceSocketEvidence,
    cfg: SocketConfigSummary | None,
    pool_hashes: set[int] | None,
    equipped: bool,
    scopes: list[tuple[str, PlugSetCollection, PlugSetSocketRef]],
) -> list[ConfigurationOption]:
    options: list[ConfigurationOption] = []
    role = POOL_ROLE.get(cfg.classification, "unknown") if cfg else None
    pool_state = (
        ConfigStateRecord(
            state="manifest_possible",
            source=ConfigSource(
                kind="manifest_plug_set", plug_set_hash=cfg.set_hash if cfg else None
            ),
            pool=role,
        )
        if cfg is not None and cfg.set_hash is not None
        else None
    )
    listed_by_hash = {value.plug_item_hash: value for value in socket.options}
    ordered = list(
        dict.fromkeys(
            [
                *([socket.equipped_plug_hash] if socket.equipped_plug_hash is not None else []),
                *listed_by_hash,
            ]
        )
    )
    for plug_hash in ordered:
        states: list[ConfigStateRecord] = []
        if pool_state is not None and pool_hashes is not None and plug_hash in pool_hashes:
            states.append(pool_state)
        if plug_hash == socket.equipped_plug_hash:
            states.append(
                ConfigStateRecord(
                    state="equipped_now" if equipped else "socketed_not_equipped",
                    source=_source(evidence, socket.socket_index, "ItemSockets"),
                    enabled=socket.equipped_enabled,
                    reasons=list(socket.equipped_enable_fail_reasons),
                )
            )
        listed = listed_by_hash.get(plug_hash)
        if listed is not None:
            states.append(
                ConfigStateRecord(
                    state="instance_listed",
                    source=_source(evidence, socket.socket_index, "ItemReusablePlugs"),
                    can_insert=listed.can_insert,
                    enabled=listed.enabled,
                )
            )
            if listed.can_insert is False or listed.insert_fail_indexes:
                states.append(
                    ConfigStateRecord(
                        state="insertion_blocked_reported",
                        source=_source(evidence, socket.socket_index, "ItemReusablePlugs"),
                        can_insert=listed.can_insert,
                        enabled=listed.enabled,
                        reasons=list(listed.insert_fail_reasons),
                    )
                )
        for scope, collection, ref in scopes:
            states.extend(_scope_states(scope, collection, ref, plug_hash))
        options.append(
            ConfigurationOption(
                item_hash=evidence.item_hash,
                socket_index=socket.socket_index,
                plug_item_hash=plug_hash,
                plug_name=listed.name if listed is not None else None,
                category=cfg.category if cfg else None,
                states=states,
                limitations=[LIMITATIONS],
                instance_id=item.instance_id,
            )
        )
    return options


# --- tool-shaped section (not wired to any tool) -----------------------------------------------


def instance_configuration_section(
    join: InstanceJoin,
    evidence: InstancePlugEvidence,
    *,
    max_rows: int = MAX_ROWS,
    max_chars: int = MAX_SECTION_CHARS,
) -> dict[str, Any]:
    """A bounded, identifier-free summary for a future tool; counts omissions instead of hiding."""

    section: dict[str, Any] = {
        "join": join.join,
        "evidence_status": join.evidence_status,
        "scope": join.scope,
        "actionable": "unknown",
        "sockets": [],
    }
    if join.join != "joined":
        return section
    section.update({"basis": BASIS, "limitations": LIMITATIONS})
    used = len(json.dumps(section))
    omitted = 0
    for row in join.sockets:
        detail = evidence.socket(row.socket_index)
        entry = row.model_dump(mode="json", exclude_none=True)
        entry = {key: value for key, value in entry.items() if value not in (False, 0)}
        if detail is not None:
            blocked = [v for v in detail.options if v.can_insert is False or v.insert_fail_indexes]
            disabled = [v for v in detail.options if v.enabled is False and v not in blocked]
            if blocked:
                entry["insertion_blocked_reported"] = [
                    {
                        "plug": v.name or f"plug {v.plug_item_hash}",
                        "reasons": v.insert_fail_reasons or None,
                        **({"reasons_unresolved": True} if None in v.insert_fail_reasons else {}),
                    }
                    for v in blocked[:MAX_NOTABLE]
                ]
            if disabled:
                entry["not_enabled_reported"] = [
                    v.name or f"plug {v.plug_item_hash}" for v in disabled[:MAX_NOTABLE]
                ]
            names = [v.name for v in detail.options if v.name][:MAX_SAMPLE]
            if names:
                entry["listed_sample"] = names
            if detail.duplicate_entries:
                entry["duplicate_entries"] = detail.duplicate_entries
            if detail.malformed_entries:
                entry["malformed_entries"] = detail.malformed_entries
            if detail.omitted_options:
                entry["counts_are_lower_bounds"] = True
        size = len(json.dumps(entry))
        if len(section["sockets"]) >= max_rows or used + size > max_chars:
            omitted += 1
            continue
        used += size
        section["sockets"].append(entry)
    if omitted:
        section["omitted_sockets"] = omitted
    if evidence.omitted_sockets:
        section["sockets_not_retained"] = evidence.omitted_sockets
    return section
