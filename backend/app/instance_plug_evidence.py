"""Parser for item-instance plug evidence from a GetItem response (offline; no fetching).

Knowledge Engine v0.2, M5C1. A future bounded GetItem call returns a `DestinyItemResponse`; this
module interprets the parts of it that describe one item *instance's* sockets. It performs no
request and nothing in the application calls it yet.

Documented structures (from Bungie's OpenAPI definitions, obtained in M5A)
-------------------------------------------------------------------------
* `item`            (ItemCommonData): `data.itemHash`, `data.itemInstanceId`.
* `sockets`         (ItemSockets, 305): `data.sockets` is a list indexed by socket index; each
  state has `plugHash`, `isEnabled`, `isVisible`, `enableFailIndexes`.
* `reusablePlugs`   (ItemReusablePlugs, 310): `data.plugs` maps a socket index (as a string key)
  to a list of DestinyItemPlugBase: `plugItemHash`, `canInsert`, `enabled`, `insertFailIndexes`,
  `enableFailIndexes`, stack sizes. "If a socket doesn't have any reusable plugs defined at the
  item scope, there will be no entry for that socket": a missing entry therefore means the
  options are *not described here* (they may come from plug sets or inventory), never "none".
* `plugObjectives`  (ItemPlugObjectives, 309): `data.objectivesPerPlug` maps a plug hash to
  its objective progress; only its presence is kept, as a flag.
* There is **no** ItemPlugStates (308) in a GetItem response; 308 appears only in profile and
  character item component sets, keyed by plug hash. It is not read here.

Not verified against a real account: that a real response matches these shapes, what 310 lists
for a randomly rolled perk column (the rolled options only, or alternatives), whether
`canInsert` accounts for material cost, and how crafted or enhanced weapons populate it.
Consequently everything missing, malformed, duplicated or truncated is reported as such and
the join (`instance_configuration`) never turns it into availability.

`insertFailIndexes` index the plug's `plug.insertionRules`; `enableFailIndexes` index its
`plug.enabledRules`. `enabled=false` is not `canInsert=false`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_serializer

from app.plug_set_evidence import resolve_reasons

ComponentState = Literal["returned", "not_returned", "malformed"]
EvidenceStatus = Literal["complete", "partial", "not_returned", "identity_mismatch"]
Identity = Literal["verified", "requested_only", "mismatch"]

SOURCE = "GetItem:ItemSockets+ItemReusablePlugs"
MAX_SOCKETS = 16
MAX_OPTIONS_PER_SOCKET = 24


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value else None


def _flag(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _indexes(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, int) and not isinstance(item, bool)]


class InstancePlugOption(BaseModel):
    """One plug listed for one socket of one instance (component 310)."""

    plug_item_hash: int
    can_insert: bool | None = None
    enabled: bool | None = None
    insert_fail_indexes: list[int] = Field(default_factory=list)
    enable_fail_indexes: list[int] = Field(default_factory=list)
    insert_fail_reasons: list[str | None] = Field(default_factory=list)
    enable_fail_reasons: list[str | None] = Field(default_factory=list)
    name: str | None = None
    has_objectives: bool = False
    # Extra copies Bungie listed for this plug; True when those copies disagreed on a flag, in
    # which case the disagreeing flags are reset to unknown rather than picked.
    duplicates: int = 0
    conflicting_duplicates: bool = False

    @model_serializer(mode="wrap")
    def _omit_empty(self, handler: Any) -> dict[str, Any]:
        data = handler(self)
        return {
            key: value
            for key, value in data.items()
            if key == "plug_item_hash" or value not in (None, [], 0, False)
        }


class InstanceSocketEvidence(BaseModel):
    socket_index: int
    # Equipped state from the same response (component 305); None when 305 was not usable.
    equipped_plug_hash: int | None = None
    equipped_enabled: bool | None = None
    equipped_enable_fail_indexes: list[int] = Field(default_factory=list)
    equipped_enable_fail_reasons: list[str | None] = Field(default_factory=list)
    # "listed": component 310 had an entry for this socket. "no_entry": it did not, which does
    # not mean the socket has no options. "not_returned": 310 itself was unusable.
    reusable_listing: Literal["listed", "no_entry", "not_returned"] = "no_entry"
    options: list[InstancePlugOption] = Field(default_factory=list)
    listed: int = 0
    omitted_options: int = 0
    duplicate_entries: int = 0
    malformed_entries: int = 0

    @model_serializer(mode="wrap")
    def _omit_empty(self, handler: Any) -> dict[str, Any]:
        data = handler(self)
        keep = {"socket_index", "reusable_listing"}
        return {
            key: value for key, value in data.items() if key in keep or value not in (None, [], 0)
        }


class InstancePlugEvidence(BaseModel):
    """What one GetItem response said about one instance's sockets, with its completeness."""

    item_hash: int
    # Internal handle for joining to the profile's item; never serialized for the model.
    instance_id: str = Field(exclude=True)
    source: str = SOURCE
    identity: Identity = "requested_only"
    status: EvidenceStatus = "not_returned"
    components: dict[str, ComponentState] = Field(default_factory=dict)
    sockets: list[InstanceSocketEvidence] = Field(default_factory=list)
    omitted_sockets: int = 0
    sockets_without_listing: int = 0
    malformed_entries: int = 0
    # DestinyItemResponse carries no minted timestamp, so none is recorded.

    def socket(self, index: int) -> InstanceSocketEvidence | None:
        return next((value for value in self.sockets if value.socket_index == index), None)

    def to_tool(self) -> dict[str, Any]:
        """Bounded, identifier-free form: counts and option lists, no instance handle."""

        data = self.model_dump(mode="json")
        return {key: value for key, value in data.items() if value not in (None, [], {}, 0)}


# --- parsing ----------------------------------------------------------------------------------


def _component(response: dict[str, Any], key: str) -> tuple[ComponentState, Any]:
    """('returned', data) | ('not_returned', None) | ('malformed', None)."""

    if key not in response or response[key] is None:
        return "not_returned", None
    component = response[key]
    if not isinstance(component, dict):
        return "malformed", None
    data = component.get("data")
    if data is None:
        return "not_returned", None
    return ("returned", data) if isinstance(data, dict) else ("malformed", None)


def _definition(definitions: dict[int, dict[str, Any]] | None, plug_hash: int | None) -> Any:
    return (definitions or {}).get(plug_hash) if plug_hash is not None else None


def _name(definition: dict[str, Any] | None) -> str | None:
    value = ((definition or {}).get("displayProperties") or {}).get("name")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _merge_duplicate(first: InstancePlugOption, raw: dict[str, Any]) -> None:
    first.duplicates += 1
    for attribute, key in (("can_insert", "canInsert"), ("enabled", "enabled")):
        if getattr(first, attribute) != _flag(raw.get(key)):
            setattr(first, attribute, None)
            first.conflicting_duplicates = True
    if _indexes(raw.get("insertFailIndexes")) != first.insert_fail_indexes or (
        _indexes(raw.get("enableFailIndexes")) != first.enable_fail_indexes
    ):
        first.conflicting_duplicates = True


def _options(
    listing: Any,
    objectives: set[int],
    plug_definitions: dict[int, dict[str, Any]] | None,
) -> InstanceSocketEvidence | None:
    """Options for one socket from one 310 list; None if the list is not a list at all."""

    if not isinstance(listing, list):
        return None
    result = InstanceSocketEvidence(socket_index=0, reusable_listing="listed")
    seen: dict[int, InstancePlugOption] = {}
    for raw in listing:
        plug_hash = _int(raw.get("plugItemHash")) if isinstance(raw, dict) else None
        if plug_hash is None:
            result.malformed_entries += 1
            continue
        if plug_hash in seen:
            result.duplicate_entries += 1
            _merge_duplicate(seen[plug_hash], raw)
            continue
        definition = _definition(plug_definitions, plug_hash)
        insert = _indexes(raw.get("insertFailIndexes"))
        enable = _indexes(raw.get("enableFailIndexes"))
        seen[plug_hash] = InstancePlugOption(
            plug_item_hash=plug_hash,
            can_insert=_flag(raw.get("canInsert")),
            enabled=_flag(raw.get("enabled")),
            insert_fail_indexes=insert,
            enable_fail_indexes=enable,
            insert_fail_reasons=resolve_reasons(insert, definition, "insertionRules")
            if insert
            else [],
            enable_fail_reasons=resolve_reasons(enable, definition, "enabledRules")
            if enable
            else [],
            name=_name(definition),
            has_objectives=plug_hash in objectives,
        )
    ordered = list(seen.values())
    # Entries that carry a restriction or an unreported flag are kept before plain ones.
    ordered.sort(
        key=lambda value: (
            not (
                value.can_insert is not True
                or value.enabled is not True
                or value.insert_fail_indexes
            )
        )
    )
    result.listed = len(ordered)
    result.options = ordered[:MAX_OPTIONS_PER_SOCKET]
    result.omitted_options = len(ordered) - len(result.options)
    return result


def parse_item_plug_response(
    response: Any,
    *,
    item_hash: int,
    instance_id: str,
    plug_definitions: dict[int, dict[str, Any]] | None = None,
    focus_indexes: set[int] | frozenset[int] = frozenset(),
) -> InstancePlugEvidence:
    """Interpret one GetItem response for a known (item hash, instance) pair.

    `plug_definitions` maps plug hashes to Manifest definitions for names and rule messages; a
    plug without one keeps unresolved indexes. `focus_indexes` are sockets reported even when
    310 has no entry for them (so their equipped plug is still recorded).
    """

    evidence = InstancePlugEvidence(item_hash=item_hash, instance_id=instance_id)
    if not isinstance(response, dict):
        evidence.components = {"sockets": "malformed", "reusablePlugs": "malformed"}
        evidence.malformed_entries = 1
        return evidence
    state, item = _component(response, "item")
    if state == "returned":
        returned_hash = _int(item.get("itemHash"))
        returned_instance = item.get("itemInstanceId")
        if returned_hash != item_hash or (
            returned_instance is not None and str(returned_instance) != instance_id
        ):
            evidence.identity = "mismatch"
            evidence.status = "identity_mismatch"
            return evidence
        evidence.identity = "verified"

    sockets_state, sockets = _component(response, "sockets")
    reusable_state, reusable = _component(response, "reusablePlugs")
    objectives_state, objectives = _component(response, "plugObjectives")
    evidence.components = {
        "sockets": sockets_state,
        "reusablePlugs": reusable_state,
        "plugObjectives": objectives_state,
    }
    socket_list = sockets.get("sockets") if sockets is not None else None
    if sockets is not None and not isinstance(socket_list, list):
        evidence.components["sockets"] = "malformed"
        socket_list = None
    plugs = reusable.get("plugs") if reusable is not None else None
    if reusable is not None and not isinstance(plugs, dict):
        evidence.components["reusablePlugs"] = "malformed"
        plugs = None
    per_plug = objectives.get("objectivesPerPlug") if objectives is not None else None
    if objectives is not None and not isinstance(per_plug, dict):
        evidence.components["plugObjectives"] = "malformed"
        per_plug = None
    with_objectives = {
        key
        for key, value in (per_plug or {}).items()
        if isinstance(value, list) and value and isinstance(key, str) and key.isdigit()
    }
    objective_hashes = {int(key) for key in with_objectives}

    by_index: dict[int, InstanceSocketEvidence] = {}
    for key, listing in (plugs or {}).items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            evidence.malformed_entries += 1
            continue
        if index < 0:
            evidence.malformed_entries += 1
            continue
        parsed = _options(listing, objective_hashes, plug_definitions)
        if parsed is None:
            evidence.malformed_entries += 1
            continue
        parsed.socket_index = index
        by_index[index] = parsed
    for index in focus_indexes:
        by_index.setdefault(
            index,
            InstanceSocketEvidence(
                socket_index=index,
                reusable_listing="no_entry" if plugs is not None else "not_returned",
            ),
        )
    for index, socket in by_index.items():
        if socket_list is not None and 0 <= index < len(socket_list):
            state_raw = socket_list[index]
            if isinstance(state_raw, dict):
                socket.equipped_plug_hash = _int(state_raw.get("plugHash"))
                socket.equipped_enabled = _flag(state_raw.get("isEnabled"))
                socket.equipped_enable_fail_indexes = _indexes(state_raw.get("enableFailIndexes"))
                socket.equipped_enable_fail_reasons = (
                    resolve_reasons(
                        socket.equipped_enable_fail_indexes,
                        _definition(plug_definitions, socket.equipped_plug_hash),
                        "enabledRules",
                    )
                    if socket.equipped_enable_fail_indexes
                    else []
                )
            else:
                evidence.malformed_entries += 1
    ordered = [by_index[index] for index in sorted(by_index)]
    evidence.sockets = ordered[:MAX_SOCKETS]
    evidence.omitted_sockets = len(ordered) - len(evidence.sockets)
    evidence.sockets_without_listing = sum(
        value.reusable_listing != "listed" for value in evidence.sockets
    )
    evidence.malformed_entries += sum(value.malformed_entries for value in evidence.sockets)

    usable = {
        name for name in ("sockets", "reusablePlugs") if evidence.components[name] == "returned"
    }
    if not usable:
        evidence.status = "not_returned"
    elif (
        len(usable) == 2
        and evidence.malformed_entries == 0
        and evidence.omitted_sockets == 0
        and not any(value.omitted_options for value in evidence.sockets)
    ):
        evidence.status = "complete"
    else:
        evidence.status = "partial"
    return evidence
