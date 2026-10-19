"""Account- and character-scoped plug-set evidence from the profile response already fetched.

Knowledge Engine v0.2, M5B. This module reads two fields of the GetProfile response that arrive
with the ItemSockets component (305) and that the application previously discarded:

* `profilePlugSets`:   {"data": {"plugs": {"<plugSetHash>": [DestinyItemPlug, ...]}}}
* `characterPlugSets`: {"data": {"<characterId>": {"plugs": {"<plugSetHash>": [...]}}}}

Neither is a flat dictionary: a set's entries sit under `data.plugs[hash]`, and character sets
add a character level. Each entry is a DestinyItemPlug (`plugItemHash`, `canInsert`, `enabled`,
`insertFailIndexes`, `enableFailIndexes`). Everything here is pure: no network, no lookups. The
normalizer supplies already-loaded Manifest definitions.

What this is not
----------------
A plug listed in a set is not proof the player can use it now. `canInsert=false` is a reported
insertion restriction; `enabled=false` only says a plug is not providing its benefits. Material
cost, armor energy, fragment slot capacity, activity restrictions and instance state are not
read. Missing data is never converted into an empty, verified set.
"""

from __future__ import annotations

from typing import Any

from app.models import (
    PlugSetCollection,
    PlugSetEntry,
    PlugSetEvidence,
    PlugSetScope,
    PlugSetSocketRef,
)
from app.socket_configuration import (
    SOURCE_CHARACTER_PLUG_SET,
    SOURCE_NAMES,
    SOURCE_PROFILE_PLUG_SET,
    socket_entries,
)

MAX_SET_ENTRIES = 200
MAX_SAMPLE = 3
PLUG_SOURCE_MASK = SOURCE_PROFILE_PLUG_SET | SOURCE_CHARACTER_PLUG_SET


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value else None


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, int) and not isinstance(item, bool)]


def _plugs_by_set(container: Any) -> dict[int, list[dict[str, Any]]] | None:
    """`{"plugs": {"<hash>": [entry, ...]}}` -> {hash: [entry, ...]}; None when malformed."""

    plugs = container.get("plugs") if isinstance(container, dict) else None
    if not isinstance(plugs, dict):
        return None
    found: dict[int, list[dict[str, Any]]] = {}
    for key, entries in plugs.items():
        try:
            set_hash = int(key)
        except (TypeError, ValueError):
            continue
        if isinstance(entries, list):
            found[set_hash] = [value for value in entries if isinstance(value, dict)]
    return found


def returned_sets(
    profile: dict[str, Any], scope: PlugSetScope, character_id: str | None = None
) -> dict[int, list[dict[str, Any]]] | None:
    """The raw plug sets Bungie returned for a scope, or None if none were returned.

    None means "not returned" (never requested, withheld by privacy, or malformed): it must not
    be read as "no plugs". An empty dict means a component was returned with no sets in it.
    """

    if scope == "profile":
        component = profile.get("profilePlugSets")
        data = component.get("data") if isinstance(component, dict) else None
        return _plugs_by_set(data)
    component = profile.get("characterPlugSets")
    data = component.get("data") if isinstance(component, dict) else None
    if not isinstance(data, dict) or character_id is None:
        return None
    return _plugs_by_set(data.get(character_id))


def socket_plug_set_refs(
    definition: dict[str, Any],
    skip: set[int] | frozenset[int] = frozenset(),
    category_names: dict[int, str] | None = None,
    plug_set_definitions: dict[int, dict[str, Any]] | None = None,
) -> list[PlugSetSocketRef]:
    """Sockets of an item definition fed by profile or character plug sets.

    The join is exact: the socket's own `reusablePlugSetHash` and `plugSources` bitmask. The
    bitmask combines flags, so every declared source is listed. Cosmetic sockets are skipped
    (`skip`), as their sets are very large and irrelevant to a build.
    """

    category_by_index: dict[int, int] = {}
    categories = (definition.get("sockets") or {}).get("socketCategories")
    for category in categories if isinstance(categories, list) else []:
        category_hash = (
            _int(category.get("socketCategoryHash")) if isinstance(category, dict) else None
        )
        for index in (category.get("socketIndexes") or []) if isinstance(category, dict) else []:
            if category_hash and isinstance(index, int):
                category_by_index[index] = category_hash
    refs: list[PlugSetSocketRef] = []
    for index, entry in enumerate(socket_entries(definition)):
        if index in skip:
            continue
        bits = _int(entry.get("plugSources")) or 0
        set_hash = _int(entry.get("reusablePlugSetHash"))
        if not bits & PLUG_SOURCE_MASK or set_hash is None:
            continue
        options = None
        definition_of_set = (plug_set_definitions or {}).get(set_hash)
        if definition_of_set is not None:
            options = len(_manifest_options(definition_of_set))
        refs.append(
            PlugSetSocketRef(
                socket_index=index,
                category=(category_names or {}).get(category_by_index.get(index, 0)),
                sources=[name for flag, name in SOURCE_NAMES.items() if bits & flag],
                plug_set_hash=set_hash,
                manifest_options=options,
            )
        )
    return refs


def _manifest_options(plug_set: dict[str, Any]) -> list[int]:
    found: list[int] = []
    for option in plug_set.get("reusablePlugItems") or []:
        value = _int(option.get("plugItemHash")) if isinstance(option, dict) else None
        if value is not None and value not in found:
            found.append(value)
    return found


def resolve_reasons(
    indexes: list[int], plug_definition: dict[str, Any] | None, rule_key: str
) -> list[str | None]:
    """Manifest messages for failure indexes; an index that cannot be resolved stays None.

    `insertFailIndexes` index the plug's `plug.insertionRules`; `enableFailIndexes` index its
    `plug.enabledRules`. The two arrays are different and must not be mixed.
    """

    rules = ((plug_definition or {}).get("plug") or {}).get(rule_key)
    reasons: list[str | None] = []
    for index in indexes:
        rule = rules[index] if isinstance(rules, list) and 0 <= index < len(rules) else None
        message = rule.get("failureMessage") if isinstance(rule, dict) else None
        reasons.append(message.strip() if isinstance(message, str) and message.strip() else None)
    return reasons


def _entry(raw: dict[str, Any]) -> PlugSetEntry | None:
    plug_hash = _int(raw.get("plugItemHash"))
    if plug_hash is None:
        return None
    can_insert, enabled = raw.get("canInsert"), raw.get("enabled")
    return PlugSetEntry(
        plug_item_hash=plug_hash,
        can_insert=can_insert if isinstance(can_insert, bool) else None,
        enabled=enabled if isinstance(enabled, bool) else None,
        insert_fail_indexes=_int_list(raw.get("insertFailIndexes")),
        enable_fail_indexes=_int_list(raw.get("enableFailIndexes")),
    )


def is_flagged(entry: PlugSetEntry) -> bool:
    """Entries that are not simply insertable and enabled; their detail is worth keeping."""

    return (
        entry.can_insert is False
        or entry.enabled is False
        or bool(entry.insert_fail_indexes or entry.enable_fail_indexes)
    )


def is_clear(entry: PlugSetEntry) -> bool:
    """Both flags explicitly reported true and no failure indexes: nothing else to keep."""

    return entry.can_insert is True and entry.enabled is True and not is_flagged(entry)


def flagged_and_sample_hashes(raw_entries: list[dict[str, Any]]) -> set[int]:
    """Plug hashes whose definitions are needed for names and rule messages."""

    found: set[int] = set()
    plain = 0
    for raw in raw_entries:
        entry = _entry(raw)
        if entry is None:
            continue
        if is_flagged(entry):
            found.add(entry.plug_item_hash)
        elif plain < MAX_SAMPLE:
            found.add(entry.plug_item_hash)
            plain += 1
    return found


def build_evidence(
    scope: PlugSetScope,
    character_id: str | None,
    plug_set_hash: int,
    raw_entries: list[dict[str, Any]],
    manifest_set: dict[str, Any] | None,
    plug_definitions: dict[int, dict[str, Any]],
) -> PlugSetEvidence:
    """Compact, faithful evidence for one returned plug set."""

    entries = [value for raw in raw_entries if (value := _entry(raw)) is not None]
    options = set(_manifest_options(manifest_set)) if manifest_set is not None else None
    unmatched = sum(value.plug_item_hash not in options for value in entries) if options else 0
    sample: list[str] = []
    for value in entries:
        definition = plug_definitions.get(value.plug_item_hash)
        name = ((definition or {}).get("displayProperties") or {}).get("name")
        if is_flagged(value):
            value.name = name or None
            value.insert_fail_reasons = resolve_reasons(
                value.insert_fail_indexes, definition, "insertionRules"
            )
            value.enable_fail_reasons = resolve_reasons(
                value.enable_fail_indexes, definition, "enabledRules"
            )
        elif name and len(sample) < MAX_SAMPLE:
            sample.append(name)
    detailed = [value for value in entries if not is_clear(value)][:MAX_SET_ENTRIES]
    room = MAX_SET_ENTRIES - len(detailed)
    clear = list(dict.fromkeys(value.plug_item_hash for value in entries if is_clear(value)))
    retained_clear = clear[: max(room, 0)]
    return PlugSetEvidence(
        plug_set_hash=plug_set_hash,
        scope=scope,
        character_id=character_id,
        entries=detailed,
        clear_plug_hashes=retained_clear,
        listed=len(entries),
        omitted_entries=max(0, len(entries) - len(detailed) - len(retained_clear)),
        manifest_options=len(options) if options is not None else None,
        manifest_status="resolved" if manifest_set is not None else "unresolved",
        unmatched_entries=unmatched,
        sample=sample,
    )


def build_collection(
    profile: dict[str, Any],
    scope: PlugSetScope,
    character_id: str | None,
    wanted: set[int],
    manifest_sets: dict[int, dict[str, Any]],
    plug_definitions: dict[int, dict[str, Any]],
) -> PlugSetCollection:
    """Evidence for the plug sets in `wanted` only; the rest of the account's sets are dropped."""

    minted = profile.get("responseMintedTimestamp")
    minted_at = minted if isinstance(minted, str) else None
    raw = returned_sets(profile, scope, character_id)
    if raw is None:
        return PlugSetCollection(status="not_returned", minted_at=minted_at)
    sets = [
        build_evidence(
            scope,
            character_id,
            set_hash,
            raw[set_hash],
            manifest_sets.get(set_hash),
            plug_definitions,
        )
        for set_hash in sorted(wanted)
        if set_hash in raw
    ]
    return PlugSetCollection(
        status="returned",
        minted_at=minted_at,
        sets=sets,
        missing_set_hashes=sorted(wanted - set(raw)),
    )
