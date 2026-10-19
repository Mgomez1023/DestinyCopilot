"""M5C1: instance reusable-plug evidence, parsed and joined offline.

Fixtures follow the documented `DestinyItemResponse` shapes (item, sockets/305, reusablePlugs/310,
plugObjectives/309); nothing here has been checked against a real account. Joins against real
Manifest definitions are skipped without a local Manifest. No test touches the network.
"""

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from app import instance_configuration, instance_plug_evidence
from app.configuration_evidence import configuration_section
from app.guardian_tools import GuardianToolService
from app.instance_configuration import (
    MAX_ROWS,
    instance_configuration_section,
    join_instance_evidence,
)
from app.instance_plug_evidence import (
    MAX_OPTIONS_PER_SOCKET,
    MAX_SOCKETS,
    parse_item_plug_response,
)
from app.mechanics_evidence import equipped_mechanics_section
from app.models import (
    GuardianContext,
    ItemSummary,
    PlugSetCollection,
    PlugSetEntry,
    PlugSetEvidence,
    SocketConfigSummary,
    SocketedPlugSummary,
)
from app.socket_configuration import (
    build_specs,
    cosmetic_socket_indexes,
    describe_sockets,
    name_hashes_needed,
    plug_set_hashes,
    referenced_definition_hashes,
    summary_entries,
)
from tests.test_objective_requirements import REAL, needs_manifest
from tests.test_plug_set_evidence import character_of, normalized

ITEM_HASH, INSTANCE = 5001, "inst-A"


def plug_def(name: str, insertion: list[str] = (), enabled: list[str] = ()) -> dict[str, Any]:  # noqa: B006
    return {
        "displayProperties": {"name": name, "description": ""},
        "plug": {
            "insertionRules": [{"failureMessage": text} for text in insertion],
            "enabledRules": [{"failureMessage": text} for text in enabled],
        },
    }


DEFINITIONS = {
    11: plug_def("Alpha"),
    12: plug_def("Beta", insertion=["Needs a quest.", "Needs rank."], enabled=["Wrong mode."]),
    13: plug_def("Gamma", enabled=["Disabled rule one.", "Disabled rule two."]),
    14: plug_def("Delta"),
}


def option(plug: int, *, can_insert: Any = True, enabled: Any = True, **extra: Any) -> dict:
    value: dict[str, Any] = {"plugItemHash": plug, **extra}
    if can_insert is not None:
        value["canInsert"] = can_insert
    if enabled is not None:
        value["enabled"] = enabled
    return value


def response(
    *,
    sockets: list[Any] | None = None,
    reusable: dict[str, Any] | None = None,
    objectives: dict[str, Any] | None = None,
    item: dict[str, Any] | None = ...,  # type: ignore[assignment]
) -> dict[str, Any]:
    body: dict[str, Any] = {}
    if item is ...:
        item = {"itemHash": ITEM_HASH, "itemInstanceId": INSTANCE}
    if item is not None:
        body["item"] = {"data": item}
    if sockets is not None:
        body["sockets"] = {"data": {"sockets": sockets}}
    if reusable is not None:
        body["reusablePlugs"] = {"data": {"plugs": reusable}}
    if objectives is not None:
        body["plugObjectives"] = {"data": {"objectivesPerPlug": objectives}}
    return body


def parse(body: Any, **extra: Any):
    return parse_item_plug_response(
        body,
        item_hash=ITEM_HASH,
        instance_id=INSTANCE,
        plug_definitions=extra.pop("plug_definitions", DEFINITIONS),
        **extra,
    )


def socket_state(plug: int | None, **extra: Any) -> dict[str, Any]:
    value: dict[str, Any] = {"isEnabled": True, "isVisible": True, **extra}
    if plug is not None:
        value["plugHash"] = plug
    return value


def full_response() -> dict[str, Any]:
    return response(
        sockets=[socket_state(None), socket_state(11), socket_state(13, isEnabled=False)],
        reusable={
            "1": [option(11), option(12, can_insert=False, insertFailIndexes=[0, 7])],
            "2": [option(13, enabled=False, enableFailIndexes=[1]), option(14)],
        },
        objectives={"12": [{"objectiveHash": 1}], "14": []},
    )


# --- Parser: documented shapes ---


def test_the_documented_shape_is_parsed_with_flags_names_and_provenance() -> None:
    evidence = parse(full_response())

    assert evidence.status == "complete" and evidence.identity == "verified"
    assert evidence.components == {
        "sockets": "returned",
        "reusablePlugs": "returned",
        "plugObjectives": "returned",
    }
    assert evidence.source == "GetItem:ItemSockets+ItemReusablePlugs"
    assert evidence.item_hash == ITEM_HASH
    one = evidence.socket(1)
    assert one is not None and one.reusable_listing == "listed" and one.listed == 2
    assert one.equipped_plug_hash == 11 and one.equipped_enabled is True
    alpha, beta = one.options
    # Options with a restriction sort ahead of plain ones.
    assert [v.plug_item_hash for v in one.options] == [12, 11]
    assert beta.name == "Alpha" and alpha.name == "Beta"
    assert alpha.can_insert is False and alpha.insert_fail_indexes == [0, 7]
    assert alpha.has_objectives is True and beta.has_objectives is False


def test_failure_indexes_resolve_against_their_own_rule_arrays() -> None:
    evidence = parse(full_response())
    blocked = next(v for v in evidence.socket(1).options if v.plug_item_hash == 12)  # type: ignore[union-attr]
    disabled = next(v for v in evidence.socket(2).options if v.plug_item_hash == 13)  # type: ignore[union-attr]

    assert blocked.insert_fail_reasons == ["Needs a quest.", None]  # index 7 does not exist
    assert blocked.enable_fail_reasons == []
    # enableFailIndexes [1] reads enabledRules[1], and is not an insertion restriction.
    assert disabled.enable_fail_reasons == ["Disabled rule two."]
    assert disabled.can_insert is True and disabled.enabled is False
    assert disabled.insert_fail_indexes == [] and disabled.insert_fail_reasons == []


def test_the_equipped_plugs_enable_failure_is_resolved_on_the_socket_state() -> None:
    body = response(
        sockets=[socket_state(None), socket_state(12), socket_state(13, enableFailIndexes=[0, 4])],
        reusable={"2": [option(13)]},
    )
    socket = parse(body).socket(2)

    assert socket is not None
    assert socket.equipped_enable_fail_indexes == [0, 4]
    assert socket.equipped_enable_fail_reasons == ["Disabled rule one.", None]


def test_without_definitions_indexes_stay_unresolved_and_names_are_absent() -> None:
    evidence = parse(full_response(), plug_definitions={})
    blocked = next(v for v in evidence.socket(1).options if v.plug_item_hash == 12)  # type: ignore[union-attr]

    assert blocked.insert_fail_reasons == [None, None] and blocked.name is None


# --- Parser: identity, missing and malformed data ---


def test_identity_is_verified_requested_only_or_mismatched() -> None:
    assert parse(full_response()).identity == "verified"
    assert parse(response(sockets=[], reusable={}, item=None)).identity == "requested_only"
    wrong_hash = parse(response(sockets=[socket_state(11)], reusable={}, item={"itemHash": 9}))
    wrong_instance = parse(
        response(sockets=[], reusable={}, item={"itemHash": ITEM_HASH, "itemInstanceId": "other"})
    )

    for bad in (wrong_hash, wrong_instance):
        assert (
            bad.status == "identity_mismatch" and bad.identity == "mismatch" and bad.sockets == []
        )


def test_missing_components_are_not_returned_or_partial_never_empty() -> None:
    nothing = parse(response())
    only_sockets = parse(response(sockets=[socket_state(11)]), focus_indexes={0})
    only_reusable = parse(response(reusable={"0": [option(11)]}))

    assert nothing.status == "not_returned" and nothing.sockets == []
    assert only_sockets.status == "partial"
    assert only_sockets.socket(0).reusable_listing == "not_returned"  # type: ignore[union-attr]
    assert only_reusable.status == "partial"
    assert only_reusable.socket(0).equipped_plug_hash is None  # type: ignore[union-attr]


def test_a_socket_without_a_310_entry_is_no_entry_not_no_options() -> None:
    evidence = parse(
        response(sockets=[socket_state(11), socket_state(12)], reusable={"1": [option(12)]}),
        focus_indexes={0},
    )

    assert evidence.socket(0).reusable_listing == "no_entry"  # type: ignore[union-attr]
    assert evidence.socket(0).options == []  # type: ignore[union-attr]
    assert evidence.socket(0).equipped_plug_hash == 11  # type: ignore[union-attr]
    assert evidence.sockets_without_listing == 1


@pytest.mark.parametrize("bad", [None, [], "text", 5])
def test_a_non_dict_response_is_not_returned(bad: Any) -> None:
    evidence = parse(bad)

    assert evidence.status == "not_returned" and evidence.sockets == []


def test_malformed_pieces_are_counted_and_make_the_evidence_partial() -> None:
    body = response(
        sockets=[socket_state(11), "not-a-dict", socket_state(12)],
        reusable={
            "0": [option(11), "junk", {"plugItemHash": "x"}, {"canInsert": True}],
            "-3": [option(11)],
            "abc": [option(11)],
            "2": "not-a-list",
            "1": [option(12, can_insert="yes", enabled=1)],
        },
    )
    evidence = parse(body)

    assert evidence.status == "partial"
    assert evidence.malformed_entries >= 6
    assert evidence.socket(0).malformed_entries == 3  # type: ignore[union-attr]
    flags = evidence.socket(1).options[0]  # type: ignore[union-attr]
    assert flags.can_insert is None and flags.enabled is None  # non-booleans are unknown


@pytest.mark.parametrize(
    "body",
    [
        {"sockets": {"data": {"sockets": "nope"}}, "reusablePlugs": {"data": {"plugs": {}}}},
        {"sockets": {"data": {"sockets": []}}, "reusablePlugs": {"data": {"plugs": []}}},
        {"sockets": "x", "reusablePlugs": {"data": "x"}},
    ],
)
def test_malformed_components_are_reported_and_never_trusted(body: dict[str, Any]) -> None:
    evidence = parse(body)

    assert "malformed" in evidence.components.values()
    assert evidence.status in {"partial", "not_returned"} and evidence.status != "complete"


# --- Parser: duplicates and truncation ---


def test_identical_duplicates_collapse_and_are_counted() -> None:
    evidence = parse(response(sockets=[socket_state(11)], reusable={"0": [option(11), option(11)]}))
    socket = evidence.socket(0)

    assert socket is not None and socket.listed == 1 and socket.duplicate_entries == 1
    assert socket.options[0].duplicates == 1 and not socket.options[0].conflicting_duplicates


def test_conflicting_duplicates_reset_the_disagreeing_flags_to_unknown() -> None:
    body = response(
        sockets=[socket_state(11)],
        reusable={"0": [option(11, can_insert=True), option(11, can_insert=False)]},
    )
    one = parse(body).socket(0).options[0]  # type: ignore[union-attr]

    assert one.can_insert is None and one.enabled is True and one.conflicting_duplicates is True


def test_options_and_sockets_are_capped_and_the_overflow_is_counted() -> None:
    many = [option(1000 + index) for index in range(MAX_OPTIONS_PER_SOCKET + 6)]
    wide = {str(index): [option(11)] for index in range(1, MAX_SOCKETS + 5)}
    evidence = parse(response(sockets=[], reusable={"0": many, **wide}))

    first = evidence.socket(0)
    assert first is not None and first.listed == MAX_OPTIONS_PER_SOCKET + 6
    assert len(first.options) == MAX_OPTIONS_PER_SOCKET and first.omitted_options == 6
    assert len(evidence.sockets) == MAX_SOCKETS and evidence.omitted_sockets == 5
    assert evidence.status == "partial"


def test_restricted_options_survive_the_cap() -> None:
    many = [option(1000 + index) for index in range(MAX_OPTIONS_PER_SOCKET + 6)]
    many.append(option(12, can_insert=False, insertFailIndexes=[0]))
    first = parse(response(reusable={"0": many})).socket(0)

    assert first is not None and first.options[0].plug_item_hash == 12


# --- Serialization and privacy ---


def test_the_instance_handle_is_never_serialized() -> None:
    evidence = parse(full_response())

    assert "instance_id" not in evidence.model_dump(mode="json")
    assert INSTANCE not in json.dumps(evidence.to_tool())
    assert INSTANCE not in evidence.model_dump_json()
    assert evidence.instance_id == INSTANCE  # the internal join handle remains


def test_serialization_omits_empty_fields() -> None:
    plain = parse(response(sockets=[socket_state(11)], reusable={"0": [option(11)]}))
    option_json = plain.socket(0).options[0].model_dump(mode="json")  # type: ignore[union-attr]

    assert option_json == {
        "plug_item_hash": 11,
        "can_insert": True,
        "enabled": True,
        "name": "Alpha",
    }


# --- Join: identity ---


def profile_item(
    *,
    item_hash: int = ITEM_HASH,
    instance: str | None = INSTANCE,
    location: str = "equipped",
    plugs: dict[int, int] | None = None,
) -> ItemSummary:
    return ItemSummary(
        item_hash=item_hash,
        instance_id=instance,
        name="Sample",
        item_type="Weapon",
        location=location,  # type: ignore[arg-type]
        is_equipped=location == "equipped",
        socketed_plug_details=[
            SocketedPlugSummary(item_hash=plug, name=str(plug), socket_index=index)
            for index, plug in (plugs or {}).items()
        ],
    )


def cfg(index: int, classification: str, **extra: Any) -> SocketConfigSummary:
    return SocketConfigSummary(index=index, classification=classification, **extra)


def test_the_join_requires_the_exact_hash_and_instance() -> None:
    evidence = parse(full_response())

    assert join_instance_evidence(evidence, profile_item(), None).join == "joined"
    assert join_instance_evidence(evidence, profile_item(item_hash=9), None).join == (
        "identity_mismatch"
    )
    assert join_instance_evidence(evidence, profile_item(instance="other"), None).join == (
        "identity_mismatch"
    )
    assert join_instance_evidence(evidence, profile_item(instance=None), None).join == (
        "identity_mismatch"
    )
    assert join_instance_evidence(evidence, None, None).join == "identity_mismatch"
    assert join_instance_evidence(parse(response()), profile_item(), None).join == "not_returned"
    mismatch = parse(response(item={"itemHash": 3}))
    assert join_instance_evidence(mismatch, profile_item(), None).options == []


def test_two_copies_of_one_definition_do_not_share_instance_evidence() -> None:
    titan_copy = parse(full_response())
    hunter_copy = profile_item(instance="hunter-copy")

    joined = join_instance_evidence(titan_copy, hunter_copy, None)

    assert joined.join == "identity_mismatch" and joined.options == []


# --- Join: separate evidence states ---


def states(join: Any, socket: int, plug: int) -> list[str]:
    option_ = next(v for v in join.options if v.socket_index == socket and v.plug_item_hash == plug)
    return option_.state_names()


def test_instance_listed_is_not_manifest_possible_or_actionable() -> None:
    evidence = parse(full_response())
    config = [cfg(1, "randomized_roll_pool", options=22, set_hash=700, category="WEAPON PERKS")]

    joined = join_instance_evidence(evidence, profile_item(plugs={1: 11}), config)

    assert states(joined, 1, 11) == ["equipped_now", "instance_listed"]
    assert states(joined, 1, 12) == ["instance_listed", "insertion_blocked_reported"]
    row = next(v for v in joined.sockets if v.socket_index == 1)
    assert row.manifest_pool == {"role": "randomized_roll_pool", "options": 22}
    assert row.listed == 2  # two instance options, not the 22-roll pool
    assert joined.actionable == "unknown"
    assert all(v.actionable == "unknown" for v in joined.options)


def test_a_randomized_pool_never_makes_pool_members_selectable() -> None:
    pool = set(range(100, 122))
    evidence = parse(response(sockets=[socket_state(11)], reusable={"0": [option(11)]}))
    config = [cfg(0, "randomized_roll_pool", options=22, set_hash=700)]

    joined = join_instance_evidence(
        evidence, profile_item(plugs={0: 11}), config, manifest_pool_hashes={700: pool | {11}}
    )

    assert [v.plug_item_hash for v in joined.options] == [11]  # only what the instance listed
    assert states(joined, 0, 11) == ["manifest_possible", "equipped_now", "instance_listed"]
    assert joined.sockets[0].listed_outside_manifest_pool == 0


def test_instance_options_outside_the_manifest_pool_are_counted() -> None:
    evidence = parse(response(sockets=[socket_state(11)], reusable={"0": [option(11), option(14)]}))
    config = [cfg(0, "randomized_roll_pool", options=3, set_hash=700)]

    joined = join_instance_evidence(
        evidence, profile_item(plugs={0: 11}), config, manifest_pool_hashes={700: {11, 12, 13}}
    )

    assert joined.sockets[0].listed_outside_manifest_pool == 1
    assert "manifest_possible" not in states(joined, 0, 14)
    assert "manifest_possible" in states(joined, 0, 11)


def test_manifest_possible_needs_a_known_pool_membership() -> None:
    evidence = parse(response(sockets=[socket_state(11)], reusable={"0": [option(11)]}))
    config = [cfg(0, "reusable_plug_pool", options=5, set_hash=701)]

    joined = join_instance_evidence(evidence, profile_item(plugs={0: 11}), config)

    assert "manifest_possible" not in states(joined, 0, 11)  # membership not verified
    assert joined.sockets[0].manifest_pool == {"role": "reusable_plug_pool", "options": 5}
    assert joined.sockets[0].listed_outside_manifest_pool is None


def test_enabled_false_is_not_an_insertion_restriction() -> None:
    evidence = parse(full_response())
    joined = join_instance_evidence(evidence, profile_item(plugs={2: 13}), None)

    assert "insertion_blocked_reported" not in states(joined, 2, 13)
    assert states(joined, 2, 13) == ["equipped_now", "instance_listed"]
    listed = next(
        v
        for v in next(o for o in joined.options if o.plug_item_hash == 13).states
        if v.state == "instance_listed"
    )
    assert listed.can_insert is True and listed.enabled is False
    section = instance_configuration_section(joined, evidence)
    row = next(v for v in section["sockets"] if v["socket_index"] == 2)
    assert row["not_enabled_reported"] == ["Gamma"] and "insertion_blocked_reported" not in row


def test_insertion_failure_reasons_are_carried_with_unresolved_ones_flagged() -> None:
    evidence = parse(full_response())
    joined = join_instance_evidence(evidence, profile_item(), None)
    blocked = next(
        v
        for v in next(o for o in joined.options if o.plug_item_hash == 12).states
        if v.state == "insertion_blocked_reported"
    )
    section = instance_configuration_section(joined, evidence)
    row = next(v for v in section["sockets"] if v["socket_index"] == 1)

    assert blocked.reasons == ["Needs a quest.", None]
    assert row["insertion_blocked_reported"][0]["reasons_unresolved"] is True


def test_a_vaulted_instance_is_socketed_not_equipped() -> None:
    evidence = parse(full_response())
    vault = profile_item(location="vault", plugs={1: 11})

    joined = join_instance_evidence(evidence, vault, None)

    assert joined.scope == "not_equipped"
    assert states(joined, 1, 11) == ["socketed_not_equipped", "instance_listed"]
    assert all("equipped_now" not in v.state_names() for v in joined.options)
    assert instance_configuration_section(joined, evidence)["scope"] == "not_equipped"


def test_a_difference_from_the_profile_snapshot_is_flagged() -> None:
    evidence = parse(full_response())  # socket 1 holds plug 11
    joined = join_instance_evidence(evidence, profile_item(plugs={1: 14}), None)

    row = next(v for v in joined.sockets if v.socket_index == 1)
    assert row.equipped_differs_from_profile is True


# --- Join: account and character evidence ---


def returned(set_hash: int, scope: str, *plugs: int) -> PlugSetCollection:
    return PlugSetCollection(
        status="returned",
        sets=[
            PlugSetEvidence(
                plug_set_hash=set_hash,
                scope=scope,  # type: ignore[arg-type]
                character_id="x" if scope == "character" else None,
                clear_plug_hashes=list(plugs),
                listed=len(plugs),
            )
        ],
    )


def test_plug_set_evidence_joins_only_where_it_was_retained() -> None:
    evidence = parse(response(sockets=[socket_state(11)], reusable={"0": [option(11), option(14)]}))
    config = [
        cfg(0, "reusable_plug_pool", options=5, set_hash=701, plug_sources=["profile_plug_set"])
    ]
    item = profile_item(plugs={0: 11})

    retained = join_instance_evidence(
        evidence, item, config, profile_sets=returned(701, "profile", 11)
    )
    missing = join_instance_evidence(evidence, item, config, profile_sets=PlugSetCollection())

    assert states(retained, 0, 11) == ["equipped_now", "instance_listed", "account_listed"]
    assert states(retained, 0, 14) == ["instance_listed", "unknown"]  # set retained, plug not
    assert missing.sockets[0].account_listing == "not_retained"
    assert "account_listed" not in states(missing, 0, 11)
    assert "unknown" not in states(missing, 0, 11)  # no state is manufactured


def test_character_and_profile_scopes_stay_separate() -> None:
    evidence = parse(response(sockets=[socket_state(11)], reusable={"0": [option(11)]}))
    config = [
        cfg(
            0,
            "reusable_plug_pool",
            options=5,
            set_hash=701,
            plug_sources=["profile_plug_set", "character_plug_set"],
        )
    ]
    joined = join_instance_evidence(
        evidence,
        profile_item(plugs={0: 11}),
        config,
        profile_sets=returned(701, "profile", 11),
        character_sets=returned(701, "character"),  # returned, but 11 is not in it
    )

    assert states(joined, 0, 11) == [
        "equipped_now",
        "instance_listed",
        "account_listed",
        "unknown",
    ]


# --- Section ---


def test_the_section_is_bounded_identifier_free_and_always_actionable_unknown() -> None:
    reusable = {str(index): [option(11), option(12, can_insert=False)] for index in range(12)}
    evidence = parse(response(sockets=[socket_state(11)] * 12, reusable=reusable))
    joined = join_instance_evidence(evidence, profile_item(), None)

    section = instance_configuration_section(joined, evidence, max_rows=MAX_ROWS)

    assert section["actionable"] == "unknown" and "basis" in section
    assert len(section["sockets"]) <= MAX_ROWS
    assert section["omitted_sockets"] == 12 - len(section["sockets"])
    assert INSTANCE not in json.dumps(section)
    tiny = instance_configuration_section(joined, evidence, max_chars=900)
    assert tiny["omitted_sockets"] >= 1


def test_a_truncated_socket_reports_lower_bounds() -> None:
    many = [option(1000 + index) for index in range(MAX_OPTIONS_PER_SOCKET + 5)]
    evidence = parse(response(sockets=[socket_state(11)], reusable={"0": many}))
    section = instance_configuration_section(
        join_instance_evidence(evidence, profile_item(), None), evidence
    )

    assert section["sockets"][0]["counts_are_lower_bounds"] is True
    assert section["sockets"][0]["omitted_options"] == 5


def test_non_joined_results_carry_no_basis_or_sockets() -> None:
    evidence = parse(response())
    section = instance_configuration_section(
        join_instance_evidence(evidence, profile_item(), None), evidence
    )

    assert section == {
        "join": "not_returned",
        "evidence_status": "not_returned",
        "scope": "equipped",
        "actionable": "unknown",
        "sockets": [],
    }


# --- Integration boundary ---


def test_nothing_in_the_application_calls_the_new_layer_or_a_fetch() -> None:
    root = Path(instance_plug_evidence.__file__).parent
    allowed = {"instance_plug_evidence.py", "instance_configuration.py"}
    for path in root.rglob("*.py"):
        if path.name in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        assert "instance_plug_evidence" not in text and "instance_configuration" not in text
    for module in (instance_plug_evidence, instance_configuration):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert not re.search(r"\bhttpx\b|BungieClient|\.get_profile|\.get_item|requests\.", source)


def test_profile_models_carry_no_instance_plug_evidence() -> None:
    assert not any("instance" in name and "plug" in name for name in GuardianContext.model_fields)
    assert not any("reusable" in name for name in ItemSummary.model_fields)


def test_the_join_leaves_m3_m4_and_m5b_outputs_unchanged() -> None:
    context = normalized()
    titan = character_of(context, "Titan")
    before = (
        equipped_mechanics_section(titan),
        configuration_section(titan, context.profile_plug_sets),
        context.model_dump_json(),
    )
    item = titan.equipped_gear[0]
    evidence = parse_item_plug_response(
        response(sockets=[socket_state(11)], reusable={"0": [option(11)]}, item=None),
        item_hash=item.item_hash,
        instance_id=item.instance_id or "",
    )

    join_instance_evidence(evidence, item, item.socket_configuration or None)

    after = (
        equipped_mechanics_section(titan),
        configuration_section(titan, context.profile_plug_sets),
        context.model_dump_json(),
    )
    assert before == after


def test_existing_plug_set_entry_serialization_is_unchanged() -> None:
    assert PlugSetEntry(plug_item_hash=1).model_dump(mode="json") == {"plug_item_hash": 1}
    assert GuardianToolService  # the existing tool surface still imports without the new layer


# --- Real Manifest ---


def _get(table: str, entity_hash: int) -> dict[str, Any] | None:
    assert REAL is not None
    signed = entity_hash - 2**32 if entity_hash >= 2**31 else entity_hash
    with sqlite3.connect(str(REAL)) as connection:
        row = connection.execute(f'SELECT json FROM "{table}" WHERE id = ?', (signed,)).fetchone()
    return json.loads(row[0]) if row else None


def manifest_config(item_hash: int) -> tuple[dict[str, Any], list[SocketConfigSummary]]:
    """The M3 socket configuration for a definition, built as the normalizer builds it."""

    definition = _get("DestinyInventoryItemDefinition", item_hash) or {}
    wanted = referenced_definition_hashes(definition)
    socket_types = {h: _get("DestinySocketTypeDefinition", h) or {} for h in wanted["socket_types"]}
    categories = {h: _get("DestinySocketCategoryDefinition", h) or {} for h in wanted["categories"]}
    cosmetic = cosmetic_socket_indexes(definition, socket_types, categories)
    sets = {
        h: _get("DestinyPlugSetDefinition", h) or {} for h in plug_set_hashes(definition, cosmetic)
    }
    specs = build_specs(definition, sets, cosmetic)
    names = {
        h: _get("DestinyInventoryItemDefinition", h) or {}
        for h in name_hashes_needed(specs, categories)
    }
    described = describe_sockets(definition, specs, socket_types, categories, names)
    return definition, [SocketConfigSummary(**v) for v in summary_entries(described)]


def pool_hashes(set_hash: int) -> set[int]:
    plug_set = _get("DestinyPlugSetDefinition", set_hash) or {}
    return {v["plugItemHash"] for v in plug_set.get("reusablePlugItems", [])}


def real_instance(
    item_hash: int, reusable: dict[str, Any], *, location: str = "equipped"
) -> tuple[Any, ItemSummary, Any]:
    evidence = parse_item_plug_response(
        {
            "item": {"data": {"itemHash": item_hash, "itemInstanceId": "real-inst"}},
            "sockets": {"data": {"sockets": []}},
            "reusablePlugs": {"data": {"plugs": reusable}},
        },
        item_hash=item_hash,
        instance_id="real-inst",
    )
    item = ItemSummary(
        item_hash=item_hash,
        instance_id="real-inst",
        name="Real",
        item_type="Weapon",
        location=location,  # type: ignore[arg-type]
    )
    return evidence, item, manifest_config(item_hash)[1]


RANDOMIZED, CRAFTABLE, FIXED = 2429822977, 3055790362, 2907129557
VARIANT_A, VARIANT_B, EXOTIC_ARMOR = 2188764214, 3654674561, 241462142


def first_pool_socket(
    config: list[SocketConfigSummary], classification: str
) -> SocketConfigSummary:
    return next(v for v in config if v.classification == classification and v.set_hash)


@needs_manifest
def test_real_randomized_weapon_instance_options_are_a_subset_not_the_pool() -> None:
    config = manifest_config(RANDOMIZED)[1]
    column = first_pool_socket(config, "randomized_roll_pool")
    pool = pool_hashes(column.set_hash)  # type: ignore[arg-type]
    members = sorted(pool)
    outsider = 999_999_999
    evidence, item, _ = real_instance(
        RANDOMIZED,
        {str(column.index): [option(members[0]), option(members[1]), option(outsider)]},
    )

    joined = join_instance_evidence(
        evidence,
        item,
        config,
        manifest_pool_hashes={column.set_hash: pool},  # type: ignore[dict-item]
    )
    row = joined.sockets[0]

    assert row.manifest_pool is not None and row.manifest_pool["role"] == "randomized_roll_pool"
    assert row.manifest_pool["options"] == len(pool) > 3
    assert row.listed == 3 and row.listed_outside_manifest_pool == 1
    assert len(joined.options) == 3  # never one option per pool member
    assert "manifest_possible" in states(joined, column.index, members[0])
    assert "manifest_possible" not in states(joined, column.index, outsider)


@needs_manifest
def test_real_fixed_weapon_has_no_pool_to_choose_from() -> None:
    config = manifest_config(FIXED)[1]
    fixed = [v for v in config if v.classification == "fixed_by_manifest"]
    assert fixed
    evidence, item, _ = real_instance(FIXED, {})

    joined = join_instance_evidence(evidence, item, config)

    # Both components came back empty: complete for what was returned, with nothing to join.
    assert evidence.status == "complete" and joined.options == [] and evidence.sockets == []


@needs_manifest
def test_real_crafted_capable_weapon_can_list_several_options_on_one_socket() -> None:
    config = manifest_config(CRAFTABLE)[1]
    column = first_pool_socket(config, "randomized_roll_pool")
    members = sorted(pool_hashes(column.set_hash))[:5]  # type: ignore[arg-type]
    evidence, item, _ = real_instance(
        CRAFTABLE, {str(column.index): [option(value) for value in members]}
    )

    joined = join_instance_evidence(evidence, item, config)

    assert joined.sockets[0].listed == 5 and len(joined.options) == 5
    assert all(
        v.state_names() == ["instance_listed"] for v in joined.options
    )  # no equipped state without 305 data, no pool state without membership
    assert joined.actionable == "unknown"


@needs_manifest
def test_real_variant_definitions_share_a_name_but_never_share_a_join() -> None:
    first, config_a = manifest_config(VARIANT_A)
    second, config_b = manifest_config(VARIANT_B)

    assert first["displayProperties"]["name"] == second["displayProperties"]["name"]
    assert {v.index: v.classification for v in config_a} != {
        v.index: v.classification for v in config_b
    }
    evidence, item_a, _ = real_instance(VARIANT_A, {"0": [option(11)]})
    # Evidence for one definition never joins an item of the other, by exact hash.
    other = item_a.model_copy(update={"item_hash": VARIANT_B})
    assert join_instance_evidence(evidence, other, config_b).join == "identity_mismatch"
    assert join_instance_evidence(evidence, item_a, config_a).join == "joined"


@needs_manifest
def test_real_exotic_armor_mod_sockets_join_account_evidence_when_retained() -> None:
    from app.plug_set_evidence import socket_plug_set_refs

    definition, config = manifest_config(EXOTIC_ARMOR)
    ref = next(
        v
        for v in socket_plug_set_refs(definition)
        if "profile_plug_set" in v.sources and "inventory" in v.sources
    )
    members = sorted(pool_hashes(ref.plug_set_hash))
    evidence, item, _ = real_instance(
        EXOTIC_ARMOR,
        {str(ref.socket_index): [option(members[0]), option(members[1], can_insert=False)]},
    )
    item = item.model_copy(update={"item_type": "Armor", "plug_set_sockets": [ref]})
    profile = returned(ref.plug_set_hash, "profile", members[0])
    character = returned(ref.plug_set_hash, "character", members[0])

    joined = join_instance_evidence(
        evidence, item, config, profile_sets=profile, character_sets=character
    )

    assert definition["itemType"] == 2
    assert states(joined, ref.socket_index, members[0]) == [
        "instance_listed",
        "account_listed",
        "character_listed",
    ]
    assert states(joined, ref.socket_index, members[1]) == [
        "instance_listed",
        "insertion_blocked_reported",
        "unknown",
    ]
    section = instance_configuration_section(joined, evidence)
    assert section["actionable"] == "unknown" and "armor energy" in section["limitations"]
    # Without the retained set, nothing is claimed about the account.
    bare = join_instance_evidence(evidence, item, config, profile_sets=PlugSetCollection())
    assert bare.sockets[0].account_listing == "not_retained"
    assert "account_listed" not in states(bare, ref.socket_index, members[0])


@needs_manifest
def test_real_vaulted_weapon_instance_is_not_called_equipped() -> None:
    config = manifest_config(RANDOMIZED)[1]
    column = first_pool_socket(config, "randomized_roll_pool")
    member = sorted(pool_hashes(column.set_hash))[0]  # type: ignore[arg-type]
    evidence, item, _ = real_instance(
        RANDOMIZED, {str(column.index): [option(member)]}, location="vault"
    )
    item = item.model_copy(
        update={
            "socketed_plug_details": [
                SocketedPlugSummary(item_hash=member, name="x", socket_index=column.index)
            ]
        }
    )
    evidence = parse_item_plug_response(
        {
            "item": {"data": {"itemHash": RANDOMIZED, "itemInstanceId": "real-inst"}},
            "sockets": {
                "data": {"sockets": [socket_state(member) for _ in range(column.index + 1)]}
            },
            "reusablePlugs": {"data": {"plugs": {str(column.index): [option(member)]}}},
        },
        item_hash=RANDOMIZED,
        instance_id="real-inst",
    )

    joined = join_instance_evidence(evidence, item, config)

    assert joined.scope == "not_equipped"
    assert states(joined, column.index, member) == ["socketed_not_equipped", "instance_listed"]
