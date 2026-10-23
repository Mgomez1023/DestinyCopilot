"""search_inventory matches socketed perk names and says which fields a search could see."""

from app.guardian_tools import GuardianToolService, InventorySearchRequest
from app.models import SocketedPlugSummary
from tests.test_guardian_tools import guardian_context, item


def plug(name: str, plug_type: str = "Weapon Perk") -> SocketedPlugSummary:
    return SocketedPlugSummary(item_hash=abs(hash(name)) % 2**32, name=name, plug_type=plug_type)


def tools_with(*extra) -> GuardianToolService:  # noqa: ANN002
    context = guardian_context()
    context.inventory.items.extend(extra)
    return GuardianToolService(context)


def vault_weapon(name: str, *perks: str, character_id: str | None = None, location="vault"):  # noqa: ANN001, ANN201
    return item(
        name,
        "Weapon",
        subtype="Auto Rifle",
        location=location,
        character_id=character_id,
        equipped=False,
        plugs=[plug(value) for value in perks],
    )


def names(result: dict) -> list[str]:
    return [value["name"] for value in result["items"]]


def test_a_perk_name_query_finds_items_by_socketed_perk() -> None:
    tools = tools_with(
        vault_weapon("Quillstorm", "Zephyr Loop", "Brimming Hush"),
        vault_weapon("Dullwater", "Plain Barrel"),
    )
    result = tools.search_inventory(InventorySearchRequest(query="zephyr loop"))
    assert names(result) == ["Quillstorm"]
    assert result["items"][0]["matched_on"] == ["perk: Zephyr Loop"]
    assert result["total_matching"] == 1


def test_name_and_subtype_queries_still_work_and_report_what_matched() -> None:
    tools = tools_with(vault_weapon("Quillstorm", "Zephyr Loop"))
    by_name = tools.search_inventory(InventorySearchRequest(query="quillstorm"))
    assert by_name["items"][0]["matched_on"] == ["name"]
    by_subtype = tools.search_inventory(InventorySearchRequest(query="auto rifle"))
    assert "Quillstorm" in names(by_subtype)
    assert (
        "subtype" in next(v for v in by_subtype["items"] if v["name"] == "Quillstorm")["matched_on"]
    )


def test_a_perk_on_an_item_held_by_another_character_is_found_unscoped() -> None:
    held = vault_weapon("Elsewhere Gun", "Zephyr Loop", character_id="char-2", location="character")
    result = tools_with(held).search_inventory(InventorySearchRequest(query="zephyr loop"))
    assert names(result) == ["Elsewhere Gun"]
    assert result["items"][0]["character_id"] == "char-2"
    assert result["items"][0]["location"] == "character"


def test_a_character_scoped_search_reports_matches_held_by_other_characters() -> None:
    held = vault_weapon("Elsewhere Gun", "Zephyr Loop", character_id="char-2", location="character")
    result = tools_with(held).search_inventory(
        InventorySearchRequest(query="zephyr loop", character_class="Titan")
    )
    assert names(result) == []
    assert result["matches_on_other_characters"] == 1


def test_an_equipped_perk_is_found_with_a_perk_filter_on_other_fields() -> None:
    equipped = item(
        "Held Rifle",
        "Weapon",
        subtype="Auto Rifle",
        plugs=[plug("Zephyr Loop")],
    )
    context = guardian_context()
    context.characters[0].equipped_gear.append(equipped)
    result = GuardianToolService(context).search_inventory(
        InventorySearchRequest(query="zephyr loop", equipped_only=True)
    )
    assert names(result) == ["Held Rifle"]


def test_a_zero_result_states_the_search_scope_and_missing_socket_data() -> None:
    tools = tools_with(
        vault_weapon("Quillstorm", "Zephyr Loop"),
        vault_weapon("Bare Gun"),
    )
    result = tools.search_inventory(InventorySearchRequest(query="nonexistent perk"))
    assert result["items"] == []
    assert result["searched_fields"] == ["name", "subtype", "socketed perk and plug names"]
    coverage = result["socket_data_coverage"]
    assert coverage["without_socket_data"] >= 1
    assert coverage["with_socket_data"] >= 1
    assert coverage["items_searched"] == (
        coverage["with_socket_data"] + coverage["without_socket_data"]
    )
    assert "could not be searched" in result["search_note"]
    assert "does not show" in result["search_note"]


def test_a_zero_result_has_no_caveat_when_every_item_had_socket_data() -> None:
    context = guardian_context()
    for gear in context.characters[0].equipped_gear:
        gear.socketed_plugs = gear.socketed_plugs or ["Some Plug"]
    context.inventory.items.clear()
    result = GuardianToolService(context).search_inventory(
        InventorySearchRequest(query="nonexistent perk")
    )
    assert "search_note" not in result


def test_a_search_without_a_query_does_not_add_search_metadata() -> None:
    result = tools_with(vault_weapon("Quillstorm")).search_inventory(InventorySearchRequest())
    assert "searched_fields" not in result
    assert "matched_on" not in result["items"][0]
