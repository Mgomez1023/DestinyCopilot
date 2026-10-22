"""The held-out scenarios stay unseen, runnable, and separate from the main bank."""

import json
import pathlib

from evals.heldout import HELDOUT, KNOWN_ISSUES
from evals.heldout_retired import HELDOUT_RETIRED
from evals.model import AXES
from evals.profiles import PROFILES, WEAPON_PERKS, WEAPONS, profile_facts
from evals.scenarios import BANK, SUBSETS
from tests.test_eval_bank import _tool_names

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _strings(value: object) -> set[str]:
    if isinstance(value, str):
        return {value.strip().lower()}
    if isinstance(value, dict):
        return set().union(*(_strings(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_strings(item) for item in value), set())
    return set()


def test_heldout_is_a_separate_subset_of_fifteen_runnable_scenarios() -> None:
    tools = _tool_names()
    assert SUBSETS["heldout"] == HELDOUT
    assert len(HELDOUT) == 15
    assert len({value.id for value in HELDOUT}) == len(HELDOUT)
    assert not {value.id for value in HELDOUT} & {value.id for value in BANK}
    assert all(value not in SUBSETS["full"] for value in HELDOUT)
    for value in HELDOUT:
        assert value.id.startswith("heldout."), value.id
        assert value.profile in PROFILES, value.id
        assert set(value.expect.tools_any) <= tools, value.id
        for axis, tag in value.tags.items():
            assert tag in AXES[axis], (value.id, axis, tag)


def test_heldout_covers_multi_turn_and_unavailable_data() -> None:
    assert sum(len(value.turns) >= 2 for value in HELDOUT) >= 5
    trials = next(value for value in HELDOUT if value.id.endswith("trials"))
    assert trials.expect.by_turn[0].answers_any
    assert {value.expect.style for value in HELDOUT} >= {"fact", "planning", "build"}
    assert len(KNOWN_ISSUES) == 5


def test_no_heldout_prompt_was_seen_in_the_bank_or_saved_drafts() -> None:
    seen = {turn.strip().lower() for value in BANK for turn in value.turns}
    for path in FIXTURES.glob("live_playtest_drafts*.json"):
        seen |= _strings(json.loads(path.read_text("utf-8")))
    for value in HELDOUT:
        for turn in value.turns:
            assert turn.strip().lower() not in seen, (value.id, turn)


def test_retired_scenarios_are_a_separate_regression_subset() -> None:
    retired = {value.id for value in HELDOUT_RETIRED}
    assert SUBSETS["heldout_retired"] == HELDOUT_RETIRED
    assert retired == {
        "heldout.frenzy_somewhere",
        "heldout.coil_still_worth_farming",
        "heldout.exotic_armor_stacking",
        "heldout.pvp_then_dungeon_build",
    }
    assert not retired & {value.id for value in HELDOUT}
    assert all(value not in SUBSETS["full"] for value in HELDOUT_RETIRED)
    assert all(value.note.startswith("retired held-out") for value in HELDOUT_RETIRED)
    tools = _tool_names()
    for value in HELDOUT_RETIRED:
        assert value.profile in PROFILES, value.id
        assert set(value.expect.tools_any) <= tools, value.id


def test_replacements_do_not_reuse_retired_prompts_or_subjects() -> None:
    retired_prompts = {turn.strip().lower() for value in HELDOUT_RETIRED for turn in value.turns}
    banned = ("frenzy", "coil", "inmost", "synthoceps", "duality", "shattered throne")
    for value in HELDOUT:
        for turn in value.turns:
            assert turn.strip().lower() not in retired_prompts, (value.id, turn)
    replacements = [
        value
        for value in HELDOUT
        if value.id.removeprefix("heldout.")
        in {
            "envious_assassin_elsewhere",
            "xur_stock_now",
            "two_exotic_weapons",
            "onslaught_build_then_waves",
        }
    ]
    assert len(replacements) == 4
    for value in replacements:
        assert not any(word in turn.lower() for turn in value.turns for word in banned), value.id


def test_replacement_expected_facts_match_the_profiles() -> None:
    titan = profile_facts(PROFILES["titan_veteran"]())
    assert titan.owned_items["Edge Transit"][1] == "equipped:Hunter"
    assert WEAPON_PERKS["Edge Transit"][0] == "Envious Assassin"
    envious = [name for name, perks in WEAPON_PERKS.items() if "Envious Assassin" in perks]
    assert envious == ["Edge Transit"]
    for name in ("Rose", "Calus Mini-Tool", "Apex Predator"):
        assert "Envious Assassin" not in WEAPON_PERKS[name]

    hunter = profile_facts(PROFILES["hunter_veteran"]())
    assert hunter.owned_items["Sunshot"][1] == "equipped:Hunter"
    assert hunter.owned_items["Star-Eater Scales"][1] == "equipped:Hunter"
    for name in ("Ace of Spades", "Gjallarhorn", "Le Monarque"):
        assert hunter.owned_items[name][1] == "vault", name
    for name in ("Ace of Spades", "Sunshot", "Gjallarhorn"):
        assert WEAPONS[name][2] == "Exotic"
    assert WEAPONS["Ace of Spades"][1] != WEAPONS["Sunshot"][1]
    assert WEAPONS["Gjallarhorn"][1] == "Power Weapons"

    warlock = PROFILES["warlock_veteran"]()
    activities = {
        item.name for character in warlock.characters for item in character.available_activities
    }
    assert any("Onslaught" in name for name in activities)
    equipped = profile_facts(warlock).owned_items
    for name in ("Sunbracers", "Sunshot", "Rose", "Falling Guillotine"):
        assert equipped[name][1] == "equipped:Warlock", name


def test_the_curated_guides_do_not_cover_the_no_guide_activity() -> None:
    guides = (pathlib.Path(__file__).parents[1] / "app" / "data" / "destiny_guides.json").read_text(
        "utf-8"
    )
    assert "onslaught" not in guides.lower()
