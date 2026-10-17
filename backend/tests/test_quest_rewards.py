"""Quest / bounty type labels and Manifest-declared reward names through Guardian
normalization, the refresh cache, the quest tool and planning candidates."""

from typing import Any

import pytest
from test_guardian import CountingResolver, _definition, _normalize_profile
from test_guardian_refresh import make_service
from test_guardian_tools import guardian_context

from app.guardian_tools import GuardianToolService
from app.models import QuestSummary
from app.session_planning import build_session_planning_context

ARC_SCOURGE = 1405737718
LEGENDARY_PAYOUT = 2738452221
RETURNERS_GIFTS = 4134078229


def _arc_scourge() -> dict[str, Any]:
    """Mirrors the real Manifest bounty: specific type label, hash-0 reward placeholders."""

    return _definition(
        "Arc Scourge",
        itemType=26,
        itemTypeDisplayName="Duality Order",
        value={
            "itemValue": [
                {"itemHash": LEGENDARY_PAYOUT, "quantity": 1},
                {"itemHash": RETURNERS_GIFTS, "quantity": 1},
                {"itemHash": 0, "quantity": 0},
                {"itemHash": 0, "quantity": 0},
            ]
        },
    )


def _bounty_profile(*item_hashes: int) -> dict[str, Any]:
    return {
        "characters": {"data": {"c1": {"classHash": 10, "light": 0, "minutesPlayedTotal": "0"}}},
        "characterInventories": {
            "data": {
                "c1": {
                    "items": [
                        {"itemHash": value, "itemInstanceId": f"i{value}", "bucketHash": 1}
                        for value in item_hashes
                    ]
                }
            }
        },
    }


def test_arc_scourge_bounty_keeps_type_label_and_resolves_rewards_in_one_batch() -> None:
    definitions = {
        "DestinyClassDefinition": {10: _definition("Titan")},
        "DestinyInventoryItemDefinition": {
            ARC_SCOURGE: _arc_scourge(),
            LEGENDARY_PAYOUT: _definition("Legendary Payout"),
            RETURNERS_GIFTS: _definition("Returners Gifts Unlocked"),
        },
    }
    resolver = CountingResolver(definitions)

    context = _normalize_profile(definitions, _bounty_profile(ARC_SCOURGE), resolver=resolver)

    quest = context.characters[0].quests[0]
    assert quest.name == "Arc Scourge"
    assert quest.item_type_name == "Duality Order"
    assert quest.reward_names == ["Legendary Payout", "Returners Gifts Unlocked"]
    assert quest.reward_total == 2
    assert quest.rewards_unresolved == 0
    item_calls = [
        hashes
        for entity_type, hashes in resolver.calls
        if entity_type == "DestinyInventoryItemDefinition"
    ]
    # One batch for referenced items, one batch for every not-yet-resolved reward.
    assert item_calls == [{ARC_SCOURGE}, {LEGENDARY_PAYOUT, RETURNERS_GIFTS}]


def test_reward_names_are_capped_and_missing_definitions_are_counted_not_invented() -> None:
    rewards = [601, 602, 603, 604, 605, 606, 607]
    definitions = {
        "DestinyClassDefinition": {10: _definition("Titan")},
        "DestinyInventoryItemDefinition": {
            600: _definition(
                "Generous Bounty",
                itemType=26,
                itemTypeDisplayName="Weekly Bounty",
                value={"itemValue": [{"itemHash": value} for value in rewards]},
            ),
            **{value: _definition(f"Reward {value}") for value in rewards[:5]},
            # 606 has an empty display name; 607 has no definition at all.
            606: {"displayProperties": {"name": "", "description": ""}},
        },
    }

    context = _normalize_profile(definitions, _bounty_profile(600))

    quest = context.characters[0].quests[0]
    assert quest.reward_names == ["Reward 601", "Reward 602", "Reward 603", "Reward 604"]
    assert quest.reward_total == 7
    assert quest.rewards_unresolved == 2


def test_missing_type_label_and_rewards_stay_empty() -> None:
    definitions = {
        "DestinyClassDefinition": {10: _definition("Titan")},
        "DestinyInventoryItemDefinition": {700: _definition("Plain Bounty", itemType=26)},
    }

    quest = _normalize_profile(definitions, _bounty_profile(700)).characters[0].quests[0]

    assert quest.item_type_name is None
    assert quest.reward_names == []
    assert quest.reward_total == 0
    assert quest.rewards_unresolved == 0


def test_quest_status_reads_rewards_from_the_current_step_then_the_quest() -> None:
    definitions = {
        "DestinyClassDefinition": {10: _definition("Titan")},
        "DestinyInventoryItemDefinition": {
            800: _definition(
                "A Spark of Hope",
                itemType=15,
                itemTypeDisplayName="Exotic Quest",
                value={"itemValue": [{"itemHash": 803}]},
            ),
            801: _definition(
                "A Spark of Hope",
                itemType=12,
                itemTypeDisplayName="Quest Step",
                value={"itemValue": [{"itemHash": 802}, {"itemHash": 0}]},
            ),
            802: _definition("Riskrunner"),
            803: _definition("Quest Line Reward"),
            810: _definition(
                "Stepless Quest", itemType=15, value={"itemValue": [{"itemHash": 803}]}
            ),
        },
    }
    profile = {
        "characters": {"data": {"c1": {"classHash": 10, "light": 0, "minutesPlayedTotal": "0"}}},
        "characterProgressions": {
            "data": {
                "c1": {
                    "quests": [
                        {"questHash": 800, "stepHash": 801, "stepObjectives": []},
                        {"questHash": 810, "stepHash": 0, "stepObjectives": []},
                    ]
                }
            }
        },
    }

    quests = {
        value.quest_hash: value
        for value in _normalize_profile(definitions, profile).characters[0].quests
    }

    assert quests[800].item_type_name == "Exotic Quest"
    assert quests[800].reward_names == ["Riskrunner"]
    assert quests[810].item_type_name is None
    assert quests[810].reward_names == ["Quest Line Reward"]


def _context_with_arc_scourge() -> Any:
    context = guardian_context()
    context.characters[0].quests.append(
        QuestSummary(
            quest_hash=ARC_SCOURGE,
            step_hash=ARC_SCOURGE,
            name="Arc Scourge",
            step_name="Arc Scourge",
            description="Defeat combatants with Arc damage.",
            character_id="char-1",
            item_type_name="Duality Order",
            reward_names=["Legendary Payout", "Returners Gifts Unlocked"],
            reward_total=3,
            rewards_unresolved=1,
        )
    )
    return context


def test_active_quests_tool_exposes_type_label_and_rewards() -> None:
    result = GuardianToolService(_context_with_arc_scourge()).get_active_quests()

    quest = next(value for value in result["quests"] if value["name"] == "Arc Scourge")
    assert quest["item_type_name"] == "Duality Order"
    assert quest["reward_names"] == ["Legendary Payout", "Returners Gifts Unlocked"]
    assert quest["reward_total"] == 3
    assert quest["rewards_unresolved"] == 1
    assert "do not describe quantities" in result["limitations"][0]


def test_active_quests_tool_omits_reward_limitation_without_rewards() -> None:
    result = GuardianToolService(guardian_context()).get_active_quests()

    assert all(value["reward_names"] == [] for value in result["quests"])
    assert "limitations" not in result


def test_planning_quest_candidate_carries_type_label_and_reward_names() -> None:
    planning = build_session_planning_context(
        _context_with_arc_scourge(), "What should I do next?", "general", None
    )

    candidate = next(value for value in planning.candidates if value.name == "Arc Scourge")
    assert candidate.item_type_name == "Duality Order"
    assert candidate.reward_names == ["Legendary Payout", "Returners Gifts Unlocked"]
    assert planning.version == 6


@pytest.mark.asyncio
async def test_quest_type_label_and_rewards_survive_cache_round_trip() -> None:
    service, guardian, _ = make_service(context=_context_with_arc_scourge())

    await service.get_context("session-a", "token")
    warm = await service.get_context("session-a", "token")

    assert guardian.full_calls == 1
    assert warm.cache_hit
    quest = next(
        value for value in warm.context.characters[0].quests if value.name == "Arc Scourge"
    )
    assert quest.item_type_name == "Duality Order"
    assert quest.reward_names == ["Legendary Payout", "Returners Gifts Unlocked"]
    assert quest.reward_total == 3
    assert quest.rewards_unresolved == 1
