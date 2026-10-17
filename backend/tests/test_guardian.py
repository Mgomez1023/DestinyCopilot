import asyncio
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

import httpx

from app.bungie.client import BungieClient
from app.bungie.guardian import GuardianNormalizer, select_membership
from app.bungie.oauth import BungieOAuth
from app.config import Settings


def test_selects_cross_save_primary_membership() -> None:
    data = {
        "primaryMembershipId": "222",
        "destinyMemberships": [
            {"membershipId": "111", "membershipType": 2},
            {"membershipId": "222", "membershipType": 3},
        ],
    }
    assert select_membership(data)["membershipType"] == 3


def test_falls_back_to_first_membership() -> None:
    data = {"destinyMemberships": [{"membershipId": "111", "membershipType": 2}]}
    assert select_membership(data)["membershipId"] == "111"


def test_oauth_authorization_uses_https_callback() -> None:
    settings = Settings(
        bungie_api_key="api-key",
        bungie_client_id="client-id",
        bungie_client_secret="client-secret",
        bungie_redirect_uri="https://localhost:8000/api/auth/callback",
    )
    oauth = BungieOAuth(settings, cast(Any, None))
    query = parse_qs(urlparse(oauth.authorization_url("csrf-state")).query)

    assert query["redirect_uri"] == ["https://localhost:8000/api/auth/callback"]
    assert query["state"] == ["csrf-state"]


def test_profile_and_activity_history_use_official_read_endpoints() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"ErrorCode": 1, "Response": {}})

    async def exercise() -> None:
        settings = Settings(bungie_api_key="api-key")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = BungieClient(settings, http)
            await client.get_profile(3, "membership", "access-token")
            await client.get_activity_history(3, "membership", "character", "access-token", count=5)

    asyncio.run(exercise())

    assert requests[0].url.path == "/Platform/Destiny2/3/Profile/membership/"
    components = set(requests[0].url.params["components"].split(","))
    assert {"ProfileInventories", "CharacterProgressions", "ItemObjectives"} <= components
    assert {"Collectibles", "Records", "Craftables"} <= components
    assert requests[1].url.path == (
        "/Platform/Destiny2/3/Account/membership/Character/character/Stats/Activities/"
    )
    assert requests[1].url.params["count"] == "5"
    assert requests[1].url.params["page"] == "0"


class FakeResolver:
    def __init__(self, definitions: dict[str, dict[int, dict[str, Any]]]) -> None:
        self.definitions = definitions

    async def resolve_many(
        self, entity_type: str, entity_hashes: set[int]
    ) -> dict[int, dict[str, Any]]:
        table = self.definitions.get(entity_type, {})
        return {value: table[value] for value in entity_hashes if value in table}

    def release_table(self, entity_type: str) -> None:
        del entity_type


def _definition(name: str, **extra: Any) -> dict[str, Any]:
    return {
        "displayProperties": {"name": name, "description": f"{name} description"},
        **extra,
    }


def _objective_definition(progress_description: str = "", **extra: Any) -> dict[str, Any]:
    """Mirror the Manifest shape: an empty display name plus progressDescription."""

    return {
        "displayProperties": {"name": "", "description": "", "hasIcon": False},
        "progressDescription": progress_description,
        **extra,
    }


def _node(name: str, parent: int | None = None, **extra: Any) -> dict[str, Any]:
    return {
        "displayProperties": {"name": name, "description": ""},
        "parentNodeHashes": [parent] if parent else [],
        **extra,
    }


def test_normalizes_private_profile_components_without_raw_envelopes() -> None:
    definitions = {
        "DestinyClassDefinition": {10: _definition("Warlock")},
        "DestinyRaceDefinition": {11: _definition("Awoken")},
        "DestinyGenderDefinition": {12: _definition("Female")},
        "DestinyInventoryItemDefinition": {
            100: _definition(
                "Test Rifle",
                itemType=3,
                itemTypeDisplayName="Auto Rifle",
                inventory={"tierTypeName": "Legendary"},
            ),
            101: _definition("Pattern Two", itemType=30),
            200: _definition("Test Quest", itemType=15),
            300: _definition("Glimmer", itemType=1, inventory={"maxStackSize": 500000}),
            400: _definition("Solar", itemType=16),
            401: _definition("Test Perk", itemType=19),
        },
        "DestinyInventoryBucketDefinition": {
            21: _definition("Kinetic Weapons"),
            22: _definition("Subclass"),
        },
        "DestinyObjectiveDefinition": {
            # Real objective definitions usually leave displayProperties.name empty.
            500: _objective_definition("Defeat targets", completionValue=10),
        },
        "DestinyActivityDefinition": {
            600: _definition(
                "Test Strike",
                activityTypeHash=601,
                destinationHash=602,
                placeHash=603,
                activityLightLevel=2000,
                tier=4,
            )
        },
        "DestinyActivityTypeDefinition": {601: _definition("Strike")},
        "DestinyDestinationDefinition": {602: _definition("Cosmodrome")},
        "DestinyPlaceDefinition": {603: _definition("Earth")},
        "DestinyMilestoneDefinition": {700: _definition("Weekly Test")},
        "DestinyProgressionDefinition": {800: _definition("Season Rank")},
        "DestinyRecordDefinition": {
            900: _definition("Almost There"),
            1580882372: {
                "displayProperties": {
                    "name": "The Final Shape - Legendary",
                    "description": "Complete all campaign missions on Legendary difficulty.",
                },
                "scope": 0,
            },
            3854636015: {
                "displayProperties": {
                    "name": "At Fate's Edge",
                    "description": 'Complete the main quest "The Edge of Fate" on Kepler.',
                },
                "scope": 1,
            },
        },
    }
    profile = {
        "profile": {
            "data": {
                "dateLastPlayed": "2026-09-10T10:00:00Z",
                "minutesPlayedTotal": "120",
            }
        },
        "characters": {
            "data": {
                "c1": {
                    "characterId": "c1",
                    "classHash": 10,
                    "raceHash": 11,
                    "genderHash": 12,
                    "classType": 2,
                    "raceType": 1,
                    "genderType": 1,
                    "light": 2010,
                    "minutesPlayedTotal": "120",
                }
            }
        },
        "profileInventory": {
            "data": {
                "items": [
                    {
                        "itemHash": 100,
                        "itemInstanceId": "vault-item",
                        "quantity": 1,
                        "location": 2,
                        "bucketHash": 21,
                        "state": 1,
                    }
                ]
            }
        },
        "profileCurrencies": {"data": {"items": [{"itemHash": 300, "quantity": 1234}]}},
        "profileProgression": {"data": {}},
        "characterInventories": {
            "data": {
                "c1": {
                    "items": [
                        {
                            "itemHash": 200,
                            "itemInstanceId": "quest-item",
                            "quantity": 1,
                            "location": 1,
                            "state": 2,
                        }
                    ]
                }
            }
        },
        "characterEquipment": {
            "data": {
                "c1": {
                    "items": [
                        {
                            "itemHash": 400,
                            "itemInstanceId": "subclass-item",
                            "quantity": 1,
                            "bucketHash": 22,
                            "state": 8,
                        }
                    ]
                }
            }
        },
        "characterProgressions": {
            "data": {
                "c1": {
                    "progressions": {
                        "800": {
                            "progressionHash": 800,
                            "level": 12,
                            "currentProgress": 250,
                            "progressToNextLevel": 50,
                            "nextLevelAt": 300,
                        }
                    },
                    "factions": {},
                    "quests": [],
                    "milestones": {
                        "700": {
                            "startDate": "2026-09-08T17:00:00Z",
                            "endDate": "2026-09-15T17:00:00Z",
                            "activities": [{"activityHash": 600}],
                        }
                    },
                }
            }
        },
        "characterActivities": {
            "data": {
                "c1": {
                    "availableActivities": [
                        {
                            "activityHash": 600,
                            "isNew": True,
                            "canLead": True,
                            "canJoin": True,
                            "isVisible": True,
                            "isCompleted": False,
                            "difficultyTier": 3,
                            "recommendedLight": 1990,
                            "challenges": [
                                {
                                    "objective": {
                                        "objectiveHash": 500,
                                        "progress": 8,
                                        "completionValue": 10,
                                        "complete": False,
                                        "visible": True,
                                    }
                                }
                            ],
                        }
                    ]
                }
            }
        },
        "itemComponents": {
            "instances": {
                "data": {
                    "vault-item": {"primaryStat": {"value": 2000}},
                    "subclass-item": {
                        "isEquipped": True,
                        "primaryStat": {"value": -10},
                    },
                }
            },
            "objectives": {
                "data": {
                    "quest-item": {
                        "objectives": [
                            {
                                "objectiveHash": 500,
                                "progress": 8,
                                "completionValue": 10,
                                "complete": False,
                                "visible": True,
                            }
                        ]
                    }
                }
            },
            "sockets": {"data": {"subclass-item": {"sockets": [{"plugHash": 401}]}}},
            "stats": {"data": {}},
        },
        "profileCollectibles": {
            "data": {
                "collectibles": {
                    "1": {"state": 0},
                    "2": {"state": 1},
                    "3": {"state": 4},
                }
            }
        },
        "characterCollectibles": {"data": {}},
        "profileRecords": {
            "data": {
                "records": {
                    "900": {
                        "state": 4,
                        "objectives": [
                            {
                                "objectiveHash": 500,
                                "progress": 8,
                                "completionValue": 10,
                                "complete": False,
                                "visible": True,
                            }
                        ],
                    },
                    "1580882372": {
                        "state": 1,
                        "objectives": [
                            {
                                "objectiveHash": 500,
                                "progress": 10,
                                "completionValue": 10,
                                "complete": True,
                                "visible": True,
                            }
                        ],
                    },
                }
            }
        },
        "characterRecords": {
            "data": {
                "c1": {
                    "records": {
                        "3854636015": {
                            "state": 0,
                            "objectives": [
                                {
                                    "objectiveHash": 500,
                                    "progress": 10,
                                    "completionValue": 10,
                                    "complete": True,
                                    "visible": True,
                                }
                            ],
                        }
                    }
                }
            }
        },
        "characterCraftables": {
            "data": {
                "c1": {
                    "craftables": {
                        "100": {"visible": True, "failedRequirementIndexes": []},
                        "101": {"visible": True, "failedRequirementIndexes": [0]},
                    }
                }
            }
        },
    }
    history = {
        "activities": [
            {
                "period": "2026-09-10T10:00:00Z",
                "activityDetails": {
                    "directorActivityHash": 600,
                    "instanceId": "activity-instance",
                    "mode": 3,
                },
                "values": {
                    "completed": {"basic": {"value": 1}},
                    "activityDurationSeconds": {"basic": {"value": 900}},
                },
            }
        ]
    }
    membership_data = {"bungieNetUser": {"displayName": "Fallback"}}
    membership = {
        "membershipId": "123",
        "membershipType": 3,
        "bungieGlobalDisplayName": "Guardian",
        "bungieGlobalDisplayNameCode": 42,
    }

    context = asyncio.run(
        GuardianNormalizer(cast(Any, FakeResolver(definitions))).normalize(
            membership_data, membership, profile, {"c1": history}
        )
    )

    assert context.bungie_display_name == "Guardian#0042"
    assert context.inventory.vault_items == 1
    assert context.inventory.items[0].name == "Test Rifle"
    assert context.inventory.items[0].is_locked is True
    assert context.currencies[0].quantity == 1234
    assert context.characters[0].subclass is not None
    assert context.characters[0].subclass.name == "Solar"
    assert context.characters[0].subclass.power is None
    assert context.characters[0].subclass.socketed_plugs == ["Test Perk"]
    assert context.characters[0].subclass.instance_data_available is True
    assert context.characters[0].subclass.socket_data_available is True
    assert context.characters[0].subclass.socket_count == 1
    assert context.characters[0].subclass.empty_socket_count == 0
    assert context.characters[0].quests[0].objectives[0].progress_percent == 80.0
    assert context.characters[0].quests[0].objectives[0].name == "Defeat targets"
    assert context.characters[0].available_activities[0].name == "Test Strike"
    assert context.characters[0].available_activities[0].activity_type == "Strike"
    assert context.characters[0].available_activities[0].destination == "Cosmodrome"
    assert context.characters[0].available_activities[0].place == "Earth"
    assert context.characters[0].recent_activities[0].place == "Earth"
    assert context.characters[0].available_activities[0].difficulty == "Challenging"
    assert context.characters[0].available_activities[0].recommended_power == 1990
    assert context.characters[0].recent_activities[0].duration_seconds == 900
    assert context.characters[0].recent_activities[0].difficulty == "Hard"
    assert context.collectibles.model_dump() == {"total_visible": 2, "acquired": 1}
    assert context.records.near_completion[0].name == "Almost There: Defeat targets"
    records = {value.record_hash: value for value in context.records.records}
    assert records[1580882372].scope == "profile"
    assert records[1580882372].character_id is None
    assert records[1580882372].completed is True
    assert records[1580882372].redeemed is True
    assert records[3854636015].scope == "character"
    assert records[3854636015].character_id == "c1"
    assert records[3854636015].completed is True
    assert context.crafting.requirements_met == 1
    assert context.crafting.incomplete_pattern_names == ["Pattern Two"]
    serialized = context.model_dump()
    assert "itemComponents" not in serialized
    assert "Response" not in serialized


class CountingResolver(FakeResolver):
    def __init__(self, definitions: dict[str, dict[int, dict[str, Any]]]) -> None:
        super().__init__(definitions)
        self.calls: list[tuple[str, set[int]]] = []

    async def resolve_many(
        self, entity_type: str, entity_hashes: set[int]
    ) -> dict[int, dict[str, Any]]:
        self.calls.append((entity_type, set(entity_hashes)))
        return await super().resolve_many(entity_type, entity_hashes)


def _normalize_profile(
    definitions: dict[str, dict[int, dict[str, Any]]],
    profile_extra: dict[str, Any],
    histories: dict[str, dict[str, Any] | None] | None = None,
    resolver: FakeResolver | None = None,
) -> Any:
    profile = {
        "profile": {"data": {"minutesPlayedTotal": "0"}},
        "characters": {"data": {}},
        **profile_extra,
    }
    membership = {"membershipId": "123", "membershipType": 3, "displayName": "Guardian"}
    return asyncio.run(
        GuardianNormalizer(cast(Any, resolver or FakeResolver(definitions))).normalize(
            {}, membership, profile, histories
        )
    )


def _near_record_profile(record_hash: int, objective_hash: int) -> dict[str, Any]:
    return {
        "profileRecords": {
            "data": {
                "records": {
                    str(record_hash): {
                        "state": 4,
                        "intervalObjectives": [
                            {
                                "objectiveHash": objective_hash,
                                "progress": 246,
                                "completionValue": 250,
                                "complete": False,
                                "visible": True,
                            }
                        ],
                    }
                }
            }
        }
    }


def test_objective_label_prefers_display_name_then_progress_description() -> None:
    normalizer = GuardianNormalizer(cast(Any, FakeResolver({})))
    normalizer.definitions = {
        "DestinyObjectiveDefinition": {
            1: _definition("Named objective"),
            2: _objective_definition("Waves completed"),
            3: _objective_definition(""),
        }
    }

    def name(objective_hash: int, prefix: str = "") -> str:
        return normalizer._normalize_objective(
            {"objectiveHash": objective_hash}, prefix=prefix
        ).name

    assert name(1) == "Named objective"
    assert name(2) == "Waves completed"
    assert name(3) == "Unknown"
    assert name(404) == "Unknown"
    assert name(2, "Altars of Sorrow: Waves upon Waves") == (
        "Altars of Sorrow: Waves upon Waves: Waves completed"
    )
    assert name(3, "Altars of Sorrow: Nightmares") == "Altars of Sorrow: Nightmares"
    assert name(404, "Some Record") == "Some Record"


def test_near_complete_interval_record_keeps_manifest_category_context() -> None:
    # Shaped like the Manifest: an interval Triumph whose objective has no label, and
    # whose location is only expressed by its presentation-node ancestry.
    definitions = {
        "DestinyRecordDefinition": {
            3001: {
                "displayProperties": {
                    "name": "Altars of Sorrow: Nightmares",
                    "description": "Defeat Nightmares at the Altars of Sorrow.",
                },
                "parentNodeHashes": [4001],
                "intervalInfo": {
                    "intervalObjectives": [{"intervalObjectiveHash": 5001, "intervalScoreValue": 0}]
                },
            }
        },
        "DestinyObjectiveDefinition": {5001: _objective_definition("", completionValue=250)},
        "DestinyPresentationNodeDefinition": {
            4001: _node("Altars of Sorrow", 4002),
            4002: _node("Activities", 4003),
            4003: _node("The Moon", 4004),
            4004: _node("Triumphs", 4005),
            4005: _node("Triumphs"),
        },
    }

    context = _normalize_profile(definitions, _near_record_profile(3001, 5001))

    record = context.records.near_completion[0]
    assert record.name == "Altars of Sorrow: Nightmares"
    assert ": Unknown" not in record.name
    assert (record.progress, record.completion_value) == (246, 250)
    assert record.record_name == "Altars of Sorrow: Nightmares"
    assert record.record_description == "Defeat Nightmares at the Altars of Sorrow."
    assert record.context_path == ["Triumphs", "The Moon", "Activities", "Altars of Sorrow"]
    # Path context is never converted into an inferred destination.
    assert record.destination_name is None
    assert record.activity_name is None


def test_record_context_path_is_depth_limited_and_batched_per_level() -> None:
    nodes = {
        6000 + index: _node(f"Node {index}", 6000 + index + 1 if index < 10 else None)
        for index in range(1, 11)
    }
    definitions = {
        "DestinyRecordDefinition": {
            3002: {**_definition("Deep Record"), "parentNodeHashes": [6001]},
            3003: {**_definition("Sibling Record"), "parentNodeHashes": [6001]},
        },
        "DestinyObjectiveDefinition": {5002: _objective_definition("Progress")},
        "DestinyPresentationNodeDefinition": nodes,
    }
    profile = _near_record_profile(3002, 5002)
    profile["profileRecords"]["data"]["records"]["3003"] = profile["profileRecords"]["data"][
        "records"
    ]["3002"]
    resolver = CountingResolver(definitions)

    context = _normalize_profile(definitions, profile, resolver=resolver)

    for record in context.records.near_completion:
        assert record.context_path == [f"Node {index}" for index in range(6, 0, -1)]
    node_calls = [
        hashes
        for entity_type, hashes in resolver.calls
        if entity_type == "DestinyPresentationNodeDefinition"
    ]
    assert len(node_calls) == 6
    assert node_calls[0] == {6001}
    assert all(len(hashes) == 1 for hashes in node_calls)


def test_record_context_path_is_cycle_safe() -> None:
    definitions = {
        "DestinyRecordDefinition": {
            3004: {**_definition("Loop Record"), "parentNodeHashes": [7001]}
        },
        "DestinyObjectiveDefinition": {5004: _objective_definition("Progress")},
        "DestinyPresentationNodeDefinition": {
            7001: _node("Inner", 7002),
            7002: _node("Outer", 7001),
        },
    }

    context = _normalize_profile(definitions, _near_record_profile(3004, 5004))

    assert context.records.near_completion[0].context_path == ["Outer", "Inner"]


def test_record_context_nodes_are_never_treated_as_titles() -> None:
    definitions = {
        "DestinyRecordDefinition": {
            3005: {**_definition("Category Record"), "parentNodeHashes": [8001]},
        },
        "DestinyObjectiveDefinition": {5005: _objective_definition("Progress")},
        "DestinyPresentationNodeDefinition": {
            # Worst case: a category node that looks Title-shaped.
            8001: _node(
                "Category",
                completionRecordHash=3005,
                children={"records": [{"recordHash": 3005}]},
            ),
        },
    }

    context = _normalize_profile(definitions, _near_record_profile(3005, 5005))

    assert context.records.near_completion[0].context_path == ["Category"]
    assert context.records.titles == []
    assert context.records.title_data_available is False


def test_activity_place_resolves_without_overwriting_empty_destination() -> None:
    definitions = {
        "DestinyClassDefinition": {10: _definition("Titan")},
        "DestinyActivityDefinition": {
            610: _definition(
                "Haunted Arena",
                activityTypeHash=611,
                destinationHash=612,
                placeHash=613,
            )
        },
        "DestinyActivityTypeDefinition": {611: _definition("Seasonal Arena")},
        "DestinyDestinationDefinition": {
            612: {"displayProperties": {"name": "", "description": ""}}
        },
        "DestinyPlaceDefinition": {613: _definition("The Moon")},
    }
    profile_extra = {
        "characters": {"data": {"c1": {"classHash": 10, "light": 0, "minutesPlayedTotal": "0"}}},
        "characterActivities": {
            "data": {"c1": {"availableActivities": [{"activityHash": 610, "isVisible": True}]}}
        },
    }
    history = {"activities": [{"activityDetails": {"directorActivityHash": 610}, "values": {}}]}

    context = _normalize_profile(definitions, profile_extra, {"c1": history})

    available = context.characters[0].available_activities[0]
    recent = context.characters[0].recent_activities[0]
    assert available.place == "The Moon"
    assert recent.place == "The Moon"
    # An unnamed destination definition is represented as unknown, never as the
    # "Unknown" placeholder string and never filled in from place.
    assert available.destination is None
    assert recent.destination is None


def test_items_keep_manifest_class_restriction_and_specific_plug_type() -> None:
    definitions = {
        "DestinyClassDefinition": {10: _definition("Titan")},
        "DestinyInventoryItemDefinition": {
            100: _definition(
                "Synthoceps",
                itemType=2,
                itemTypeDisplayName="Gauntlets",
                classType=0,
                inventory={"tierTypeName": "Exotic"},
            ),
            101: _definition(
                "Celestial Nighthawk",
                itemType=2,
                itemTypeDisplayName="Helmet",
                classType=1,
                inventory={"tierTypeName": "Exotic"},
            ),
            102: _definition(
                "Test Hand Cannon", itemType=3, itemTypeDisplayName="Hand Cannon", classType=3
            ),
            103: _definition("Legacy Helm", itemType=2, itemTypeDisplayName="Helmet"),
            200: _definition(
                "Rampage",
                itemType=19,
                itemTypeDisplayName="Trait",
                plug={"plugCategoryIdentifier": "frames"},
            ),
        },
    }
    profile_extra = {
        "characters": {"data": {"c1": {"classHash": 10, "light": 0, "minutesPlayedTotal": "0"}}},
        "profileInventory": {
            "data": {
                "items": [
                    {"itemHash": 101, "itemInstanceId": "i101", "location": 2},
                    {"itemHash": 102, "itemInstanceId": "i102", "location": 2},
                    {"itemHash": 103, "itemInstanceId": "i103", "location": 2},
                ]
            }
        },
        "characterEquipment": {
            "data": {"c1": {"items": [{"itemHash": 100, "itemInstanceId": "i100"}]}}
        },
        "itemComponents": {
            "sockets": {"data": {"i102": {"sockets": [{"plugHash": 200}]}}},
        },
    }

    context = _normalize_profile(definitions, profile_extra)

    items = {
        value.name: value
        for value in [*context.inventory.items, *context.characters[0].equipped_gear]
    }
    assert items["Synthoceps"].class_restriction == "Titan"
    assert items["Celestial Nighthawk"].class_restriction == "Hunter"
    assert items["Test Hand Cannon"].class_restriction == "Any"
    assert items["Legacy Helm"].class_restriction is None
    rampage = items["Test Hand Cannon"].socketed_plug_details[0]
    assert rampage.name == "Rampage"
    assert rampage.item_type == "Mod"
    assert rampage.plug_type == "Trait"
    assert rampage.category_identifier == "frames"


def test_missing_private_components_are_reported_instead_of_guessed() -> None:
    profile = {
        "profile": {"data": {"minutesPlayedTotal": "0"}},
        "characters": {"data": {}},
    }
    membership = {"membershipId": "123", "membershipType": 3, "displayName": "Guardian"}

    context = asyncio.run(
        GuardianNormalizer(cast(Any, FakeResolver({}))).normalize({}, membership, profile)
    )

    assert context.inventory.total_items == 0
    assert context.characters == []
    assert "ProfileInventories" in context.data_availability.unavailable_components
    assert "ItemObjectives" in context.data_availability.unavailable_components


def test_record_identity_retention_is_bounded_and_prioritizes_terminal_rules() -> None:
    completed = {str(value): {"state": 0} for value in range(1000, 1070)}
    completed["3854636015"] = {"state": 4}
    completed["999"] = {"state": 4}
    profile = {
        "profileRecords": {"data": {"records": completed}},
        "characterRecords": {"data": {}},
    }
    normalizer = GuardianNormalizer(cast(Any, FakeResolver({})))

    result = normalizer._normalize_records(profile)

    assert len(result.records) == 64
    assert result.records_truncated is True
    terminal = next(value for value in result.records if value.record_hash == 3854636015)
    assert terminal.completed is False
    assert all(value.record_hash != 999 for value in result.records)


def test_normalizes_bounded_title_progress_from_seal_presentation_nodes() -> None:
    definitions = {
        "DestinyPresentationNodeDefinition": {
            1: {
                "displayProperties": {"name": "Seals"},
                "children": {"presentationNodes": [{"presentationNodeHash": 2}]},
            },
            2: {
                "displayProperties": {"name": "Dredgen Seal"},
                "completionRecordHash": 100,
                "children": {
                    "presentationNodes": [],
                    "records": [{"recordHash": 101}, {"recordHash": 102}],
                },
            },
        },
        "DestinyRecordDefinition": {
            100: _definition(
                "Complete Dredgen",
                titleInfo={"titlesByGender": {"Male": "Dredgen", "Female": "Dredgen"}},
            ),
            101: _definition("First requirement"),
            102: _definition("Second requirement"),
        },
        "DestinyObjectiveDefinition": {500: _definition("Win matches")},
    }
    profile = {
        "profileRecords": {
            "data": {
                "recordSealsRootNodeHash": 1,
                "records": {
                    "100": {"state": 4},
                    "101": {"state": 0},
                    "102": {
                        "state": 4,
                        "objectives": [
                            {
                                "objectiveHash": 500,
                                "progress": 8,
                                "completionValue": 10,
                                "complete": False,
                                "visible": True,
                            }
                        ],
                    },
                },
            }
        },
        "characterRecords": {"data": {}},
    }
    membership = {"membershipId": "123", "membershipType": 3, "displayName": "Guardian"}

    context = asyncio.run(
        GuardianNormalizer(cast(Any, FakeResolver(definitions))).normalize({}, membership, profile)
    )

    assert context.records.title_data_available is True
    assert len(context.records.titles) == 1
    title = context.records.titles[0]
    assert title.name == "Dredgen"
    assert title.completed is False
    assert title.completed_records == 1
    assert title.total_records == 2
    assert title.remaining_records == 1
    assert title.remaining[0].name == "Second requirement"
    assert title.remaining[0].objectives[0].progress_percent == 80
    assert title.data_complete is True
