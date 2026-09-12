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
            await client.get_activity_history(
                3, "membership", "character", "access-token", count=5
            )

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
            500: _definition("Defeat targets", completionValue=10),
        },
        "DestinyActivityDefinition": {
            600: _definition(
                "Test Strike",
                activityTypeHash=601,
                destinationHash=602,
                activityLightLevel=2000,
                tier=4,
            )
        },
        "DestinyActivityTypeDefinition": {601: _definition("Strike")},
        "DestinyDestinationDefinition": {602: _definition("Cosmodrome")},
        "DestinyMilestoneDefinition": {700: _definition("Weekly Test")},
        "DestinyProgressionDefinition": {800: _definition("Season Rank")},
        "DestinyRecordDefinition": {900: _definition("Almost There")},
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
            "sockets": {
                "data": {"subclass-item": {"sockets": [{"plugHash": 401}]}}
            },
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
                    }
                }
            }
        },
        "characterRecords": {"data": {}},
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
    assert context.characters[0].quests[0].objectives[0].progress_percent == 80.0
    assert context.characters[0].available_activities[0].name == "Test Strike"
    assert context.characters[0].available_activities[0].activity_type == "Strike"
    assert context.characters[0].available_activities[0].destination == "Cosmodrome"
    assert context.characters[0].available_activities[0].difficulty == "Challenging"
    assert context.characters[0].available_activities[0].recommended_power == 1990
    assert context.characters[0].recent_activities[0].duration_seconds == 900
    assert context.characters[0].recent_activities[0].difficulty == "Hard"
    assert context.collectibles.model_dump() == {"total_visible": 2, "acquired": 1}
    assert context.records.near_completion[0].name == "Almost There: Defeat targets"
    assert context.crafting.requirements_met == 1
    assert context.crafting.incomplete_pattern_names == ["Pattern Two"]
    serialized = context.model_dump()
    assert "itemComponents" not in serialized
    assert "Response" not in serialized


def test_missing_private_components_are_reported_instead_of_guessed() -> None:
    profile = {
        "profile": {"data": {"minutesPlayedTotal": "0"}},
        "characters": {"data": {}},
    }
    membership = {"membershipId": "123", "membershipType": 3, "displayName": "Guardian"}

    context = asyncio.run(
        GuardianNormalizer(cast(Any, FakeResolver({}))).normalize(
            {}, membership, profile
        )
    )

    assert context.inventory.total_items == 0
    assert context.characters == []
    assert "ProfileInventories" in context.data_availability.unavailable_components
    assert "ItemObjectives" in context.data_availability.unavailable_components
