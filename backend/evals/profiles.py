"""Deterministic synthetic Guardian profiles.

Each profile is a complete GuardianContext built from real Destiny item, activity, and
quest names with class-correct armor. Hashes are stable (CRC32), so fixtures are identical
across runs. `PROFILE_FACTS` records the ground truth the evaluators check answers against.
"""

import zlib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from app.models import (
    AvailableActivitySummary,
    CharacterSummary,
    DataAvailability,
    GuardianContext,
    InventorySummary,
    ItemStatSummary,
    ItemSummary,
    MilestoneSummary,
    ObjectiveSummary,
    ProgressionSummary,
    QuestSummary,
    RecentActivitySummary,
    RecordObjectiveSummary,
    RecordProgressSummary,
    SocketedPlugSummary,
)

NOW = datetime(2026, 10, 17, 18, tzinfo=UTC)
CLASSES = ("Titan", "Hunter", "Warlock")

# Exotic armor by class, from the Manifest (classType 0/1/2).
EXOTIC_ARMOR = {
    "Synthoceps": ("Titan", "Gauntlets"),
    "Heart of Inmost Light": ("Titan", "Chest Armor"),
    "Cuirass of the Falling Star": ("Titan", "Chest Armor"),
    "Phoenix Cradle": ("Titan", "Leg Armor"),
    "Celestial Nighthawk": ("Hunter", "Helmet"),
    "Star-Eater Scales": ("Hunter", "Leg Armor"),
    "Gyrfalcon's Hauberk": ("Hunter", "Chest Armor"),
    "Wormhusk Crown": ("Hunter", "Helmet"),
    "Sunbracers": ("Warlock", "Gauntlets"),
    "Phoenix Protocol": ("Warlock", "Chest Armor"),
    "Contraverse Hold": ("Warlock", "Gauntlets"),
    "Necrotic Grip": ("Warlock", "Gauntlets"),
}
# Weapons: (subtype, bucket, tier, damage), as the Manifest defines them.
WEAPONS = {
    "Ace of Spades": ("Hand Cannon", "Kinetic Weapons", "Exotic", "Kinetic"),
    "Sunshot": ("Hand Cannon", "Energy Weapons", "Exotic", "Solar"),
    "Rose": ("Hand Cannon", "Kinetic Weapons", "Legendary", "Kinetic"),
    "Fatebringer": ("Hand Cannon", "Kinetic Weapons", "Legendary", "Kinetic"),
    "Funnelweb": ("Submachine Gun", "Energy Weapons", "Legendary", "Void"),
    "Riskrunner": ("Submachine Gun", "Energy Weapons", "Exotic", "Arc"),
    "Calus Mini-Tool": ("Submachine Gun", "Energy Weapons", "Legendary", "Solar"),
    "Gjallarhorn": ("Rocket Launcher", "Power Weapons", "Exotic", "Solar"),
    "Apex Predator": ("Rocket Launcher", "Power Weapons", "Legendary", "Solar"),
    "Edge Transit": ("Grenade Launcher", "Power Weapons", "Legendary", "Void"),
    "Witherhoard": ("Grenade Launcher", "Kinetic Weapons", "Exotic", "Kinetic"),
    "Heritage": ("Shotgun", "Kinetic Weapons", "Legendary", "Kinetic"),
    "Taipan-4fr": ("Linear Fusion Rifle", "Power Weapons", "Legendary", "Void"),
    "Falling Guillotine": ("Sword", "Power Weapons", "Legendary", "Void"),
    "Outbreak Perfected": ("Pulse Rifle", "Kinetic Weapons", "Exotic", "Kinetic"),
    "Le Monarque": ("Combat Bow", "Energy Weapons", "Exotic", "Void"),
    "Polaris Lance": ("Scout Rifle", "Energy Weapons", "Exotic", "Solar"),
}
WEAPON_PERKS = {
    "Rose": ["Rampage", "Opening Shot"],
    "Fatebringer": ["Explosive Payload", "Firefly"],
    "Funnelweb": ["Subsistence", "Frenzy"],
    "Calus Mini-Tool": ["Incandescent", "Unrelenting"],
    "Apex Predator": ["Reconstruction", "Bait and Switch"],
    "Edge Transit": ["Envious Assassin", "Bait and Switch"],
    "Heritage": ["Reconstruction", "Recombination"],
    "Taipan-4fr": ["Firing Line", "Bait and Switch"],
    "Falling Guillotine": ["Relentless Strikes", "Whirlwind Blade"],
}
# Exotic weapons' fixed sockets (name, plug category) from the Manifest definition; every
# copy has exactly these.
EXOTIC_SOCKETS = {
    "Ace of Spades": [
        ("Memento Mori", "intrinsics"),
        ("Corkscrew Rifling", "barrels"),
        ("High-Caliber Rounds", "magazines"),
        ("Firefly", "frames"),
    ],
    "Sunshot": [
        ("Sunburn", "intrinsics"),
        ("Chambered Compensator", "barrels"),
        ("Accurized Rounds", "magazines"),
        ("Sun Blast", "frames"),
    ],
    "Riskrunner": [
        ("Arc Conductor", "intrinsics"),
        ("Arrowhead Brake", "barrels"),
        ("Extended Mag", "magazines"),
        ("Superconductor", "frames"),
    ],
    "Gjallarhorn": [
        ("Wolfpack Rounds", "intrinsics"),
        ("Alloy Casing", "magazines_gl"),
        ("Pack Hunter", "frames"),
    ],
    "Witherhoard": [
        ("Primeval's Torment", "intrinsics"),
        ("Black Powder", "magazines_gl"),
        ("Break the Bank", "frames"),
    ],
    "Outbreak Perfected": [
        ("The Corruption Spreads", "intrinsics"),
        ("Extended Barrel", "barrels"),
        ("Accurized Rounds", "magazines"),
        ("Parasitism", "frames"),
    ],
    "Le Monarque": [
        ("Poison Arrows", "intrinsics"),
        ("Natural String", "bowstrings"),
        ("Compact Arrow Shaft", "arrows"),
        ("Snapshot Sights", "frames"),
    ],
    "Polaris Lance": [
        ("The Perfect Fifth", "intrinsics"),
        ("Smallbore", "barrels"),
        ("Tactical Mag", "magazines"),
        ("Zen Moment", "frames"),
    ],
}
SUBCLASSES = {
    "Titan": {"Solar": "Sunbreaker", "Arc": "Striker", "Void": "Sentinel"},
    "Hunter": {"Solar": "Gunslinger", "Arc": "Arcstrider", "Void": "Nightstalker"},
    "Warlock": {"Solar": "Dawnblade", "Arc": "Stormcaller", "Void": "Voidwalker"},
}
# Real Manifest plug names: one grenade and one aspect per class and element, and two
# fragments per element (fragments are shared by every class of that element).
SUBCLASS_GRENADES = {
    "Titan": {"Solar": "Thermite Grenade", "Arc": "Pulse Grenade", "Void": "Suppressor Grenade"},
    "Hunter": {"Solar": "Swarm Grenade", "Arc": "Flux Grenade", "Void": "Vortex Grenade"},
    "Warlock": {"Solar": "Healing Grenade", "Arc": "Storm Grenade", "Void": "Vortex Grenade"},
}
SUBCLASS_ASPECTS = {
    "Titan": {"Solar": "Roaring Flames", "Arc": "Knockout", "Void": "Bastion"},
    "Hunter": {"Solar": "Knock 'Em Down", "Arc": "Flow State", "Void": "Stylish Executioner"},
    "Warlock": {"Solar": "Heat Rises", "Arc": "Electrostatic Mind", "Void": "Chaos Accelerant"},
}
SUBCLASS_FRAGMENTS = {
    "Solar": ("Ember of Ashes", "Ember of Empyrean"),
    "Arc": ("Spark of Ions", "Spark of Magnitude"),
    "Void": ("Echo of Persistence", "Echo of Expulsion"),
}
ARMOR_BUCKETS = ("Helmet", "Gauntlets", "Chest Armor", "Leg Armor", "Class Armor")
LEGENDARY_ARMOR = {
    "Titan": [
        "Iron Forerunner Helm",
        "Iron Forerunner Gauntlets",
        "Iron Forerunner Plate",
        "Iron Forerunner Greaves",
        "Iron Forerunner Mark",
    ],
    "Hunter": [
        "Iron Forerunner Mask",
        "Iron Forerunner Grips",
        "Iron Forerunner Vest",
        "Iron Forerunner Strides",
        "Iron Forerunner Cloak",
    ],
    "Warlock": [
        "Iron Forerunner Hood",
        "Iron Forerunner Gloves",
        "Iron Forerunner Vestments",
        "Iron Forerunner Boots",
        "Iron Forerunner Bond",
    ],
}
# One real armor mod per slot, in that slot's Manifest plug category.
ARMOR_MODS = {
    "Helmet": ("Hands-On", "enhancements.v2_head"),
    "Gauntlets": ("Heavy Handed", "enhancements.v2_arms"),
    "Chest Armor": ("Charged Up", "enhancements.v2_chest"),
    "Leg Armor": ("Recuperation", "enhancements.v2_legs"),
    "Class Armor": ("Bomber", "enhancements.v2_class_item"),
}


def _hash(*parts: object) -> int:
    return zlib.crc32("|".join(str(value) for value in parts).encode()) or 1


def _plug(name: str, category: str, plug_type: str) -> SocketedPlugSummary:
    return SocketedPlugSummary(
        item_hash=_hash("plug", name),
        name=name,
        description=f"Manifest description for {name}.",
        item_type="Mod",
        plug_type=plug_type,
        category_identifier=category,
    )


SOCKET_LABELS = {
    "intrinsics": "Intrinsic",
    "barrels": "Barrel",
    "magazines": "Magazine",
    "magazines_gl": "Magazine",
    "frames": "Trait",
    "bowstrings": "Bowstring",
    "arrows": "Arrow",
}


def weapon(name: str, location: str, character_id: str | None, equipped: bool = False):
    subtype, bucket, tier, damage = WEAPONS[name]
    perks = [
        _plug(value, category, SOCKET_LABELS[category])
        for value, category in EXOTIC_SOCKETS.get(name, [])
    ] or [_plug(value, "frames", "Trait") for value in WEAPON_PERKS.get(name, [])]
    return ItemSummary(
        item_hash=_hash("weapon", name),
        instance_id=f"inst-{_hash(name, location, character_id)}",
        name=name,
        item_type="Weapon",
        item_subtype=subtype,
        tier=tier,
        class_restriction="Any",
        bucket_name="General" if location == "vault" else bucket,
        equipment_slot=bucket,
        damage_type=damage,
        power=2010,
        is_equipped=equipped,
        instance_data_available=True,
        socket_data_available=True,
        stat_data_available=True,
        socket_count=len(perks),
        empty_socket_count=0,
        socketed_plugs=[value.name for value in perks],
        socketed_plug_details=perks,
        location=location,  # type: ignore[arg-type]
        character_id=character_id,
    )


def armor(
    name: str,
    location: str,
    character_id: str | None,
    *,
    class_name: str | None = None,
    bucket: str | None = None,
    equipped: bool = False,
) -> ItemSummary:
    exotic = name in EXOTIC_ARMOR
    owner_class, exotic_bucket = EXOTIC_ARMOR.get(name, (class_name, bucket))
    mod = _plug(*ARMOR_MODS[exotic_bucket or "Helmet"], "Armor Mod")
    return ItemSummary(
        item_hash=_hash("armor", name, owner_class),
        instance_id=f"inst-{_hash(name, location, character_id)}",
        name=name,
        item_type="Armor",
        item_subtype=exotic_bucket,
        tier="Exotic" if exotic else "Legendary",
        class_restriction=owner_class,  # type: ignore[arg-type]
        bucket_name="General" if location == "vault" else exotic_bucket,
        equipment_slot=exotic_bucket,
        power=2010,
        is_equipped=equipped,
        instance_data_available=True,
        socket_data_available=True,
        stat_data_available=True,
        socket_count=1,
        empty_socket_count=0,
        stats=[ItemStatSummary(name="Health", value=20), ItemStatSummary(name="Grenade", value=25)],
        socketed_plugs=[mod.name],
        socketed_plug_details=[mod],
        location=location,  # type: ignore[arg-type]
        character_id=character_id,
    )


def subclass(class_name: str, element: str, character_id: str) -> ItemSummary:
    name = SUBCLASSES[class_name][element]
    lower = element.casefold()
    plugs = [
        _plug(
            SUBCLASS_GRENADES[class_name][element],
            f"shared.{lower}.grenades",
            f"{element} Grenade",
        ),
        _plug(
            SUBCLASS_ASPECTS[class_name][element],
            f"{class_name.casefold()}.{lower}.aspects",
            f"{element} Aspect",
        ),
        *(
            _plug(fragment, f"shared.{lower}.fragments", f"{element} Fragment")
            for fragment in SUBCLASS_FRAGMENTS[element]
        ),
    ]
    return ItemSummary(
        item_hash=_hash("subclass", name),
        instance_id=f"inst-{_hash(name, character_id)}",
        name=name,
        item_type="Subclass",
        tier="Common",
        class_restriction=class_name,  # type: ignore[arg-type]
        bucket_name="Subclass",
        equipment_slot="Subclass",
        damage_type=element,
        is_equipped=True,
        socket_data_available=True,
        socketed_plugs=[value.name for value in plugs],
        socketed_plug_details=plugs,
        location="equipped",
        character_id=character_id,
    )


def objective(name: str, progress: int, total: int, activity: str | None = None):
    return ObjectiveSummary(
        objective_hash=_hash("objective", name),
        name=name,
        progress=progress,
        completion_value=total,
        progress_percent=round(progress / total * 100, 1),
        complete=progress >= total,
        activity_name=activity,
    )


def quest(name: str, character_id: str, step: str, objectives: list[ObjectiveSummary], **extra):
    return QuestSummary(
        quest_hash=_hash("quest", name),
        step_hash=_hash("step", name, step),
        name=name,
        step_name=step,
        description=f"{name}: {step}.",
        character_id=character_id,
        objectives=objectives,
        **extra,
    )


def activity(name: str, character_id: str, activity_type: str, destination: str | None):
    return AvailableActivitySummary(
        activity_hash=_hash("activity", name),
        name=name,
        character_id=character_id,
        activity_type=activity_type,
        destination=destination,
        can_lead=True,
        can_join=True,
    )


def near_record(name: str, objective_label: str, progress: int, total: int, path: list[str]):
    return RecordObjectiveSummary(
        objective_hash=_hash("record-objective", name),
        name=objective_label,
        progress=progress,
        completion_value=total,
        progress_percent=round(progress / total * 100, 1),
        record_name=name,
        record_description=f"{objective_label}.",
        context_path=path,
    )


@dataclass
class CharacterSpec:
    class_name: str
    element: str = "Solar"
    weapons: tuple[str, str, str] = ("Rose", "Calus Mini-Tool", "Apex Predator")
    exotic_armor: str | None = None
    quests: list[QuestSummary] = field(default_factory=list)
    activities: list[str] = field(default_factory=list)
    recent: list[str] = field(default_factory=list)
    power: int = 2010
    last_played: datetime = NOW


ACTIVITY_TYPES = {
    "Duality: Standard": ("Dungeon", "Derelict Leviathan"),
    "Nightfall: The Lightblade": ("Nightfall", "Cosmodrome"),
    "Vanguard Ops": ("Vanguard Op", "The Tower"),
    "Crucible: Control": ("Control", None),
    "Gambit": ("Gambit", None),
    "Haunted Altars of Sorrow": ("Story", None),
    "The Coil": ("Seasonal Arena", "The Dreaming City"),
    "Onslaught: Legend": ("Onslaught", None),
}


def character(spec: CharacterSpec) -> CharacterSummary:
    character_id = f"{spec.class_name.casefold()}-1"
    kinetic, energy, power = spec.weapons
    gear = [
        subclass(spec.class_name, spec.element, character_id),
        *(weapon(value, "equipped", character_id, True) for value in (kinetic, energy, power)),
    ]
    armor_names = LEGENDARY_ARMOR[spec.class_name]
    for name, bucket in zip(armor_names, ARMOR_BUCKETS, strict=True):
        if spec.exotic_armor and EXOTIC_ARMOR[spec.exotic_armor][1] == bucket:
            gear.append(armor(spec.exotic_armor, "equipped", character_id, equipped=True))
        else:
            gear.append(
                armor(
                    name,
                    "equipped",
                    character_id,
                    class_name=spec.class_name,
                    bucket=bucket,
                    equipped=True,
                )
            )
    return CharacterSummary(
        character_id=character_id,
        class_name=spec.class_name,
        race_name="Human",
        gender_name="Female",
        power=spec.power,
        last_played=spec.last_played,
        minutes_played_total=60_000,
        subclass=gear[0],
        equipped_gear=gear,
        quests=spec.quests,
        milestones=[
            MilestoneSummary(
                milestone_hash=_hash("milestone", "Weekly Vanguard", character_id),
                name="Weekly Vanguard",
                character_id=character_id,
                activity_names=["Vanguard Ops"],
                objectives=[objective("Complete Vanguard Ops activities", 1, 3)],
            )
        ],
        progressions=[
            ProgressionSummary(
                progression_hash=_hash("progression", "Vanguard Rank", character_id),
                name="Vanguard Rank",
                scope="character",
                character_id=character_id,
                level=9,
                progress_to_next_level=300,
                next_level_at=1000,
            )
        ],
        available_activities=[
            activity(name, character_id, *ACTIVITY_TYPES[name]) for name in spec.activities
        ],
        recent_activities=[
            RecentActivitySummary(
                activity_hash=_hash("activity", name),
                name=name,
                character_id=character_id,
                period=spec.last_played - timedelta(hours=index + 1),
                activity_type=ACTIVITY_TYPES.get(name, (None, None))[0],
                completed=True,
                duration_seconds=1500,
            )
            for index, name in enumerate(spec.recent)
        ],
    )


def context(
    characters: list[CharacterSpec],
    vault: list[ItemSummary],
    near: list[RecordObjectiveSummary] | None = None,
    *,
    notes: list[str] | None = None,
    unavailable: dict[str, str] | None = None,
    last_played: datetime = NOW,
) -> GuardianContext:
    built = [character(value) for value in characters]
    return GuardianContext(
        bungie_display_name="EvalGuardian#0001",
        membership_id="eval-membership",
        membership_type=3,
        platform_name="Steam",
        last_played=last_played,
        total_minutes_played=sum(value.minutes_played_total for value in built),
        current_season_name="Season of the Eval",
        characters=built,
        inventory=InventorySummary(
            total_items=len(vault),
            vault_items=len(vault),
            unique_item_hashes=len({value.item_hash for value in vault}),
            items=vault,
            returned_items=len(vault),
        ),
        records=RecordProgressSummary(
            total_visible=900, completed=420, near_completion=list(near or [])
        ),
        data_availability=DataAvailability(
            fetched_at=NOW,
            available_components=["Profiles", "Characters", "CharacterInventories"],
            unavailable_components=dict(unavailable or {}),
            notes=list(notes or []),
        ),
    )


ALTARS_RECORD = near_record(
    "Altars of Sorrow: Waves upon Waves",
    "Waves of Altars of Sorrow completed",
    46,
    50,
    ["Triumphs", "The Moon", "Activities", "Altars of Sorrow"],
)
DUALITY_RECORD = near_record(
    "Duality: Gallery of Riches",
    "Duality chests opened",
    9,
    10,
    ["Triumphs", "Dungeons", "Duality"],
)
CAMPAIGN_STEP = quest(
    "The Edge of Fate",
    "titan-1",
    "Kepler Uplink",
    [objective('Complete the mission "Kepler Uplink"', 0, 1)],
    tracked=True,
    item_type_name="Campaign Quest",
)


def titan_veteran() -> GuardianContext:
    return context(
        [
            CharacterSpec(
                "Titan",
                weapons=("Rose", "Calus Mini-Tool", "Apex Predator"),
                exotic_armor="Synthoceps",
                quests=[CAMPAIGN_STEP],
                activities=[
                    "Duality: Standard",
                    "Nightfall: The Lightblade",
                    "Vanguard Ops",
                    "Crucible: Control",
                    "Haunted Altars of Sorrow",
                ],
                recent=["Vanguard Ops", "Vanguard Ops", "Crucible: Control"],
            ),
            CharacterSpec(
                "Hunter",
                element="Void",
                weapons=("Ace of Spades", "Funnelweb", "Edge Transit"),
                exotic_armor="Celestial Nighthawk",
                activities=["Nightfall: The Lightblade", "Gambit"],
                last_played=NOW - timedelta(days=40),
            ),
            CharacterSpec(
                "Warlock",
                element="Arc",
                weapons=("Rose", "Riskrunner", "Taipan-4fr"),
                exotic_armor="Sunbracers",
                activities=["Vanguard Ops"],
                last_played=NOW - timedelta(days=90),
            ),
        ],
        vault=[
            weapon("Fatebringer", "vault", None),
            weapon("Gjallarhorn", "vault", None),
            weapon("Heritage", "vault", None),
            armor("Heart of Inmost Light", "vault", None),
            armor("Star-Eater Scales", "vault", None),
            armor("Phoenix Protocol", "vault", None),
        ],
        near=[ALTARS_RECORD, DUALITY_RECORD],
    )


def hunter_veteran() -> GuardianContext:
    return context(
        [
            CharacterSpec(
                "Hunter",
                element="Solar",
                weapons=("Rose", "Sunshot", "Apex Predator"),
                exotic_armor="Star-Eater Scales",
                quests=[
                    quest(
                        "Rite of the Nine",
                        "hunter-1",
                        "Tribute",
                        [objective("Complete the dungeon", 0, 1, "Duality: Standard")],
                    )
                ],
                activities=["Duality: Standard", "Nightfall: The Lightblade", "Gambit"],
                recent=["Gambit", "Gambit", "Gambit"],
            ),
            CharacterSpec(
                "Titan",
                weapons=("Rose", "Funnelweb", "Apex Predator"),
                last_played=NOW - timedelta(days=60),
            ),
        ],
        vault=[
            armor("Celestial Nighthawk", "vault", None),
            armor("Wormhusk Crown", "vault", None),
            weapon("Ace of Spades", "vault", None),
            weapon("Gjallarhorn", "vault", None),
            weapon("Heritage", "vault", None),
            weapon("Le Monarque", "vault", None),
        ],
        near=[DUALITY_RECORD],
    )


def warlock_veteran() -> GuardianContext:
    return context(
        [
            CharacterSpec(
                "Warlock",
                element="Solar",
                weapons=("Rose", "Sunshot", "Falling Guillotine"),
                exotic_armor="Sunbracers",
                activities=["Nightfall: The Lightblade", "Onslaught: Legend", "Vanguard Ops"],
                recent=["Onslaught: Legend", "Onslaught: Legend"],
            ),
            CharacterSpec(
                "Titan",
                weapons=("Rose", "Calus Mini-Tool", "Edge Transit"),
                exotic_armor="Heart of Inmost Light",
                last_played=NOW - timedelta(days=12),
            ),
            CharacterSpec(
                "Hunter",
                element="Arc",
                weapons=("Rose", "Riskrunner", "Apex Predator"),
                last_played=NOW - timedelta(days=30),
            ),
        ],
        vault=[
            armor("Contraverse Hold", "vault", None),
            weapon("Outbreak Perfected", "vault", None),
            armor("Phoenix Protocol", "vault", None),
            weapon("Witherhoard", "vault", None),
        ],
        near=[ALTARS_RECORD],
    )


def returning_player() -> GuardianContext:
    long_ago = NOW - timedelta(days=500)
    return context(
        [
            CharacterSpec(
                "Warlock",
                element="Void",
                weapons=("Rose", "Funnelweb", "Edge Transit"),
                power=1810,
                quests=[
                    quest(
                        "The Final Shape",
                        "warlock-1",
                        "Transgression",
                        [objective('Complete the mission "Transgression"', 0, 1)],
                        item_type_name="Campaign Quest",
                    )
                ],
                activities=["Vanguard Ops", "Crucible: Control"],
                last_played=long_ago,
            )
        ],
        vault=[weapon("Riskrunner", "vault", None)],
        last_played=long_ago,
        notes=["Last played over a year ago."],
    )


def new_player() -> GuardianContext:
    return context(
        [
            CharacterSpec(
                "Hunter",
                element="Arc",
                weapons=("Rose", "Riskrunner", "Apex Predator"),
                power=1650,
                activities=["Vanguard Ops", "Crucible: Control"],
                recent=["Vanguard Ops"],
            )
        ],
        vault=[],
    )


def large_vault() -> GuardianContext:
    vault = [
        *(weapon(name, "vault", None) for name in WEAPONS),
        *(armor(name, "vault", None) for name in EXOTIC_ARMOR),
    ]
    return context(
        [
            CharacterSpec(
                "Titan",
                element="Arc",
                weapons=("Heritage", "Riskrunner", "Apex Predator"),
                exotic_armor="Cuirass of the Falling Star",
                activities=["Duality: Standard", "Nightfall: The Lightblade", "Vanguard Ops"],
            ),
            CharacterSpec(
                "Hunter",
                weapons=("Ace of Spades", "Funnelweb", "Apex Predator"),
                last_played=NOW - timedelta(days=5),
            ),
        ],
        vault=vault,
    )


def near_complete() -> GuardianContext:
    return context(
        [
            CharacterSpec(
                "Titan",
                weapons=("Rose", "Calus Mini-Tool", "Apex Predator"),
                activities=["Duality: Standard", "Vanguard Ops", "Crucible: Control"],
            ),
        ],
        vault=[weapon("Fatebringer", "vault", None)],
        near=[
            ALTARS_RECORD,
            DUALITY_RECORD,
            near_record(
                "Crucible: Clutch Victories",
                "Final-round wins",
                18,
                20,
                ["Triumphs", "Crucible", "Lifetime"],
            ),
            near_record(
                "Vanguard: Strike Specialist",
                "Strikes completed",
                96,
                100,
                ["Triumphs", "Vanguard", "Lifetime"],
            ),
        ],
    )


def no_next_step() -> GuardianContext:
    return context(
        [
            CharacterSpec(
                "Titan",
                weapons=("Rose", "Funnelweb", "Apex Predator"),
                activities=["Vanguard Ops", "Crucible: Control"],
            )
        ],
        vault=[],
    )


def missing_class() -> GuardianContext:
    return context(
        [
            CharacterSpec(
                "Titan",
                weapons=("Rose", "Calus Mini-Tool", "Apex Predator"),
                exotic_armor="Synthoceps",
                activities=["Vanguard Ops"],
            ),
            CharacterSpec(
                "Warlock",
                element="Void",
                weapons=("Rose", "Funnelweb", "Edge Transit"),
                activities=["Vanguard Ops"],
                last_played=NOW - timedelta(days=3),
            ),
        ],
        vault=[armor("Celestial Nighthawk", "vault", None)],
    )


def wrong_class_vault() -> GuardianContext:
    return context(
        [
            CharacterSpec(
                "Titan",
                weapons=("Rose", "Calus Mini-Tool", "Apex Predator"),
                activities=["Nightfall: The Lightblade", "Vanguard Ops"],
            )
        ],
        vault=[
            armor("Celestial Nighthawk", "vault", None),
            armor("Star-Eater Scales", "vault", None),
            armor("Sunbracers", "vault", None),
            armor("Phoenix Protocol", "vault", None),
        ],
    )


def stale_manifest() -> GuardianContext:
    value = titan_veteran()
    unknown = ItemSummary(
        item_hash=_hash("unknown-item"),
        instance_id="inst-unknown",
        name="Unknown item",
        item_type="Weapon",
        item_subtype=None,
        tier=None,
        class_restriction=None,
        location="vault",
        socket_data_available=False,
        instance_data_available=False,
    )
    value.inventory.items.append(unknown)
    value.inventory.total_items += 1
    value.data_availability.unavailable_components = {
        "ItemSockets": "Bungie did not return socket data for this profile.",
    }
    value.data_availability.notes = ["Some Manifest definitions could not be resolved."]
    for item in value.characters[0].equipped_gear:
        item.socket_data_available = False
        item.socketed_plugs = []
        item.socketed_plug_details = []
    return value


PROFILES: dict[str, Callable[[], GuardianContext]] = {
    "titan_veteran": titan_veteran,
    "hunter_veteran": hunter_veteran,
    "warlock_veteran": warlock_veteran,
    "returning_player": returning_player,
    "new_player": new_player,
    "large_vault": large_vault,
    "near_complete": near_complete,
    "no_next_step": no_next_step,
    "missing_class": missing_class,
    "wrong_class_vault": wrong_class_vault,
    "stale_manifest": stale_manifest,
}

PROFILE_DESCRIPTIONS = {
    "titan_veteran": "Titan main, active Edge of Fate step, near-complete Altars and Duality "
    "records, cross-class Exotics in the vault, Hunter and Warlock alts.",
    "hunter_veteran": "Hunter main with an active dungeon quest; Hunter Exotics and Ace of "
    "Spades in the vault.",
    "warlock_veteran": "Warlock main, three characters, near-complete Altars record.",
    "returning_player": "One Warlock, last played over a year ago, Final Shape step active.",
    "new_player": "One low-Power Hunter, empty vault, no quests, no near-complete records.",
    "large_vault": "Titan and Hunter with every eval weapon and Exotic armor in the vault.",
    "near_complete": "Titan with four near-complete records and no active quests.",
    "no_next_step": "Titan with no quests, no near-complete records, empty vault.",
    "missing_class": "Titan and Warlock only; no Hunter character; a Hunter Exotic in the vault.",
    "wrong_class_vault": "Titan only; the vault holds only Hunter and Warlock Exotic armor.",
    "stale_manifest": "Titan veteran with missing socket data and an unresolved item.",
}


@dataclass(frozen=True)
class ProfileFacts:
    """Ground truth used by the behavioral properties and the judge."""

    classes: tuple[str, ...]
    owned_items: dict[str, tuple[str | None, str]]  # name -> (class restriction, location)
    active_quests: dict[str, tuple[str, ...]]  # class -> quest names
    near_complete: tuple[str, ...]
    available_activities: dict[str, tuple[str, ...]]


def profile_facts(value: GuardianContext) -> ProfileFacts:
    owned: dict[str, tuple[str | None, str]] = {}
    for item in value.inventory.items:
        owned.setdefault(item.name, (item.class_restriction, "vault"))
    for current in value.characters:
        for item in current.equipped_gear:
            if item.item_type in {"Weapon", "Armor"}:
                owned.setdefault(
                    item.name, (item.class_restriction, f"equipped:{current.class_name}")
                )
    return ProfileFacts(
        classes=tuple(value.class_name for value in value.characters),
        owned_items=owned,
        active_quests={
            current.class_name: tuple(
                quest.name for quest in current.quests if not (quest.completed and quest.redeemed)
            )
            for current in value.characters
        },
        near_complete=tuple(
            value.record_name or value.name for value in value.records.near_completion
        ),
        available_activities={
            current.class_name: tuple(activity.name for activity in current.available_activities)
            for current in value.characters
        },
    )


def known_gear_names() -> dict[str, str | None]:
    """Every eval gear name with its class restriction (weapons are any-class)."""

    return {
        **{name: None for name in WEAPONS},
        **{name: owner for name, (owner, _) in EXOTIC_ARMOR.items()},
    }
