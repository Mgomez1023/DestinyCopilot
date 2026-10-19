"""Canonical equipment identity read from Manifest definitions.

An item has two unrelated "buckets":

* the **storage** bucket: where this copy sits right now. A vaulted weapon or helmet reports
  Bungie's vault bucket ("General"), which says nothing about what kind of gear it is;
* the **equipment slot**: where the item can be equipped. It is a property of the item's
  *definition* (`inventory.bucketTypeHash`) and is the same wherever the copy is stored.

Slots are never inferred from names, subtypes or memory. A definition yields a slot only when
its bucket is a real equipment-slot bucket (`category == 3`, "Equippable") and the definition
is equippable; anything else is reported as not equippable or unresolved, never guessed.
"""

from typing import Any, Literal

# DestinyInventoryBucketDefinition.category: 0 Invisible, 1 Item, 2 Currency, 3 Equippable.
EQUIPPABLE_BUCKET_CATEGORY = 3
ENGRAM_TRAIT = "item.engram"

SlotState = Literal["known", "not_equippable", "unresolved", "not_collected"]


def bucket_name(bucket_definition: dict[str, Any] | None) -> str | None:
    name = ((bucket_definition or {}).get("displayProperties") or {}).get("name")
    return name.strip() if isinstance(name, str) and name.strip() else None


def definition_bucket_hash(definition: dict[str, Any]) -> int | None:
    """The bucket this definition belongs to, which is its equipment slot when equippable."""

    value = (definition.get("inventory") or {}).get("bucketTypeHash")
    return value if isinstance(value, int) and not isinstance(value, bool) and value else None


def equipment_slot(
    definition: dict[str, Any], bucket_definition: dict[str, Any] | None
) -> tuple[str | None, SlotState]:
    """The equipment slot of an item definition, and how sure we are.

    * ("Helmet", "known"): equippable, in an Equippable-category bucket;
    * (None, "not_equippable"): the definition says it cannot be equipped, or its bucket is
      not an equipment slot (a dummy, a consumable, an engram);
    * (None, "unresolved"): the definition or its bucket could not be read, so nothing is said.
    """

    if not definition:
        return None, "unresolved"
    bucket_hash = definition_bucket_hash(definition)
    if bucket_hash is None:
        return None, "unresolved"
    if definition.get("equippable") is False:
        return None, "not_equippable"
    if not bucket_definition or definition.get("equippable") is not True:
        return None, "unresolved"
    if bucket_definition.get("category") != EQUIPPABLE_BUCKET_CATEGORY:
        return None, "not_equippable"
    name = bucket_name(bucket_definition)
    return (name, "known") if name else (None, "unresolved")


def is_engram_variant(definition: dict[str, Any]) -> bool:
    """A non-equippable dummy that shares a real item's name (the engram preview entry)."""

    trait_ids = definition.get("traitIds")
    if isinstance(trait_ids, list) and ENGRAM_TRAIT in trait_ids:
        return True
    return int(definition.get("itemType") or 0) == 0 and not definition.get("equippable")
