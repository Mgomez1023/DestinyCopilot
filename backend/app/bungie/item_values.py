"""Shared readers for Manifest item `value` blocks (quest and bounty rewards)."""

from typing import Any

MAX_REWARD_NAMES = 4
# Distinct reward hashes read per definition; bounds the batched name resolution.
MAX_REWARD_HASHES = 8
MAX_REWARD_NAME_CHARS = 120
MAX_ITEM_TYPE_NAME_CHARS = 80


def reward_hashes(definition: dict[str, Any], limit: int = MAX_REWARD_HASHES) -> list[int]:
    """Distinct non-placeholder reward item hashes declared in `value.itemValue`.

    Bungie pads `itemValue` with hash-0 placeholder entries; those are not rewards.
    """

    found: list[int] = []
    for entry in (definition.get("value") or {}).get("itemValue", []) or []:
        if not isinstance(entry, dict):
            continue
        try:
            item_hash = int(entry.get("itemHash") or 0)
        except (TypeError, ValueError):
            continue
        if item_hash and item_hash not in found:
            found.append(item_hash)
            if len(found) >= limit:
                break
    return found


def item_type_name(definition: dict[str, Any]) -> str | None:
    """The definition's own specific type label (e.g. "Duality Order"); never inferred."""

    value = str(definition.get("itemTypeDisplayName") or "").strip()
    return value[:MAX_ITEM_TYPE_NAME_CHARS] or None


def reward_summary(
    definition: dict[str, Any], resolved: dict[int, dict[str, Any]]
) -> tuple[list[str], int, int]:
    """Return (reward names capped at MAX_REWARD_NAMES, declared total, unresolved count).

    `resolved` maps item hash to definition. A declared reward whose definition is missing
    or has no display name is counted as unresolved rather than given an invented label.
    """

    hashes = reward_hashes(definition)
    names: list[str] = []
    unresolved = 0
    for item_hash in hashes:
        name = str(
            ((resolved.get(item_hash) or {}).get("displayProperties") or {}).get("name") or ""
        ).strip()
        if not name:
            unresolved += 1
        elif name not in names and len(names) < MAX_REWARD_NAMES:
            names.append(name[:MAX_REWARD_NAME_CHARS])
    return names, len(hashes), unresolved
