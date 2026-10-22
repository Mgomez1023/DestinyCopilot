"""Claims about one named item's definition that must rest on a retrieved result.

A class restriction, an equipment slot, and whether a barrel or perk is fixed or selectable are
facts of an item's definition. The model remembers them wrongly, so an answer that states one for
a named item needs that item in a result retrieved this turn (an item lookup, the player's
inventory, or loadout). Nothing here knows any particular item: it reads the claim's wording and
checks the item's name against what tools returned.
"""

import re
from typing import Any

from app.ownership_evidence import (
    _CAPITALIZED_RUN,
    _LIST_MARKER,
    _SENTENCE_SPLIT,
    _generic,
    _is_sentence_opener,
)

MAX_RESULT_DEPTH = 6
MAX_RETRIEVED_NAMES = 2000
MIN_NAME_CHARS = 4

_CLASS = r"(?:Titans?|Hunters?|Warlocks?)"
_SLOT = (
    r"(?:kinetic|energy|power|heavy|special|helmet|gauntlets|arms|chest(?:\s+armor)?|"
    r"legs?(?:\s+armor)?|boots|class\s+item)"
)
_ITEM_PART = r"(?:barrels?|magazines?|mags?|perks?|traits?|frames?|intrinsics?|rolls?|sockets?)"
CLAIM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (kind, re.compile(pattern, re.IGNORECASE))
    for kind, pattern in (
        (
            "class restriction",
            rf"\b(?:is|are)\s+(?:an?\s+|the\s+)?{_CLASS}(?:-only|\s+(?:only|exclusive|class|"
            r"exotic|helmet|gauntlets|chest|leg|armor|piece)\b)",
        ),
        (
            "class restriction",
            rf"\b(?:can|may)\s+only\s+be\s+(?:equipped|worn|used)\s+by\s+(?:an?\s+|the\s+)?{_CLASS}",
        ),
        (
            "class restriction",
            r"\b(?:is|are)\s+(?:class[- ]locked(?:\s+to)?|"
            rf"(?:restricted|limited|exclusive)\s+to)\s+(?:the\s+)?{_CLASS}",
        ),
        (
            "class restriction",
            r"\b(?:isn't|is not|can't be|cannot be|aren't|are not)\s+(?:usable|equipped|worn|"
            rf"used|wearable|available)\s+(?:by|on|for)\s+(?:an?\s+|the\s+|your\s+)?{_CLASS}",
        ),
        ("class restriction", rf"\bfor\s+(?:an?\s+|the\s+)?{_CLASS}\s+only\b"),
        (
            "equipment slot",
            rf"\b(?:is|are|goes? in|sits? in|occupies|takes up|uses)\s+(?:an?\s+|the\s+|your\s+)?"
            rf"{_SLOT}\s+(?:weapon\s+)?slot\b",
        ),
        (
            "fixed or selectable perks",
            rf"\b{_ITEM_PART}\b[^.;]{{0,25}}\b(?:is|are|can be|cannot be|can't be)\s+"
            r"(?:fixed|locked|selectable|random(?:ly)?(?:\s+rolled)?|randomi[sz]ed|changed|"
            r"swapped|re-?rolled|customi[sz]ed)\b",
        ),
        (
            "fixed or selectable perks",
            rf"\b(?:fixed|selectable|random(?:ly rolled)?)\s+{_ITEM_PART}\b",
        ),
    )
)


def collect_item_names(result: Any, into: set[str]) -> None:
    """Add every name a tool result returned, folded, so an item the answer cites can be found."""

    def walk(value: Any, depth: int) -> None:
        if depth > MAX_RESULT_DEPTH or len(into) >= MAX_RETRIEVED_NAMES:
            return
        if isinstance(value, dict):
            for key, inner in value.items():
                if key in {"name", "item_name", "plug", "plug_name"} and isinstance(inner, str):
                    folded = inner.strip().casefold()
                    if len(folded) >= MIN_NAME_CHARS:
                        into.add(folded)
                elif isinstance(inner, (dict, list)):
                    walk(inner, depth + 1)
        elif isinstance(value, list):
            for inner in value:
                walk(inner, depth + 1)

    walk(result, 0)


def _candidate_names(sentence: str) -> list[str]:
    names: list[str] = []
    for match in _CAPITALIZED_RUN.finditer(sentence):
        text = match.group(0).strip()
        possessive = re.search(r"['’]s$", text) is not None
        text = re.sub(r"['’]s$", "", text)
        words = text.split()
        if all(_generic(word) for word in words if word[:1].isupper()):
            continue
        # A possessive ("Name's barrel") names its owner even at the start of a sentence.
        if not possessive and _is_sentence_opener(sentence, match.start(), match.end()):
            continue
        names.append(text)
    return names


def _retrieved(name: str, retrieved: set[str]) -> bool:
    folded = name.casefold()
    return any(folded == value or folded in value or value in folded for value in retrieved)


def _plain(text: str) -> str:
    return text.replace("’", "'")


def unverified_item_claims(answer: str, retrieved: set[str]) -> list[tuple[str, str]]:
    """(item name, kind of claim) pairs where a definition fact is stated without a lookup."""

    found: list[tuple[str, str]] = []
    for sentence in _SENTENCE_SPLIT.split(_plain(answer)):
        sentence = sentence.strip()
        if not sentence or sentence.endswith("?"):
            continue
        kinds = [kind for kind, pattern in CLAIM_PATTERNS if pattern.search(sentence)]
        if not kinds:
            continue
        names = _candidate_names(sentence)
        if not names or any(_retrieved(name, retrieved) for name in names):
            continue
        found.append((names[0], kinds[0]))
    return found


def remove_unverified_item_claims(answer: str, retrieved: set[str]) -> str | None:
    """The answer without the sentences that state an unverified definition fact.

    Returns None when nothing was removed or what is left would not stand as an answer.
    """

    claims = {name for name, _ in unverified_item_claims(answer, retrieved)}
    if not claims:
        return None
    kept: list[str] = []
    removed = False
    for line in answer.splitlines():
        pieces = re.split(r"(?<=[.!?])\s+", line)
        left = [
            piece
            for piece in pieces
            if not (
                piece.strip()
                and unverified_item_claims(piece, retrieved)
                and not piece.strip().endswith("?")
            )
        ]
        removed = removed or len(left) != len(pieces)
        text = " ".join(left).strip()
        if text and _LIST_MARKER.sub("", text, count=1).strip():
            kept.append(text)
        elif not line.strip():
            kept.append("")
    cleaned = "\n".join(kept).strip()
    if not removed or len(re.findall(r"\w+", cleaned)) < 6:
        return None
    return cleaned
