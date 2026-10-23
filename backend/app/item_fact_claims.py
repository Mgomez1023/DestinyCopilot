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


# --- Claims that contradict what a tool returned ---------------------------------------------
# A name found in some result is not enough: the class or slot the answer states for it must
# also agree with the retrieved definition (this turn's tools, or the player's gear in context).

Definitions = dict[str, dict[str, set[str]]]
_Contradiction = tuple[str, str, str, int, int, str]

_CLASS_WORD = re.compile(r"\b(?:Titan|Hunter|Warlock)s?\b", re.IGNORECASE)
_SLOT_NOUNS = {
    "helmet": r"helm(?:et)?",
    "gauntlets": r"gauntlets?|gloves",
    "chest": r"chest(?:\s+armor|\s+piece)?",
    "legs": r"legs?(?:\s+armor)?|boots",
    "class item": r"class\s+items?|bonds?|marks?|cloaks?",
    "kinetic": r"kinetic",
    "energy": r"energy",
    "power": r"power|heavy",
}
_SLOT_GROUPS = {key: f"s{index}" for index, key in enumerate(_SLOT_NOUNS)}
# Whole words only: "Legendary" holds "leg", "powerful" holds "power", "energetic" holds "energy".
_SLOT_WORD = re.compile(
    r"\b(?:"
    + "|".join(f"(?P<{_SLOT_GROUPS[key]}>{value})" for key, value in _SLOT_NOUNS.items())
    + r")\b",
    re.IGNORECASE,
)
_ARMOR_NOUN = (
    r"(?:helm(?:et)?|gauntlets?|gloves|chest(?:\s+armor|\s+piece)?|legs|leg\s+armor|boots|"
    r"armor|class\s+item|bond|mark|cloak)"
)
_CLASS_CLAIM = re.compile(
    rf"\b(?P<cls>Titan|Hunter|Warlock)s?(?:-only|\s+(?:only|exclusive)\b|[- ]{_ARMOR_NOUN}\b)",
    re.IGNORECASE,
)
_NEGATOR = re.compile(r"\b(?:not|isn't|aren't|can't|cannot|never|no)\b", re.IGNORECASE)
_SLOT_OF_ITEM = re.compile(
    r"\b(?:the|its|this)\s+(?P<slot>helm(?:et)?|gauntlets?|gloves|chest(?:\s+armor|\s+piece)?|"
    r"leg\s+armor|boots|class\s+item|bond|mark|cloak)(?:'s|\s+(?:ability|abilities|perk|"
    r"perks|intrinsic|effect|exotic))",
    re.IGNORECASE,
)
_SLOT_PHRASE = {
    "helmet": "helmet",
    "gauntlets": "gauntlets",
    "chest": "chest armor",
    "legs": "leg armor",
    "class item": "class item",
    "kinetic": "kinetic",
    "energy": "energy",
    "power": "power",
}


def canonical_class(text: str) -> str:
    folded = text.strip().casefold()
    for name in ("titan", "hunter", "warlock"):
        if name in folded:
            return name
    return "any" if folded.startswith("any") else ""


def canonical_slot(text: str) -> str:
    """helmet, gauntlets, chest, legs, class item, kinetic, energy, or power; "" if unknown."""

    match = _SLOT_WORD.search(text)
    if match is None:
        return ""
    return next(key for key, group in _SLOT_GROUPS.items() if match.group(group))


def collect_item_definitions(result: Any, into: Definitions) -> None:
    """Record the class restriction and slot each named item was returned with.

    `into` maps a folded item name to {"classes": {...}, "slots": {...}}. Several copies or
    variants of one name accumulate, so a claim matching any returned variant is consistent.
    """

    def walk(value: Any, current: dict[str, set[str]] | None, depth: int) -> None:
        if depth > MAX_RESULT_DEPTH or len(into) >= MAX_RETRIEVED_NAMES:
            return
        if isinstance(value, dict):
            name = value.get("name") or value.get("item_name")
            if isinstance(name, str) and len(name.strip()) >= MIN_NAME_CHARS:
                current = into.setdefault(
                    name.strip().casefold(), {"classes": set(), "slots": set()}
                )
            if current is not None:
                restriction = value.get("class_restriction")
                if isinstance(restriction, str) and canonical_class(restriction):
                    current["classes"].add(canonical_class(restriction))
                for key in ("equipment_slot", "bucket"):
                    slot = value.get(key)
                    if isinstance(slot, str) and canonical_slot(slot):
                        current["slots"].add(canonical_slot(slot))
            for inner in value.values():
                if isinstance(inner, (dict, list)):
                    walk(inner, current, depth + 1)
        elif isinstance(value, list):
            for inner in value:
                walk(inner, current, depth + 1)

    walk(result, None, 0)


def _named_items(sentence: str, definitions: Definitions) -> list[str]:
    folded = sentence.casefold()
    found = [name for name in definitions if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", folded)]
    return [name for name in found if not any(name != other and name in other for other in found)]


def _contradictions(sentence: str, definitions: Definitions) -> list[_Contradiction]:
    """(name, kind, claimed, start, end, actual) for each claim a retrieved definition refutes.

    Only a sentence naming exactly one known item is read, and a negated class mention is left
    alone ("isn't a Titan helmet" agrees with a Hunter item).
    """

    names = _named_items(sentence, definitions)
    if len(names) != 1:
        return []
    name = names[0]
    classes, slots = definitions[name]["classes"], definitions[name]["slots"]
    at = sentence.casefold().find(name)
    found: list[_Contradiction] = []

    def check_class(word: str, start: int, end: int) -> None:
        claimed = canonical_class(word)
        if classes and claimed and claimed not in classes:
            found.append(
                (name, "class restriction", claimed, start, end, "/".join(sorted(classes)))
            )

    def check_slot(word: str, start: int, end: int) -> None:
        claimed = canonical_slot(word)
        if slots and claimed and claimed not in slots:
            found.append((name, "equipment slot", claimed, start, end, "/".join(sorted(slots))))

    appositive = re.match(r"(?:'s)?\s*\((?P<text>[^)]*)\)", sentence[at + len(name) :])
    if appositive:
        base = at + len(name) + appositive.start("text")
        for match in _CLASS_WORD.finditer(appositive.group("text")):
            check_class(match.group(0), base + match.start(), base + match.end())
        for match in _SLOT_WORD.finditer(appositive.group("text")):
            check_slot(match.group(0), base + match.start(), base + match.end())
    for match in _CLASS_CLAIM.finditer(sentence):
        if _NEGATOR.search(sentence[max(0, match.start() - 30) : match.start()]):
            continue
        check_class(match.group("cls"), match.start("cls"), match.end("cls"))
    for match in _SLOT_OF_ITEM.finditer(sentence):
        between = sentence[at + len(name) : match.start()]
        # "<item> ... the helmet's ability": the slot phrase must be about that item.
        if match.start() > at and len(between) <= 60 and not re.search(r"[.;,]", between):
            check_slot(match.group("slot"), match.start("slot"), match.end("slot"))
    # One claim can match two patterns (an appositive and "Hunter helm"): fix it once.
    return list({(value[1], value[3], value[4]): value for value in found}.values())


def _statements(answer: str) -> list[str]:
    return [
        value.strip()
        for value in _SENTENCE_SPLIT.split(_plain(answer))
        if value.strip() and not value.strip().endswith("?")
    ]


def contradicted_item_facts(
    answer: str, definitions: Definitions
) -> list[tuple[str, str, str, str]]:
    """(item name, kind, claimed, retrieved) where the answer contradicts a returned definition."""

    if not definitions:
        return []
    return [
        (name, kind, claimed, actual)
        for sentence in _statements(answer)
        for name, kind, claimed, _start, _end, actual in _contradictions(sentence, definitions)
    ]


def correct_item_facts(answer: str, definitions: Definitions) -> str | None:
    """The answer with contradicted facts stated as retrieved, or None.

    A class word becomes the retrieved class, and a slot word inside an appositive becomes the
    retrieved slot. A sentence whose claim cannot be fixed in place is dropped. None when
    nothing contradicted a result, or when too little would remain.
    """

    if not definitions or not contradicted_item_facts(answer, definitions):
        return None
    pieces = re.split(r"(?<=[.!?])(\s+)", answer)
    out: list[str] = []
    for index in range(0, len(pieces), 2):
        sentence = pieces[index]
        spaces = pieces[index + 1] if index + 1 < len(pieces) else ""
        plain = _plain(sentence)
        keep: str | None = sentence
        statement = bool(plain.strip()) and not plain.strip().endswith("?")
        edits = _contradictions(plain, definitions) if statement else []
        for _name, kind, _claimed, start, end, actual in sorted(
            edits, key=lambda value: value[3], reverse=True
        ):
            if keep is None:
                break
            if kind == "class restriction" and "/" not in actual and actual != "any":
                keep = keep[:start] + actual.capitalize() + keep[end:]
            elif (
                kind == "equipment slot"
                and actual in _SLOT_PHRASE
                and "(" in plain[:start]
                and ")" in plain[end:]
            ):
                keep = keep[:start] + _SLOT_PHRASE[actual] + keep[end:]
            else:
                keep = None
        if keep is not None:
            out.append(keep + spaces)
    text = "".join(out).strip()
    return text if len(re.findall(r"\w+", text)) >= 6 else None
