"""Possession and equipment evidence for build answers.

A build answer may say that the player has, owns, or has equipped an item only when a trusted
Guardian tool result proves it. Trusted sources are the equipped loadout, current-build
analysis, and inventory search results; any of them proves possession of what it lists, and
only equipped entries prove "equipped". This module supports possession and equipment claims
only. It never vouches for what an item does, how strong it is, or what is current.

Claim detection is deliberately conservative: a sentence is a claim only when it uses
possession or equipment wording AND names an item-like proper noun. "You have high Grenade"
names no item and is not a claim.
"""

import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

ClaimType = Literal["possession", "equipped"]
EvidenceSource = Literal["equipped_loadout", "current_build_analysis", "inventory_search"]

SOURCE_BY_TOOL: dict[str, EvidenceSource] = {
    "get_equipped_loadout": "equipped_loadout",
    "get_build_details": "equipped_loadout",
    "analyze_current_build": "current_build_analysis",
    "find_build_alternatives": "inventory_search",
    "search_inventory": "inventory_search",
}
# Result keys whose entries are the character's equipped gear.
EQUIPPED_KEYS = frozenset({"equipped_weapons", "equipped_armor", "equipped_exotics"})
ITEM_LIST_KEYS = (
    "equipped_weapons",
    "equipped_armor",
    "equipped_exotics",
    "weapons",
    "armor",
    "items",
    "candidates",
)
SUBCLASS_GROUP_KEYS = ("abilities", "aspects", "fragments", "mods")
MAX_NAME_CHARS = 80
UNKNOWN_CHARACTER = "unknown"
MAX_REPORTED_CLAIMS = 12
MAX_SUMMARY_NAMES = 24
# After removing unsupported sentences, at least this much of the answer must remain.
MIN_REPAIRED_WORDS = 12
MAX_REMOVED_FRACTION = 0.5

# Sentences that recommend an action on the item are the recommendation itself, not a
# removable aside.
RECOMMENDATION = re.compile(
    r"\b(?:equip|use|using|run|running|slot|swap|switch(?:\s+to)?|pair|try|go\s+with|"
    r"replace|add|install|bring|wear|put\s+on|pick|grab|chase|farm|keep|consider|recommend)\b",
    re.IGNORECASE,
)
_POSSESSION_VERB = re.compile(
    r"\b(?:you|your\s+(?:guardian|titan|hunter|warlock))\s+"
    r"(?:already\s+|also\s+|currently\s+|still\s+)*"
    r"(?:own|owns|have|has|possess|possesses|got)\b|\byou['’]ve\s+(?:got|owned)\b",
    re.IGNORECASE,
)
_EQUIPPED = re.compile(
    r"\b(?:(?:is|are|was|were|['’]s)\s+)?(?:currently\s+|already\s+|still\s+)?equipped\b",
    re.IGNORECASE,
)
_CLAUSE_BREAK = re.compile(
    r";|—|–|\s-\s|\b(?:but|so|because|which|while|although|though|since)\b|"
    r",\s+and\s+(?:it|this|that|they|these|those|its|your|you|the)\b",
    re.IGNORECASE,
)
# "Witherhoard, which you have": the claimed item precedes the possession verb.
_RELATIVE_GAP = re.compile(r"^[\s,]*(?:which|that)?[\s,]*$", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CAPITALIZED_RUN = re.compile(r"[A-Z][\w'’\-]*(?:(?:\s+(?:of|the|and|in|for))?\s+[A-Z][\w'’\-]*)*")
_CONNECTORS = frozenset({"of", "the", "and", "in", "for"})
_BEFORE_GAP_LIMIT = 3
# A possession verb claims the items right after it, not a whole trailing clause.
_OBJECT_WINDOW = 80
_CONTRACTION = re.compile(r"['’](?:ll|d|m|ve|re|s|t)$", re.IGNORECASE)

# Capitalized words that are not item names: sentence openers, classes, elements, stats,
# rarities, slots, and common activity vocabulary.
_NOT_ITEM_WORDS = (
    "You Your The This That It Its These Those Both Also Currently Right Now Keep Swap "
    "Replace Use Run Try Equip Switch Add Remove Drop Pair Slot Consider Note Since If "
    "When Because Overall However Instead Weakest Biggest First Second Third Next Main "
    "Primary Backup Quick Short Why What How Which Who Where I We And Or But So Then "
    "Here There Yes No Okay Sure Good Great Best Better Top "
    "Titan Hunter Warlock Guardian Sunbreaker Striker Sentinel Arcstrider Gunslinger "
    "Nightstalker Dawnblade Stormcaller Voidwalker Behemoth Revenant Shadebinder "
    "Berserker Threadrunner Broodweaver Prismatic Solar Arc Void Stasis Strand Kinetic "
    "Health Melee Grenade Super Class Weapons Weapon Resilience Recovery Discipline "
    "Intellect Strength Mobility Exotic Legendary Rare Common Uncommon Helmet Gauntlets "
    "Chest Armor Legs Boots Power Energy Heavy Special Mods Mod Aspect Aspects Fragment "
    "Fragments Abilities Ability Perk Perks Scout Pulse Hand Cannon Rifle Sniper Shotgun "
    "Sword Bow Glaive Sidearm Launcher Rocket Grenade Machine Gun Fusion Linear Trace "
    "Submachine PvE PvP Crucible Gambit Nightfall Vanguard Ops Strike Strikes Raid "
    "Raids Dungeon Dungeons Trials Osiris Iron Banner Lost Sector Edge Fate Final Shape "
    "Destiny Bungie Light Darkness Tower Moon Earth Mars Mercury Europa Neomuna "
    "Season Triumph Triumphs Quest Bounty Patrol "
    "Current Equipped Owned Verified Recommended Suggested Let Fix Fixes Option Options "
    "Step Steps Tip Tips Result Results Summary Verdict Bottom Line Plan Goal Reason "
    "Reasons Change Changes Candidate Candidates Evidence Gear Loadout Build Kit Stats Stat "
    "Slots Weakness Weaknesses Strengths Analysis Notes Ownership Possession Unknown"
)
_NOT_ITEMS = frozenset(value.casefold() for value in _NOT_ITEM_WORDS.split())


class OwnershipEvidence(BaseModel):
    """Items the player is proven to possess, and which of them are equipped."""

    possessed: dict[str, list[str]] = Field(default_factory=dict)
    equipped: dict[str, list[str]] = Field(default_factory=dict)
    # Which character each equipped item is equipped on. UNKNOWN_CHARACTER means the result
    # did not say, which only a claim with no character context can rely on.
    equipped_on: dict[str, list[str]] = Field(default_factory=dict)
    display: dict[str, str] = Field(default_factory=dict)

    def add(self, name: Any, source: str, *, equipped: bool, character: str | None = None) -> None:
        if not isinstance(name, str):
            return
        text = name.strip()
        if not text or len(text) > MAX_NAME_CHARS:
            return
        key = text.casefold()
        self.display.setdefault(key, text)
        sources = self.possessed.setdefault(key, [])
        if source not in sources:
            sources.append(source)
        if equipped:
            equipped_sources = self.equipped.setdefault(key, [])
            if source not in equipped_sources:
                equipped_sources.append(source)
            owners = self.equipped_on.setdefault(key, [])
            owner = character or UNKNOWN_CHARACTER
            if owner not in owners:
                owners.append(owner)

    def source_for(
        self, name: str, claim_type: ClaimType, character: str | None = None
    ) -> str | None:
        """The evidence source that proves this claim type for this item, if any.

        Possession is account-wide. "Equipped" is per character: when the claim is about a
        character, only evidence for that character (or evidence that did not say which)
        counts, so an item equipped on the Titan never proves it is equipped on the Hunter.

        Only possession and equipment are supported here; there is deliberately no way to ask
        whether an item's effect, strength, or meta standing is supported.
        """

        key = name.casefold()
        if claim_type == "possession":
            sources = self.possessed.get(key)
            return sources[0] if sources else None
        sources = self.equipped.get(key)
        if not sources:
            return None
        if character is not None:
            owners = self.equipped_on.get(key, [])
            if character not in owners and UNKNOWN_CHARACTER not in owners:
                return None
        return sources[0]

    def equipped_characters(self) -> dict[str, int]:
        """How many equipped items were recorded per character (for traces)."""

        counts: dict[str, int] = {}
        for owners in self.equipped_on.values():
            for owner in owners:
                counts[owner] = counts.get(owner, 0) + 1
        return dict(sorted(counts.items()))

    def equipped_names(self, character: str | None) -> list[str]:
        """Display names equipped on the character (or equipped anywhere when unscoped)."""

        return [
            self.display[key]
            for key in self.equipped
            if character is None
            or character in self.equipped_on.get(key, [])
            or UNKNOWN_CHARACTER in self.equipped_on.get(key, [])
        ]

    def counts_by_source(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for sources in self.possessed.values():
            for source in sources:
                counts[source] = counts.get(source, 0) + 1
        return dict(sorted(counts.items()))


def _plug_names(item: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for key in ("socketed_plug_names", "perks_and_sockets"):
        values = item.get(key)
        if isinstance(values, list):
            names.extend(value for value in values if isinstance(value, str))
    for key in ("socketed_plugs", "relevant_perks"):
        values = item.get(key)
        if isinstance(values, list):
            names.extend(
                value["name"]
                for value in values
                if isinstance(value, dict) and isinstance(value.get("name"), str)
            )
    return names


def record_ownership_evidence(
    evidence: OwnershipEvidence,
    tool_name: str,
    result: dict[str, Any],
    character_classes: dict[str, str] | None = None,
) -> None:
    """Record what one trusted Guardian tool result proves about possession and equipment.

    `character_classes` maps opaque character IDs to class names so equipped items can be
    attributed to their character; IDs are used only for this lookup and never stored.
    """

    source = SOURCE_BY_TOOL.get(tool_name)
    if source is None or not isinstance(result, dict) or "error" in result:
        return
    classes = character_classes or {}
    # Results about one character say which; unscoped inventory results tag each item.
    result_class = result.get("character_class")
    if not isinstance(result_class, str):
        header = result.get("character")
        result_class = header.get("class") if isinstance(header, dict) else None
    result_class = result_class if isinstance(result_class, str) else None
    for key in ITEM_LIST_KEYS:
        values = result.get(key)
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict):
                continue
            equipped = bool(
                key in EQUIPPED_KEYS
                or item.get("equipped") is True
                or item.get("location") == "equipped"
            )
            owner = classes.get(str(item.get("character_id"))) or result_class
            evidence.add(item.get("name"), source, equipped=equipped, character=owner)
            for plug in _plug_names(item):
                evidence.add(plug, source, equipped=equipped, character=owner)
    # A character-scoped search also names matches held by other characters, each with its owner.
    others = result.get("other_character_matches")
    for entry in others if isinstance(others, list) else []:
        if isinstance(entry, dict):
            owner = entry.get("character_class")
            evidence.add(
                entry.get("name"),
                source,
                equipped=entry.get("equipped") is True,
                character=owner if isinstance(owner, str) else None,
            )
    # Alternatives results name the character's currently equipped Exotics (not merely locked
    # items) so they can be preserved; that is direct evidence of what is equipped.
    preservation = result.get("preservation")
    if isinstance(preservation, dict):
        for key in ("equipped_exotic_weapon", "equipped_exotic_armor"):
            values = preservation.get(key)
            if isinstance(values, list):
                for name in values:
                    evidence.add(name, source, equipped=True, character=result_class)
    mods = result.get("armor_mods")
    if source != "inventory_search" and isinstance(mods, list):
        for mod in mods:
            if isinstance(mod, dict):
                evidence.add(mod.get("name"), source, equipped=True, character=result_class)
    subclass = result.get("subclass")
    if source != "inventory_search" and isinstance(subclass, dict):
        evidence.add(subclass.get("name"), source, equipped=True, character=result_class)
        for group in SUBCLASS_GROUP_KEYS:
            values = subclass.get(group)
            if not isinstance(values, list):
                continue
            for value in values:
                name = value.get("name") if isinstance(value, dict) else value
                evidence.add(name, source, equipped=True, character=result_class)


@dataclass(frozen=True, slots=True)
class OwnershipClaim:
    item: str
    claim_type: ClaimType
    supported: bool
    source: str | None
    start: int  # Bounds of the containing sentence in the answer.
    end: int
    removable: bool

    def trace(self, action: str) -> dict[str, Any]:
        return {
            "item": self.item,
            "claim_type": self.claim_type,
            "supported": self.supported,
            "evidence": self.source,
            "action": action,
        }


@dataclass(frozen=True, slots=True)
class _Mention:
    text: str
    start: int
    end: int
    known: bool


def _sentences(answer: str) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    offset = 0
    for line in answer.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        position = 0
        for piece in _SENTENCE_SPLIT.split(stripped):
            index = stripped.find(piece, position)
            if piece.strip():
                spans.append((piece, offset + index, offset + index + len(piece)))
            position = index + len(piece)
        offset += len(line)
    return spans


_PLAIN_COMPOUND = re.compile(r"^[A-Z][a-z]+(?:-[a-z]+)+$")


def _generic(word: str) -> bool:
    lowered = word.casefold()
    base = _CONTRACTION.sub("", lowered)
    return (
        base in _NOT_ITEMS
        or word.isupper()
        or lowered in _NOT_ITEMS
        or (lowered.endswith("s") and lowered[:-1] in _NOT_ITEMS)
        # "Super-focused", "Grenade-heavy": an ordinary compound adjective, not an item name
        # (item names capitalize each part, as in "Wish-Ender", or carry digits).
        or bool(_PLAIN_COMPOUND.match(word))
    )


_MIN_PARTIAL_WORD_CHARS = 5
_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])?\s*")
_NAME_PREDICATE = re.compile(r"^\s*(?:is|are|was|were|has|have|will|can|['’]s)\b", re.IGNORECASE)


def _is_sentence_opener(sentence: str, start: int, end: int) -> bool:
    """A lone capitalized word that only looks like a name because it opens the sentence.

    Capitalization at the start of a sentence carries no information ("Brief equipped
    loadout: …"). The word is treated as an item name only when it is followed by a verb that
    predicates something of it ("Witherhoard is equipped"), as a real item claim would be.
    """

    if start != _LIST_MARKER.match(sentence).end() or " " in sentence[start:end].strip():
        return False
    return _NAME_PREDICATE.match(sentence[end:]) is None


def _partial_name(text: str, evidence: OwnershipEvidence) -> str | None:
    """The evidence item that a shortened reference points to ("Nighthawk" for
    "Celestial Nighthawk"), or None.

    A reference counts only when it is a run of whole words inside exactly one known name and
    is distinctive: two or more words, or a single word of at least five letters. A shortened
    reference to an item the evidence knows is a reference to that item, not a new one.
    """

    words = re.findall(r"[\w'’]+", text.casefold())
    if not words or (len(words) == 1 and len(words[0]) < _MIN_PARTIAL_WORD_CHARS):
        return None
    matches: list[str] = []
    for key, display in evidence.display.items():
        key_words = re.findall(r"[\w'’]+", key)
        if len(words) >= len(key_words):
            continue
        for index in range(len(key_words) - len(words) + 1):
            if key_words[index : index + len(words)] == words:
                matches.append(display)
                break
    return matches[0] if len(matches) == 1 else None


def _mentions(sentence: str, evidence: OwnershipEvidence) -> list[_Mention]:
    """Item-like names in a sentence: names the evidence knows, then unknown proper nouns."""

    mentions: list[_Mention] = []
    taken = [False] * len(sentence)
    for key in sorted(evidence.display, key=len, reverse=True):
        for match in re.finditer(rf"(?<![\w]){re.escape(key)}(?![\w])", sentence, re.IGNORECASE):
            if any(taken[match.start() : match.end()]):
                continue
            for index in range(match.start(), match.end()):
                taken[index] = True
            mentions.append(_Mention(evidence.display[key], match.start(), match.end(), True))
    for match in _CAPITALIZED_RUN.finditer(sentence):
        if any(taken[match.start() : match.end()]):
            continue
        # Trim generic openers and closers ("Your Sunshot", "Use Witherhoard") off the run.
        words = [
            (word.group(), word.start(), word.end()) for word in re.finditer(r"\S+", match.group())
        ]
        while words and (_generic(words[0][0]) or words[0][0].casefold() in _CONNECTORS):
            words.pop(0)
        while words and words[-1][0].casefold() in _CONNECTORS:
            words.pop()
        if not words:
            continue
        start = match.start() + words[0][1]
        end = match.start() + words[-1][2]
        text = sentence[start:end]
        resolved = _partial_name(text, evidence)
        if resolved is None and _is_sentence_opener(sentence, start, end):
            continue
        mentions.append(_Mention(resolved or text, start, end, resolved is not None))
    return sorted(mentions, key=lambda value: value.start)


def _bind(
    bound: dict[str, ClaimType], order: list[_Mention], mention: _Mention, claim_type: ClaimType
) -> None:
    key = mention.text.casefold()
    if bound.get(key) == "equipped":
        return
    if key not in bound:
        order.append(mention)
    bound[key] = claim_type


def _clause_end(sentence: str, start: int) -> int:
    match = _CLAUSE_BREAK.search(sentence, start)
    return match.start() if match else len(sentence)


def find_ownership_claims(
    answer: str, evidence: OwnershipEvidence, character: str | None = None
) -> list[OwnershipClaim]:
    """Every possession or equipment claim about a named item in the answer.

    `character` is the character the answer is about; "equipped" claims must be proven for it.
    """

    claims: list[OwnershipClaim] = []
    for sentence, start, end in _sentences(answer):
        possession = _POSSESSION_VERB.search(sentence)
        equipped_matches = list(_EQUIPPED.finditer(sentence))
        if not (possession or equipped_matches):
            continue
        mentions = _mentions(sentence, evidence)
        if not mentions:
            continue
        bound: dict[str, ClaimType] = {}
        order: list[_Mention] = []

        if possession is not None:
            limit = min(_clause_end(sentence, possession.end()), possession.end() + _OBJECT_WINDOW)
            for mention in mentions:
                if possession.end() <= mention.start < limit or (
                    mention.end <= possession.start()
                    and _RELATIVE_GAP.match(sentence[mention.end : possession.start()])
                ):
                    _bind(bound, order, mention, "possession")
        for trigger in equipped_matches:
            for mention in mentions:
                before = sentence[mention.end : trigger.start()]
                after = sentence[trigger.end() : mention.start]
                ends_before = (
                    mention.end <= trigger.start()
                    and len(before) <= _BEFORE_GAP_LIMIT
                    and not before.strip(" (,")
                )
                starts_after = (
                    mention.start >= trigger.end() and len(after) <= 2 and not after.strip()
                )
                if ends_before or starts_after:
                    _bind(bound, order, mention, "equipped")
        recommending = bool(RECOMMENDATION.search(sentence))
        for mention in order:
            claim_type = bound[mention.text.casefold()]
            source = evidence.source_for(mention.text, claim_type, character)
            claims.append(
                OwnershipClaim(
                    item=mention.text,
                    claim_type=claim_type,
                    supported=source is not None,
                    source=source,
                    start=start,
                    end=end,
                    removable=not recommending,
                )
            )
    return claims


def unsupported_claims(claims: list[OwnershipClaim]) -> list[OwnershipClaim]:
    return [value for value in claims if not value.supported]


def repair_unsupported_ownership(
    answer: str, claims: list[OwnershipClaim]
) -> tuple[str, list[OwnershipClaim]]:
    """Remove only the sentences carrying unsupported, non-recommending ownership claims.

    Returns the original answer unchanged (and no removals) when any unsupported claim is
    part of a recommendation, or when removing the sentences would leave too little of the
    answer. In those cases the core answer is unsupported and the caller must fail closed.
    """

    bad = unsupported_claims(claims)
    if not bad or not all(value.removable for value in bad):
        return answer, []
    spans = sorted({(value.start, value.end) for value in bad})
    kept: list[str] = []
    cursor = 0
    for start, end in spans:
        kept.append(answer[cursor:start])
        cursor = end
    kept.append(answer[cursor:])
    repaired = "".join(kept)
    # Drop list markers or blank lines orphaned by the removal.
    lines = [line.rstrip() for line in repaired.splitlines()]
    lines = [line for line in lines if re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()]
    repaired = re.sub(r"[ \t]{2,}", " ", "\n".join(lines)).strip()
    before_words = len(re.findall(r"\w+", answer))
    after_words = len(re.findall(r"\w+", repaired))
    if after_words < MIN_REPAIRED_WORDS or (before_words - after_words) > (
        before_words * MAX_REMOVED_FRACTION
    ):
        return answer, []
    return repaired, bad


def evidence_summary(evidence: OwnershipEvidence, character: str | None = None) -> str:
    """Bounded, structured statement of the possession facts already established.

    Equipped names are limited to the character the answer is about, so the model is never
    told another character's gear is equipped.
    """

    equipped = evidence.equipped_names(character)
    equipped_keys = {name.casefold() for name in equipped}
    owned_only = [evidence.display[key] for key in evidence.possessed if key not in equipped_keys]
    if not equipped and not owned_only:
        return ""
    parts = ["Verified Guardian possession evidence (no further lookup needed for these):"]
    if equipped:
        label = f"equipped on the {character}" if character else "equipped"
        parts.append(f"{label}: {', '.join(equipped[:MAX_SUMMARY_NAMES])}.")
    if owned_only:
        parts.append(f"owned, not equipped: {', '.join(owned_only[:MAX_SUMMARY_NAMES])}.")
    parts.append(
        "This proves only possession and what is equipped, nothing about what items do or how "
        "strong they are. Use it only to decide which ownership statements to keep: state "
        "ownership only for these names, delete any other ownership claim instead of "
        "rewriting the answer, and do not list, quote, or mention this evidence block."
    )
    return " ".join(parts)
