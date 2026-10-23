import re
from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field

from app.absence_claims import absent_measure_values
from app.build_analysis import is_explicit_build_request, replaces_item
from app.item_fact_claims import contradicted_item_facts, unverified_item_claims
from app.ownership_evidence import (
    OwnershipClaim,
    OwnershipEvidence,
    evidence_summary,
    find_ownership_claims,
    repair_unsupported_ownership,
    unsupported_claims,
)
from app.session_planning import (
    REWRITE_RULE,
    PlanningAnswerValidator,
    PlanningViolation,
    SessionPlanningContext,
    _normalize,
    contains_recommendation,
)

ResponseMode = Literal[
    "direct_fact",
    "recommendation",
    "comparison",
    "walkthrough",
    "build_advice",
    "account_summary",
    "troubleshooting",
]


ViolationGroup = Literal["evidence", "grounding_safety", "style"]
BuildFallbackKind = Literal["evidence", "grounding_safety", "style_only"]

# Required Guardian evidence was never gathered for a personalized build answer.
EVIDENCE_VIOLATION_CODES = frozenset(
    {
        "missing_current_build_analysis",
        "missing_owned_inventory_evidence",
    }
)
# Presentation-only problems. An answer whose only remaining issues are listed here is
# still grounded, so it must not be replaced by a claim that data retrieval failed.
STYLE_VIOLATION_CODES = frozenset(
    {
        "answer_not_first",
        "extreme_verbosity",
        "followup_restarts_plan",
        "internal_jargon",
        "inventory_dump",
        "mechanical_structure",
        "repeated_content",
        "too_many_build_changes",
        "too_many_followups",
        "unnecessary_clarification",
        "unnecessary_read_only_permission",
        # A draft that is only a question gets one rewrite that assumes and answers; if it
        # still asks, that is kept (no canned replacement).
        "clarification_only",
        # Read-only work the draft asked about or claimed instead of doing; see below.
        "account_data_requested",
        "unperformed_lookup",
        "unresearched_guidance",
        # Removed deterministically before delivery if a correction leaves it in place.
        "unsupported_capability",
    }
)
# A draft that asks permission for, offers, or claims read-only work instead of doing it.
# While tool rounds remain, these are retried with tools available rather than corrected
# tool-less, so the lookup can actually happen.
INITIATIVE_VIOLATION_CODES = frozenset(
    {
        "account_data_requested",
        "unnecessary_read_only_permission",
        "unperformed_lookup",
        "unresearched_guidance",
    }
)
# Claims that are unsupported, unsafe, or conflict with explicit constraints. These are
# always blocked. Unlisted codes are treated as this group so new checks fail closed.
GROUNDING_SAFETY_VIOLATION_CODES = frozenset(
    {
        "active_quest_only",
        "activity_mode_constraint",
        "build_constraint_conflict",
        "character_scope",
        "excluded_candidate_recommended",
        "chill_constraint",
        "fake_build_score",
        "false_authentication_guidance",
        "goal_item_replaced",
        "group_constraint",
        "implied_bungie_write",
        "incompatible_exotic_recommendation",
        "known_time_window_exceeded",
        "locked_item_replaced",
        "misdescribed_item",
        "activity_record_conflation",
        "deprecated_armor_stat",
        "element_fragment_as_prismatic",
        "incorrect_perk_mechanic",
        "false_inventory_denial",
        "invented_duration",
        "unverified_current_source",
        "unsupported_range_claim",
        "unsupported_replacement",
        "unsupported_solo_claim",
        "unsupported_stat_effect",
        "unsupported_exotic_replacement",
        "unsupported_loot_claim",
        "near_completion_only",
        "raw_build_identifier",
        "rejected_candidate_recommended",
        # Re-asking a preference the user already stated ignores explicit constraints.
        "reasked_known_activity_mode",
        "reasked_known_character",
        "reasked_known_duration",
        "reasked_known_fireteam",
        "reasked_known_goal",
        "reasked_known_intensity",
        "reasked_known_time",
        "solo_constraint",
        "ungrounded_current_build_claim",
        "unknown_completion_as_known",
        "unrelated_account_evidence",
        "unrequested_timed_itinerary",
        "unsupported_ownership_claim",
        "unsupported_stat_threshold",
        "unverified_item_fact",
        "contradicted_item_fact",
        "absence_reported_as_value",
    }
)


def extreme_max_words(mode: "ResponseModeContext") -> int:
    """Length beyond which an answer in this mode is verbose, unless detail was requested."""

    if mode.focused_build:
        return 220
    return {
        "direct_fact": 160,
        "account_summary": 220,
        "recommendation": 320,
        "comparison": 320,
        "build_advice": 380,
        "walkthrough": 700,
        "troubleshooting": 700,
    }[mode.mode]


def violation_group(code: str) -> ViolationGroup:
    if code in EVIDENCE_VIOLATION_CODES:
        return "evidence"
    if code in STYLE_VIOLATION_CODES:
        return "style"
    return "grounding_safety"


def build_fallback_kind(violations: list[PlanningViolation]) -> BuildFallbackKind:
    """Choose the build fallback for violations that remain after the single correction."""

    groups = {violation_group(value.code) for value in violations}
    if "evidence" in groups:
        return "evidence"
    if "grounding_safety" in groups:
        return "grounding_safety"
    return "style_only"


# What an unverified draft was missing, in plain words, for the keep-your-build fallback.
_UNVERIFIED_BY_CODE = {
    "unsupported_ownership_claim": "that you own a piece my draft relied on",
    "locked_item_replaced": "a change that keeps the item you asked to keep",
    "goal_item_replaced": "a change that keeps the item you asked to use",
    "unsupported_stat_effect": "what a stat change would do",
    "unsupported_stat_threshold": "the stat threshold my draft cited",
    "incompatible_exotic_recommendation": "an Exotic your class can wear",
}


class ResponseModeContext(BaseModel):
    mode: ResponseMode
    target_min_words: int | None
    target_max_words: int | None
    detail_level: Literal["concise", "moderate", "detailed", "as_needed"]
    source_limit: int
    detailed_requested: bool = False
    timed_itinerary_requested: bool = False
    continuation: bool = False
    focused_build: bool = False


class BuildResponseValidationContext(BaseModel):
    is_build_request: bool = False
    requires_current_build_analysis: bool = False
    requires_owned_inventory: bool = False
    current_build_analysis_used: bool = False
    guardian_build_data_used: bool = False
    inventory_ownership_checked: bool = False
    preserve_equipped_exotics: bool = False
    owned_item_names: list[str] = Field(default_factory=list)
    locked_item_names: list[str] = Field(default_factory=list)
    equipped_exotic_weapons: list[str] = Field(default_factory=list)
    equipped_exotic_armor: list[str] = Field(default_factory=list)
    owned_exotic_weapons: list[str] = Field(default_factory=list)
    owned_exotic_armor: list[str] = Field(default_factory=list)
    # Everything the loadout results show equipped, Exotic or not.
    equipped_item_names: list[str] = Field(default_factory=list)
    # Items the player said they want to get or use; recommendations must not replace them.
    goal_item_names: list[str] = Field(default_factory=list)
    # Items proven possessed or equipped by trusted Guardian results (not effects or meta).
    ownership: OwnershipEvidence = Field(default_factory=OwnershipEvidence)
    # The character this conversation is about; "equipped" claims must be proven for it.
    active_character: str | None = None
    current_external_grounding: bool = False
    activity_mode: str = "unspecified"
    fireteam: str = "unspecified"
    normal_change_limit: int | None = 3


_ALWAYS_CURRENT = re.compile(
    r"\b(?:meta|currently strongest|current(?:ly)? best|nerfed|buffed|current artifact|"
    r"this season(?:'s)? best)\b",
    re.IGNORECASE,
)
_BEST_DPS = re.compile(r"\bbest dps\b", re.IGNORECASE)
_SUPERLATIVE_OPTION = re.compile(
    r"\b(?:best|top|highest|strongest|most powerful|optimal)\b(?:\s+\w+){0,3}?\s+"
    r"(?:weapons?|options?|builds?|dps|damage|picks?|choices?|exotics?|loadouts?|"
    r"subclass(?:es)?|perks?)\b",
    re.IGNORECASE,
)
_TIME_ANCHOR = re.compile(
    r"\b(?:right now|currently|this season|this week|this patch|at the moment|these days|"
    r"in the current|today)\b",
    re.IGNORECASE,
)
# Wording that compares things within the player's own build rather than the game at large.
_RELATIVE_TO_PLAYER = re.compile(
    r"\byour\b|\byou(?:'ve| have| own| run| use)\b|\bequipped\b", re.IGNORECASE
)
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")


def _has_current_claim(answer: str) -> bool:
    """Whether the answer asserts something about the current game state or meta.

    Ordinary build analysis ("your weakest stat", "this slot contributes the least", "your
    best DPS is the rocket you have equipped") is not a current claim. Absolute superlatives
    about options in the game are, when time-anchored ("top PvE option right now").
    """

    for sentence in _SENTENCE_BREAK.split(answer):
        if _ALWAYS_CURRENT.search(sentence):
            return True
        relative = bool(_RELATIVE_TO_PLAYER.search(sentence))
        anchored = bool(_TIME_ANCHOR.search(sentence))
        if _BEST_DPS.search(sentence) and (anchored or not relative):
            return True
        if _SUPERLATIVE_OPTION.search(sentence) and anchored and not relative:
            return True
    return False


_TRADEOFF = re.compile(
    r"\b(?:give[s]? up|gives? up|lose|loses|losing|cost|costs|trade-?off|in exchange|"
    r"at the expense|you(?:'d| would) (?:lose|miss)|downside)\b",
    re.IGNORECASE,
)


class ReadOnlyActionValidationContext(BaseModel):
    # The user explicitly asked to verify, confirm, or research something this turn.
    research_requested: bool = False
    # What actually ran this turn before the draft: any tool or web search, and Guardian data
    # (a Guardian tool, or the planning context built from Guardian data).
    lookups_performed: bool = False
    guardian_lookup_performed: bool = False
    # A search of the player's owned gear (search_inventory / find_build_alternatives) ran.
    inventory_checked: bool = False
    # The user asked for a current source, and whether web or live evidence was retrieved.
    current_source_requested: bool = False
    current_evidence_performed: bool = False
    # A recommendation turn: loot-quality claims need retrieved evidence.
    recommendation_turn: bool = False
    # Stays true for the whole turn (guidance_requested is cleared after the retry).
    guidance_asked: bool = False
    # The player's active quests, which are no evidence about an unrelated unlock goal.
    account_quest_names: list[str] = Field(default_factory=list)
    # Owned items by name: (rarity tier, weapon subtype), as Guardian reports them.
    item_facts: dict[str, tuple[str | None, str | None]] = Field(default_factory=dict)
    # Names (folded) that any tool returned this turn: the items a definition fact may be about.
    retrieved_item_names: set[str] = Field(default_factory=set)
    # Class restriction and slot each returned item carried, from this turn's tools and from the
    # player's gear in context (so an earlier turn's lookup still counts): a stated class or
    # slot must agree with it.
    item_definitions: dict[str, dict[str, set[str]]] = Field(default_factory=dict)
    # Stemmed words of every key and text value the tools returned this turn: a measure none of
    # them mention was never returned, so it cannot be stated as zero.
    evidence_terms: set[str] = Field(default_factory=set)
    # The user asked how to unlock or obtain something, and whether this turn retrieved any
    # Manifest, Guide, Live, or web knowledge (Guardian data alone cannot supply the path).
    guidance_requested: bool = False
    knowledge_lookup_performed: bool = False


# "How do I unlock X", "what's the fastest way to get Y": an acquisition or unlock path, which
# must come from retrieved knowledge rather than memory.
_GUIDANCE_REQUEST = re.compile(
    r"\b(?:how (?:do|can|would|should) (?:i|we)|how to|where (?:do|can|would) (?:i|we)|"
    r"what(?:'s| is) the (?:fastest|best|quickest|easiest) way to)\b[^.?!]{0,60}?"
    r"\b(?:unlock|get|obtain|acquire|earn|farm|complete|finish)\b",
    re.IGNORECASE,
)
# Questions that one item's Manifest definition settles: where it comes from (when the user
# asks for verification) and which of its sockets are fixed or roll randomly.
_ITEM_SOURCE_QUESTION = re.compile(
    r"\b(?:drops?|dropped|drop source|comes? from|obtain(?:ed|able)?|acquir(?:e|ed|able)|"
    r"farm(?:able|ed)?)\b",
    re.IGNORECASE,
)
_ITEM_CONFIGURATION_QUESTION = re.compile(
    r"\b(?:change|swap|switch|re-?roll|replace|get (?:a )?different|different)\b[^.?!]{0,40}"
    r"\b(?:barrels?|magazines?|mags?|perks?|traits?|frames?|intrinsics?|rolls?|grips?|scopes?|"
    r"sights?|stocks?|bowstrings?|arrows?)\b|"
    r"\b(?:barrels?|magazines?|perks?|rolls?)\b[^.?!]{0,30}\b(?:fixed|random(?:ized)?|"
    r"re-?roll(?:ed|able)?)\b",
    re.IGNORECASE,
)
# "Rose is a strong exotic sidearm": an item described as another rarity or weapon type.
_WEAPON_TYPES = (
    "linear fusion rifle|fusion rifle|hand cannon|sidearm|auto rifle|pulse rifle|scout rifle|"
    "shotgun|sniper rifle|rocket launcher|grenade launcher|submachine gun|machine gun|sword|"
    "bow|trace rifle|glaive"
)


def misdescribed_items(answer: str, facts: dict[str, tuple[str | None, str | None]]) -> list[str]:
    """Owned items the answer calls the wrong rarity or weapon type ("Rose is an Exotic").

    Only a sentence naming exactly one owned item and describing it with "is a/an ..." counts,
    so lists, comparisons, and mentions of other things never match.
    """

    found: list[str] = []
    for sentence in _SENTENCE_PARTS.split(_plain(answer))[::2]:
        names = [name for name in facts if re.search(rf"\b{re.escape(name)}\b", sentence)]
        if len(names) != 1:
            continue
        name = names[0]
        tier, subtype = facts[name]
        exotic_phrasing = re.search(
            _EXOTIC_BEFORE.format(types=_WEAPON_TYPES, name=re.escape(name)),
            sentence,
            re.IGNORECASE,
        ) or re.search(_EXOTIC_AFTER.format(name=re.escape(name)), sentence, re.IGNORECASE)
        if tier and tier != "Exotic" and exotic_phrasing:
            found.append(f"{name} is {tier}, not Exotic")
            continue
        described = re.search(
            rf"\b{re.escape(name)}\b[^.;:]{{0,20}}?\b(?:is|are|was)\s+(?:an?|the)\s+"
            r"(?P<phrase>[^.;:,!?]{1,60})",
            sentence,
        )
        if not described:
            continue
        phrase = described.group("phrase").casefold()
        words = phrase.split()[:4]
        found_kind = re.search(rf"\b({_WEAPON_TYPES})\b", phrase)
        kind = found_kind.group(1) if found_kind else ""
        if tier and "exotic" in words and "not" not in words and tier != "Exotic":
            found.append(f"{name} is {tier}, not Exotic")
        if kind and subtype and kind != subtype.casefold():
            found.append(f"{name} is a {subtype}, not a {kind}")
    return list(dict.fromkeys(found))


# Ember/Echo/Spark/Whisper/Thread of X are Solar/Void/Arc/Stasis/Strand fragments; Prismatic
# fragments are Facets. One equipped on another subclass is no evidence about Prismatic.
_ELEMENT_FRAGMENT = re.compile(r"\b(?:Ember|Echo|Spark|Whisper|Thread) of [A-Z][\w']+")
_PRISMATIC_CONTRAST = re.compile(
    r"\b(?:not|isn't|aren't|doesn't|don't|no|nothing|unrelated|different|rather|instead)\b|"
    r"\bsays? nothing\b",
    re.IGNORECASE,
)
# Loot-quality superlatives ("highest-value loot", "best chance at drops") about an activity.
_LOOT_SUPERLATIVE = re.compile(
    r"\b(?:highest|higher|best|better|top)[- ](?:value|tier|quality|chance|odds)?"
    r"[^.?!\n]{0,25}\b(?:loot|drops?|rewards?)\b|"
    r"\b(?:award|give|offer)s?\s+(?:higher|better|the best)\b[^.?!\n]{0,25}"
    r"\b(?:loot|drops?|rewards?|tier)\b|"
    r"\b(?:drop|loot|rewards?|engrams?)\b[^.?!\n]{0,40}\bmore\s+(?:often|frequently)\b|"
    r"\bmore\s+(?:often|frequently)\b[^.?!\n]{0,40}\b(?:drop|loot|rewards?|engrams?)\b|"
    r"\b(?:more|extra|bonus|additional)\s+(?:direct\s+)?(?:loot|drops?|rewards?|engrams?)\b|"
    r"\b(?:stronger|greater|richer|better|more|less)\s+(?:raw\s+)?(?:loot|drops?|rewards?)\s+"
    r"(?:potential|variety|opportunit\w+|chances?)\b|\bloot potential\b|"
    r"\bmultiply\s+(?:your\s+)?(?:loot|drops?|rewards?)\b",
    re.IGNORECASE,
)
_LOOT_HEDGED = re.compile(
    r"\b(?:unknown|not established|isn't established|unverified|not confirmed|no data|"
    r"no evidence|whether)\b|"
    # "does not establish / prove / show ...": a negated verb of knowing, not any negation.
    r"\b(?:do(?:es)?\s*not|don't|doesn't|did\s*not|didn't|can\s*not|cannot|can't|isn't|"
    r"is\s*not|aren't|are\s*not|won't)\s+(?:\w+\s+){0,2}?(?:establish\w*|show\w*|prove\w*|"
    r"confirm\w*|verify|verified|say|know|guarantee\w*|compare\w*|support\w*|indicate\w*|"
    r"tell|rank\w*)\b",
    re.IGNORECASE,
)
# "Haunted Altars of Sorrow (solo, ~30 min runs)": nothing says an activity is playable strictly
# alone; the data only shows it can be launched.
_SOLO_CLAIM = re.compile(
    r"\b(?:solo[- ](?:friendly|able|playable|clear)|playable (?:solo|alone)|(?:can|could) be "
    r"(?:played|done|run) (?:solo|alone)|by yourself)\b|\(\s*(?:strictly\s+)?solo\b|"
    r"\bsolo,\s*~?\d",
    re.IGNORECASE,
)
_SOLO_HEDGED = re.compile(
    r"\b(?:unknown|not established|unverified|whether|matchmaking)\b|"
    r"\b(?:isn't|aren't|wasn't|weren't|is not|are not)\s+established\b|"
    r"\b(?:can't|cannot|doesn't|does not|don't|do not)\s+(?:\w+\s+){0,2}?"
    r"(?:confirm|establish|show)",
    re.IGNORECASE,
)
# Kill-stacking perks only count qualifying kills with the weapon that has them, so another
# weapon cannot raise their uptime.
_KILL_PERKS = r"(?:Rampage|Kill Clip|Multikill Clip)"
_OTHER_WEAPON = (
    r"(?:pair\w*|energy (?:weapon|hand cannon|hc|smg)|second\w* weapon|other weapon|"
    r"another weapon|secondary|your (?:energy|power) weapon)"
)
_CROSS_WEAPON_PERK = re.compile(
    rf"\b{_KILL_PERKS}\b[^.?!\n]{{0,120}}\b{_OTHER_WEAPON}\b|"
    rf"\b{_OTHER_WEAPON}\b[^.?!\n]{{0,120}}\b(?:keep|maintain|sustain|string|stack|uptime)\w*"
    rf"[^.?!\n]{{0,60}}\b{_KILL_PERKS}\b|"
    rf"\b(?:keep|maintain|sustain|string|stack)\w*[^.?!\n]{{0,60}}\b{_KILL_PERKS}\b[^.?!\n]{{0,60}}"
    rf"\b{_OTHER_WEAPON}\b",
    re.IGNORECASE,
)
_PERK_OWN_WEAPON = re.compile(
    r"\b(?:only|itself|its own|same weapon|with (?:that|the) weapon|does not|doesn't|"
    r"do not|don't|won't)\b",
    re.IGNORECASE,
)
# Two weapons of one type do not automatically cover more range.
_RANGE_COVERAGE = re.compile(
    r"\b(?:hand cannon|hc)s?\b[^.?!\n]{0,100}\b(?:range|coverage|overlap)\b|"
    r"\b(?:range|coverage|overlap)\b[^.?!\n]{0,100}\b(?:hand cannon|hc)s?\b",
    re.IGNORECASE,
)
_RANGE_PAIRING = re.compile(
    r"\b(?:pair\w*|energy (?:hand cannon|hc)|two hand cannons|second hand cannon|"
    r"another hand cannon|reduces? (?:awkward )?(?:range )?overlap)\b",
    re.IGNORECASE,
)
# A line that only makes sense after the sentence it explained was removed.
_ORPHAN_START = re.compile(
    r"^\s*(?:[-*•]\s|\d{1,2}[.)]\s)|^\W*(?:why|what to do|how|note|backup)\b[^.\n]{0,25}:",
    re.IGNORECASE,
)
LOOT_LIMIT_NOTE = "Loot quality and solo clear time aren't established by your data."
# Used when only a loot ranking was removed: the note names just the claim that was cut.
LOOT_ONLY_LIMIT_NOTE = "Loot quality isn't established by your data."
# A stated duration ("~15-30 minutes", "takes about 25 minutes") about an activity.
_DURATION_CLAIM = re.compile(
    r"(?:\b(?:in|within|under|about|around|roughly|approximately|takes?|take)\s+)?(?:~\s*)?"
    r"(\d{1,3})(?:\s*[\u2013-]\s*(\d{1,3}))?\s*(?:minutes?|mins?)\b",
    re.IGNORECASE,
)
# Ways a draft calls one of the player's owned weapons Exotic: "Exotic hand cannon "Rose"",
# "Exotic Rose", or "Rose (the Exotic ...)".
_EXOTIC_BEFORE = r"\bexotic\s+(?:(?:{types}|weapon)(?:\s*/\s*\w+)?\s+)?[\"'\u201c]?{name}\b"
_EXOTIC_AFTER = r"\b{name}\b[\"'\u201d,]*\s*[(,\u2014-]\s*(?:the\s+|an?\s+)?exotic\b"


def _fragment_misread(answer: str) -> bool:
    """A sentence that presents an element fragment as a Prismatic one."""

    return any(
        _ELEMENT_FRAGMENT.search(sentence)
        and re.search(r"\bPrismatic\b", sentence)
        and not _PRISMATIC_CONTRAST.search(sentence)
        for sentence in _SENTENCE_PARTS.split(_plain(answer))[::2]
    )


_RESEARCH_REQUEST = re.compile(
    r"\b(?:verify|confirm|double[- ]check|fact[- ]check|look (?:it |that |this )?up|research|"
    r"search (?:for|online|the web)|cite|citations?|with (?:a |an )?sources?|"
    r"(?:current|recent|official|reliable|up[- ]to[- ]date) sources?)\b",
    re.IGNORECASE,
)
_LOOKUP_VERBS = (
    r"(?:check|scan|fetch|retrieve|look(?:\s+\w+)?\s+up|look into|search|inspect|review|"
    r"analy[sz]e|compare|verify|confirm|research|pull up|dig into|find out|pick|show|find|"
    r"identify)"
)
# Armor 3.0 stats are Weapons, Health, Class, Grenade, Melee, and Super. These pre-3.0 names no
# longer exist, so recommending them as armor stats is wrong whatever the tools returned.
_OLD_STATS = "Recovery|Resilience|Discipline|Intellect|Strength|Mobility"
_DEPRECATED_STAT = re.compile(
    rf"\b(?:{_OLD_STATS})\b\s*(?:/\s*(?:{_OLD_STATS})\s*)?(?:stats?\b|stat bars?\b|points?\b|\d)|"
    rf"(?-i:\b(?:{_OLD_STATS})\b)",
    re.IGNORECASE,
)
_ARMOR_STAT_CONTEXT = re.compile(
    r"\b(?:stat|stats|bar|bars|armor|piece|chest|legs|helmet|gauntlets?|class item|higher|more|"
    r"raise|boost|increase|shift\w*|stack\w*|target\w*|invest\w*|low|lower|lack\w*|swap|"
    r"replace|equip)\b",
    re.IGNORECASE,
)
# "Recovery not listed but implied low": a stat the tools did not return is unknown, not low.
_INFERRED_LOW_STAT = re.compile(
    r"\b(?:not (?:listed|shown|returned|provided)|isn't (?:listed|shown)|missing|unlisted)\b"
    r"[^.?!\n]{0,60}\b(?:implied|implies|likely|probably|presumably|assum\w+)\b[^.?!\n]{0,30}"
    r"\b(?:low|lacking|weak|poor)\b|\b(?:implied|likely|probably|presumably) low\b",
    re.IGNORECASE,
)
_STAT_EFFECT = re.compile(
    r"(?-i:\b(?:Weapons|Health|Class|Grenade|Melee|Super)\b)\s+stat\b[^.?!\n]{0,80}\b(?:faster|"
    r"recharge\w*|regenerat\w*|cooldowns?|reduc\w*|increas\w*|boost\w*|improv\w*|shorter)\b|"
    r"\b(?:faster|recharge\w*|regenerat\w*|cooldowns?|reduc\w*|boost\w*|improv\w*)\b"
    r"[^.?!\n]{0,60}\b(?:Weapons|Health|Class|Grenade|Melee|Super)\s+stat\b",
    re.IGNORECASE,
)


# The tools can search the player's gear, so "I can't scan your collection" is false.
def claims_cross_weapon_perk(answer: str) -> bool:
    """Says another weapon keeps Rampage (or a similar kill perk) up."""

    return any(
        _CROSS_WEAPON_PERK.search(sentence) and not _PERK_OWN_WEAPON.search(sentence)
        for sentence in re.split(r"(?<=[.!?])[ \t]+|\n+", _plain(answer))
    )


# A sentence that itself says the two activities are different is the correction, not the error.
_DISTINCT_ACTIVITY = re.compile(
    r"\b(?:different|distinct|separate|not the same|only the (?:regular|normal|original)|"
    r"(?:does not|doesn't|do not|don't|won't|will not)\s+(?:count|advance|progress|apply))\b",
    re.IGNORECASE,
)
# Words that name a difficulty or playlist of the same activity, not a different one.
_SAME_ACTIVITY_WORDS = frozenset(
    [
        "standard",
        "master",
        "normal",
        "hard",
        "legend",
        "customize",
        "matchmade",
        "playlist",
        "contest",
        "prestige",
        "expert",
        "heroic",
        "adept",
        "grandmaster",
    ]
)


def mentions_deprecated_armor_stat(answer: str) -> bool:
    """A pre-Armor-3.0 stat used as an armor stat, or a stat that was not returned called low."""

    text = _plain(answer)
    return bool(
        any(
            _DEPRECATED_STAT.search(sentence) and _ARMOR_STAT_CONTEXT.search(sentence)
            for sentence in re.split(r"(?<=[.!?])[ \t]+|\n+", text)
        )
        or _INFERRED_LOW_STAT.search(text)
    )


def denies_inventory_search(answer: str) -> bool:
    """Says the player's owned gear cannot be searched, which the inventory tools contradict."""

    return bool(_INVENTORY_DENIAL.search(_plain(answer)))


_INVENTORY_DENIAL = re.compile(
    r"\b(?:don't have access to|do not have access to|can't|cannot|am unable to|unable to|"
    r"not able to)\s+(?:\w+\s+){0,3}?(?:scan|search|check|inspect|see|view|read|look through|"
    r"browse|list)\s+(?:through\s+|into\s+)?(?:your|the)\s+(?:\w+\s+){0,2}?"
    r"(?:inventory|collection|vault|gear|armor|items|stash)\b",
    re.IGNORECASE,
)
# Asking permission for, or offering a choice between, read-only lookups. Those are always
# authorized, so the draft should run the lookup (or drop the offer) instead of asking.
_READ_ONLY_OFFER = re.compile(
    r"\b(?:want me to|would you like me to|should i|shall i|can i|may i|is it (?:ok|okay)|"
    r"okay if i|ok if i|let me know if you(?:'d| would)? (?:like|want) me to)\b"
    rf"[^?.!\n]{{0,160}}\b{_LOOKUP_VERBS}\b|"
    # Also the bulleted form: "If you want, I can:\n- Look up ...".
    rf"\bif you(?:'d| would)? (?:like|want)(?: me to)?,? i (?:can|could|will|'ll)"
    rf"[:\s\-*•]+(?:also\s+)?{_LOOKUP_VERBS}\b|"
    rf"\bi (?:can|could) (?:also\s+)?{_LOOKUP_VERBS}\b[^?\n]{{0,160}}"
    r"(?:\bwhich (?:do you|would you|one|option)\b|\bprefer\b|\bif you(?:'d| would)? "
    r"(?:like|want)\b)",
    re.IGNORECASE,
)
# "Which quest should I look up?" asks for a missing name, which is allowed, not permission.
_WHICH_TARGET_QUESTION = re.compile(
    r"(?:^|(?<=[.!?\n]))[ \t]*(?:which|what)\b[^.?!\n]{0,80}\b(?:should|shall|can) i\b[^?\n]*\?",
    re.IGNORECASE,
)
# A first-person claim that a lookup happened; false when nothing ran this turn.
_LOOKUP_CLAIM = re.compile(
    r"\b(?:i|i've|i have|we've|we have)\s+(?:just\s+|already\s+)?(?:checked|searched|looked up|"
    r"looked into|verified|confirmed|researched|scanned|reviewed|pulled up)\b|"
    r"\b(?:after|from) (?:checking|searching|looking up|verifying)\b",
    re.IGNORECASE,
)
# Asking the player to supply account data that Guardian tools may already hold.
_ACCOUNT_DATA_REQUEST = re.compile(
    r"\b(?:tell me|let me know|list|type(?: out| in)?|share|send|paste|give me|post)\b"
    r"[^.?!\n]{0,60}\b(?:fragments?|aspects?|loadout|inventory|vault|gear|weapons?|armor|"
    r"exotics?|perks?|rolls?|stats?|quests?|triumphs?|titles?|seals?|progress|unlocks?|"
    r"unlocked)\b[^.?!\n]{0,40}\byou\b|"
    r"\b(?:tell me|let me know|type(?: out| in)?|share|send|paste|give me|post)\b"
    r"[^.?!\n]{0,20}\byour\b[^.?!\n]{0,30}\b(?:fragments?|aspects?|loadout|inventory|vault|"
    r"gear|weapons?|armor|exotics?|perks?|rolls?|stats?|quests?|triumphs?|progress|unlocks?)\b",
    re.IGNORECASE,
)
# The chat is text-only and the application is read-only: no images or files, and no
# changes to the player's items. Offers of either describe abilities the app lacks.
_UNSUPPORTED_CAPABILITY = re.compile(
    # Sending Copilot an image or file ("upload a screenshot", "screenshot it and I'll check").
    # Screenshots the player takes for themselves in game are fine.
    r"\b(?:upload|attach|send|share|post|paste|show)\b[^.?!\n]{0,40}\b(?:screen ?(?:shots?|grabs?|"
    r"caps?)|images?|photos?|pictures?|pics?|files?|clips?|videos?)\b|"
    r"\bscreen ?(?:shots?|grabs?|caps?)\b[^.?!\n]{0,60}\b(?:and|so|then)\s+i(?:'ll| will| can| "
    r"could)\b|"
    # Offering to inspect which fragments or aspects are unlocked or missing: the build tools
    # show only what is equipped, so that is not something Copilot can look at.
    r"\b(?:i can|i could|i'll|i will|let me|want me to|shall i|should i)\s+(?:\w+\s+){0,3}?"
    r"(?:check|inspect|see|look at|read|pull|fetch|list|produce|report|run|scan)\b"
    r"(?:[^.?!\n]{0,80}\b(?:fragments?|facets?|aspects?)\b[^.?!\n]{0,60}\b(?:owner\w*|owned|own|"
    r"unlock\w*|missing|lack\w*|locked|haven't)\b|"
    r"[^.?!\n]{0,80}\b(?:owned|unlocked|missing|locked)\b[^.?!\n]{0,40}\b(?:fragments?|facets?|"
    r"aspects?)\b)|"
    # Offering to work out Crucible results from matches: no tool returns a match history.
    r"\b(?:i can|i could|i'll|i will|let me|want me to|shall i|should i)\s+(?:\w+\s+){0,3}?"
    r"(?:fetch|pull|compute|calculate|work out|check|get|look up|run|scan|estimate)\b"
    r"[^.?!\n]{0,80}\b(?:k/?d|kill[- ]?death|match history|match results|(?:crucible|pvp) "
    r"(?:matches|games|history))\b|"
    r"\b(?:i can|i could|i'll|i will|let me|want me to|shall i)\s+(?:\w+\s+){0,3}?(?:check|"
    r"fetch|pull|produce|build|get)\b[^.?!\n]{0,40}\b(?:fragments?|facets?|prismatic)\s+"
    r"checklist\b|"
    # The same offer as a bullet: "- Fetch an exact checklist of every fragment you're missing".
    r"^[\s*•-]*(?:fetch|produce|pull up|check|report|list)\b[^.?!\n]{0,60}\b(?:checklist|"
    r"fragments?|facets?)\b[^.?!\n]{0,50}\b(?:missing|unlocked|owned|lack)\b|"
    r"\b(?:want me to|would you like me to|shall i|should i|i can|i could|i'll|i will|"
    r"let me)\s+(?:also\s+|go ahead and\s+|quickly\s+)?(?:move(?! on\b)|transfer|vault|"
    r"pull(?! up\b)|equip|unequip|swap(?! (?:over )?to\b)|infuse|dismantle|delete|"
    r"lock(?! in\b)|unlock|reroll|masterwork|launch)\b",
    re.IGNORECASE,
)
# A sentence that says the ability is missing, rather than offering it.
_CAPABILITY_DENIAL = re.compile(
    r"\b(?:i|we|copilot|this (?:chat|app))\s+(?:can'?t|cannot|can not|won'?t|am unable to|"
    r"is unable to|am not able to|isn'?t able to|don'?t|doesn'?t)\b|"
    r"\byou\s+(?:can'?t|cannot|can not)\b|"
    r"\b(?:no way to|(?:is|are)n'?t supported|not supported|read[- ]only)\b",
    re.IGNORECASE,
)
# A completed change to the player's items, which no tool can make.
_IMPLIED_WRITE = re.compile(
    r"(?<!do )(?<!did )\b(?:i(?:'ve| have)?|we(?:'ve| have)?)\s+(?:just\s+|now\s+|already\s+)?"
    r"(?:moved|transferred|vaulted|equipped|unequipped|infused|dismantled|deleted|"
    r"locked(?! in\b)|unlocked|swapped|claimed)\b",
    re.IGNORECASE,
)
_SENTENCE_PARTS = re.compile(r"((?<=[.!?])[ \t]+|\n+)")
_CLAUSE_BREAK = re.compile(r"[,;:—]|\s-\s|\b(?:so|but|and|then|though|however)\b", re.I)
_TRAILING_CONNECTOR = re.compile(
    r"(?:[\s,;:—-]|\b(?:so|but|and|then|though|however)\b)+$", re.IGNORECASE
)
# "Do you want me to check...", "If you want me to look up...": words allowed before an offer
# that opens its clause. Anything else means the clause also carries information.
_LIST_MARKER = r"(?:\s*(?:\d{1,2}[.)]|[-*•]))?"
_OFFER_LEAD_IN = re.compile(
    _LIST_MARKER + r"(?:\W*\b(?:do|you|if|so|also|and|or|just|then|but)\b)*\W*", re.I
)
# What may precede an offer without carrying information ("If you want, I can..."): a clause
# made only of these words is dropped with the offer instead of being kept as a sentence.
_OFFER_PREAMBLE = re.compile(
    _LIST_MARKER + r"(?:\W*\b(?:if|you|you'd|would|want|like|prefer|need|so|also|and|or|just|"
    r"then|but|do|sure|okay|ok)\b)*\W*",
    re.IGNORECASE,
)
# "If you want, I'll list ..." / "Tell me 'story' or 'triumph' and I'll commit ...": a closing
# hand-back that adds nothing the answer needed.
_IF_YOU_WANT = re.compile(
    r"\W*(?:if you(?:'d| would)? (?:like|want)|want me to)\b[^.?!\n]*"
    r"\bI(?:'ll| will| can| could)\b",
    re.IGNORECASE,
)
_TELL_ME_OR = re.compile(
    r"\W*(?:tell me|say|reply|let me know)\b[^.?!\n]{0,60}\bor\b[^.?!\n]{0,40}\bI(?:'ll| will)\b",
    re.IGNORECASE,
)
_BULLET = re.compile(r"\s*(?:[-*•]|\d{1,2}[.)])\s+")
# "Pick one (and say PvE or PvP)": still a question, not an answer.
_SOLICITATION = re.compile(
    r"\W*(?:pick|tell me|say|let me know|choose|reply|answer|specify)\b", re.IGNORECASE
)
_OFFER_HEADER = re.compile(
    r"[^\n]{0,40}\b(?:i can|if you (?:want|like|'d like)|want me to|would you like)\b[^\n]{0,40}"
    r":?\s*$",
    re.IGNORECASE,
)
_BARE_LIST_MARKER = re.compile(r"\s*(?:\d{1,2}[.)]|[-*•])\s*")


def _sentence_parts(text: str) -> list[str]:
    """Sentences alternating with their separators; "6." stays with the item it numbers."""

    parts = _SENTENCE_PARTS.split(text)
    index = 0
    while index + 2 < len(parts):
        if _BARE_LIST_MARKER.fullmatch(parts[index]):
            parts[index : index + 3] = ["".join(parts[index : index + 3])]
        else:
            index += 2
    return parts


def _kept_before(text: str, start: int) -> str | None:
    """The informative part of a sentence before its offending clause, or None."""

    kept = _TRAILING_CONNECTOR.sub("", text[:start]).strip()
    if len(re.findall(r"\w+", kept)) < 3 or _OFFER_PREAMBLE.fullmatch(_plain(kept)):
        return None
    return kept if kept.endswith((".", "!", "?")) else f"{kept}."


# Closing hand-backs: a bare "Which do you want?" after an offer, or, at the end of a
# recommendation, a request to pick between the options it already ranked.
_SHORT_HANDBACK = re.compile(
    r"\W*(?:so,?\s+)?(?:which (?:one |option )?(?:do|would) you (?:want|like|prefer)"
    r"(?: me to do)?|want (?:me to|that)|interested|sound good)\W*",
    re.IGNORECASE,
)
# A closing question that offers more instead of finishing: "Want a build tuned to...?"
_CLOSING_OFFER = re.compile(
    r"\W*(?:so,?\s+)?(?:want|would you like|do you want|do you prefer|prefer|shall i|should i)\b",
    re.IGNORECASE,
)
_PICK_ONE = re.compile(
    r"\b(?:which|what)\b[^.?!\n]{0,40}\b(?:do|would) you (?:want|like|prefer)\b[^?\n]*\bor\b"
    r"[^?\n]*\?|\bwhich (?:one|of (?:these|them|those))\b[^?\n]*\?|\bpick one\b",
    re.IGNORECASE,
)


# Copilot declining a change it cannot make ("I can't move items"): the answer to an action
# request, which needs no lookup to be complete.
_DECLINES_CHANGE = re.compile(
    r"\b(?:i|copilot)\s+(?:can't|cannot|can not|am unable to|am not able to|isn't able to)\s+"
    r"(?:\w+\s+){0,3}?(?:move|transfer|vault|equip|change|swap|make|modify|edit|lock|unlock|"
    r"dismantle|infuse|reroll|launch)\b",
    re.IGNORECASE,
)


def _plain(text: str) -> str:
    """Typographic apostrophes as ASCII, so "can’t" reads like "can't". Same length."""

    return text.replace("’", "'").replace("‑", "-")


def declines_change(answer: str) -> bool:
    return bool(_DECLINES_CHANGE.search(_plain(answer)))


def is_research_request(message: str) -> bool:
    """Whether the user explicitly asked to verify, confirm, look up, or source something."""

    return bool(_RESEARCH_REQUEST.search(message))


# "How do I finish that quest?": no named target yet, so a clarification comes before research.
_VAGUE_TARGET = re.compile(
    r"\b(?:that|this|those|these)\s+(?:quests?|ones?|items?|activit(?:y|ies)|missions?|"
    r"weapons?|exotics?|things?|steps?)\b|"
    r"\b(?:unlock|get|obtain|acquire|earn|farm|complete|finish)\s+(?:it|that|this|them)\b",
    re.IGNORECASE,
)


def is_guidance_request(message: str) -> bool:
    """Whether the user asked how to unlock, obtain, or complete something they named."""

    return bool(_GUIDANCE_REQUEST.search(message) and not _VAGUE_TARGET.search(message))


def item_definition_lookup(message: str) -> str | None:
    """The Manifest tool that settles a question about one item's definition, if any.

    "Can you change the barrel on my Sunshot?" is answered by the item's socket configuration
    (fixed or randomized); "can you verify X drops from Y?" starts from the item's sources.
    """

    if _ITEM_CONFIGURATION_QUESTION.search(message):
        return "get_item_details"
    if is_research_request(message) and _ITEM_SOURCE_QUESTION.search(message):
        return "find_item_source"
    return None


def _clause_start(sentence: str, position: int) -> int:
    breaks = [value.end() for value in _CLAUSE_BREAK.finditer(sentence, 0, position)]
    return breaks[-1] if breaks else 0


def _capability_offer_start(sentence: str) -> int | None:
    """Where the clause offering a missing ability begins; a denial in that clause excuses it.

    "I can't see your unlocks, so send me a screenshot" is still an offer: the denial belongs
    to the first clause, not to the request.
    """

    for match in _UNSUPPORTED_CAPABILITY.finditer(sentence):
        start = _clause_start(sentence, match.start())
        if not _CAPABILITY_DENIAL.search(sentence[start : match.end()]):
            return start
    return None


def _misstated_capability_start(sentence: str) -> int | None:
    """Clause start of an unsupported offer or of a claimed change to the player's items."""

    starts = [_capability_offer_start(sentence)]
    starts.extend(
        _clause_start(sentence, value.start()) for value in _IMPLIED_WRITE.finditer(sentence)
    )
    found = [value for value in starts if value is not None]
    return min(found) if found else None


UNSUPPORTED_CAPABILITY_FALLBACK = (
    "I can't do that from here: Copilot reads your Guardian data through text chat only and "
    "can't change your items. I can walk you through doing it in game."
)


def _is_loot_claim(sentence: str) -> bool:
    """A statement ranking loot; a question ("Want more loot?") and a stated limit are not."""

    return bool(
        _LOOT_SUPERLATIVE.search(sentence)
        and not _LOOT_HEDGED.search(sentence)
        and not sentence.rstrip().endswith("?")
    )


def _qualify_loot(sentence: str) -> str | None:
    """The sentence without its loot ranking, or None when nothing useful is left.

    A parenthetical ("(highest-value loot)") or a trailing clause (", since Nightfalls give
    higher-tier rewards") is cut out so a headline recommendation survives; a sentence that
    is only the claim is dropped.
    """

    claim = _LOOT_SUPERLATIVE.search(sentence)
    if claim is None:
        return sentence
    for span in re.finditer(r"\s*\([^()]*\)", sentence):
        if span.start() <= claim.start() and claim.end() <= span.end():
            cut = sentence[: span.start()] + sentence[span.end() :]
            return cut if len(re.findall(r"\w+", cut)) >= 3 else None
    clause = [
        value.start()
        for value in re.finditer(
            r"\s[—–-]\s|,\s(?:which|so|since|because|as)\s|\s(?:because|since)\s",
            sentence,
        )
        if value.start() < claim.start()
    ]
    if clause:
        head = sentence[: clause[-1]].rstrip(" ,;:—-")
        if len(re.findall(r"\w+", head)) >= 5:
            return head if head.endswith((".", "!", "?")) else f"{head}."
    # "Best loot in your constraints: Haunted Altars of Sorrow": the claim is only the label,
    # so keep the pick that follows it.
    labelled = re.match(r"^[^:]{0,60}:\s*(?P<pick>\S.*)$", sentence)
    if labelled and labelled.start("pick") > claim.start():
        return f"Recommended: {labelled.group('pick')}"
    return None


def _without_solo_claim(sentence: str) -> str | None:
    """The sentence without a "(solo, ...)" parenthetical or a trailing solo clause."""

    cut = re.sub(r"\s*\([^()]*\bsolo\b[^()]*\)", "", sentence)
    if cut != sentence:
        return cut
    cut = re.sub(
        r",?\s*(?:and\s+)?(?:is\s+)?(?:solo[- ](?:friendly|able|playable|clear)|playable "
        r"(?:solo|alone)|by yourself)[^.?!\n]*",
        "",
        sentence,
        flags=re.IGNORECASE,
    )
    cut = cut.rstrip(" ,;:")
    if len(re.findall(r"\w+", cut)) < 3:
        return None
    return cut if cut.endswith((".", "!", "?")) else f"{cut}."


def _rejoin(parts: list[str], replaced: list[str | None]) -> str:
    """Rebuild sentences, dropping the None ones and keeping the strongest separator that
    stood between two kept sentences so a dropped line never merges its neighbours."""

    kept: list[str] = []
    pending: list[str] = []
    for position, sentence in enumerate(replaced):
        if position:
            pending.append(parts[position * 2 - 1])
        if sentence is None or not sentence.strip():
            continue
        if kept:
            breaks = [value for value in pending if "\n" in value]
            kept.append(max(breaks, key=len) if breaks else pending[-1])
        kept.append(sentence)
        pending = []
    return "".join(kept).strip()


def _without_misstated_capability(sentence: str) -> str | None:
    """The sentence up to the offending clause, or None when nothing informative precedes it."""

    start = _misstated_capability_start(_plain(sentence))
    if start is None:
        return sentence
    return _kept_before(sentence, start)


# A revised answer must read as if it had been right the first time. These are the ways a draft
# narrates the repair instead: a "Quick correction:" label, or a stand-alone sentence that only
# talks about the correction.
_CORRECTION_LABEL = re.compile(
    r"^(?P<lead>\s*(?:[-*•]\s+)?)(?:\*\*)?(?:(?:quick|small|brief|minor|one)\s+)?"
    r"(?:correction|clarification|clarifier|heads[- ]up)(?:\*\*)?\s*[:—–-]\s*(?:\*\*)?",
    re.IGNORECASE,
)
_CORRECTION_ONLY_SENTENCE = re.compile(
    r"^\s*(?:[-*•]\s+)?(?:just\s+|so\s+)?(?:(?:please\s+)?(?:don'?t|do not)\s+mix\s+"
    r"(?:\w+\s+){0,2}up|to\s+(?:clarify|correct\s+(?:that|myself|this))|"
    r"(?:sorry|apologies|my\s+mistake|I\s+misspoke|correcting\s+myself)|"
    r"(?:to\s+)?keep\s+(?:them|those|these|the\s+two)\s+straight)\b[^.!?\n]{0,60}[.!?]?\s*$",
    re.IGNORECASE,
)


_FILLER_OPENER = re.compile(
    r"^\s*(?:based on (?:the|your) (?:available )?data|short answer|primary recommendation)"
    r"\s*[,:—–-]\s*",
    re.IGNORECASE,
)
_MONTH = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
    r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
_CITED_FACT = re.compile(
    rf"https?://[^\s)\]>]+|\b\d{{4}}-\d{{2}}-\d{{2}}\b|"
    rf"\b{_MONTH}\.?\s+\d{{1,2}}(?:,\s*\d{{4}})?\b",
    re.IGNORECASE,
)


def strip_filler_opener(answer: str) -> str | None:
    """The answer without a filler lead-in ("Based on the available data, ..."), or None.

    The sentence after the label already is the answer, so nothing needs rewriting; only the
    label goes. Returns None when there is no such opener or nothing would remain.
    """

    match = _FILLER_OPENER.match(_plain(answer))
    if match is None:
        return None
    rest = answer[match.end() :]
    if len(re.findall(r"\w+", rest)) < 3:
        return None
    return rest[:1].upper() + rest[1:]


def loses_cited_facts(draft: str, rewritten: str) -> bool:
    """Whether a rewrite drops a source link or a date the draft stated.

    A rewrite meant only to reorder or tighten an answer must keep what the answer cites; one
    that drops them has replaced the evidence with something weaker.
    """

    kept = rewritten.casefold()
    return any(
        value.group(0).rstrip(".,;").casefold() not in kept for value in _CITED_FACT.finditer(draft)
    )


def remove_correction_narration(answer: str) -> str:
    """The answer without lines that narrate a correction instead of stating the fact.

    A "Quick correction:" label is dropped and the fact after it kept; a sentence that only
    talks about the correction ("Don't mix them up.") is dropped. Everything else is untouched,
    and an answer that would be left with no content is returned as it was.
    """

    parts = _sentence_parts(answer)
    replaced: list[str | None] = []
    changed = False
    for index in range(0, len(parts), 2):
        sentence = parts[index]
        plain = _plain(sentence)
        if _CORRECTION_ONLY_SENTENCE.match(plain):
            replaced.append(None)
            changed = True
            continue
        label = _CORRECTION_LABEL.match(plain)
        if label:
            rest = sentence[label.end() :]
            if len(re.findall(r"\w+", rest)) >= 3:
                rest = rest[:1].upper() + rest[1:]
                replaced.append(label.group("lead") + rest)
            else:
                replaced.append(None)
            changed = True
            continue
        replaced.append(sentence)
    if not changed:
        return answer
    text = _rejoin(parts, replaced).strip()
    return text if len(re.findall(r"\w+", text)) >= 3 else answer


def remove_unsupported_capability_offers(answer: str) -> str:
    """Remove offers of images, uploads, or item changes, and claims that an item was changed.

    The last line of defense after any correction. Only the offending clause is removed: a
    sentence keeps any information before it ("Rose is in your vault, and I can equip it" keeps
    "Rose is in your vault."), honest denials stay, and everything else is returned unchanged.
    """

    parts = _sentence_parts(answer)
    if all(_misstated_capability_start(_plain(value)) is None for value in parts[::2]):
        return answer
    # Between two kept sentences, keep the strongest separator that stood between them, so a
    # dropped sentence never merges a following line or bullet into the previous one.
    kept: list[str] = []
    pending: list[str] = []
    for index in range(0, len(parts), 2):
        if index:
            pending.append(parts[index - 1])
        sentence = _without_misstated_capability(parts[index])
        if sentence is None:
            continue
        if kept:
            breaks = [value for value in pending if "\n" in value]
            kept.append(max(breaks, key=len) if breaks else pending[-1])
        kept.append(sentence)
        pending = []
    cleaned = "".join(kept).strip()
    # "Done!" alone after removing "I've moved Rose to your vault." would still mislead.
    return cleaned if len(re.findall(r"\w+", cleaned)) >= 3 else UNSUPPORTED_CAPABILITY_FALLBACK


_DETAILED = re.compile(
    r"\b(?:detailed|in[- ]depth|comprehensive|thorough|step[- ]by[- ]step)\b",
    re.IGNORECASE,
)
_TIMED_PLAN = re.compile(
    r"\b(?:minute[- ]by[- ]minute|timed plan|timeboxed plan|schedule|itinerary)\b|"
    r"\bstep[- ]by[- ]step\b.{0,40}\b(?:minutes?|hours?)\b|"
    r"\b(?:minutes?|hours?)\b.{0,40}\bstep[- ]by[- ]step\b",
    re.IGNORECASE,
)
_COMPARISON = re.compile(
    r"\b(?:compare|versus|vs\.?)\b|\b(?:should|would) i\b.{0,180}\bor\b|"
    r"\bgiven\b.{0,180}\bor\b|\bwhich\b.{0,100}\b(?:better|next)\b",
    re.IGNORECASE,
)
_TROUBLESHOOTING = re.compile(
    r"\b(?:error|bug|broken|not working|won't work|can'?t connect|cannot connect|"
    r"troubleshoot|login issue|oauth issue|failed to)\b",
    re.IGNORECASE,
)
_WALKTHROUGH = re.compile(
    r"\b(?:walkthrough|guide|step[- ]by[- ]step|guide me through|how (?:do|can) i|"
    r"how to|quest steps?|encounter mechanics?)\b",
    re.IGNORECASE,
)
_DIRECT_ACCOUNT = re.compile(
    r"\b(?:equipped|currently using|subclass am i using|do i own|what do i own|"
    r"what subclass.{0,40}using|have i (?:started|completed|finished)|how far am i|"
    r"what quests? (?:are|is) active)\b",
    re.IGNORECASE,
)
_ACCOUNT_SUMMARY = re.compile(
    r"\b(?:summarize|summary|overview)\b.{0,40}\b(?:account|guardian|character|inventory)\b|"
    r"\bhow (?:is|does) my (?:account|guardian|character)\b|"
    r"\bwhat (?:quests?|activities) (?:do i have|are available)\b",
    re.IGNORECASE,
)
_EXTERNAL_KNOWLEDGE_REQUEST = re.compile(
    r"\b(?:how (?:do|can) i get|where (?:do|can) i get|what does\b|"
    r"unlocks?|rewards?|perks?|traits?|patch(?:es| notes)?|meta|drop source|"
    r"is (?:it|that|this) good|better|upgrade|replace)\b",
    re.IGNORECASE,
)


def classify_response_mode(
    message: str, *, session_planning: bool, build_request: bool | None = None
) -> ResponseMode:
    if _TROUBLESHOOTING.search(message):
        return "troubleshooting"
    if build_request is True or (build_request is None and is_explicit_build_request(message)):
        return "build_advice"
    if _COMPARISON.search(message):
        return "comparison"
    if _WALKTHROUGH.search(message):
        return "walkthrough"
    if _DIRECT_ACCOUNT.search(message):
        return "direct_fact"
    if _ACCOUNT_SUMMARY.search(message):
        return "account_summary"
    if session_planning:
        return "recommendation"
    return "direct_fact"


def is_account_fact_only(message: str) -> bool:
    """Return true only for personal facts fully answerable by Guardian tools."""

    return bool(
        (_DIRECT_ACCOUNT.search(message) or _ACCOUNT_SUMMARY.search(message))
        and not _EXTERNAL_KNOWLEDGE_REQUEST.search(message)
    )


def response_mode_context(
    message: str,
    *,
    session_planning: bool,
    continuation: bool,
    build_request: bool | None = None,
    focused_build: bool = False,
) -> ResponseModeContext:
    mode = classify_response_mode(
        message,
        session_planning=session_planning,
        build_request=build_request,
    )
    detailed = bool(_DETAILED.search(message))
    timed = bool(_TIMED_PLAN.search(message))
    targets: dict[ResponseMode, tuple[int | None, int | None, str, int]] = {
        "direct_fact": (20, 80, "concise", 4),
        "account_summary": (40, 120, "concise", 4),
        "recommendation": (80, 180, "moderate", 4),
        "comparison": (80, 180, "moderate", 4),
        "build_advice": (100, 220, "detailed", 6),
        "walkthrough": (None, None, "as_needed", 8),
        "troubleshooting": (None, None, "as_needed", 8),
    }
    minimum, maximum, detail_level, source_limit = targets[mode]
    if mode == "build_advice" and focused_build and not detailed:
        minimum, maximum, detail_level, source_limit = (50, 120, "concise", 4)
    if detailed:
        detail_level = "detailed"
        source_limit = 8
    return ResponseModeContext(
        mode=mode,
        target_min_words=minimum,
        target_max_words=maximum,
        detail_level=detail_level,  # type: ignore[arg-type]
        source_limit=source_limit,
        detailed_requested=detailed,
        timed_itinerary_requested=timed,
        continuation=continuation,
        focused_build=bool(mode == "build_advice" and focused_build and not detailed),
    )


def response_mode_instruction(context: ResponseModeContext) -> str:
    if context.target_min_words is None:
        range_text = "Use enough detail to solve the request; no fixed word target applies."
    else:
        range_text = (
            f"A typical useful answer is about {context.target_min_words}-"
            f"{context.target_max_words} words, but this is guidance, not a truncation rule."
        )
    shapes = {
        "direct_fact": (
            "Answer in the first sentence. Include only the requested fact and, if useful, one "
            "brief clarifier. Avoid headings, caveat lists, closing offers, and unrelated account "
            "metadata."
        ),
        "account_summary": (
            "Lead with the useful account summary, then include only the few facts relevant to "
            "the request. Avoid raw metadata and exhaustive dumps."
        ),
        "recommendation": (
            "Lead with one clear primary route, give a few concrete actions and a brief reason, "
            "then optionally one compatible backup, and a follow-up only if the choice depends "
            "on something unknown. Moderate actionable detail is welcome; do not compress it "
            "into only two or three sentences."
        ),
        "comparison": (
            "Lead with the grounded choice or conditional answer, compare both sides briefly, "
            "state important uncertainty naturally, and optionally give one next step."
        ),
        "build_advice": (
            "Lead with the item choice and one or two grounded reasons; keep the answer focused on "
            "the requested gear decision."
            if context.focused_build
            else "Lead with the main build change, then explain the relevant gear, subclass, "
            "stats, and tradeoffs. Short bullets are useful when they improve scanning; skip "
            "general build theory the player did not ask for."
        ),
        "walkthrough": (
            "Give ordered, actionable steps. Numbered steps or short bullets are appropriate, and "
            "the answer may be longer when the task genuinely needs it."
        ),
        "troubleshooting": (
            "Lead with the likely fix or first diagnostic, then give the minimum ordered checks "
            "needed to resolve the issue."
        ),
    }
    continuation = (
        "This is a conversational follow-up: adapt the prior route directly instead of restarting "
        "with a mechanical Goal/Primary plan template. "
        if context.continuation
        else ""
    )
    timed = (
        "The user explicitly requested a timed plan, so a grounded timed structure is allowed."
        if context.timed_itinerary_requested
        else (
            "Do not create an unrequested minute-by-minute itinerary. A stated time window is a "
            "fit constraint, not permission to invent prep/activity/cleanup durations."
        )
    )
    return (
        f"Response mode: {context.mode}. {range_text} {shapes[context.mode]} "
        f"{continuation}{timed} Answer first; do not open with 'Based on the available data,' "
        "'Primary recommendation,' 'Short answer,' or generic throat-clearing. Preserve useful "
        "uncertainty, steps, and citations even when that exceeds the typical range."
    )


class ResponseQualityValidator:
    _BAD_OPENERS = (
        "based on the available data",
        "based on the data",
        "primary recommendation",
        "short answer",
        "there are several things to consider",
    )
    _INTERNAL_JARGON = re.compile(
        r"\b(?:guardian|manifest|guide|live) tools?\b|\bapi (?:returned|response|says)\b|"
        r"\b(?:provider terminology|function_call|evidence_state|known_true|known_false|"
        r"not_returned|completion_state|profile_completion_state|active_quest_status|"
        r"planning context|resolver)\b|\b(?:manifest|canonical|guide) "
        r"(?:match(?:es)?|entity|resolution)\b|\bmatched alias\b",
        re.IGNORECASE,
    )
    _DEBUG_REQUEST = re.compile(r"\b(?:debug|api|tool|provider|trace)\b", re.IGNORECASE)
    _TIME_SEGMENT = re.compile(r"\b(\d{1,3})\s*[- ]?\s*(minutes?|mins?|hours?|hrs?)\b", re.I)

    @classmethod
    def _segment_minutes(cls, text: str) -> list[int]:
        return [
            int(amount) * (60 if unit.casefold().startswith(("hour", "hr")) else 1)
            for amount, unit in cls._TIME_SEGMENT.findall(text)
        ]

    def __init__(self) -> None:
        self.planning = PlanningAnswerValidator()

    def validate(
        self,
        answer: str,
        mode: ResponseModeContext,
        user_message: str,
        planning_context: SessionPlanningContext | None,
        build_context: BuildResponseValidationContext | None = None,
        read_only_context: ReadOnlyActionValidationContext | None = None,
    ) -> list[PlanningViolation]:
        # Read-only checks: match "can’t" and "I’ve" the same as "can't" and "I've".
        answer, user_message = _plain(answer), _plain(user_message)
        violations = (
            self.planning.validate(answer, planning_context) if planning_context is not None else []
        )
        normalized = answer.casefold().strip()
        if answer.count("?") > 1 and not any(
            value.code == "too_many_followups" for value in violations
        ):
            violations.append(
                PlanningViolation(
                    code="too_many_followups",
                    correction="Ask at most one targeted follow-up question.",
                )
            )
        if any(normalized.startswith(value) for value in self._BAD_OPENERS):
            violations.append(
                PlanningViolation(
                    code="answer_not_first",
                    correction="Put the useful answer or recommendation in the first sentence.",
                )
            )
        if not self._DEBUG_REQUEST.search(user_message) and self._INTERNAL_JARGON.search(answer):
            violations.append(
                PlanningViolation(
                    code="internal_jargon",
                    correction=(
                        "Remove API, tool, provider, and internal evidence-state terminology; "
                        "express uncertainty naturally."
                    ),
                )
            )
        # Restating the player's own session length ("fits your 30-minute window") is not an
        # invented segment duration; only other durations can make an itinerary.
        known_minutes = set(self._segment_minutes(user_message))
        if planning_context is not None and planning_context.preferences.time_minutes:
            known_minutes.add(planning_context.preferences.time_minutes)
        time_segments = [
            value for value in self._segment_minutes(answer) if value not in known_minutes
        ]
        schedule_language = bool(
            re.search(
                r"\b(?:prep|cleanup|then|first|next|finish with|schedule|itinerary)\b",
                answer,
                re.I,
            )
        )
        if not mode.timed_itinerary_requested and (
            len(time_segments) >= 3 or (len(time_segments) >= 2 and schedule_language)
        ):
            violations.append(
                PlanningViolation(
                    code="unrequested_timed_itinerary",
                    correction=(
                        "Use the known session length as a fit constraint; remove the unrequested "
                        "minute-by-minute schedule and any invented segment durations."
                    ),
                )
            )
        extreme_max = extreme_max_words(mode)
        word_count = len(re.findall(r"\b[\w'-]+\b", answer))
        if not mode.detailed_requested and word_count > extreme_max:
            violations.append(
                PlanningViolation(
                    code="extreme_verbosity",
                    correction=(
                        f"Remove redundant material while preserving grounded detail; this "
                        f"{mode.mode} answer is far beyond its useful detail range."
                    ),
                )
            )
        sentences = [
            re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
            for value in re.split(r"(?<=[.!?])\s+|[\r\n]+", answer)
        ]
        substantive = [value for value in sentences if len(value.split()) >= 6]
        if len(substantive) != len(set(substantive)):
            violations.append(
                PlanningViolation(
                    code="repeated_content",
                    correction="Remove the repeated recommendation or reason.",
                )
            )
        heading_count = sum(
            bool(re.match(r"^\s*(?:#{1,3}\s+|\*\*[^*]+\*\*|[A-Z][^.!?]{1,30}:)\s*", line))
            for line in answer.splitlines()
            if line.strip()
        )
        compact_modes = {"direct_fact", "account_summary", "recommendation", "comparison"}
        if mode.mode in compact_modes and heading_count >= 4:
            violations.append(
                PlanningViolation(
                    code="mechanical_structure",
                    correction=(
                        "Replace repetitive sections with a natural paragraph or short list."
                    ),
                )
            )
        if mode.continuation and re.match(
            r"\s*(?:goal|primary plan|primary recommendation)\s*:", answer, re.I
        ):
            violations.append(
                PlanningViolation(
                    code="followup_restarts_plan",
                    correction=(
                        "Adapt the previous recommendation directly instead of restarting it."
                    ),
                )
            )
        if planning_context is not None and self.invented_durations(
            answer, user_message, planning_context
        ):
            violations.append(
                PlanningViolation(
                    code="invented_duration",
                    correction=(
                        "Do not state how long an activity takes unless the player's own recent "
                        "runs or a retrieved source show it. Say the duration is unknown, or use "
                        "the player's stated session length only as a fit constraint."
                    ),
                )
            )
        if planning_context is not None:
            for first, second in self.activity_record_conflations(answer, planning_context):
                violations.append(
                    PlanningViolation(
                        code="activity_record_conflation",
                        correction=(
                            f"{first} is a different activity from {second}. Do not apply "
                            f"progress on a {second} record or Triumph to {first}, or treat "
                            "them as the same."
                        ),
                    )
                )
                break
            if self.unsupported_solo_claims(answer, planning_context):
                violations.append(
                    PlanningViolation(
                        code="unsupported_solo_claim",
                        correction=(
                            "Nothing shows an activity can be played strictly alone; the data "
                            "only shows it can be launched. Say whether solo or queuing alone "
                            "into matchmaking is established, and do not claim solo play for it."
                        ),
                    )
                )
        asks_only = not [
            sentence
            for sentence in _sentence_parts(answer)[::2]
            if len(re.findall(r"\w+", sentence)) >= 4
            and not sentence.rstrip().endswith("?")
            and not _SOLICITATION.match(sentence)
        ]
        names_owned_item = bool(
            read_only_context
            and any(
                re.search(rf"\b{re.escape(name)}\b", user_message)
                for name in read_only_context.item_facts
            )
        )
        if (
            asks_only
            and "?" in answer
            and ((build_context is not None and build_context.is_build_request) or names_owned_item)
        ):
            violations.append(
                PlanningViolation(
                    code="unnecessary_clarification",
                    correction=(
                        "Answer instead of asking. Use what the Guardian data shows, keep the "
                        "player's equipped Exotic and the weapon they said they like unless they "
                        "asked to change them, and give one smaller improvement you can verify, "
                        "or say plainly that the current evidence justifies no particular "
                        "change. Do not ask permission."
                    ),
                )
            )
        elif asks_only and "?" in answer:
            violations.append(
                PlanningViolation(
                    code="clarification_only",
                    correction=(
                        "The draft only asks a question. Answer instead: take the most "
                        "reasonable reading of the request, state that assumption in one "
                        "clause (for example 'Assuming you mean your Titan, ...'), and give "
                        "the answer. Ask a question only if no reasonable assumption exists."
                    ),
                )
            )
        if build_context is not None and build_context.is_build_request:
            violations.extend(
                self._validate_build_answer(answer, mode, user_message, build_context)
            )
        violations.extend(self._validate_capabilities(answer, read_only_context))
        return list({value.code: value for value in violations}.values())

    @staticmethod
    def _validate_capabilities(
        answer: str, read_only: ReadOnlyActionValidationContext | None
    ) -> list[PlanningViolation]:
        """What the app can do: read-only lookups are done, not offered; nothing is written."""

        read_only = read_only or ReadOnlyActionValidationContext()
        violations: list[PlanningViolation] = []
        if _READ_ONLY_OFFER.search(_WHICH_TARGET_QUESTION.sub(" ", answer)):
            violations.append(
                PlanningViolation(
                    code="unnecessary_read_only_permission",
                    correction=(
                        "Read-only lookups never need permission. Do not ask whether to check "
                        "something, offer a choice of sources, or confirm a result format: run "
                        "a lookup that helps answer the request, or drop the offer."
                    ),
                )
            )
        # Planning context is Guardian data the model already has, so "I checked your quests"
        # on a planning turn is true; a requested verification still needs a real lookup.
        looked_up = read_only.lookups_performed or read_only.guardian_lookup_performed
        if (read_only.research_requested and not read_only.lookups_performed) or (
            not looked_up and _LOOKUP_CLAIM.search(answer)
        ):
            violations.append(
                PlanningViolation(
                    code="unperformed_lookup",
                    correction=(
                        "No lookup has run in this turn, so nothing has been checked or "
                        "verified yet. Run the relevant lookups now and report what they "
                        "establish; never say you checked or verified something you did not."
                    ),
                )
            )
        if not read_only.guardian_lookup_performed and _ACCOUNT_DATA_REQUEST.search(answer):
            violations.append(
                PlanningViolation(
                    code="account_data_requested",
                    correction=(
                        "Check the player's Guardian data with the relevant Guardian tool "
                        "before asking them to type or send account details. Ask for one "
                        "specific name only if the retrieved data still cannot answer."
                    ),
                )
            )
        sentences = _sentence_parts(answer)[::2]
        if any(_capability_offer_start(value) is not None for value in sentences):
            violations.append(
                PlanningViolation(
                    code="unsupported_capability",
                    correction=(
                        "This chat is text-only and Copilot cannot change items. Remove any "
                        "request for screenshots, images, or files and any offer to move, "
                        "equip, vault, or otherwise change gear or to check unlocked fragments; "
                        "keep the most useful supported next step from the information already "
                        "gathered, and explain the in-game steps when that helps."
                    ),
                )
            )
        if _IMPLIED_WRITE.search(answer):
            violations.append(
                PlanningViolation(
                    code="implied_bungie_write",
                    correction=(
                        "Copilot is read-only and changed nothing. Do not say an item was moved, "
                        "equipped, or otherwise changed; say you can't make the change and give "
                        "the in-game steps."
                    ),
                )
            )
        unrelated = [
            name
            for name in read_only.account_quest_names
            if read_only.guidance_asked
            for sentence in _SENTENCE_PARTS.split(_plain(answer))[::2]
            if name.casefold() in sentence.casefold()
            and re.search(r"\b(?:fragments?|unlock\w*)\b", sentence, re.I)
        ]
        if unrelated:
            violations.append(
                PlanningViolation(
                    code="unrelated_account_evidence",
                    correction=(
                        f"Do not present the player's active quest ({unrelated[0]}) as a route to "
                        "the unlock they asked about: no retrieved source connects them. Use only "
                        "routes the researched sources name."
                    ),
                )
            )
        stat_text = _plain(answer)
        if mentions_deprecated_armor_stat(answer):
            violations.append(
                PlanningViolation(
                    code="deprecated_armor_stat",
                    correction=(
                        "Armor stats are now Weapons, Health, Class, Grenade, Melee, and Super; "
                        "Recovery, Resilience, Discipline, Intellect, Strength, and Mobility no "
                        "longer exist. Use only stats the tools returned, and never treat a "
                        "stat that was not returned as low."
                    ),
                )
            )
        if not read_only.knowledge_lookup_performed and any(
            _STAT_EFFECT.search(sentence) and not _LOOT_HEDGED.search(sentence)
            for sentence in _SENTENCE_PARTS.split(stat_text)[::2]
        ):
            violations.append(
                PlanningViolation(
                    code="unsupported_stat_effect",
                    correction=(
                        "Nothing retrieved this turn says what that armor stat does. Describe "
                        "stats by their returned values only, or look the stat up first."
                    ),
                )
            )
        if _INVENTORY_DENIAL.search(stat_text) and (
            not read_only.inventory_checked or _READ_ONLY_OFFER.search(stat_text)
        ):
            violations.append(
                PlanningViolation(
                    code="false_inventory_denial",
                    correction=(
                        "The player's owned gear can be searched with the inventory tools. Do "
                        "not say you cannot scan it, and do not offer to while denying it: run "
                        "the search, name the owned replacement it finds, or say none was found."
                    ),
                )
            )
        if _fragment_misread(answer):
            violations.append(
                PlanningViolation(
                    code="element_fragment_as_prismatic",
                    correction=(
                        "Ember, Spark, Echo, Whisper, and Thread fragments belong to element "
                        "subclasses, not Prismatic (whose fragments are Facets). Do not present "
                        "them as the player's Prismatic fragments or as unlock progress."
                    ),
                )
            )
        loot_claims = (
            [
                sentence.strip()
                for sentence in _SENTENCE_PARTS.split(_plain(answer))[::2]
                if _is_loot_claim(sentence)
            ]
            if read_only.recommendation_turn and not read_only.knowledge_lookup_performed
            else []
        )
        if loot_claims:
            quoted = " | ".join(value[:160] for value in loot_claims[:2])
            violations.append(
                PlanningViolation(
                    code="unsupported_loot_claim",
                    correction=(
                        f"Nothing retrieved this turn supports the loot ranking in: {quoted}. "
                        "Remove or reword only that claim; keep the rest of the recommendation. "
                        "Do not add a general disclaimer about loot or clear time: if the answer "
                        "still compares loot, say in one clause that this comparison is not "
                        "established, and otherwise say nothing about it."
                    ),
                )
            )
        wrong = misdescribed_items(answer, read_only.item_facts)
        if wrong:
            violations.append(
                PlanningViolation(
                    code="misdescribed_item",
                    correction="Correct the item description: " + "; ".join(wrong) + ".",
                )
            )
        unverified = (
            []
            if read_only.current_evidence_performed
            else unverified_item_claims(answer, read_only.retrieved_item_names)
        )
        if unverified:
            names = ", ".join(dict.fromkeys(name for name, _ in unverified))
            kinds = ", ".join(dict.fromkeys(kind for _, kind in unverified))
            violations.append(
                PlanningViolation(
                    code="unverified_item_fact",
                    correction=(
                        f"The draft states the {kinds} of {names}, but no tool returned that "
                        "item this turn. Look the item up (get_item_details, or the player's "
                        "inventory) and state only what the result shows; if it cannot be "
                        "retrieved, drop the claim."
                    ),
                )
            )
        contradicted = contradicted_item_facts(answer, read_only.item_definitions)
        if contradicted:
            facts = "; ".join(
                f"{name} has {kind} {actual}, not {claimed}"
                for name, kind, claimed, actual in dict.fromkeys(contradicted)
            )
            violations.append(
                PlanningViolation(
                    code="contradicted_item_fact",
                    correction=f"A retrieved result contradicts the draft: {facts}.",
                )
            )
        absent = absent_measure_values(answer, read_only.evidence_terms)
        if absent:
            measures = ", ".join(dict.fromkeys(metric for _, metric in absent))
            violations.append(
                PlanningViolation(
                    code="absence_reported_as_value",
                    correction=(
                        f"Nothing returned this turn includes {measures}, yet the draft states "
                        "a zero or none value for it. Missing data is not zero: say the "
                        "measure is unavailable from the data."
                    ),
                )
            )
        if read_only.current_source_requested and not read_only.current_evidence_performed:
            violations.append(
                PlanningViolation(
                    code="unverified_current_source",
                    correction=(
                        "The player asked for a current source, and the Manifest's acquisition "
                        "hints are static, not current. Run a web search and report what a current "
                        "source says, or state plainly that no current source was found and that "
                        "the Manifest shows only a possible source."
                    ),
                )
            )
        if read_only.guidance_requested and not read_only.knowledge_lookup_performed:
            violations.append(
                PlanningViolation(
                    code="unresearched_guidance",
                    correction=(
                        "This asks for an unlock or acquisition path, and nothing this turn "
                        "retrieved one. Look it up with Manifest, Guide, or web research and give "
                        "the specific activities, locations, and steps the sources name; if no "
                        "source covers it, say so briefly instead of listing generic categories."
                    ),
                )
            )
        return violations

    @classmethod
    def activity_record_conflations(
        cls, answer: str, planning_context: SessionPlanningContext
    ) -> list[tuple[str, str]]:
        """(activity, record home) pairs where a record is applied to a different activity."""

        return [(a, h) for a, h, _ in cls.conflating_sentences(answer, planning_context)]

    @staticmethod
    def conflating_sentences(
        answer: str, planning_context: SessionPlanningContext
    ) -> list[tuple[str, str, str]]:
        """(activity, record home, sentence) where a record is applied to a different activity.

        A record under "Altars of Sorrow" says nothing about "Haunted Altars of Sorrow": the
        longer name contains the shorter one but they are separate activities.
        """

        records = [
            (_normalize(value.name).removesuffix(" completed"), _normalize(value.context_path[-1]))
            for value in planning_context.candidates
            if "record" in value.source_kinds and value.context_path
        ]
        homes = {
            _normalize(value.context_path[-1]): value.context_path[-1]
            for value in planning_context.candidates
            if "record" in value.source_kinds and value.context_path
        }
        activities = [
            value.name
            for value in planning_context.candidates
            if "available_activity" in value.source_kinds
        ]
        found: list[tuple[str, str, str]] = []
        # The activity the answer is currently talking about, so a sentence that says only "it
        # advances the record" is read against the activity named just before it.
        current: tuple[str, str, list[str]] | None = None
        for sentence in _SENTENCE_PARTS.split(_plain(answer))[::2]:
            label = _normalize(sentence)
            for activity in activities:
                key = _normalize(activity)
                for home_key, home in homes.items():
                    if key == home_key or home_key not in key:
                        continue
                    # "Haunted Altars" or "Haunted Altars of Sorrow": the extra words that make
                    # it a different activity, plus the shared first word, are enough.
                    extra = [
                        word
                        for word in key.replace(home_key, "").split()
                        if word not in _SAME_ACTIVITY_WORDS
                    ]
                    words = [*extra, home_key.split()[0]]
                    if extra and all(re.search(rf"\b{re.escape(word)}\b", label) for word in words):
                        current = (activity, home, extra)
            if current is None:
                continue
            activity, home, extra = current
            home_key = _normalize(home)
            names_activity = all(re.search(rf"\b{re.escape(word)}\b", label) for word in extra)
            # Leave the record's own name out: "Waves of Altars of Sorrow" is not the activity.
            bare = label
            for core, owner in records:
                if owner == home_key:
                    bare = bare.replace(core, " ")
            if not names_activity and re.search(rf"\b{re.escape(home_key)}\b", bare):
                current = None  # this sentence is about the regular activity itself
                continue
            if (
                re.search(r"\b(?:triumphs?|records?|waves)\b", sentence, re.IGNORECASE)
                and (names_activity or home_key in label)
                and not _DISTINCT_ACTIVITY.search(sentence)
            ):
                found.append((activity, home, sentence))
        return found

    @classmethod
    def repair_activity_conflations(
        cls, answer: str, planning_context: SessionPlanningContext
    ) -> str | None:
        """The answer without the sentences that apply one activity's record to another.

        Only the conflating sentences go; the rest of the plan stays. Returns None when nothing
        conflated, or when what remains opens mid-thought or holds no recommendation.
        """

        bad = {
            _plain(value).strip()
            for _, _, value in cls.conflating_sentences(answer, planning_context)
        }
        if not bad:
            return None
        parts = _sentence_parts(answer)
        replaced: list[str | None] = [
            None if _plain(value).strip() in bad else value for value in parts[::2]
        ]
        from app.local_repairs import clean_layout, drop_dependents

        text = clean_layout(_rejoin(parts, drop_dependents(parts, replaced)))
        if not text or _ORPHAN_START.match(text):
            return None
        return text if contains_recommendation(_plain(text)) else None

    @staticmethod
    def unsupported_solo_claims(answer: str, planning_context: SessionPlanningContext) -> list[str]:
        """Sentences that call an available activity solo-playable with nothing to show it."""

        activities = [
            _normalize(value.name)
            for value in planning_context.candidates
            if "available_activity" in value.source_kinds
        ]
        return [
            sentence
            for sentence in _SENTENCE_PARTS.split(_plain(answer))[::2]
            if _SOLO_CLAIM.search(sentence)
            and not _SOLO_HEDGED.search(sentence)
            and any(name in _normalize(sentence) for name in activities)
        ]

    @classmethod
    def allowed_minutes(
        cls, user_message: str, planning_context: SessionPlanningContext
    ) -> set[int]:
        """Minute figures the player or their own recent runs support."""

        allowed = set(cls._segment_minutes(user_message))
        if planning_context.preferences.time_minutes:
            allowed.add(planning_context.preferences.time_minutes)
        for run in planning_context.recent_activities:
            seconds = run.get("duration_seconds")
            if isinstance(seconds, int) and seconds > 0:
                minutes = round(seconds / 60)
                allowed.update({minutes - 1, minutes, minutes + 1})
        return allowed

    @classmethod
    def invented_durations(
        cls, answer: str, user_message: str, planning_context: SessionPlanningContext
    ) -> list[str]:
        """Stated durations that neither the player nor their own recent runs support."""

        allowed = cls.allowed_minutes(user_message, planning_context)
        found: list[str] = []
        for match in _DURATION_CLAIM.finditer(_plain(answer)):
            numbers = [int(value) for value in match.groups() if value]
            if any(value not in allowed for value in numbers):
                found.append(match.group(0).strip())
        return found

    @staticmethod
    def repair_planning_claims(
        answer: str, allowed_minutes: set[int] | None = None, *, note_loot: bool = True
    ) -> str | None:
        """Remove unsupported loot comparisons, solo claims, and invented durations.

        Each sentence gets every fix that applies, in turn. A parenthetical or explanatory tail
        is cut out; a sentence that would be left as a fragment is dropped whole. An uncertainty
        note is added once if a loot ranking was removed and `note_loot` (the player asked about
        loot; otherwise the cut is silent). Returns None when nothing was
        unsupported, or when what remains would open mid-thought or hold no recommendation.
        """

        allowed = allowed_minutes or set()

        def unsupported_duration(text: str) -> bool:
            return any(
                any(int(value) not in allowed for value in match.groups() if value)
                for match in _DURATION_CLAIM.finditer(text)
            )

        parts = _sentence_parts(answer)
        replaced: list[str | None] = []
        changed = removed_loot = removed_solo = False
        for index in range(0, len(parts), 2):
            current: str | None = _plain(parts[index])
            touched = False
            if _is_loot_claim(current):
                current, touched, removed_loot = _qualify_loot(current), True, True
            if (
                current is not None
                and _SOLO_CLAIM.search(current)
                and not _SOLO_HEDGED.search(current)
            ):
                current, touched, removed_solo = _without_solo_claim(current), True, True
            if current is not None and unsupported_duration(current):
                touched = removed_solo = True
                # The headline keeps its pick without the number. Elsewhere a parenthetical
                # carrying the number is cut and the rest kept; otherwise the sentence goes.
                bracketed = re.sub(
                    r"\s*\([^()]*\)",
                    lambda m: "" if unsupported_duration(m.group(0)) else m.group(0),
                    current,
                )
                bracketed = re.sub(r"\s+([:;,.])", r"\1", bracketed)
                if index == 0:
                    current = re.sub(r"\s{2,}", " ", _DURATION_CLAIM.sub("", current, count=1))
                elif (
                    bracketed != current
                    and not unsupported_duration(bracketed)
                    and len(re.findall(r"\w+", bracketed)) >= 4
                ):
                    current = bracketed
                else:
                    current = None
            replaced.append(current if touched else parts[index])
            changed = changed or touched
        if not changed:
            return None
        from app.local_repairs import clean_layout, drop_dependents

        text = clean_layout(_rejoin(parts, drop_dependents(parts, replaced)))
        # Never hand back an answer that now opens mid-thought ("Why: ...").
        if not text or _ORPHAN_START.match(text):
            return None
        if removed_loot:
            # Say only what was cut, and not again when the answer already states the limit.
            stated = any(
                _LOOT_HEDGED.search(part)
                and re.search(r"\b(?:loot|drops?|rewards?)\b", part, re.IGNORECASE)
                for part in _SENTENCE_PARTS.split(_plain(text))[::2]
            )
            if not stated and note_loot:
                text = f"{text}\n{LOOT_LIMIT_NOTE if removed_solo else LOOT_ONLY_LIMIT_NOTE}"
        return text if contains_recommendation(_plain(text)) else None

    @staticmethod
    def trim_trailing_handback(
        answer: str, *, offers: bool, choices: bool, account_checked: bool = True
    ) -> str | None:
        """The answer without the closing hand-back, when what remains stands on its own.

        Removed from the end, sentence by sentence: an offer of read-only work (when `offers`)
        or of an ability the app lacks, a bare "Which do you want?" after one, and (when
        `choices`) a request to pick between options the answer already ranked. An offer that
        follows information in its sentence only loses its own clause. Before the account was
        checked, a request for account details is left for the tool-enabled retry. Returns
        None when nothing was removed or what remains would not stand alone.
        """

        # Matching reads the apostrophe-normalized copy; cuts use the same indices in the
        # original, so the delivered text keeps its characters.
        parts = _sentence_parts(answer.rstrip())
        end = len(parts)
        removed_offer = removed_choice = removed_other = False
        while end >= 3:
            last, plain = parts[end - 1], _plain(parts[end - 1])
            # "If you want, I can:" followed by bullets that only list more offers.
            head = end - 1
            while head >= 2 and _BULLET.match(parts[head]):
                head -= 2
            if (offers or choices) and head < end - 1 and _OFFER_HEADER.match(_plain(parts[head])):
                end, removed_offer = head, True
                continue
            if not account_checked and _ACCOUNT_DATA_REQUEST.search(plain):
                break
            if choices and _PICK_ONE.search(plain):
                end -= 2
                removed_choice = True
                continue
            if _SHORT_HANDBACK.fullmatch(plain):
                end -= 2
                removed_other = True
                continue
            if (offers or choices) and _IF_YOU_WANT.match(plain):
                end -= 2
                removed_offer = True
                continue
            if choices and _TELL_ME_OR.match(plain):
                end -= 2
                removed_choice = True
                continue
            if choices and plain.rstrip().endswith("?") and _CLOSING_OFFER.match(plain):
                end -= 2
                removed_choice = True
                continue
            offer = _READ_ONLY_OFFER.search(plain) if offers else None
            clause = _clause_start(plain, offer.start()) if offer else 0
            if offer and _OFFER_LEAD_IN.fullmatch(plain[clause : offer.start()]):
                start: int | None = clause
            else:
                start = _capability_offer_start(plain)
            if start is None:
                break
            removed_offer = True
            kept = _kept_before(last, start)
            if kept is not None:
                parts[end - 1] = kept
                break
            end -= 2
        rest = "".join(parts[:end]).rstrip()
        plain_rest = _plain(rest)
        if not (removed_offer or removed_choice or removed_other) or (
            len(re.findall(r"\w+", rest)) < 3
        ):
            return None
        if offers and _READ_ONLY_OFFER.search(_WHICH_TARGET_QUESTION.sub(" ", plain_rest)):
            return None
        if removed_choice and not contains_recommendation(plain_rest):
            return None
        return rest

    @staticmethod
    def needs_initiative_retry(
        violations: list[PlanningViolation], read_only: ReadOnlyActionValidationContext
    ) -> bool:
        """Whether the draft skipped read-only work that a tool-enabled round could do.

        A permission question after lookups already ran is handled by the usual tool-less
        correction (answer from what was gathered); before any lookup, the work itself is due.
        """

        codes = {value.code for value in violations}
        return bool(
            codes
            & {
                "unperformed_lookup",
                "account_data_requested",
                "unresearched_guidance",
                "unverified_current_source",
                "unverified_item_fact",
            }
            or ("false_inventory_denial" in codes and not read_only.inventory_checked)
            or ("unsupported_replacement" in codes and not read_only.inventory_checked)
            # A bare offer to look something up is work to do, unless the offered lookup is one
            # no tool provides (fragment ownership): a tool-enabled round would only grab
            # unrelated tools, so that draft is corrected instead.
            or (
                "unnecessary_read_only_permission" in codes
                and not read_only.lookups_performed
                and "unsupported_capability" not in codes
            )
        )

    @staticmethod
    def initiative_instruction(violations: list[PlanningViolation]) -> str:
        corrections = " ".join(value.correction for value in violations)
        return (
            "Read-only lookups are already authorized by the user's request; do the work instead "
            "of asking or offering. Call the tool whose result can answer the question (only "
            "that tool; never an unrelated one to look busy), trying another source if the first "
            "cannot answer, then answer with what it establishes. If no available source can "
            "establish it, say so in one sentence and give the most useful next step. "
            "If your draft already answers the "
            "request, keep that answer and only drop the offer or question; never replace it "
            f"with a menu of options for the user to pick from. {corrections} {REWRITE_RULE}"
        )

    @staticmethod
    def _validate_build_answer(
        answer: str,
        mode: ResponseModeContext,
        user_message: str,
        context: BuildResponseValidationContext,
    ) -> list[PlanningViolation]:
        normalized = answer.casefold()
        violations: list[PlanningViolation] = []
        missing_evidence_acknowledged = bool(
            re.search(
                r"\b(?:can(?:not|'t)|could(?: not|n'?t)|unable|need|would need)\b"
                r"[^.!?]{0,120}\b(?:verify|inspect|check|review|see)\b",
                answer,
                re.IGNORECASE,
            )
        )
        if (
            context.requires_current_build_analysis
            and not context.current_build_analysis_used
            and not missing_evidence_acknowledged
        ):
            violations.append(
                PlanningViolation(
                    code="missing_current_build_analysis",
                    correction=(
                        "Do not judge or recommend changes to the current setup because its "
                        "equipped build was not verified. State briefly that personalized advice "
                        "is unavailable until the current setup is checked."
                    ),
                )
            )
        if (
            context.requires_owned_inventory
            and not context.inventory_ownership_checked
            and not missing_evidence_acknowledged
        ):
            violations.append(
                PlanningViolation(
                    code="missing_owned_inventory_evidence",
                    correction=(
                        "Do not name or recommend an owned or vault option because no owned "
                        "candidates or instance perks were verified. State briefly that a "
                        "personalized choice is unavailable until owned gear is checked."
                    ),
                )
            )
        if re.search(
            r"\b(?:sign in(?: again)?|log in(?: again)?|connect|reconnect|link)\b"
            r"[^.!?]{0,50}\b(?:bungie|account|platform)\b|"
            r"\bprovide\b[^.!?]{0,40}\b(?:platform|account)\b",
            answer,
            re.IGNORECASE,
        ):
            violations.append(
                PlanningViolation(
                    code="false_authentication_guidance",
                    correction=(
                        "Guardian context is already authenticated. Remove sign-in, account-link, "
                        "and platform guidance; describe only the specific build or character "
                        "lookup limitation."
                    ),
                )
            )
        # A possession or "equipped" claim about a named item needs Guardian evidence for that
        # item (equipped loadout, current-build analysis, or inventory results). Wording that
        # names no item ("you have high Grenade") is not a claim.
        unsupported = unsupported_claims(
            find_ownership_claims(answer, context.ownership, context.active_character)
        )
        if unsupported:
            listed = ", ".join(dict.fromkeys(value.item for value in unsupported))
            violations.append(
                PlanningViolation(
                    code="unsupported_ownership_claim",
                    correction=(
                        "Delete the statements that say the player has, owns, or has equipped: "
                        f"{listed}. No Guardian result proves that. Keep the rest of the answer; "
                        "separate verified owned options from aspirational items."
                    ),
                )
            )

        for item in context.locked_item_names:
            # The verb must govern the item ("replace Synthoceps"), so "Keep Synthoceps and make
            # one change" is not read as changing it.
            if replaces_item(answer, item) or re.search(
                rf"\b(?:change|stop using)\s+(?:your\s+|the\s+|my\s+)?{re.escape(item)}\b",
                answer,
                re.IGNORECASE,
            ):
                violations.append(
                    PlanningViolation(
                        code="locked_item_replaced",
                        correction=f"Keep {item}; the user explicitly locked that item.",
                    )
                )
        for sentence in _SENTENCE_PARTS.split(_plain(answer))[::2]:
            if _CROSS_WEAPON_PERK.search(sentence) and not _PERK_OWN_WEAPON.search(sentence):
                violations.append(
                    PlanningViolation(
                        code="incorrect_perk_mechanic",
                        correction=(
                            "Rampage, Kill Clip, and Multikill Clip build only from qualifying "
                            "kills with the weapon that has the perk; kills with another weapon "
                            "do not maintain them. Do not claim a different weapon raises that "
                            "perk's uptime."
                        ),
                    )
                )
                break
        for sentence in _SENTENCE_PARTS.split(_plain(answer))[::2]:
            if _RANGE_COVERAGE.search(sentence) and _RANGE_PAIRING.search(sentence):
                violations.append(
                    PlanningViolation(
                        code="unsupported_range_claim",
                        correction=(
                            "Two weapons of the same type do not automatically cover more range. "
                            "Range depends on each weapon's stats and perks, which have not been "
                            "compared; do not claim it."
                        ),
                    )
                )
                break
        equipped_exotics = [*context.equipped_exotic_weapons, *context.equipped_exotic_armor]
        equipped_all = list(dict.fromkeys([*equipped_exotics, *context.equipped_item_names]))
        # A replacement has to be something owned and not already equipped.
        candidates = [name for name in context.owned_item_names if name not in equipped_all]
        for item in equipped_all:
            if item in context.locked_item_names or not replaces_item(answer, item):
                continue
            others = [name for name in candidates if re.search(rf"\b{re.escape(name)}\b", answer)]
            if item not in equipped_exotics:
                if not others:
                    violations.append(
                        PlanningViolation(
                            code="unsupported_replacement",
                            correction=(
                                f"Do not recommend replacing {item} without naming a specific "
                                "owned replacement from the returned inventory, and never call "
                                "an unidentified replacement the single biggest improvement. If "
                                "none can be established, say so and give the best smaller "
                                "change you can verify, or say no change is justified."
                            ),
                        )
                    )
                continue
            if not others or not _TRADEOFF.search(answer):
                violations.append(
                    PlanningViolation(
                        code="unsupported_exotic_replacement",
                        correction=(
                            f"Do not recommend replacing {item} unless you name a specific owned "
                            "replacement from the returned inventory and say what the player "
                            "loses by the change. Otherwise recommend a smaller change you can "
                            "verify, or say no change is clearly better."
                        ),
                    )
                )
        for item in context.goal_item_names:
            if replaces_item(answer, item):
                violations.append(
                    PlanningViolation(
                        code="goal_item_replaced",
                        correction=(
                            f"Keep {item}: the player said they want to get or use it. "
                            "Recommend an improvement that works with it instead."
                        ),
                    )
                )

        if not context.current_external_grounding and _has_current_claim(answer):
            violations.append(
                PlanningViolation(
                    code="ungrounded_current_build_claim",
                    correction=(
                        "Remove current-meta, patch, artifact, buff, or nerf claims unless Live "
                        "or web research grounded them."
                    ),
                )
            )

        if re.search(
            r"\bbuild score\s*:?\s*\d{1,3}(?:\s*/\s*100|\s*%)?|"
            r"\b[abcds][+-]?\s*[- ]?tier\b",
            answer,
            re.IGNORECASE,
        ):
            violations.append(
                PlanningViolation(
                    code="fake_build_score",
                    correction=(
                        "Remove universal numeric scores and tier ratings; discuss only "
                        "goal-specific dimensions supported by evidence."
                    ),
                )
            )

        if re.search(
            r"\b(?:i(?:'ve| have)?|we(?:'ve| have)?)\s+"
            r"(?:equipped|switched|changed|applied|infused|dismantled|spent|claimed)\b",
            answer,
            re.IGNORECASE,
        ):
            violations.append(
                PlanningViolation(
                    code="implied_bungie_write",
                    correction=(
                        "Use recommendation language only; do not imply that any in-game change "
                        "was performed."
                    ),
                )
            )

        if not re.search(r"\b(?:debug|identifier|id)\b", user_message, re.IGNORECASE) and (
            re.search(r"\b(?:character|item|instance)[ _-]?id\b", answer, re.IGNORECASE)
            or re.search(r"\b\d{16,}\b", answer)
        ):
            violations.append(
                PlanningViolation(
                    code="raw_build_identifier",
                    correction="Remove raw character, item, instance, and hash identifiers.",
                )
            )

        if (
            re.search(
                r"\b\d{1,3}\s+(?:resilience|recovery|mobility|discipline|intellect|strength)\b"
                r"[^.!?]{0,50}\b(?:bad|good|must|need|target|threshold|too low|enough)\b",
                answer,
                re.IGNORECASE,
            )
            and not context.current_external_grounding
        ):
            violations.append(
                PlanningViolation(
                    code="unsupported_stat_threshold",
                    correction=(
                        "Describe the observed stat distribution without asserting an exact "
                        "threshold unless current mechanics were grounded externally."
                    ),
                )
            )

        mentioned_owned = sum(
            item.casefold() in normalized for item in set(context.owned_item_names)
        )
        inventory_lines = sum(
            bool(re.match(r"\s*(?:[-*]|\d+[.)])\s+", line)) for line in answer.splitlines()
        )
        if not mode.detailed_requested and (mentioned_owned > 8 or inventory_lines > 12):
            violations.append(
                PlanningViolation(
                    code="inventory_dump",
                    correction=(
                        "Return only a bounded shortlist and the highest-impact changes, not a "
                        "large inventory dump."
                    ),
                )
            )

        if context.normal_change_limit is not None and not mode.detailed_requested:
            change_lines = [
                line
                for line in answer.splitlines()
                if re.match(r"\s*(?:[-*]|\d+[.)])\s+", line)
                and re.search(
                    r"\b(?:swap|replace|change|equip|switch|remove|add|use instead)\b",
                    line,
                    re.IGNORECASE,
                )
            ]
            if len(change_lines) > context.normal_change_limit:
                violations.append(
                    PlanningViolation(
                        code="too_many_build_changes",
                        correction=(
                            f"Prioritize at most {context.normal_change_limit} high-impact changes "
                            "unless the user explicitly requested a complete detailed rebuild."
                        ),
                    )
                )

        exotic_groups = (
            (
                context.equipped_exotic_weapons,
                context.owned_exotic_weapons,
                "Exotic weapon",
            ),
            (
                context.equipped_exotic_armor,
                context.owned_exotic_armor,
                "Exotic armor piece",
            ),
        )
        locked_normalized = {value.casefold() for value in context.locked_item_names}
        for equipped, owned, label in exotic_groups:
            preserved = [value for value in equipped if value.casefold() in locked_normalized]
            if not preserved:
                continue
            for candidate in owned:
                if candidate.casefold() in {value.casefold() for value in preserved}:
                    continue
                if candidate.casefold() in normalized and re.search(
                    rf"\b(?:equip|run|use|swap (?:to|in))\b[^.!?]{{0,60}}"
                    rf"\b{re.escape(candidate)}\b",
                    answer,
                    re.IGNORECASE,
                ):
                    violations.append(
                        PlanningViolation(
                            code="incompatible_exotic_recommendation",
                            correction=(
                                f"Do not recommend a second {label} while preserving the "
                                f"equipped one ({', '.join(preserved)})."
                            ),
                        )
                    )

        constraint_counts = Counter()
        if context.activity_mode == "pve" and re.search(
            r"\b(?:for|in|play|recommend)\s+(?:the\s+)?(?:pvp|crucible)\b",
            answer,
            re.IGNORECASE,
        ):
            constraint_counts["activity"] += 1
        if context.activity_mode == "pvp" and re.search(
            r"\b(?:for|in|play|recommend)\s+(?:the\s+)?(?:pve|campaign|dungeon)\b",
            answer,
            re.IGNORECASE,
        ):
            constraint_counts["activity"] += 1
        if context.fireteam == "solo" and re.search(
            r"\b(?:requires? a fireteam|group[- ]only|for group support)\b",
            answer,
            re.IGNORECASE,
        ):
            constraint_counts["fireteam"] += 1
        if constraint_counts:
            violations.append(
                PlanningViolation(
                    code="build_constraint_conflict",
                    correction=(
                        "Keep the advice consistent with the explicit PvE/PvP and solo/group "
                        "constraints."
                    ),
                )
            )
        return violations

    @staticmethod
    def safe_build_answer(context: BuildResponseValidationContext) -> str:
        """Return a build-specific evidence fallback without activity-planning copy."""

        if context.requires_owned_inventory:
            return (
                "I couldn't retrieve a grounded shortlist of your owned gear and its returned "
                "perk data, so I don't want to guess."
            )
        if context.requires_current_build_analysis:
            return (
                "I couldn't inspect the required build data, so I can't give a grounded "
                "personalized loadout recommendation yet."
            )
        return (
            "I couldn't retrieve the required build evidence, so I can't give a grounded "
            "personalized recommendation yet."
        )

    @staticmethod
    def safe_build_no_swap_answer(context: "BuildResponseValidationContext") -> str:
        """The honest result when a proposed swap has no verified owned replacement."""

        keep = [*context.equipped_exotic_weapons, *context.equipped_exotic_armor]
        keep += [value for value in context.goal_item_names if value not in keep]
        kept = f" I'd keep {', '.join(keep[:3])} as they are." if keep else ""
        return (
            "I couldn't find an owned replacement that is shown to be better than your current "
            f"gear, so I'm not recommending a swap.{kept} A change would need a specific owned "
            "piece to compare against what you have equipped."
        )

    @staticmethod
    def safe_build_grounding_answer(
        context: "BuildResponseValidationContext | None" = None,
        remaining_codes: tuple[str, ...] | list[str] = (),
    ) -> str:
        """Fallback for an unsupported or constraint-breaking draft; data was available.

        The build was read, so the answer keeps it, says what could not be verified, and offers
        at most one owned, unequipped item to compare. Without a loadout it can only decline.
        """

        equipped = list(dict.fromkeys(context.equipped_item_names)) if context else []
        if not equipped:
            return (
                "I couldn't put together a recommendation I could fully verify against your "
                "gear and the constraints you gave, so I'd rather not guess. Narrowing it to "
                "one slot or goal usually helps."
            )
        unverified = [
            _UNVERIFIED_BY_CODE[code]
            for code in dict.fromkeys(remaining_codes)
            if code in _UNVERIFIED_BY_CODE
        ]
        what = unverified[0] if unverified else "a change that fits your constraints"
        owned_other = [
            name
            for name in dict.fromkeys(context.owned_item_names)
            if name not in equipped and name not in context.locked_item_names
        ]
        if owned_other:
            check = (
                f"One thing worth checking: {owned_other[0]} is in your inventory but not "
                "equipped. I haven't compared it with what you're wearing, so I can't say it's "
                "an upgrade."
            )
        else:
            check = (
                "If you name the slot or goal you want to improve, I can compare your owned "
                "options for it."
            )
        return (
            f"I'd keep your current build as it is ({', '.join(equipped[:4])}). I couldn't "
            f"verify {what}, so I'm not recommending a change. {check}"
        )

    @staticmethod
    def correction_instruction(
        violations: list[PlanningViolation],
        build_context: BuildResponseValidationContext | None = None,
    ) -> str:
        """The no-tools correction request, with the possession facts already established."""

        instruction = PlanningAnswerValidator.correction_instruction(violations)
        if build_context is None or not build_context.is_build_request:
            return instruction
        summary = evidence_summary(build_context.ownership, build_context.active_character)
        return f"{instruction} {summary}".strip() if summary else instruction

    @staticmethod
    def ownership_claims(
        answer: str, context: BuildResponseValidationContext
    ) -> list[OwnershipClaim]:
        return find_ownership_claims(answer, context.ownership, context.active_character)

    @staticmethod
    def repair_ownership(
        answer: str, context: BuildResponseValidationContext
    ) -> tuple[str, list[OwnershipClaim]]:
        """Remove only removable unsupported possession sentences; never rewrites the rest."""

        return repair_unsupported_ownership(
            answer, find_ownership_claims(answer, context.ownership, context.active_character)
        )
