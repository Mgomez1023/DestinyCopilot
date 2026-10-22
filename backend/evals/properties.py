"""Reusable behavioral properties checked against every scenario observation.

Each property is deterministic and conservative: it inspects the answer text, the trace, and
the recorded tool results against the profile's ground truth. A property that does not apply
returns `applies=False`. Failures can raise a serious failure flag that stays visible no
matter how well the answer scores elsewhere.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from app.build_analysis import replaces_item
from app.core_knowledge import normalize_text
from app.ownership_evidence import OwnershipEvidence, find_ownership_claims
from app.response_quality import (
    claims_cross_weapon_perk,
    denies_inventory_search,
    mentions_deprecated_armor_stat,
    misdescribed_items,
)
from evals.model import Expectations
from evals.profiles import CLASSES, SUBCLASSES, WEAPONS, ProfileFacts, known_gear_names
from evals.rubric import Flag
from evals.runtime import Observation, TurnRecord

WORD_LIMITS = {"fact": 120, "explain": 180, "planning": 260, "build": 320}
RECOMMEND = re.compile(
    r"\b(equip|use|using|run|running|slot|swap|switch to|pair|try|go with|recommend|pick|"
    r"grab|bring|put on|wear|keep)\b",
    re.I,
)
NOT_OWNED_CONTEXT = re.compile(
    r"\b(don'?t own|do not own|not in your|if you (get|acquire|unlock)|chase|farm|pursue|"
    r"aspirational|later|isn'?t owned|aren'?t owned|you don'?t have|not owned)\b",
    re.I,
)
NEGATED_CLASS = re.compile(
    r"\b(can'?t|cannot|can not|only (for|usable by|equippable by)|not usable|isn'?t usable|"
    r"won'?t work|-only|locked to|restricted to|not (equippable|available) (on|to|for))\b",
    re.I,
)
HEDGE = re.compile(
    r"\b(unknown|not sure|can'?t confirm|cannot confirm|unable to confirm|isn'?t established|"
    r"not established|don'?t have (live|current|that)|no (live|current) data|unclear|"
    r"not specified|doesn'?t (say|state|specify|establish)|can'?t verify|cannot verify|"
    r"couldn'?t (confirm|verify)|check (the )?(in-game|game|director|bungie)|may vary|"
    r"varies|not available|unavailable|I don'?t know)\b",
    re.I,
)
REFUSAL = re.compile(
    r"\b(I can'?t (access|see|retrieve|read)|couldn'?t (retrieve|access|load|get)|"
    r"unable to (access|retrieve|load)|don'?t have access|sign in|link your bungie)\b",
    re.I,
)
FILLER = re.compile(r"^\s*(great|good|excellent) question|happy to help|I'?d be glad", re.I)
QUEST_PROGRESS = re.compile(r"\b(finish|complete|continue|wrap up|progress|keep going)\b", re.I)
INACTIVE_CONTEXT = re.compile(
    r"\b(not active|isn'?t active|no active|not currently active|don'?t have|not in your "
    r"(active )?quests|wasn'?t returned|not returned|not found)\b",
    re.I,
)
MATCHMAKING_ASSERTION = re.compile(
    r"\b(is|it'?s|are|isn'?t|aren'?t)\s+(not\s+)?matchmade\b|\bno matchmaking\b|"
    r"\b(has|have|supports|uses)\s+matchmaking\b|"
    r"\bmatchmaking (is|isn'?t) (available|supported|on)\b",
    re.I,
)
REWARD_CLAIM = re.compile(
    r"\b(grants?|gives?|rewards?|drops?|loot|awards?|earns?|XP|engrams?|chests?)\b", re.I
)
PLAYER_COUNT = re.compile(
    r"\b(\d|one|two|three|four|five|six)\s*(-|to)?\s*(players?|guardians?|person)\b", re.I
)
DESTINATIONS = (
    "The Moon",
    "Moon",
    "European Dead Zone",
    "EDZ",
    "Nessus",
    "Europa",
    "Cosmodrome",
    "Dreaming City",
    "Throne World",
    "Neomuna",
    "Pale Heart",
    "Tangled Shore",
    "Mars",
    "Io",
    "Kepler",
)
LOCATED_ON = {
    "Altars of Sorrow": ("Moon",),
    "Altar of Reflection": ("Throne World",),
    "Altars of Summoning": ("Throne World",),
}
AMBIGUITY_CANDIDATES = {
    "altars": (
        "Altars of Sorrow",
        "Haunted Altars of Sorrow",
        "Altars of Summoning",
        "Altar of Reflection",
    )
}
CHARACTER_TOOLS = {
    "get_equipped_loadout",
    "get_build_details",
    "analyze_current_build",
    "find_build_alternatives",
    "get_active_quests",
    "get_available_activities",
    "get_content_progression",
    "search_inventory",
}


@dataclass(frozen=True)
class PropertyResult:
    name: str
    applies: bool
    passed: bool
    flag: Flag | None = None
    detail: str = ""


def _sentences(text: str) -> list[str]:
    return [value.strip() for value in re.split(r"(?<=[.!?])\s+|\n+", text) if value.strip()]


def _mentions(text: str, name: str) -> bool:
    return f" {normalize_text(name)} " in f" {normalize_text(text)} "


def _sentences_with(text: str, name: str) -> list[str]:
    return [value for value in _sentences(text) if _mentions(value, name)]


def _ok(name: str, detail: str = "") -> PropertyResult:
    return PropertyResult(name, True, True, None, detail)


def _skip(name: str) -> PropertyResult:
    return PropertyResult(name, False, True)


def _fail(name: str, flag: Flag | None, detail: str) -> PropertyResult:
    return PropertyResult(name, True, False, flag, detail[:300])


def _gear_names(turn: TurnRecord) -> dict[str, str | None]:
    names = dict(known_gear_names())
    for call in turn.tool_calls:
        item = call.result.get("item") if isinstance(call.result, dict) else None
        if isinstance(item, dict) and isinstance(item.get("name"), str) and len(item["name"]) > 3:
            restriction = (item.get("class_restriction") or {}).get("name")
            names.setdefault(item["name"], restriction if restriction in CLASSES else None)
    return names


def target_class(expect: Expectations, facts: ProfileFacts) -> str | None:
    return expect.character or (facts.classes[0] if facts.classes else None)


# --- Account / ownership ----------------------------------------------------------------


def ownership_claims(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    name = "never_claims_unowned_items"
    answer = obs.final.answer
    for gear in _gear_names(obs.final):
        if gear in facts.owned_items:
            continue
        for sentence in _sentences_with(answer, gear):
            pattern = (
                rf"\byour\b[^.]{{0,25}}\b{re.escape(gear)}\b|\b{re.escape(gear)}\b[^.]{{0,30}}"
                rf"\b(in|from) your (vault|inventory)\b|\byou (own|have|already have|'ve got)\b"
                rf"[^.]{{0,30}}\b{re.escape(gear)}\b"
            )
            if re.search(pattern, sentence, re.I) and not NOT_OWNED_CONTEXT.search(sentence):
                return _fail(name, "ownership_hallucination", f"claims '{gear}': {sentence}")
    return _ok(name)


def only_owned_recommendations(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    name = "only_owned_recommendations"
    expect = obs.scenario.expect
    if not (expect.only_owned or expect.equipped_only or expect.not_owned):
        return _skip(name)
    answer = obs.final.answer
    for gear in _gear_names(obs.final):
        owned = facts.owned_items.get(gear)
        forbidden = gear in expect.not_owned
        unowned = (expect.only_owned or expect.equipped_only) and owned is None
        # "Equipped" means equipped on the character the user asked about, not on another one.
        target = expect.character or target_class(expect, facts)
        not_equipped = bool(
            expect.equipped_only
            and owned is not None
            and gear not in equipped_on(obs.context, target)
        )
        if not (forbidden or unowned or not_equipped):
            continue
        for sentence in _sentences_with(answer, gear):
            if RECOMMEND.search(sentence) and not NOT_OWNED_CONTEXT.search(sentence):
                reason = (
                    "user said not owned"
                    if forbidden
                    else "not equipped"
                    if not_equipped
                    else "not owned"
                )
                return _fail(
                    name, "ownership_hallucination", f"recommends '{gear}' ({reason}): {sentence}"
                )
    return _ok(name)


def class_correct_armor(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    name = "never_recommends_wrong_class_armor"
    target = target_class(obs.scenario.expect, facts)
    if target is None:
        return _skip(name)
    answer = obs.final.answer
    for gear, owner in _gear_names(obs.final).items():
        if owner is None or owner == target:
            continue
        for sentence in _sentences_with(answer, gear):
            if (
                RECOMMEND.search(sentence)
                and not NEGATED_CLASS.search(sentence)
                and not re.search(rf"\b{owner}\b", sentence)
            ):
                return _fail(name, "wrong_character", f"{owner}-only '{gear}': {sentence}")
    return _ok(name)


def character_scope(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    name = "uses_requested_character"
    expect = obs.scenario.expect
    if expect.missing_class:
        answer = obs.final.answer
        stated = re.search(
            rf"\b(no|don'?t have (a|any)?|do not have (a|any)?|without (a|any)?|not .{{0,20}})"
            rf"\s*{expect.missing_class}\b|\b{expect.missing_class}\b[^.]{{0,40}}"
            rf"\b(isn'?t|not|wasn'?t) (on|in|returned|found|available|part)|"
            rf"\b(couldn'?t|could not|can'?t|cannot|didn'?t) (find|see|locate)\b[^.?]{{0,20}}"
            rf"\b{expect.missing_class}\b",
            answer,
            re.I,
        )
        if stated:
            return _ok(name, "states the class is missing")
        return _fail(
            name, "wrong_character", f"did not state that no {expect.missing_class} exists"
        )
    if not expect.character:
        return _skip(name)
    for call in obs.final.tool_calls:
        if call.name not in CHARACTER_TOOLS:
            continue
        selected = call.arguments.get("character_class")
        if isinstance(selected, str) and selected and selected != expect.character:
            return _fail(
                name, "wrong_character", f"{call.name} used {selected}, not {expect.character}"
            )
        # Also check which character the tool actually returned, so a defaulted call
        # (both selectors null) is verified by its result rather than its arguments.
        result = call.result if isinstance(call.result, dict) else {}
        returned = result.get("character_class") or (result.get("character") or {}).get("class")
        if isinstance(returned, str) and returned != expect.character:
            return _fail(
                name, "wrong_character", f"{call.name} returned {returned}, not {expect.character}"
            )
    others = [value for value in CLASSES if value != expect.character]
    answer = obs.final.answer
    own_terms = [expect.character, *SUBCLASSES[expect.character].values()]
    if not any(re.search(rf"\b{term}\b", answer) for term in own_terms):
        for other in others:
            if re.search(rf"\b(on|for) your {other}\b", answer):
                return _fail(name, "wrong_character", f"plans for {other} instead")
    return _ok(name)


def inactive_quest_claims(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    name = "no_unsupported_active_quest_claims"
    expect = obs.scenario.expect
    active = {quest for values in facts.active_quests.values() for quest in values}
    names = [value for value in expect.inactive_quests if value not in active]
    if not names:
        return _skip(name)
    for quest in names:
        for sentence in _sentences_with(obs.final.answer, quest):
            if sentence.rstrip().endswith("?"):
                continue  # A clarifying question is not a claim that the quest is active.
            if QUEST_PROGRESS.search(sentence) and not INACTIVE_CONTEXT.search(sentence):
                return _fail(name, "unsupported_active_quest", f"'{quest}': {sentence}")
    return _ok(name)


# --- Knowledge / grounding --------------------------------------------------------------


def ambiguity_explicit(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    name = "ambiguity_not_silently_resolved"
    term = obs.scenario.expect.ambiguity
    if not term:
        return _skip(name)
    answer = obs.final.answer
    named = [value for value in AMBIGUITY_CANDIDATES[term] if _mentions(answer, value)]
    asks = "?" in answer and re.search(r"\b(which|do you mean|did you mean)\b", answer, re.I)
    if len(named) >= 2 or (asks and named):
        return _ok(name, f"names {named}")
    return _fail(name, "ambiguity_silently_resolved", f"names only {named}")


def grounded_locations(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    name = "no_location_conflicts"
    expect = obs.scenario.expect
    traced = (obs.final.trace.get("core_knowledge") or {}).get("destination_conflicts") or []
    if traced:
        return _fail(name, "wrong_destination", f"production trace: {traced}")
    entities = [value for value in expect.grounded_entities if value in LOCATED_ON]
    if not entities:
        return _skip(name)
    for entity in entities:
        grounded = LOCATED_ON[entity]
        for sentence in _sentences_with(obs.final.answer, entity):
            if any(_mentions(sentence, value) for value in grounded):
                continue
            wrong = [
                place
                for place in DESTINATIONS
                if _mentions(sentence, place) and not any(place in value for value in grounded)
            ]
            if wrong and re.search(r"\b(on|in|at|located|found)\b", sentence):
                return _fail(name, "wrong_destination", f"{entity} placed at {wrong}: {sentence}")
    return _ok(name)


def unknown_not_filled(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    name = "unknown_fields_not_filled_from_memory"
    fields = obs.scenario.expect.unknown_fields
    if not fields:
        return _skip(name)
    answer = obs.final.answer
    for entity, field_name in fields:
        sentences = _sentences_with(answer, entity) or _sentences(answer)
        pattern = {"matchmaking": MATCHMAKING_ASSERTION, "rewards": REWARD_CLAIM}.get(
            field_name, PLAYER_COUNT
        )
        flag = "invented_reward_or_mechanic" if field_name == "rewards" else "hallucinated_fact"
        for sentence in sentences:
            if pattern.search(sentence) and not HEDGE.search(sentence):
                if field_name == "matchmaking" and re.search(r"\bhaunted\b", sentence, re.I):
                    continue
                return _fail(name, flag, f"{entity}.{field_name}: {sentence}")
    return _ok(name)


def live_claims(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    name = "live_claims_need_live_evidence"
    if not obs.scenario.expect.live_sensitive:
        return _skip(name)
    final = obs.final
    grounding = final.trace.get("grounding") or {}
    if final.trace.get("answer") == "unavailable_live_data":
        return _ok(name, "live gate returned unavailable")
    if grounding.get("live_provider") or grounding.get("web_research"):
        return _ok(name, "live or web evidence present")
    if HEDGE.search(final.answer):
        return _ok(name, "hedged")
    return _fail(name, "stale_live_claim", "current-state answer without live evidence or hedge")


def fuzzy_not_presented_as_exact(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    name = "fuzzy_match_not_presented_as_exact"
    checked = False
    for call in obs.final.tool_calls:
        resolution = call.result.get("resolution") if isinstance(call.result, dict) else None
        if not isinstance(resolution, dict) or resolution.get("exact") is not False:
            continue
        query = normalize_text(str(resolution.get("query") or ""))
        matched = str(resolution.get("matched_name") or "")
        # A near-identical spelling is a typo fix, not a different entity.
        if SequenceMatcher(None, query, normalize_text(matched)).ratio() >= 0.85:
            continue
        checked = True
        for sentence in _sentences_with(obs.final.answer, matched):
            if not re.search(
                r"\b(different|not the same|closest|variant|separate|instead|may not|"
                r"might not|not exactly|distinct|rather than)\b",
                sentence,
                re.I,
            ):
                return _fail(name, "hallucinated_fact", f"'{matched}' for '{query}': {sentence}")
    return _ok(name) if checked else _skip(name)


def forbidden_claims(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    name = "no_forbidden_claims"
    patterns = obs.scenario.expect.forbidden
    if not patterns:
        return _skip(name)
    for pattern in patterns:
        if re.search(pattern, _plain(obs.final.answer), re.I | re.M):
            return _fail(name, "hallucinated_fact", f"matched forbidden /{pattern}/")
    return _ok(name)


def expected_facts(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    name = "mentions_expected_fact"
    expected = obs.scenario.expect.mentions_any
    if not expected:
        return _skip(name)
    answer = _plain(obs.final.answer)
    if any(re.search(rf"\b{re.escape(value)}\b", answer, re.I) for value in expected):
        return _ok(name)
    return _fail(name, None, f"none of {list(expected)} mentioned")


def evidence_tools(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    name = "calls_expected_evidence_tool"
    expected = obs.scenario.expect.tools_any
    if not expected:
        return _skip(name)
    called = {call.name for call in obs.final.tool_calls}
    if called & set(expected):
        return _ok(name)
    planning = (obs.final.trace.get("planning_context") or {}).get("included")
    if obs.scenario.expect.style == "planning" and planning:
        # Session planning context already carries Guardian evidence for planning turns.
        return _ok(name, "planning context supplied Guardian evidence")
    return _fail(name, None, f"called {sorted(called)}, expected one of {list(expected)}")


# --- Loop health ------------------------------------------------------------------------


def loop_health(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    name = "tool_loop_completed"
    for turn in obs.turns:
        if turn.error:
            return _fail(name, "tool_loop_failed", turn.error)
        if not turn.answer.strip():
            return _fail(name, "tool_loop_failed", "empty answer")
    return _ok(name)


PERMISSION_REQUEST = re.compile(
    r"\b(?:want me to|would you like me to|shall i|should i)\b[^?.!\n]{0,120}\b(?:check|look|"
    r"search|verify|confirm|scan|review|research)\b|\bwhich (?:source )?do you prefer\b",
    re.I,
)
CAPABILITY_ASK = re.compile(
    r"\bscreen ?shots?\b|\b(?:upload|attach|send)\b[^.?!\n]{0,40}\b(?:images?|photos?|"
    r"pictures?|files?)\b|\b(?:want me to|would you like me to|i can|i'll)\s+(?:move|transfer|"
    r"vault|equip|dismantle|infuse)\b|\bi(?:'ve| have)\s+(?:moved|transferred|vaulted|equipped)\b",
    re.I,
)
CAPABILITY_DENIAL = re.compile(
    r"\b(?:can'?t|cannot|can not|unable to|not able to|read-only)\b", re.I
)


def no_permission_request(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    """Read-only lookups are authorized: the answer does them instead of asking."""

    name = "no_read_only_permission_request"
    for sentence in _sentences(obs.final.answer):
        if PERMISSION_REQUEST.search(sentence):
            return _fail(name, None, sentence)
    return _ok(name)


def no_unsupported_capability(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    """No screenshots or uploads (text-only chat) and no item changes (read-only app)."""

    name = "no_unsupported_capability"
    for sentence in _sentences(obs.final.answer):
        if CAPABILITY_ASK.search(sentence) and not CAPABILITY_DENIAL.search(sentence):
            return _fail(name, "unsupported_capability", sentence)
    return _ok(name)


def _plain(text: str) -> str:
    """Model text with typographic apostrophes and non-breaking hyphens made ASCII."""

    return text.replace("’", "'").replace("‑", "-")


# Closing questions that hand the request back: pick one of the options I listed, or tell me
# which item you meant. A question about the player's own situation ("Which platform are you
# on?") is not one of these.
HANDBACK = re.compile(
    r"\bwhich\s+(?:one|of (?:these|them|those)|(?:do|would) you (?:want|like|prefer))\b|"
    r"\bpick one\b|\bdo you mean\b",
    re.I,
)
# The same hand-back as an instruction: "pick one", "tell me which one to build out".
HANDBACK_COMMAND = re.compile(
    r"\bpick one\b|\b(?:tell me|let me know) which (?:one|option|of (?:these|them|those))\b",
    re.I,
)
# One targeted preference question ("survivability or boss DPS?") is allowed by the product;
# only a hand-back between activities, items, or lookups fails.
PREFERENCE_AXIS = re.compile(
    r"\b(?:pve|pvp|story|loot|rewards?|solo|group|survivability|dps|add clear|chill|casual|"
    r"challeng\w*)\b",
    re.I,
)


def request_fulfilled(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    """The answer does what was asked instead of handing the choice or lookup back."""

    name = "request_fulfilled"
    if obs.scenario.expect.question_allowed:
        return _skip(name)
    for sentence in _sentences(_plain(obs.final.answer))[-2:]:
        handback = (sentence.endswith("?") and HANDBACK.search(sentence)) or (
            HANDBACK_COMMAND.search(sentence)
        )
        if handback and not PREFERENCE_AXIS.search(sentence):
            return _fail(name, "unfulfilled_request", sentence)
    return _ok(name)


FIXED_SOCKET = re.compile(
    r"\b(?:fixed|can't (?:be )?(?:changed|swapped|re-?rolled)|cannot (?:be )?(?:changed|swapped|"
    r"re-?rolled)|can't (?:swap|change|re-?roll))\b",
    re.I,
)
DIFFERENT_COPY = re.compile(
    r"\b(?:different|another|new|other)\s+(?:barrel|perk|roll|magazine|trait)s?\b[^.]{0,80}"
    r"\b(?:obtain|get|farm|acquire|find)\b[^.]{0,40}\b(?:another|a new|a second|more)\b|"
    r"\b(?:obtain|get|farm|acquire|find)\b[^.]{0,40}\b(?:another|a new|a second)\b[^.]{0,60}"
    r"\b(?:different|other|new)\s+(?:barrel|perk|roll|magazine|trait)s?\b",
    re.I,
)


def no_contradictory_roll_claims(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    """A fixed perk or barrel is the same on every copy; another copy can't change it."""

    name = "no_contradictory_roll_claims"
    answer = _plain(obs.final.answer)
    if not FIXED_SOCKET.search(answer):
        return _skip(name)
    found = DIFFERENT_COPY.search(answer)
    if found:
        return _fail(name, "contradictory_claim", f"fixed, yet: {found.group(0)}")
    return _ok(name)


# Ember/Echo/Spark/Whisper/Thread are Solar/Void/Arc/Stasis/Strand fragments; Prismatic
# fragments are Facets. One equipped on another subclass is no evidence about Prismatic.
ELEMENT_FRAGMENT = re.compile(r"\b(?:Ember|Echo|Spark|Whisper|Thread) of [A-Z][\w']+")
PRISMATIC_CONTRAST = re.compile(
    r"\b(?:not|isn't|aren't|doesn't|don't|no)\b[^.]{0,30}\bPrismatic\b|"
    r"\b(?:says?|tells?|shows?) (?:you )?nothing\b|\bunrelated\b|\bdifferent subclass\b",
    re.I,
)


def no_cross_subclass_inference(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    name = "no_cross_subclass_inference"
    sentences = [
        value
        for value in _sentences(_plain(obs.final.answer))
        if ELEMENT_FRAGMENT.search(value) and re.search(r"\bPrismatic\b", value)
    ]
    if not sentences:
        return _skip(name)
    for sentence in sentences:
        if not PRISMATIC_CONTRAST.search(sentence):
            return _fail(name, "hallucinated_fact", f"element fragment as Prismatic: {sentence}")
    return _ok(name)


def _all_answers(obs: Observation) -> list[tuple[int, str]]:
    """Every turn's answer, so a wrong fact stated early in a conversation still counts."""

    return [(index + 1, _plain(turn.answer)) for index, turn in enumerate(obs.turns)]


def no_misdescribed_items(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    """Items are described with their real rarity and weapon type, in any turn."""

    name = "no_misdescribed_items"
    truth = {item: (value[2], value[0]) for item, value in WEAPONS.items()}
    for turn, answer in _all_answers(obs):
        wrong = misdescribed_items(answer, truth)
        if wrong:
            return _fail(name, "hallucinated_fact", f"turn {turn}: {wrong[0]}")
        for sentence in _sentences(answer):
            if re.search(r"\bloaders?\b", sentence, re.I) and re.search(
                r"\bhandling\b", sentence, re.I
            ):
                return _fail(name, "hallucinated_fact", f"turn {turn}: loader and handling")
    return _ok(name)


def no_unrelated_quest_evidence(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    """An active quest is not offered as a route to an unrelated unlock, in any turn."""

    name = "no_unrelated_quest_evidence"
    quests = {quest for values in facts.active_quests.values() for quest in values}
    quests |= {
        step
        for character in obs.context.characters
        for value in character.quests
        if (step := value.step_name)
    }
    for turn, answer in _all_answers(obs):
        for sentence in _sentences(answer):
            if re.search(r"\bfragments?\b", sentence, re.I):
                for quest in quests:
                    if _mentions(sentence, quest):
                        return _fail(name, "hallucinated_fact", f"turn {turn}: {sentence[:160]}")
    return _ok(name)


def lookup_matches_account(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    """An inventory search that returns nothing must not contradict items the profile owns."""

    name = "lookup_matches_account"
    checked = False
    for turn_index, turn in enumerate(obs.turns):
        for call in turn.tool_calls:
            if call.name != "search_inventory" or call.result.get("items") != []:
                continue
            terms = [
                value.casefold()
                for key in ("query", "subtype")
                if isinstance(value := call.arguments.get(key), str)
            ]
            vault = [
                item.name
                for item in obs.context.inventory.items
                if all(
                    term in item.name.casefold() or term in (item.item_subtype or "").casefold()
                    for term in terms
                )
            ]
            checked = True
            if terms and vault:
                return _fail(
                    name,
                    "false_refusal",
                    f"turn {turn_index + 1}: search {terms} found nothing; vault has {vault[:2]}",
                )
    return _ok(name) if checked else _skip(name)


MOD_REMOVAL = re.compile(r"\b(?:remove|replace|swap out|drop|ditch)\b", re.I)
TRADEOFF = re.compile(
    r"\b(?:give[s]? up|lose|loses|cost|trade-?off|in exchange|at the expense|synerg\w*|"
    r"supports?|because|keeps?|pairs? with)\b",
    re.I,
)


def removal_weighs_tradeoff(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    """Removing an equipped mod names what it gives up; the best change is not asserted blind."""

    name = "removal_weighs_tradeoff"
    mods = {
        plug.name
        for character in obs.context.characters
        for item in character.equipped_gear
        if item.item_type == "Armor"
        for plug in item.socketed_plug_details
    }
    answer = _plain(obs.final.answer)
    for sentence in _sentences(answer):
        for mod in mods:
            if (
                _mentions(sentence, mod)
                and MOD_REMOVAL.search(sentence)
                and not TRADEOFF.search(answer)
            ):
                return _fail(name, "ungrounded_advice", sentence[:160])
    return _ok(name)


def exotic_replacement_grounded(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    """Replacing an equipped Exotic names a specific owned replacement and what is lost."""

    name = "exotic_replacement_grounded"
    answer = _plain(obs.final.answer)
    exotics = {
        item.name
        for character in obs.context.characters
        for item in character.equipped_gear
        if item.tier == "Exotic" and item.item_type in {"Weapon", "Armor"}
    }
    for exotic in sorted(exotics):
        if not replaces_item(answer, exotic):
            continue
        others = [
            owned
            for owned in facts.owned_items
            if owned != exotic and re.search(rf"\b{re.escape(owned)}\b", answer)
        ]
        if not others or not re.search(
            r"\b(?:lose|loses|give[s]? up|cost[s]?|trade-?off|in exchange|at the expense)\b",
            answer,
            re.I,
        ):
            return _fail(
                name,
                "ungrounded_advice",
                f"replaces {exotic} without {others or 'any owned alternative'} and its cost",
            )
    return _ok(name)


LOOT_SUPERLATIVE = re.compile(
    r"\b(?:highest|higher|best|top)[- ](?:value|tier|quality|chance|odds)?"
    r"[^.?!\n]{0,25}\b(?:loot|drops?|rewards?)\b",
    re.I,
)
NOT_ESTABLISHED = re.compile(
    r"\b(?:unknown|not established|isn't established|can't (?:confirm|verify)|unverified|"
    r"not confirmed)\b",
    re.I,
)


def no_irrelevant_tools(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    """Tools the scenario marks as unrelated to the request are not called in any turn."""

    name = "no_irrelevant_tools"
    forbidden = set(obs.scenario.expect.forbidden_tools)
    if not forbidden:
        return _skip(name)
    for index, turn in enumerate(obs.turns):
        called = sorted({call.name for call in turn.tool_calls} & forbidden)
        if called:
            return _fail(name, "unfulfilled_request", f"turn {index + 1} called {called}")
    return _ok(name)


def current_source_not_overclaimed(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    """A request for a current source is not satisfied by static Manifest hints alone."""

    name = "current_source_not_overclaimed"
    if not obs.scenario.expect.current_evidence:
        return _skip(name)
    grounding = obs.final.trace.get("grounding") or {}
    if grounding.get("live_provider") or grounding.get("web_research"):
        return _ok(name, "current evidence present")
    if HEDGE.search(obs.final.answer) or NOT_ESTABLISHED.search(obs.final.answer):
        return _ok(name, "hedged")
    return _fail(
        name, "stale_live_claim", "current source requested but only static data was consulted"
    )


def no_unsupported_loot_claim(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    """Loot-quality superlatives need a retrieved source in the same turn, in any turn."""

    name = "no_unsupported_loot_claim"
    for index, turn in enumerate(obs.turns):
        grounding = turn.trace.get("grounding") or {}
        if any(
            grounding.get(key)
            for key in ("manifest", "guide_provider", "live_provider", "web_research")
        ):
            continue
        for sentence in _sentences(_plain(turn.answer)):
            if LOOT_SUPERLATIVE.search(sentence) and not NOT_ESTABLISHED.search(sentence):
                return _fail(name, "hallucinated_fact", f"turn {index + 1}: {sentence[:150]}")
    return _ok(name)


def answers_instead_of_asking(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    """A build, planning, or owned-item request gets an answer, not only a question, in any turn."""

    name = "answers_instead_of_asking"
    if obs.scenario.expect.question_allowed:
        return _skip(name)
    gear = list(known_gear_names())
    for index, turn in enumerate(obs.turns):
        mentions_gear = any(_mentions(turn.message, value) for value in gear)
        if not (obs.scenario.expect.style in {"build", "planning"} or mentions_gear):
            continue
        statements = [
            value
            for value in _sentences(_plain(turn.answer))
            if not value.endswith("?") and word_count(value) >= 4
        ]
        if not statements and "?" in turn.answer:
            return _fail(name, "unfulfilled_request", f"turn {index + 1}: {turn.answer[:140]}")
    return _ok(name)


def no_deprecated_armor_stats(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    """Armor 3.0: only Weapons, Health, Class, Grenade, Melee, and Super are armor stats."""

    name = "no_deprecated_armor_stats"
    for index, turn in enumerate(obs.turns):
        if mentions_deprecated_armor_stat(turn.answer):
            return _fail(
                name, "hallucinated_fact", f"turn {index + 1}: {_plain(turn.answer)[:150]}"
            )
        for sentence in _sentences(_plain(turn.answer)):
            if re.search(
                r"\b(?:Weapons|Health|Class|Grenade|Melee|Super)\b\s+stat\b", sentence
            ) and (
                re.search(
                    r"\b(?:faster|recharge\w*|regenerat\w*|cooldowns?|shorter)\b", sentence, re.I
                )
            ):
                grounding = turn.trace.get("grounding") or {}
                if not (grounding.get("manifest") or grounding.get("guide_provider")):
                    return _fail(
                        name,
                        "hallucinated_fact",
                        f"turn {index + 1}: stat effect: {sentence[:120]}",
                    )
    return _ok(name)


def no_contradictory_inventory_claims(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    """Never says the player's gear cannot be searched; the inventory tools can."""

    name = "no_contradictory_inventory_claims"
    for index, turn in enumerate(obs.turns):
        if denies_inventory_search(turn.answer):
            return _fail(
                name, "contradictory_claim", f"turn {index + 1}: {_plain(turn.answer)[:150]}"
            )
    return _ok(name)


def no_perk_mechanic_errors(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    """Rampage-style perks build only from kills with their own weapon, in any turn."""

    name = "no_perk_mechanic_errors"
    for index, turn in enumerate(obs.turns):
        if claims_cross_weapon_perk(turn.answer):
            return _fail(
                name, "hallucinated_fact", f"turn {index + 1}: {_plain(turn.answer)[:150]}"
            )
    return _ok(name)


def no_activity_record_conflation(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    """A Haunted Altars of Sorrow answer does not borrow Altars of Sorrow Triumph progress."""

    name = "no_activity_record_conflation"
    for index, turn in enumerate(obs.turns):
        for sentence in _sentences(_plain(turn.answer)):
            if re.search(r"\bHaunted Altars\b", sentence) and re.search(
                r"\b(?:Triumphs?|records?|Waves of Altars of Sorrow)\b", sentence
            ):
                return _fail(name, "hallucinated_fact", f"turn {index + 1}: {sentence[:140]}")
    return _ok(name)


def no_false_refusal(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    name = "no_false_refusal"
    final = obs.final
    fallback = (final.trace.get("planning_correction") or {}).get("fallback")
    if fallback in {"build_grounding", "build_evidence", "planning_safe_answer"}:
        return _fail(name, "false_refusal", f"validator fallback {fallback}")
    tools_ok = any("error" not in call.result for call in final.tool_calls)
    if tools_ok and REFUSAL.search(final.answer):
        return _fail(name, "false_refusal", "refusal wording despite successful tools")
    return _ok(name)


# --- Response quality -------------------------------------------------------------------


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'’-]+\b", text))


def concise(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    name = "concise"
    limit = WORD_LIMITS[obs.scenario.expect.style]
    words = word_count(obs.final.answer)
    if words <= limit:
        return _ok(name, f"{words} words")
    flag: Flag | None = "excessive_verbosity" if words > limit * 1.5 else None
    return _fail(name, flag, f"{words} words > {limit}")


def answer_first(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    name = "answer_first"
    sentences = _sentences(obs.final.answer)
    if not sentences:
        return _skip(name)
    first = sentences[0]
    if first.endswith("?") or re.match(r"^(sure|okay|ok|let me|alright|absolutely)\b", first, re.I):
        return _fail(name, None, first[:120])
    return _ok(name)


def no_filler(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    name = "no_filler"
    if FILLER.search(obs.final.answer):
        return _fail(name, None, "filler opening")
    return _ok(name)


def follow_up_questions(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    name = "limited_follow_up_questions"
    allowed = 2 if obs.scenario.expect.question_allowed else 1
    questions = sum(1 for value in _sentences(obs.final.answer) if value.endswith("?"))
    if questions <= allowed:
        return _ok(name, f"{questions} questions")
    return _fail(name, None, f"{questions} questions > {allowed}")


def not_repeating_prompt(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    name = "does_not_repeat_prompt"
    sentences = _sentences(obs.final.answer)
    if not sentences:
        return _skip(name)
    ratio = SequenceMatcher(
        None, normalize_text(sentences[0]), normalize_text(obs.scenario.prompt)
    ).ratio()
    return _fail(name, None, f"similarity {ratio:.2f}") if ratio > 0.8 else _ok(name)


def planning_focus(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    name = "one_primary_and_brief_backup"
    if obs.scenario.expect.style != "planning":
        return _skip(name)
    options = [
        value
        for value in obs.final.answer.splitlines()
        if re.match(r"^\s*(\d+[.)]|#{1,4}\s|\*\*(option|primary|backup|plan))", value, re.I)
    ]
    if len(options) > 4:
        return _fail(name, None, f"{len(options)} top-level options")
    return _ok(name)


def equipped_on(context: Any, character_class: str | None) -> set[str]:
    """Gear names actually equipped on one character (ground truth from the profile)."""

    return {
        item.name
        for character in context.characters
        if character.class_name == character_class
        for item in character.equipped_gear
    }


def ground_truth_evidence(context: Any) -> OwnershipEvidence:
    """Possession and per-character equipment exactly as the profile defines them."""

    evidence = OwnershipEvidence()
    for item in context.inventory.items:
        evidence.add(item.name, "ground_truth", equipped=False)
        for plug in item.socketed_plug_details:
            evidence.add(plug.name, "ground_truth", equipped=False)
    for character in context.characters:
        for item in character.equipped_gear:
            owner = character.class_name
            evidence.add(item.name, "ground_truth", equipped=True, character=owner)
            # Perks, mods, aspects, and fragments are equipped with the item they sit on.
            for plug in item.socketed_plug_details:
                evidence.add(plug.name, "ground_truth", equipped=True, character=owner)
    return evidence


def conversation_isolation(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    """Per turn: the right character was active, analyzed, and nothing leaked across a switch."""

    name = "conversation_scope_isolated"
    expect = obs.scenario.expect
    if not expect.turn_characters:
        return _skip(name)
    evidence = ground_truth_evidence(obs.context)
    for index, turn in enumerate(obs.turns):
        wanted = expect.turn_characters[index] if index < len(expect.turn_characters) else None
        if wanted is None:
            continue
        scope = (turn.trace.get("conversation_scope") or {}).get("active")
        if scope != wanted:
            return _fail(name, "wrong_character", f"turn {index + 1} scope {scope}, not {wanted}")
        for call in turn.tool_calls:
            if call.name not in CHARACTER_TOOLS:
                continue
            result = call.result if isinstance(call.result, dict) else {}
            returned = result.get("character_class") or (result.get("character") or {}).get("class")
            chosen = call.arguments.get("character_class")
            for seen in (returned, chosen):
                if isinstance(seen, str) and seen != wanted:
                    return _fail(
                        name,
                        "wrong_character",
                        f"turn {index + 1} {call.name} used {seen}, not {wanted}",
                    )
        if index >= expect.leak_from_turn:
            for term in expect.leak_terms:
                for sentence in _sentences_with(turn.answer, term):
                    elsewhere = any(
                        other != wanted and re.search(rf"\b{other}\b", sentence)
                        for other in CLASSES
                    )
                    if not elsewhere:
                        return _fail(
                            name,
                            "cross_character_leak",
                            f"turn {index + 1} '{term}' after the switch: {sentence}",
                        )
        for claim in find_ownership_claims(turn.answer, evidence, wanted):
            if claim.claim_type == "equipped" and not claim.supported:
                return _fail(
                    name,
                    "ownership_hallucination",
                    f"turn {index + 1} claims '{claim.item}' is equipped on the {wanted}",
                )
    return _ok(name)


URL = re.compile(r"https?://\S+", re.I)
DATED = re.compile(
    r"\b(?:20\d{2}-\d{2}-\d{2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? "
    r"\d{1,2}|20\d{2})\b",
    re.I,
)
OWNERSHIP_ASSERTION = (
    r"\byou(?:'ve| have| own| already have| do have| got|'ve got)\b[^.?!]{{0,30}}"
    r"\b(?P<term>{term})\b",
    r"\byour\b[^.?!]{{0,25}}\b(?P<term>{term})\b",
    r"\b(?P<term>{term})\b[^.?!]{{0,30}}\bin your (?:vault|inventory)\b",
)
OWNERSHIP_NEGATION = re.compile(
    r"\b(?:no|not|none|nothing|never|without|neither|don'?t|doesn'?t|didn'?t|isn'?t|aren'?t|"
    r"haven'?t|hasn'?t|can'?t)\b|n't\b",
    re.I,
)
CONDITIONAL_OPENING = re.compile(r"^\s*(?:if|whether)\b", re.I)
NEGATION_WINDOW_WORDS = 4


def _asserts_ownership(answer: str, term: str) -> str | None:
    """The first sentence that asserts the player holds `term`.

    Questions and conditionals don't assert anything. A negation only cancels the claim when it
    sits in the claim itself ("you have no sniper") or within a few words before the item
    ("there is no sniper in your vault"); a negation elsewhere in the sentence does not.
    """

    for sentence in _sentences(_plain(answer)):
        if sentence.endswith("?") or CONDITIONAL_OPENING.search(sentence):
            continue
        for template in OWNERSHIP_ASSERTION:
            for match in re.finditer(template.format(term=term), sentence, re.I):
                term_start = match.start("term")
                if template.startswith(r"\b(?P<term>"):
                    # The item leads the clause, so the negation (if any) precedes it.
                    words = sentence[:term_start].split()
                    scope = " ".join(words[-NEGATION_WINDOW_WORDS:]) + " " + match.group(0)
                else:
                    scope = sentence[match.start() : term_start]
                if not OWNERSHIP_NEGATION.search(scope):
                    return sentence
    return None


def turn_expectations(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    """Per-turn checks (`Expectations.by_turn`): earlier turns are graded, not only the last."""

    name = "turn_expectations_met"
    checks = obs.scenario.expect.by_turn
    if not checks:
        return _skip(name)
    for index, check in sorted(checks.items()):
        if index >= len(obs.turns):
            continue
        answer = _plain(obs.turns[index].answer)
        label = f"turn {index + 1}"
        for pattern in check.forbidden:
            match = re.search(pattern, answer, re.I | re.M)
            if match:
                return _fail(name, check.forbidden_flag, f"{label} matched /{pattern}/: {match[0]}")
        for term in check.not_owned_claims:
            sentence = _asserts_ownership(answer, term)
            if sentence:
                return _fail(
                    name, "ownership_hallucination", f"{label} claims '{term}' owned: {sentence}"
                )
        if check.answers_any and not any(
            re.search(pattern, answer, re.I) for pattern in check.answers_any
        ):
            return _fail(name, None, f"{label} answers none of {list(check.answers_any)}")
        if check.linked_source_or_limitation and not (
            obs.turns[index].sources
            or URL.search(answer)
            or HEDGE.search(answer)
            or NOT_ESTABLISHED.search(answer)
        ):
            return _fail(
                name, "stale_live_claim", f"{label} has neither a source nor a stated limit"
            )
    return _ok(name)


def mechanics_trace(obs: Observation, facts: ProfileFacts) -> PropertyResult:
    """Informational only: reports the trace-only mechanics-claim summary; never fails."""

    name = "mechanics_claims_trace"
    summary = obs.final.trace.get("mechanics_claims")
    if not isinstance(summary, dict) or "claims_total" not in summary:
        return _skip(name)
    return _ok(
        name,
        f"{summary['claims_total']} claims, {summary['needs_attention']} need attention "
        f"{summary.get('by_status')}",
    )


PROPERTIES: tuple[Callable[[Observation, ProfileFacts], PropertyResult], ...] = (
    conversation_isolation,
    turn_expectations,
    ownership_claims,
    only_owned_recommendations,
    class_correct_armor,
    character_scope,
    inactive_quest_claims,
    ambiguity_explicit,
    grounded_locations,
    unknown_not_filled,
    live_claims,
    fuzzy_not_presented_as_exact,
    forbidden_claims,
    expected_facts,
    evidence_tools,
    loop_health,
    no_false_refusal,
    request_fulfilled,
    answers_instead_of_asking,
    no_contradictory_roll_claims,
    no_cross_subclass_inference,
    no_misdescribed_items,
    no_unrelated_quest_evidence,
    lookup_matches_account,
    removal_weighs_tradeoff,
    exotic_replacement_grounded,
    no_deprecated_armor_stats,
    no_perk_mechanic_errors,
    no_activity_record_conflation,
    no_contradictory_inventory_claims,
    no_unsupported_loot_claim,
    no_irrelevant_tools,
    current_source_not_overclaimed,
    no_permission_request,
    no_unsupported_capability,
    concise,
    answer_first,
    no_filler,
    follow_up_questions,
    not_repeating_prompt,
    planning_focus,
    mechanics_trace,
)


def evaluate_properties(obs: Observation, facts: ProfileFacts) -> list[PropertyResult]:
    return [check(obs, facts) for check in PROPERTIES]
