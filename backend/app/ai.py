import dataclasses
import hashlib
import json
import logging
import re
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit
from uuid import uuid4

from openai import AsyncOpenAI, OpenAIError
from pydantic import ValidationError

from app.absence_claims import collect_evidence_terms, replace_absent_values
from app.build_analysis import (
    BuildRequestContext,
    derive_build_request_context,
    forced_alternatives_relevance,
)
from app.character_selection import (
    CharacterResolutionError,
    ConversationScope,
    CurrentCharacter,
    resolve_conversation_scope,
    resolve_current_character,
    selector_source,
)
from app.chat_stream import (
    StreamStatusReporter,
    is_web_search_stream_event,
    status_for_tool,
)
from app.citations import citation_spans, clean_citation_artifacts
from app.config import Settings
from app.content_progression import mentioned_content_names, supported_content_aliases
from app.core_knowledge import DestinyKnowledgeEngine, normalize_text
from app.core_knowledge_provider import CORE_PROVIDER_CATEGORY
from app.destiny_knowledge import DestinyKnowledgeService, KnowledgeToolError
from app.guardian_tools import (
    CONVERSATION_SCOPED_TOOLS,
    CURRENT_DEFAULT_TOOLS,
    GuardianToolError,
    GuardianToolService,
)
from app.item_fact_claims import (
    collect_item_definitions,
    collect_item_names,
    correct_item_facts,
    remove_unverified_item_claims,
)
from app.live_knowledge import (
    LIVE_PROVIDER_CATEGORY,
    unavailable_live_data,
    unavailable_live_message,
    volatile_topic,
)
from app.local_repairs import (
    LOCAL_STYLE_CODES,
    dependents_of,
    is_constraint_code,
    locally_handled,
    offending_sentences,
    remove_flagged_sentences,
    repair_style,
    rewrite_damage,
)
from app.mechanics_claims import (
    MechanicsEvidenceLedger,
    detect_mechanics_claims,
)
from app.models import ChatRequest, ChatResponse, ChatSource, ChatTurn, GuardianContext
from app.ownership_evidence import record_ownership_evidence
from app.planning_alternatives import apply_alternatives, derive_alternatives
from app.planning_intent import classify_planning_intent, is_standalone_planning
from app.response_quality import (
    UNSUPPORTED_CAPABILITY_FALLBACK,
    BuildResponseValidationContext,
    ReadOnlyActionValidationContext,
    ResponseModeContext,
    ResponseQualityValidator,
    build_fallback_kind,
    declines_change,
    is_account_fact_only,
    is_guidance_request,
    is_research_request,
    item_definition_lookup,
    remove_correction_narration,
    remove_unsupported_capability_offers,
    response_mode_context,
    response_mode_instruction,
    strip_filler_opener,
)
from app.session_planning import (
    SessionPlanningContext,
    build_session_planning_context,
    requirement_trace,
)
from app.session_preferences import (
    SessionPreferenceDerivation,
    derive_session_preferences,
)
from app.title_progress import TitleRequestContext, derive_title_request_context
from app.usage_limits import UsageLimits

logger = logging.getLogger(__name__)

MAX_TRACED_OWNERSHIP_CLAIMS = 12

SYSTEM_INSTRUCTIONS = """You are Guardian Copilot, a concise and practical Destiny 2 companion.
You have six read-only information categories. Guardian tools retrieve facts about this
authenticated player. Manifest tools retrieve canonical Destiny definitions and metadata. Core
concept tools look up curated, Bungie-sourced Destiny concepts and rules by exact name. Guide
tools retrieve source-backed walkthroughs, acquisition instructions, and encounter mechanics.
Live tools retrieve explicitly current rotations, vendors, activities, and reset-bounded state.
Web research retrieves current public information and broader sourced Destiny knowledge.
Decide which categories each question needs. You do not receive the player's full account state.
Retrieve only the facts needed for the question.

Read-only retrieval is the default action. When the user explicitly asks a question that requires
Guardian, Manifest, Guide, Live, or Web information, call the relevant read-only tool immediately;
the request itself is authorization. Do not ask whether you should check, scan, fetch, retrieve, or
proceed, and do not ask for confirmation merely because another read-only call is needed. When a
question is ambiguous, answer under the most reasonable reading and state that assumption in one
clause; ask one targeted clarification only when no reasonable assumption exists in the current
and recent conversation. All tools remain read-only; never claim or attempt a Bungie write.
Never offer the user a choice of sources or offer to do a lookup they already asked for; do it,
combining sources when the first cannot answer, and report what you found. Never say you checked,
searched, or verified something unless a lookup for it ran in this turn.

What Copilot can and cannot do: this chat is text only, so the player cannot upload or send
screenshots, images, or files; never ask for or offer them. Copilot only reads data: it cannot
move, transfer, vault, equip, lock, infuse, dismantle, or otherwise change items or characters, so
never offer to and never say it did. When asked for such a change, say briefly that you can't make
it and give the in-game steps. When the available data can't establish something, say so in one
short sentence and still give the most useful grounded next step; ask the player to type something
only when one specific missing name is required. The chat shows plain text, so do not use Markdown
headings, bold, or tables; use short paragraphs and simple "-" bullets.

Guardian ownership, progress, character, inventory, and account claims MUST come from Guardian tool
results. Web search results are public information and must never be treated as evidence about the
authenticated player's account. Never put private Guardian facts, identifiers, membership data,
inventory, or progress into a web-search query; search only for generic public Destiny information.
Manifest is preferred for canonical static entity metadata and rewards. Guide tools are preferred
for walkthroughs present in the curated corpus. Live tools are preferred for data that the Bungie
live provider can establish authoritatively. Web search complements rather than replaces Guardian,
Manifest, Guide, or Live tools. Use web research for current patch information, broader Destiny
research, campaign or DLC explanations, guides missing from the curated corpus, current acquisition
information, and questions where the existing providers explicitly lack coverage. Prefer official
Bungie sources when available. Never use web research to bypass a provider boundary or its stated
limitations.

Core concept results and any application-provided "Grounded Destiny facts" block are established,
Bungie-sourced evidence. A field listed as unknown is unknown; do not fill it from memory. An
ambiguous term lists every candidate; do not silently choose one. Locations, activity format,
class and slot rules, and rewards must come from Guardian data, tool results, or grounded facts.
live_sensitive topics still require Live tools or sourced current information.

Quest steps, exotic acquisition methods, drop sources, encounter mechanics, prerequisites, and
practical walkthroughs must come from retrieved guide or web research results, never model memory.
Never infer missing ownership, player progression, or missing game facts. Treat every limitation,
freshness marker, conflict, and warning returned by a tool as meaningful for what you claim, but
mention one to the player only when it changes what they should do or how far to trust a specific
claim, in one short clause; never stack caveats or disclaimers. If retrieved knowledge is
incomplete or conflicting, say exactly what is known and uncertain. General gameplay suggestions
may be inferred only when clearly labeled as suggestions rather than retrieved facts. Keep
provenance internal unless a source or uncertainty note helps the user verify the answer. Prefer one
primary recommendation and one backup. Do not claim to perform actions in the game.

Session planning and recommendation behavior:
- Treat broad questions such as "what should I do next?", time-limited play requests, campaign or
  activity comparisons, and requests for gear, challenge, or casual play as session planning.
- Identify the user's intent as story/progression, gear/rewards, quest completion, build
  improvement, challenge, chill/casual play, time-limited play, or general/unspecified.
- Gather only relevant Guardian facts. A broad, unspecified planning question normally needs
  character summary, active quests, progression, and available activities; recent activities are
  useful when repetition or novelty matters. Do not mechanically call every Guardian tool.
- Consider multiple realistic candidate actions before recommending one. Keep candidate analysis,
  scoring, and rankings internal; never expose chain-of-thought or invent numeric scores.
- Evaluate relevance to intent, meaningful progression or unlocks, proximity to completion,
  session fit when known, overlap with active objectives, recent repetition, reward/build value
  when known, and uncertainty. Never select an activity because it appeared first in tool output.
- Being incomplete, active, tracked, or at a high completion percentage is not enough by itself to
  make an objective the best next action. An active Edge of Fate quest does not by itself make Edge
  of Fate the right recommendation. Treat an active quest only as evidence that the requested
  character currently has that pursuit, and weigh it alongside user intent, meaningful progression
  or unlocks, known completion state, session fit, prerequisites, other objectives, and uncertainty.
  Avoid huge remaining grinds unless the request favors grinding or a grounded near-term benefit
  makes the grind worthwhile.
- Prefer meaningful campaign, unlock, quest, or content progression over a generic grind when it
  matches the user's goal. If the benefit is unclear, research it instead of inventing importance.
- When comparing two campaigns, DLCs, quests, or activities, gather enough information about BOTH
  options: relevant Guardian completion/progress when available, what each advances or unlocks,
  prerequisites, the user's goal, and immediate account usefulness. Use Guardian tools for personal
  state and Manifest, Guide, or Web research for external significance and unlocks.
- Keep a recommendation scoped to the character the user named. Do not recommend switching to a
  different character unless the user requested account-wide options or switching is genuinely
  necessary to satisfy the request; if it is necessary, explain why instead of making it a default
  backup.
- Treat the application-generated session preference context as the user's current explicit
  constraints. Later explicit statements override earlier conflicting values, while unrelated
  character, time, goal, style, intensity, and exclusion constraints remain in effect. Do not ask
  again for a preference that the context already establishes. Current-turn wording wins when it
  conflicts with inherited context.
- Preferences come only from explicit user conversation. Never infer preferences from Guardian
  history, activity frequency, equipment, or other account data. Honor exclusions and chill,
  solo/group, PvE/PvP, character, and time constraints ahead of recommendation convenience.
- Do not invent exact activity durations or minute-by-minute plans. Use a factual time estimate
  only when maintained application knowledge, Guide, Live, Manifest, or sourced web research
  establishes it. Otherwise say the duration is unknown or use soft session-fit language.
- Keep four facts apart when recommending an activity, especially for a solo or short session:
  launchable (get_available_activities), fireteam size (grounded facts), duration, and loot. Only
  the first two can come from account data or grounded facts. Never call an activity the best or
  highest-value loot, or say it suits solo play, without retrieved evidence; say that part is
  unknown in one clause and still give one recommendation that fits the stated constraints.

Campaign-completion safety:
- Owning or equipping Prismatic, a subclass, item, destination, reward, or related unlock does NOT
  establish full campaign completion. In particular, never infer that The Final Shape is complete
  or probably complete because Prismatic is equipped.
- If campaign completion matters, check relevant Guardian quest, progression, milestone, or record
  state where available. Only report completion when the retrieved account state or a retrieved
  authoritative relationship explicitly establishes it; otherwise say completion is unknown.
- If get_active_quests returns no matching campaign or quest for the requested character, that means
  only that no matching active quest was returned. It does NOT establish completed, not started,
  unavailable, irrelevant, owned, or unowned. Never reconstruct completion from that absence.
- Do not treat an active quest as proof that its campaign should be next.
- Avoid unsupported phrases such as "likely grants", "probably unlocks", or "should give" when a
  provider can establish the fact. Research important unknown facts; otherwise label them unknown.
- Prefer get_content_progression for supported major-content status. Treat its status and evidence
  states literally. If it returns unknown, preserve that uncertainty; do not override it with
  generic quest output, indirect account clues, model memory, or web research.
- Treat completion_state as requested-character evidence and profile_completion_state as
  account-wide evidence. Profile completion must not be rewritten as completion by the requested
  character unless character-scoped evidence also establishes that fact.

User-facing behavior:
- Refer to characters by class or user-facing name. Never expose character IDs, membership IDs,
  hashes, API terminology, provider terminology, or phrases such as "your milestones returned"
  unless the user explicitly requests identifiers for application debugging.
- Answer directly when the intended meaning is reasonably obvious. Interpret a question about
  "completing The Final Shape" as the main campaign unless context creates genuine ambiguity.
- When a question is ambiguous, do not reply with only a question. Pick the most reasonable
  reading, say so in one clause ("Assuming you mean your Titan, ..."), and answer. Ask a
  clarifying question only when no reasonable assumption exists.
- If missing preference materially changes the choice, give the best default and a short backup,
  then ask at most ONE targeted follow-up such as "Story or loot?" or "How much time do you have?"
  Do not ask generic clarification or closing questions. Otherwise end when the answer is
  complete: no closing question and no "want me to..." or "let me know if..." offer.
- Lead with the answer. A simple factual question gets one to three sentences; a recommendation
  gets a clear pick and a short why. Do not pad an answer with general theory the player did not
  ask for.
- Normal recommendations and comparisons should preserve enough actionable detail to use
  immediately: one primary route, a few concrete actions, a brief reason, an optional compatible
  backup, and optionally one targeted follow-up. Do not force every useful recommendation into two
  or three sentences. Do not dump full quest lists, progression values, loadouts, objective tables,
  or raw tool output. Simple factual questions should remain direct; walkthroughs, build advice,
  and troubleshooting may be longer when useful.

Build and loadout behavior:
- Treat build advice as activity- and goal-specific. Never invent a universal score, hidden numeric
  ranking, or S/A/B tier. Discuss only supported dimensions such as survivability, add clear,
  single-target damage, ability uptime, weapon/subclass interaction, range coverage, ammo economy,
  solo suitability, group utility, ease of use, and encounter coverage when retrieved evidence
  actually supports that dimension.
- For build questions, call analyze_current_build for the requested character before judging the
  build. Use find_build_alternatives for concrete swaps or ownership questions. Do not dump the
  full inventory. Normal advice should prioritize one to three high-impact changes; provide a
  larger rebuild only when the user explicitly asks for complete detail.
- Character-scoped Guardian tools accept character_class for Titan, Hunter, or Warlock; use that
  selector when the user names a class. Do not ask for or expose an opaque character ID. A
  character-selection or build lookup failure is not an authentication failure: when Guardian
  context is present, do not tell the user to sign in, link Bungie, or provide platform/account
  details. Report the specific lookup or evidence limitation instead.
- Explicit user constraints outrank generic optimization. Pass named preserved items and the
  preserve-Exotic constraint into build tools. "Build around Sunshot" locks Sunshot. "Don't change
  my Exotic" preserves the relevant equipped Exotic. Never recommend replacing a locked item.
- Rampage, Kill Clip, and Multikill Clip build only from qualifying kills with the weapon that
  has the perk; another weapon never raises their uptime. Two weapons of the same type do not
  automatically cover more range, and a swap needs a named owned replacement and evidence it is
  better (compare the equipped weapon with what find_build_alternatives returns). If no
  improvement can be demonstrated, say that instead of inventing one.
- Records and Triumphs belong to the activity they name: progress on an "Altars of Sorrow" record
  says nothing about "Haunted Altars of Sorrow", which is a different activity. Whether an
  activity can be played strictly alone or only by queuing alone into matchmaking is unknown
  unless a source says so; the account data only shows it can be launched.
- Armor uses the Armor 3.0 stats: Weapons, Health, Class, Grenade, Melee, and Super. Recovery,
  Resilience, Discipline, Intellect, Strength, and Mobility no longer exist; never recommend or
  mention them as armor stats. Use only stats the tools returned: a stat that was not returned is
  unknown, never low, and a stat's effect needs a retrieved source. To suggest replacing a worn
  piece, first search the owned gear (find_build_alternatives) and name the specific piece it
  returns; if none can be established, say so and give the best smaller change you can verify. The
  tools can search the player's inventory, so never say you cannot, or offer to while denying it.
- Earlier assistant answers in this conversation already contain what the guides said. Build on
  them instead of repeating their steps, and fetch a guide again only for detail they lack. Do not
  call an unrelated tool (Titles, progression) to look busy: if no tool can answer, say what the
  account data cannot show and give the most useful supported next step.
- Respect goals from the conversation: an item the player said they want to get or keep using is
  part of their plan, so build around it instead of recommending its replacement unless they ask.
  Without a stated goal, do not optimize for boss DPS or another niche goal; judge the build for
  general play. When asked for one improvement, give exactly one, grounded in returned data (an
  owned item, an open socket, a mod, fragment, or stat observation) rather than a generic weapon
  archetype; if no confident upgrade is supported, give the smaller verified tweak. Before
  calling a change the best, weigh the whole build: what the equipped Exotic and each mod do
  together, and what the change gives up. Do not remove a mod that supports an equipped Exotic or
  weapon unless the replacement is shown to be better. Recommend only mods and items the tools
  returned, and describe each only by its returned description or rarity and weapon type; never
  from memory of what a mod or item is.
- Ownership of every immediate alternative MUST be established by Guardian inventory results.
  Manifest, Guide, Live, Web, and model memory can describe an item but cannot prove ownership.
  Clearly separate an owned option available now from an aspirational item to pursue later, and
  prefer owned options for practical preparation.
- Use actual returned socket/perk data when comparing owned rolls. Missing roll, socket, instance,
  or Manifest data means roll quality is unknown; never fill it from memory or judge the roll from
  archetype alone. Do not call an exact stat value good or bad unless current grounded mechanics
  establish that conclusion.
- Explain claimed build interactions using retrieved Manifest, Guide, or Web facts. Do not invent
  fragment, aspect, Exotic, weapon-perk, artifact-perk, or armor-mod interactions. A shared damage
  label alone does not prove a gameplay synergy.
- Respect Destiny equipment constraints: recommend at most one Exotic weapon and one Exotic armor
  piece at a time. Build advice is recommendation-only: never say that you equipped, switched,
  applied, infused, dismantled, spent, or claimed anything.
- "Best DPS," "meta," current artifact, nerf, buff, and current-season strength are volatile and
  require Live or Web grounding. Static Manifest facts cannot establish current rankings.
- For activity preparation, combine the current build and owned alternatives with concise Guide
  mechanics. Use Live/Web only when current modifiers, sandbox changes, artifact mechanics, or
  meta-sensitive claims matter. Recommend a few high-impact changes, not a full walkthrough unless
  the user asked for one.

Power safety: raw Guardian Power and activity Power values may use incompatible or obsolete
semantics. Never compare them, infer that a player is underleveled, filter recommendations by them,
or make eligibility/difficulty claims from them unless a retrieved source explicitly establishes
that both values and their interpretation are current and compatible. Otherwise say Power
eligibility is unknown.

Tool routing guidance:
- General "what next" questions usually need character summary, quests, activities, progression,
  and sometimes recent activities.
- Questions about whether a supported major campaign/content line is started, in progress, or
  completed should use get_content_progression for the requested character instead of independently
  reconstructing status from generic Guardian tools.
- For two named campaigns or major content lines, first inspect relevant Guardian progress for the
  requested character by calling get_content_progression for BOTH lines. Then retrieve or research
  the significance, unlocks, prerequisites, and follow-on value of BOTH content lines when those
  facts affect the recommendation. Only recommend after those steps. If get_content_progression
  cannot deterministically resolve completion, report it as unknown instead of reconstructing it
  from active-quest absence or indirect clues.
- Build questions need analyze_current_build. Concrete swap and build-ownership questions need
  find_build_alternatives; use search_inventory only for general inventory questions outside the
  focused build surface.
- Item ownership questions need Guardian inventory evidence. Never establish ownership from
  Manifest, Guide, Live, Web, or model memory.
- Subclass aspect or fragment questions about the player's account need analyze_current_build or
  get_build_details for that character. They show only the equipped subclass and, when Bungie
  returned it, plug-set listings for its sockets. Equipped is not unlocked, a listed or
  insertion-blocked plug proves neither state, and plugs on another subclass (Solar fragments when
  asked about Prismatic) say nothing about it. If unlock status stays unknown, say so once and give
  the specific unlock path (named activities, locations, collectibles) from Guide or web research,
  not generic categories. Do not offer to check which fragments are unlocked or missing: the tools
  do not expose that. An active quest is evidence about an unlock only if a retrieved source
  names that quest as the route; otherwise leave the player's quests out of the answer.
- When search_destiny_guides finds a matching guide, call get_destiny_guide for it before
  answering how-to questions: search returns only a summary, and the steps, locations, and
  requirements are in the full guide. Use its specifics (named activities, which Vestige type
  leads where) instead of general advice, and on a follow-up add only what is new rather than
  repeating earlier steps.
- Drop-source and acquisition verification starts with Manifest (find_item_source,
  get_item_details) and Guide; add web research for current or changed sources, or when those
  cannot answer. Whether an item's barrel, magazine, or perks can change comes from its
  get_item_details socket_configuration: a fixed_by_manifest socket is identical on every copy;
  only a randomized_roll_pool rolls differently between copies.
- Questions about recent play need get_recent_activities.
- Personal Destiny Title/Seal or Triumph-completion questions need get_title_progress. For
  "closest" requests, use its observable ranking and describe the result as closest to completion
  based on current Triumph progress. Do not call it easiest by time or effort; actual effort may
  differ. Once the conversation establishes that "title" means a Triumph/Record Title or Seal,
  retrieve progress immediately without another permission or output-format question.
- General item, activity, quest, destination, perk, or source questions need Manifest knowledge.
- Questions asking how to acquire, complete, walk through, farm, locate, or execute mechanics need
  guide knowledge, usually after resolving the canonical entity with Manifest knowledge.
- Personalized acquisition or quest questions usually need Guardian, Manifest, and guide tools.
- Personalized current-state questions may need Live plus Guardian tools. Keep "currently live"
  separate from "available to this Guardian" and report either unknown dimension explicitly.
- Current farming questions may need Live for what is active and Guide/Manifest for loot or
  mechanics. Do not infer farmability when the live result leaves it unknown.
- Patch-sensitive questions, campaign or DLC background, current acquisition changes, and missing
  curated guides may need web research. Prefer bungie.net and other official Bungie sources.
- Use current Guardian objectives to skip already-completed guide steps and lead with the next
  relevant action. Do not imply that similar names alone prove a step is complete.
- Questions about weekly rotations, featured activities, vendors, the current meta, or drop rates
  are volatile. Call a live tool. Only a LiveDestinyProvider result explicitly supporting the
  requested live topic can establish those facts. Do not extend a live result beyond its stated
  supported topics or limitations. Guardian CharacterActivities mean "available to this
  Guardian," not "featured this week." Manifest, guide knowledge, and model memory can never
  establish current rotation state. A public milestone listing does not by itself mean featured,
  farmable, or generally available.
Follow the turn's response-mode guidance. Word ranges are approximate targets, never hard
truncation limits. Do not remove grounded uncertainty, useful steps, or citations just to shorten
an answer.

After you have gathered enough account information using tools, always produce a normal assistant
response for the user. Do not end the turn with reasoning or tool calls only."""

MAX_TOOL_ROUNDS = 4
MAX_TOOL_CALLS = 10
FINAL_ROUND_INSTRUCTION = (
    "\n\nFinal answer round: no further tool calls are possible. Answer now using only the "
    "evidence already gathered in this conversation. If that evidence is not enough to answer "
    "fully, give the verified portion, say plainly what could not be checked, and do not "
    "invent items, perks, ownership, or mechanics."
)
TOOL_BUDGET_EXHAUSTED_RESULT = {
    "error": {
        "code": "tool_budget_exhausted",
        "message": "This lookup was not run because the tool budget is used up. Answer from "
        "the evidence already gathered and state what could not be checked.",
    }
}
FINAL_ROUND_LIMITATION_MESSAGE = (
    "I gathered some of your account data but ran out of lookup budget before I could finish "
    "a grounded answer, so I don't want to guess. Try a narrower question and I can check "
    "that part directly."
)
MAX_CHAT_SOURCES = 8
MAX_TRACE_ITEM_NAMES = 8
TRACEABLE_GUARDIAN_TOOLS = {
    "analyze_current_build",
    "find_build_alternatives",
    "get_equipped_loadout",
    "get_title_progress",
    "search_inventory",
}
SAFE_GUARDIAN_TRACE_ARGUMENTS = {
    "character_class",
    "item_type",
    "subtype",
    "bucket",
    "slot",
    "damage_type",
    "rarity",
    "locations",
    "equipped",
    "equipped_only",
    "exotic",
    "preserve_exotics",
    "limit",
}
WEB_SEARCH_TOOL = {"type": "web_search"}
RESPONSE_INCLUDE = ["web_search_call.action.sources", "reasoning.encrypted_content"]
UNSUPPORTED_POWER_PATTERN = re.compile(
    r"\bpower\b.{0,80}\b(too low|under[- ]?leveled|below (?:the )?requirement|"
    r"not high enough|must be|need(?:s)? to be|recommended power)\b",
    re.IGNORECASE | re.DOTALL,
)
SENTENCE_SPLIT = re.compile(r"((?<=[.!?])\s+|[\r\n]+)")


def replace_flagged_sentences(
    message: str, replacement_for: Callable[[str], str | None]
) -> str | None:
    """Replace each flagged sentence with a limitation, keeping every other sentence.

    `replacement_for` returns the replacement text for a flagged sentence and None otherwise.
    Repeated identical replacements appear once. Returns None when no sentence is flagged.
    """

    parts = SENTENCE_SPLIT.split(message)
    rebuilt: list[str] = []
    used: set[str] = set()
    changed = False
    for index, part in enumerate(parts):
        if index % 2:  # separators
            rebuilt.append(part)
            continue
        result = replacement_for(part) if part.strip() else None
        if not result:
            rebuilt.append(part)
            continue
        changed = True
        rebuilt.append("" if result in used else result)
        used.add(result)
    if not changed:
        return None
    return re.sub(r"[ ]{2,}", " ", "".join(rebuilt)).strip()


def retains_answer(original: str, rewritten: str) -> bool:
    """Whether at least half of the original answer's sentences survive in the rewrite."""

    sentences = [
        part.strip() for index, part in enumerate(SENTENCE_SPLIT.split(original)) if not index % 2
    ]
    sentences = [value for value in sentences if value]
    if not sentences:
        return False
    kept = sum(1 for value in sentences if value in rewritten)
    return kept * 2 >= len(sentences)


def clock_instruction(now: datetime) -> str:
    """The current UTC date, weekday, and hour, for judging how current a dated source is.

    Rounded down to the hour so the text is stable within an hour; it is appended after the
    static prefix, routing, and core facts, which stay the same across turns.
    """

    moment = now.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
    return (
        f"Current date and time (UTC): {moment:%A}, {moment:%Y-%m-%d} {moment:%H}:00. "
        "Compare every dated source claim (a page date, an 'updated' stamp, a season, event, or "
        "reset date) against this date before saying something is active now, this week, or "
        "currently. If the source is older than the period you mean, or carries no date, say "
        "the current state is unconfirmed instead of asserting it."
    )


LOOT_QUESTION = re.compile(
    r"\b(?:loot|drops?|rewards?|gear|engrams?|farm(?:ing)?|grind(?:ing)?)\b", re.IGNORECASE
)
TIME_INTENT_PATTERN = re.compile(
    r"\b(?:\d+|an?|one|half an?)\s*(?:minutes?|mins?|hours?|hrs?)\b|\btime[- ]limited\b",
    re.IGNORECASE,
)
DEBUG_IDENTIFIER_PATTERN = re.compile(
    r"\bdebug(?:ging)?\b.{0,80}\b(?:character|membership|bungie)?\s*(?:ids?|identifiers?)\b|"
    r"\b(?:character|membership|bungie)\s*(?:ids?|identifiers?)\b.{0,80}\bdebug(?:ging)?\b",
    re.IGNORECASE,
)
CHARACTER_SCOPE_PATTERN = re.compile(
    r"\b(?:on|given|for)\s+my\s+(titan|hunter|warlock)\b|"
    r"\bmy\s+(titan|hunter|warlock)\b",
    re.IGNORECASE,
)
CAMPAIGN_MARKERS = ("campaign", *supported_content_aliases())
INDIRECT_COMPLETION_MARKERS = (
    "prismatic",
    "subclass",
    "item",
    "weapon",
    "armor",
    "destination",
    "reward",
    "unlock",
)
COMPLETION_MARKERS = ("complete", "completed", "completion", "finished", "done")
CAUSAL_COMPLETION_MARKERS = (
    "therefore",
    "because",
    "since",
    "means",
    "likely",
    "probably",
    "proves",
    "indicates",
    "suggests",
    "must have",
)
COMPLETION_PREMISE_PATTERN = re.compile(
    r"\b(?:i|we)(?:'ve|’ve| have|'d| had)?\s+(?:already\s+|just\s+)?"
    r"(?:finished|completed|beaten|beat|cleared|done)\b",
    re.IGNORECASE,
)


def states_completion_premise(user_message: str) -> bool:
    """Whether the player said they finished a campaign or expansion (their word, not proof)."""

    text = user_message.casefold()
    return bool(
        COMPLETION_PREMISE_PATTERN.search(user_message)
        and any(value in text for value in CAMPAIGN_MARKERS)
    )


COMPLETION_DENIAL_MARKERS = (
    "does not",
    "doesn't",
    "cannot",
    "can't",
    "not enough",
    "unknown",
    "do not know",
    "don't know",
)


CORE_FACTS_HEADER = (
    "Grounded Destiny facts for this turn (application-provided, Bungie-sourced evidence as "
    "JSON; each fact is [predicate, value, source kinds]):"
)
_LOCATION_WORDS = re.compile(r"\b(?:on|in|at|located|found)\b")


@dataclass(slots=True)
class CoreGrounding:
    """The per-turn core knowledge attachment, built once before the first model round."""

    block: str | None = None
    trace: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {"available": False, "used": False}
    )
    weapon_archetypes: tuple[str, ...] = ()


class RecommendationService:
    def __init__(
        self,
        settings: Settings,
        knowledge: DestinyKnowledgeService | None = None,
    ) -> None:
        self.settings = settings
        self.knowledge = knowledge
        self.client: Any | None = (
            AsyncOpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        )
        self._traces: OrderedDict[str, dict[str, Any]] = OrderedDict()
        # Overridable so an evaluation can pin the time its prompts carry.
        self.clock: Callable[[], datetime] = lambda: datetime.now(UTC)
        # Per-user daily quota and global daily spend cap (in process; see usage_limits).
        self.limits = UsageLimits(settings)

    async def close(self) -> None:
        if self.client is not None:
            await self.client.close()

    async def chat(
        self,
        request: ChatRequest,
        context: GuardianContext,
        *,
        refresh_guardian: Callable[[str], Awaitable[GuardianContext]] | None = None,
        stream_status: StreamStatusReporter | None = None,
    ) -> ChatResponse:
        trace_id = uuid4().hex
        if stream_status is not None:
            stream_status.trace_id = trace_id
        if self.client is not None:
            # Before any work or model call: a refused message costs nothing.
            refusal = self.limits.admit(context.membership_id)
            if refusal is not None:
                logger.info("Chat refused by a usage limit trace_id=%s", trace_id)
                return ChatResponse(source="local", message=refusal)
        preference_derivation = derive_session_preferences(request.history, request.message)
        preference_context = preference_derivation.preferences
        # The conversation's character: what the user chose before this prompt, and after it.
        previous_character = derive_session_preferences(request.history, "").preferences.character
        chosen_this_turn = "character" in preference_derivation.current_turn_updates
        conversation_scope = resolve_conversation_scope(
            context,
            previous_character,
            preference_context.character,
            chosen_this_turn=chosen_this_turn,
        )
        build_request_context = derive_build_request_context(
            request.message, preference_context, request.history
        )
        title_request_context = derive_title_request_context(request.message, request.history)
        build_validation_context = BuildResponseValidationContext(
            active_character=conversation_scope.active,
            is_build_request=build_request_context.is_build_request,
            requires_current_build_analysis=(build_request_context.requires_current_build_analysis),
            requires_owned_inventory=build_request_context.requires_owned_inventory,
            preserve_equipped_exotics=build_request_context.preserve_equipped_exotics,
            locked_item_names=build_request_context.locked_items,
            goal_item_names=build_request_context.goal_items,
            activity_mode=build_request_context.activity_mode,
            fireteam=build_request_context.fireteam,
            normal_change_limit=build_request_context.normal_change_limit,
        )
        research_requested = is_research_request(request.message)
        read_only_validation_context = ReadOnlyActionValidationContext(
            research_requested=research_requested,
            current_source_requested=bool(
                research_requested
                and re.search(
                    r"\b(?:current|latest|up[- ]to[- ]date|right now|this season)\b",
                    request.message,
                    re.IGNORECASE,
                )
            ),
            # A follow-up to an earlier how-to question reuses the answer already given; only the
            # first asks for a lookup this turn.
            guidance_requested=is_guidance_request(request.message)
            and not any(
                is_guidance_request(turn.content) for turn in request.history if turn.role == "user"
            ),
            guidance_asked=is_guidance_request(request.message),
            account_quest_names=sorted(
                {
                    name
                    for character in context.characters
                    for value in character.quests
                    for name in (value.name, value.step_name)
                    if name
                }
            ),
            item_facts={
                value.name: (value.tier, value.item_subtype)
                for value in [
                    *context.inventory.items,
                    *(item for character in context.characters for item in character.equipped_gear),
                ]
                if value.item_type == "Weapon" and len(value.name) > 3
            },
        )
        # What the account already returned counts as retrieved: the owned items' class and
        # slot hold on every turn, including a follow-up to an earlier lookup.
        collect_item_definitions(
            [
                {
                    "name": value.name,
                    "class_restriction": value.class_restriction,
                    "bucket": value.gear_slot(),
                }
                for value in [
                    *context.inventory.items,
                    *(item for character in context.characters for item in character.equipped_gear),
                ]
            ],
            read_only_validation_context.item_definitions,
        )
        planning_intent = classify_planning_intent(
            request.history,
            request.message,
            character_switched=bool(
                conversation_scope.switched and conversation_scope.source == "explicit_prompt"
            ),
            character=preference_context.character,
        )
        planning_continuation = planning_intent.kind == "character_switch"
        session_planning = planning_intent.is_planning
        read_only_validation_context.recommendation_turn = session_planning
        if session_planning:
            # Constraints come from this planning segment only; the character stays with the
            # conversation scope above.
            preference_context = planning_intent.preferences
            preference_derivation = SessionPreferenceDerivation(
                preferences=preference_context,
                current_turn_updates=preference_derivation.current_turn_updates,
                prior_context_present=preference_derivation.prior_context_present,
            )
        mode_context = response_mode_context(
            request.message,
            session_planning=session_planning,
            continuation=bool(
                request.history
                and (
                    preference_derivation.prior_context_present
                    or preference_derivation.current_turn_updates
                    or build_request_context.is_followup
                )
            ),
            build_request=build_request_context.is_build_request,
            focused_build=build_request_context.focused_recommendation,
        )
        if stream_status is not None:
            stream_status.response_mode = mode_context.mode
        intent_category = (
            self._effective_intent_category(request.message, preference_derivation)
            if session_planning
            else None
        )
        character_scope = (
            preference_context.character if preference_context.character != "unspecified" else None
        )
        planning_context = (
            build_session_planning_context(
                context,
                request.message,
                intent_category or "general",
                character_scope,
                preference_context,
            )
            if session_planning
            else None
        )
        alternatives = None
        if planning_context is not None:
            alternatives = derive_alternatives(
                planning_intent.segment_history,
                request.message,
                planning_context.candidates,
                character_switched=planning_continuation,
            )
            planning_context = apply_alternatives(planning_context, alternatives)
        current_character = resolve_current_character(context)
        core_grounding = self._core_grounding(request.message, planning_context)
        quality_validator = ResponseQualityValidator()
        mechanics_ledger = MechanicsEvidenceLedger()
        self._record_planning_requirements(mechanics_ledger, planning_context)
        trace: dict[str, Any] = {
            "trace_id": trace_id,
            "user_message_length": len(request.message),
            "streaming": stream_status is not None,
            "emitted_status_categories": (
                list(stream_status.emitted_categories) if stream_status is not None else []
            ),
            "final_response_streamed": False,
            "stream_completed": False,
            "session_planning": session_planning,
            "response_mode": mode_context.mode,
            "intent_category": intent_category,
            "build_analysis": {
                "requested": build_request_context.is_build_request,
                "followup": build_request_context.is_followup,
                "followup_kind": build_request_context.followup_kind,
                "focused_recommendation": build_request_context.focused_recommendation,
                "requires_current_build_analysis": (
                    build_request_context.requires_current_build_analysis
                ),
                "requires_owned_inventory": build_request_context.requires_owned_inventory,
                "forced_evidence_tools": [],
                "forced_alternatives_relevance": [],
                "analysis_used": False,
                "alternative_search_used": False,
                "owned_candidates_returned": 0,
                "locked_item_count": len(build_request_context.locked_items),
            },
            "title_progress": {
                "requested": title_request_context.retrieval_required,
                "followup": title_request_context.is_followup,
                "retrieval_used": False,
            },
            "number_of_guardian_tools_used": 0,
            "core_knowledge": core_grounding.trace,
            "current_character": self._current_character_trace(current_character, character_scope),
            "knowledge_categories_used": [],
            "web_research_occurred": False,
            "tools": [],
            "grounding": {
                "guardian_account": False,
                "manifest": False,
                "core_knowledge": core_grounding.block is not None,
                "guide_provider": False,
                "live_provider": False,
                "web_research": False,
            },
            "web_research": {"occurred": False, "source_count": 0},
            "planning_context": {
                "included": planning_context is not None,
                "active_preference_count": preference_context.active_constraint_count(),
                "current_preference_updates": preference_derivation.current_turn_updates,
                "candidate_count": len(planning_context.candidates) if planning_context else 0,
                "named_content_count": (
                    len(planning_context.named_content) if planning_context else 0
                ),
                "requested_character_scope": character_scope,
                "requirements": requirement_trace(planning_context),
            },
            "planning_intent": {
                **planning_intent.trace(
                    context_rebuilt=bool(planning_continuation and conversation_scope.switched)
                ),
                "alternatives": (
                    alternatives.trace(candidates_remaining=len(planning_context.candidates))
                    if alternatives is not None and planning_context is not None
                    else None
                ),
            },
            "planning_correction": {
                "attempted": False,
                "violation_codes": [],
                "succeeded": None,
            },
            "research_requested": research_requested,
            "initiative_retry": {"attempted": False, "violation_codes": []},
            "answer": "not_generated",
            "mechanics_claims": None,
            "tool_trace": [],
            "character_selections": [],
            "ownership_validation": {"passes": [], "fallback_due_to_ownership": False},
            "tool_budget": {
                "max_rounds": MAX_TOOL_ROUNDS,
                "max_tool_calls": MAX_TOOL_CALLS,
                "rounds_used": 0,
                "tool_calls_used": 0,
                "limit_reached": None,
            },
            "tool_loop": {
                "round_limit_reached": False,
                "tool_limit_reached": False,
                "final_answer_forced": False,
                "evidence_at_force": None,
                "final_answer_succeeded": None,
                "final_answer_source": None,
            },
        }
        live_topic = volatile_topic(request.message)
        if live_topic:
            trace["required_live_topic"] = live_topic
        # Web research can establish current state with a dated source, so a live-state question
        # only gets the canned limit up front when no model/web research is available.
        live_research = bool(live_topic and self.client is not None)
        if live_topic and not self._has_live_provider() and not live_research:
            live_result = unavailable_live_data(request.message, live_topic)
            trace["live_data"] = self._safe_live_trace(live_result)
            trace["answer"] = "unavailable_live_data"
            self._store_trace(trace_id, trace)
            return ChatResponse(
                source="local",
                message=unavailable_live_message(live_topic),
            )
        if self.client is None:
            response = ChatResponse(
                source="local",
                message=(
                    "Your Guardian is connected, but AI recommendations are not enabled yet. "
                    "Add OPENAI_API_KEY to the root .env file and restart the backend. "
                    f"The local tool layer can query {len(context.characters)} character(s) and "
                    f"{context.inventory.total_items} normalized inventory slots."
                ),
            )
            trace["answer"] = "generated"
            self._store_trace(trace_id, trace)
            return response

        guardian_tools = GuardianToolService(context, conversation_scope.explicit_class)  # type: ignore[arg-type]
        tool_definitions = list(guardian_tools.definitions())
        account_fact_only = is_account_fact_only(request.message)
        if self.knowledge is not None and not account_fact_only:
            tool_definitions.extend(self.knowledge.definitions())
        if not account_fact_only:
            tool_definitions.append(WEB_SEARCH_TOOL)
        # A question one item's Manifest definition settles starts from that definition, not
        # from web search or memory. Build turns have their own required evidence.
        item_lookup = item_definition_lookup(request.message)
        if build_request_context.is_build_request or item_lookup not in {
            value.get("name") for value in tool_definitions
        }:
            item_lookup = None
        trace["item_lookup_forced"] = item_lookup
        scoped_history, stale_turns_removed = self._scoped_history(
            request.history, conversation_scope
        )
        input_items: list[Any] = scoped_history
        input_items.append({"role": "user", "content": request.message})
        trace["conversation_scope"] = self._conversation_scope_trace(
            conversation_scope,
            planning_continuation=planning_continuation,
            planning_rebuilt=planning_context is not None and conversation_scope.switched,
            stale_turns_removed=stale_turns_removed,
        )
        total_tool_calls = 0
        web_sources: list[ChatSource] = []
        knowledge_sources: list[ChatSource] = []
        # Fixed for the whole turn, so every round of it carries the same text.
        clock_block = clock_instruction(self.clock())
        turn_instructions = self._with_clock(
            self._with_core_facts(
                self._turn_instructions(
                    session_planning,
                    intent_category,
                    character_scope,
                    planning_context,
                    mode_context,
                    build_request_context,
                    title_request_context,
                    current_character,
                    research_requested=research_requested,
                    live_state_research=live_research and not self._has_live_provider(),
                ),
                core_grounding,
            ),
            clock_block,
        )

        forced_evidence_attempted: set[str] = set()
        # After a draft proposes replacing gear without checking the inventory, the next round
        # must run that search, rather than hoping the model chooses to.
        pending_forced_tool: str | None = None

        try:
            for round_number in range(1, MAX_TOOL_ROUNDS + 1):
                trace["tool_budget"]["rounds_used"] = round_number
                self._log_request_metadata(input_items, round_number)
                tool_choice, forced_evidence = self._round_tool_choice(
                    round_number,
                    title_request_context,
                    build_validation_context,
                    forced_evidence_attempted,
                    item_lookup,
                )
                if pending_forced_tool is not None:
                    tool_choice, forced_evidence = (
                        {"type": "function", "name": pending_forced_tool},
                        (),
                    )
                    pending_forced_tool = None
                # The last permitted round, or a spent tool budget, must produce the answer:
                # no tools, answer from the evidence already gathered.
                round_instructions = turn_instructions
                final_forced = round_number >= MAX_TOOL_ROUNDS or (
                    total_tool_calls >= MAX_TOOL_CALLS
                )
                if final_forced:
                    tool_choice, forced_evidence = "none", ()
                    round_instructions = turn_instructions + FINAL_ROUND_INSTRUCTION
                    trace["tool_loop"].update(
                        round_limit_reached=round_number >= MAX_TOOL_ROUNDS,
                        tool_limit_reached=total_tool_calls >= MAX_TOOL_CALLS,
                        final_answer_forced=True,
                        evidence_at_force={
                            "tool_calls_made": total_tool_calls,
                            "tools": list(trace["tools"]),
                        },
                    )
                    trace["tool_budget"]["limit_reached"] = (
                        "tool_calls" if total_tool_calls >= MAX_TOOL_CALLS else "rounds"
                    )
                if forced_evidence:
                    trace["build_analysis"]["forced_evidence_tools"].extend(
                        value
                        for value in forced_evidence
                        if value not in trace["build_analysis"]["forced_evidence_tools"]
                    )
                response = await self._create_response(
                    stream_status=stream_status,
                    model=self.settings.openai_model,
                    instructions=round_instructions,
                    input=input_items,
                    tools=tool_definitions,
                    include=RESPONSE_INCLUDE,
                    tool_choice=tool_choice,
                    parallel_tool_calls=True,
                    reasoning={"effort": self.settings.openai_reasoning_effort},
                    max_output_tokens=self.settings.openai_max_output_tokens,
                    truncation="disabled",
                    store=False,
                )
                response_output = self._field(response, "output", []) or []
                response_web_sources, web_research_occurred = self._extract_web_sources(
                    response_output
                )
                self._merge_sources(web_sources, response_web_sources)
                if web_research_occurred:
                    if stream_status is not None:
                        await stream_status.emit("web")
                        trace["emitted_status_categories"] = list(stream_status.emitted_categories)
                    trace["grounding"]["web_research"] = True
                    trace["web_research"]["occurred"] = True
                    mechanics_ledger.note_web_research()
                    trace["web_research_occurred"] = True
                    self._record_knowledge_category(trace, "web")
                trace["web_research"]["source_count"] = len(web_sources)
                function_calls = [
                    item for item in response_output if self._field(item, "type") == "function_call"
                ]
                if final_forced:
                    # tool_choice="none" should yield no calls; never act on one if it appears.
                    function_calls = []
                if forced_evidence:
                    # Each evidence tool is forced at most once. With a two-tool choice, only
                    # the tools actually called count, so the other can be forced next round.
                    called = {self._field(item, "name") for item in function_calls}
                    attempted = set(forced_evidence) & called
                    forced_evidence_attempted.update(
                        attempted if len(forced_evidence) > 1 and attempted else forced_evidence
                    )
                phase = "tool_calls" if function_calls else "final"
                self._log_response_metadata(response, phase)
                self._require_completed_response(response)
                if not function_calls:
                    if live_topic and not trace["grounding"]["live_provider"] and not web_sources:
                        # Neither a live provider nor web research produced anything to answer
                        # a current-state question from.
                        live_result = unavailable_live_data(request.message, live_topic)
                        trace["live_data"] = self._safe_live_trace(live_result)
                        trace["answer"] = "unavailable_live_data"
                        self._store_trace(trace_id, trace)
                        return ChatResponse(
                            source="local",
                            message=unavailable_live_message(
                                live_topic, provider_connected=self._has_live_provider()
                            ),
                        )
                    message = self._extract_assistant_text(response)
                    if not message and final_forced:
                        logger.error("Forced final round returned no assistant text")
                        trace["tool_loop"]["final_answer_succeeded"] = False
                        trace["tool_loop"]["final_answer_source"] = "limitation_fallback"
                        trace["answer"] = "generated"
                        self._store_trace(trace_id, trace)
                        return ChatResponse(source="local", message=FINAL_ROUND_LIMITATION_MESSAGE)
                    if not message:
                        logger.error(
                            "OpenAI final response contained no assistant text; "
                            "refusing to return an empty chat message"
                        )
                        raise RuntimeError(
                            "OpenAI returned a final response without assistant text."
                        )
                    build_validation_context.current_external_grounding = bool(
                        web_sources or trace["grounding"]["live_provider"]
                    )
                    read_only_validation_context.lookups_performed = bool(
                        trace["tools"] or trace["web_research_occurred"]
                    )
                    read_only_validation_context.guardian_lookup_performed = bool(
                        trace["number_of_guardian_tools_used"] or planning_context is not None
                    )
                    read_only_validation_context.current_evidence_performed = bool(
                        trace["grounding"]["web_research"] or trace["grounding"]["live_provider"]
                    )
                    read_only_validation_context.inventory_checked = bool(
                        {"search_inventory", "find_build_alternatives"} & set(trace["tools"])
                    )
                    read_only_validation_context.knowledge_lookup_performed = bool(
                        set(trace["knowledge_categories_used"]) - {"guardian"}
                    )
                    safe_fallback = False
                    message, violations = self._check_answer(
                        quality_validator,
                        message,
                        "initial",
                        mode_context,
                        request.message,
                        planning_context,
                        build_validation_context,
                        read_only_validation_context,
                        trace,
                    )
                    if (
                        not final_forced
                        and not trace["initiative_retry"]["attempted"]
                        and quality_validator.needs_initiative_retry(
                            violations, read_only_validation_context
                        )
                    ):
                        # The draft asked about, offered, or claimed read-only work instead of
                        # doing it. Send it back once with tools still available; a tool-less
                        # correction could only rephrase the missing work.
                        trace["initiative_retry"] = {
                            "attempted": True,
                            "violation_codes": [value.code for value in violations],
                            "round": round_number,
                        }
                        # One tool-enabled chance is enough: if no source can settle it, an
                        # honest "couldn't verify" afterwards is accepted, not corrected again.
                        read_only_validation_context.research_requested = False
                        read_only_validation_context.guidance_requested = False
                        read_only_validation_context.current_source_requested = False
                        if {"unsupported_replacement", "false_inventory_denial"} & {
                            value.code for value in violations
                        }:
                            pending_forced_tool = "find_build_alternatives"
                        elif "unverified_item_fact" in {value.code for value in violations} and (
                            "get_item_details" in {v.get("name") for v in tool_definitions}
                        ):
                            pending_forced_tool = "get_item_details"
                        input_items.extend(self._normalize_replay_output_items(response_output))
                        input_items.append(
                            {
                                "role": "developer",
                                "content": quality_validator.initiative_instruction(violations),
                            }
                        )
                        trace["continuation_items_normalized"] = True
                        continue
                    if stream_status is not None:
                        if mode_context.mode in {"recommendation", "comparison"}:
                            await stream_status.emit("comparison")
                        elif mode_context.mode == "build_advice":
                            await stream_status.emit("build")
                        await stream_status.emit("final")
                        trace["emitted_status_categories"] = list(stream_status.emitted_categories)
                    correction_output: list[Any] = []
                    # Presentation findings were repaired locally where possible and never
                    # justify a model rewrite; what is left is kept as the draft wrote it.
                    kept_style = locally_handled((v.code for v in violations), message)
                    for code in sorted(kept_style):
                        self._record_repair(trace, "initial", code, "keep_draft", "no_rewrite")
                    violations = [v for v in violations if v.code not in kept_style]
                    if violations:
                        trace["planning_correction"] = {
                            "attempted": True,
                            "violation_codes": [value.code for value in violations],
                            "succeeded": False,
                        }
                        trace["continuation_items_normalized"] = True
                        correction_response = await self._request_planning_correction(
                            input_items,
                            response_output,
                            turn_instructions,
                            tool_definitions,
                            quality_validator.correction_instruction(
                                violations, build_validation_context
                            ),
                            stream_status,
                        )
                        correction_output = self._field(correction_response, "output", []) or []
                        self._log_response_metadata(correction_response, "planning_correction")
                        self._require_completed_response(correction_response)
                        corrected_message = remove_correction_narration(
                            self._extract_assistant_text(correction_response)
                        )
                        corrected_violations = violations
                        if corrected_message:
                            damage = self._rewrite_damage_reason(
                                message,
                                corrected_message,
                                violations,
                                quality_validator,
                                mode_context,
                                request.message,
                                planning_context,
                                build_validation_context,
                                read_only_validation_context,
                            )
                            codes_text = ",".join(value.code for value in violations)
                            if damage is None:
                                self._record_repair(
                                    trace, "after_correction", codes_text, "rewrite", "kept"
                                )
                                corrected_message, corrected_violations = self._check_answer(
                                    quality_validator,
                                    corrected_message,
                                    "after_correction",
                                    mode_context,
                                    request.message,
                                    planning_context,
                                    build_validation_context,
                                    read_only_validation_context,
                                    trace,
                                )
                            else:
                                # The rewrite is worse than the draft: keep the draft and
                                # take out only the sentences the validators flagged. If the
                                # draft cannot be repaired that way, the rewrite is still
                                # better than a canned answer.
                                local = self._remove_flagged(
                                    message,
                                    violations,
                                    quality_validator,
                                    mode_context,
                                    request.message,
                                    planning_context,
                                    build_validation_context,
                                    read_only_validation_context,
                                )
                                if local is None:
                                    self._record_repair(
                                        trace,
                                        "after_correction",
                                        codes_text,
                                        "rewrite",
                                        f"kept_despite_{damage}_no_local_repair",
                                    )
                                    corrected_message, corrected_violations = self._check_answer(
                                        quality_validator,
                                        corrected_message,
                                        "after_correction",
                                        mode_context,
                                        request.message,
                                        planning_context,
                                        build_validation_context,
                                        read_only_validation_context,
                                        trace,
                                    )
                                else:
                                    self._record_repair(
                                        trace, "after_correction", codes_text, "rewrite", damage
                                    )
                                    self._record_repair(
                                        trace,
                                        "after_correction",
                                        codes_text,
                                        "remove_flagged_sentences",
                                        "applied",
                                    )
                                    trace["planning_correction"]["rewrite_discarded"] = damage
                                    correction_output = []
                                    corrected_message, corrected_violations = self._check_answer(
                                        quality_validator,
                                        local,
                                        "after_correction",
                                        mode_context,
                                        request.message,
                                        planning_context,
                                        build_validation_context,
                                        read_only_validation_context,
                                        trace,
                                    )
                        if corrected_message:
                            kept_style = locally_handled(
                                (v.code for v in corrected_violations), corrected_message
                            )
                            corrected_violations = [
                                v for v in corrected_violations if v.code not in kept_style
                            ]
                        if corrected_message and not corrected_violations:
                            message = corrected_message
                            trace["planning_correction"]["succeeded"] = True
                            correction_sources, _ = self._extract_web_sources(correction_output)
                            self._merge_sources(web_sources, correction_sources)
                            trace["web_research"]["source_count"] = len(web_sources)
                        elif build_validation_context.is_build_request and (
                            "unsupported_replacement" in {v.code for v in corrected_violations}
                            and build_fallback_kind(corrected_violations) != "evidence"
                            and {v.code for v in corrected_violations}
                            <= {"unsupported_replacement", "too_many_build_changes"}
                        ):
                            # The only problem left is a swap with no verified owned
                            # replacement: say that, instead of a generic refusal.
                            message = quality_validator.safe_build_no_swap_answer(
                                build_validation_context
                            )
                            safe_fallback = True
                            trace["planning_correction"]["remaining_violation_codes"] = [
                                value.code for value in corrected_violations
                            ]
                            trace["planning_correction"]["fallback"] = "build_no_swap"
                        elif build_validation_context.is_build_request:
                            fallback_kind = build_fallback_kind(corrected_violations)
                            trace["planning_correction"]["remaining_violation_codes"] = [
                                value.code for value in corrected_violations
                            ]
                            trace["ownership_validation"]["fallback_due_to_ownership"] = bool(
                                fallback_kind in {"evidence", "grounding_safety"}
                                and any(
                                    value.code == "unsupported_ownership_claim"
                                    for value in corrected_violations
                                )
                            )
                            if fallback_kind == "evidence":
                                message = quality_validator.safe_build_answer(
                                    build_validation_context
                                )
                                safe_fallback = True
                                trace["planning_correction"]["fallback"] = "build_evidence"
                            elif fallback_kind == "grounding_safety":
                                # The data was available; the draft made unsupported or
                                # constraint-breaking claims. Block it without blaming retrieval.
                                message = quality_validator.safe_build_grounding_answer(
                                    build_validation_context,
                                    [value.code for value in corrected_violations],
                                )
                                safe_fallback = True
                                trace["planning_correction"]["fallback"] = "build_grounding"
                            else:
                                # Only presentation issues remain, so the grounded answer is
                                # kept rather than replaced with a false retrieval refusal.
                                if corrected_message:
                                    message = corrected_message
                                    correction_sources, _ = self._extract_web_sources(
                                        correction_output
                                    )
                                    self._merge_sources(web_sources, correction_sources)
                                    trace["web_research"]["source_count"] = len(web_sources)
                                trace["planning_correction"]["fallback"] = "style_only_accepted"
                        elif planning_context is not None:
                            if build_fallback_kind(corrected_violations) == "style_only":
                                # Same principle as build turns: a grounded recommendation
                                # with only presentation issues is not an evidence failure.
                                if corrected_message:
                                    message = corrected_message
                                    correction_sources, _ = self._extract_web_sources(
                                        correction_output
                                    )
                                    self._merge_sources(web_sources, correction_sources)
                                    trace["web_research"]["source_count"] = len(web_sources)
                                trace["planning_correction"]["remaining_violation_codes"] = [
                                    value.code for value in corrected_violations
                                ]
                                trace["planning_correction"]["fallback"] = "style_only_accepted"
                            else:
                                message = quality_validator.planning.safe_answer(planning_context)
                                safe_fallback = True
                                trace["planning_correction"]["remaining_violation_codes"] = [
                                    value.code for value in corrected_violations
                                ]
                                trace["planning_correction"]["fallback"] = "planning_safe_answer"
                        elif corrected_message and {
                            value.code for value in corrected_violations
                        } <= {value.code for value in violations}:
                            # Neither a build nor a planning turn: the correction introduced no
                            # new problem, so it is at least as good as the draft it replaces.
                            message = corrected_message
                            correction_sources, _ = self._extract_web_sources(correction_output)
                            self._merge_sources(web_sources, correction_sources)
                            trace["web_research"]["source_count"] = len(web_sources)
                            trace["planning_correction"]["remaining_violation_codes"] = [
                                value.code for value in corrected_violations
                            ]
                            trace["planning_correction"]["fallback"] = "partial_correction_accepted"
                    # Presentation only: readable citations, and no offer of an ability the app
                    # lacks even if a correction left one in place.
                    message = remove_unsupported_capability_offers(
                        remove_correction_narration(
                            clean_citation_artifacts(
                                message, citation_spans([*response_output, *correction_output])
                            )
                        )
                    )
                    safe_message = self._enforce_power_safety(message)
                    safe_message = self._enforce_campaign_completion_safety(
                        safe_message, request.message
                    )
                    safe_message = self._enforce_campaign_comparison_safety(
                        safe_message,
                        request.message,
                        character_scope,
                    )
                    # Sources the answer cites come first, so a citation shortened to its label
                    # in the text always keeps its link within the source limit.
                    sources = self._combined_sources(
                        [
                            *self._cited_sources([*response_output, *correction_output]),
                            *web_sources,
                        ],
                        knowledge_sources,
                        limit=mode_context.source_limit,
                    )
                    trace["web_research"]["source_count"] = len(
                        [source for source in sources if source in web_sources]
                    )
                    # A generated replacement answer relies on none of the retrieved sources.
                    # A rewrite that only swapped out flagged sentences still rests on them, so
                    # its citations stay while most of the original answer does.
                    if safe_fallback or (
                        safe_message != message and not retains_answer(message, safe_message)
                    ):
                        sources = []
                    message = self._hide_internal_identifiers(
                        safe_message,
                        context,
                        allow_identifiers=bool(DEBUG_IDENTIFIER_PATTERN.search(request.message)),
                    )
                    self._record_destination_conflicts(core_grounding, message)
                    trace["mechanics_claims"] = self._mechanics_claim_trace(
                        message, mechanics_ledger
                    )
                    if final_forced:
                        trace["tool_loop"]["final_answer_succeeded"] = True
                        trace["tool_loop"]["final_answer_source"] = "model"
                    trace["answer"] = "generated"
                    self._store_trace(trace_id, trace)
                    return ChatResponse(
                        source="openai",
                        message=message,
                        sources=sources,
                    )

                input_items.extend(self._normalize_replay_output_items(response_output))
                trace["continuation_items_normalized"] = True
                for call in function_calls:
                    if total_tool_calls >= MAX_TOOL_CALLS:
                        # Over budget: skip the lookup but answer the call so the replay stays
                        # valid; the next round is the tool-less final answer.
                        trace["tool_budget"]["limit_reached"] = "tool_calls"
                        input_items.append(
                            {
                                "type": "function_call_output",
                                "call_id": call.call_id,
                                "output": json.dumps(
                                    TOOL_BUDGET_EXHAUSTED_RESULT, separators=(",", ":")
                                ),
                            }
                        )
                        continue
                    total_tool_calls += 1
                    trace["tool_budget"]["tool_calls_used"] = total_tool_calls
                    category = self._knowledge_category(call.name) or "guardian"
                    if stream_status is not None:
                        stage, label = status_for_tool(call.name, category)
                        await stream_status.emit(stage, label)
                        trace["emitted_status_categories"] = list(stream_status.emitted_categories)
                    if category == "guardian" and refresh_guardian is not None:
                        context = await refresh_guardian(call.name)
                        conversation_scope = resolve_conversation_scope(
                            context,
                            previous_character,
                            preference_context.character,
                            chosen_this_turn=chosen_this_turn,
                        )
                        build_validation_context.active_character = conversation_scope.active
                        guardian_tools = GuardianToolService(
                            context,
                            conversation_scope.explicit_class,  # type: ignore[arg-type]
                        )
                        current_character = resolve_current_character(context)
                        trace["current_character"] = self._current_character_trace(
                            current_character, character_scope
                        )
                        if planning_context is not None:
                            planning_context = build_session_planning_context(
                                context,
                                request.message,
                                intent_category or "general",
                                character_scope,
                                preference_context,
                            )
                            if alternatives is not None:
                                planning_context = apply_alternatives(
                                    planning_context,
                                    derive_alternatives(
                                        planning_intent.segment_history,
                                        request.message,
                                        planning_context.candidates,
                                        character_switched=planning_continuation,
                                    ),
                                )
                            # The core block was built once for this turn and is re-attached,
                            # not rebuilt, when refreshed planning context changes routing.
                            turn_instructions = self._with_clock(
                                self._with_core_facts(
                                    self._turn_instructions(
                                        session_planning,
                                        intent_category,
                                        character_scope,
                                        planning_context,
                                        mode_context,
                                        build_request_context,
                                        title_request_context,
                                        current_character,
                                        research_requested=research_requested,
                                        live_state_research=live_research
                                        and not self._has_live_provider(),
                                    ),
                                    core_grounding,
                                ),
                                clock_block,
                            )
                            self._record_planning_requirements(mechanics_ledger, planning_context)
                            trace["planning_context"]["requirements"] = requirement_trace(
                                planning_context
                            )
                            trace["planning_context"]["candidate_count"] = len(
                                planning_context.candidates
                            )
                            trace["planning_context"]["named_content_count"] = len(
                                planning_context.named_content
                            )
                    result = await self._execute_tool_call(
                        guardian_tools, call.name, call.arguments
                    )
                    self._record_mechanics_evidence(mechanics_ledger, call.name, category, result)
                    collect_item_names(result, read_only_validation_context.retrieved_item_names)
                    collect_item_definitions(result, read_only_validation_context.item_definitions)
                    collect_evidence_terms(result, read_only_validation_context.evidence_terms)
                    self._update_build_validation_context(
                        build_validation_context,
                        trace,
                        call.name,
                        result,
                        {
                            value.character_id: value.class_name
                            for value in guardian_tools.context.characters
                        },
                    )
                    if call.name == "find_build_alternatives" and call.name in forced_evidence:
                        # Observability only: never alters answers, validation, or tool choice.
                        trace["build_analysis"]["forced_alternatives_relevance"].append(
                            self._forced_alternatives_relevance(
                                request.message,
                                guardian_tools.context,
                                call.arguments,
                                result,
                                core_grounding.weapon_archetypes,
                            )
                        )
                    if call.name == "get_title_progress" and "error" not in result:
                        trace["title_progress"]["retrieval_used"] = True
                    trace["tools"].append(call.name)
                    selection = self._character_selection_trace(
                        call.name,
                        call.arguments,
                        character_scope,
                        current_character,
                        conversation_scope.explicit_class,
                    )
                    if selection is not None:
                        trace["character_selections"].append(selection)
                    tool_trace = self._safe_tool_trace(
                        call.name,
                        category,
                        call.arguments,
                        result,
                    )
                    safe_provenance_fields = {
                        "source": "result_source",
                        "provider": "result_provider",
                        "availability_scope": "availability_scope",
                        "current_rotation_authoritative": "current_rotation_authoritative",
                        "freshness": "freshness",
                        "live_data_available": "live_data_available",
                    }
                    for result_field, trace_field in safe_provenance_fields.items():
                        value = result.get(result_field)
                        if isinstance(value, (str, bool)):
                            tool_trace[trace_field] = value
                    if "requires_live_provider" in result:
                        tool_trace["requires_live_provider"] = bool(
                            result["requires_live_provider"]
                        )
                    supported_topics = result.get("supported_live_topics")
                    if isinstance(supported_topics, list) and all(
                        isinstance(value, str) for value in supported_topics
                    ):
                        tool_trace["supported_live_topics"] = supported_topics
                    trace["tool_trace"].append(tool_trace)
                    if category == "manifest":
                        trace["grounding"]["manifest"] = True
                        self._record_manifest_conflict(core_grounding, call.name, result)
                    elif category == CORE_PROVIDER_CATEGORY:
                        # Core results are curated evidence, never Guardian account evidence.
                        trace["grounding"]["core_knowledge"] = True
                        core_grounding.trace["used"] = True
                        core_grounding.trace["tool_used"] = True
                        tool_trace.update(self._safe_core_tool_summary(result))
                    elif category == "guide":
                        trace["grounding"]["guide_provider"] = True
                    elif category == LIVE_PROVIDER_CATEGORY:
                        result_is_current = bool(
                            result.get("live_data_available") is True
                            and result.get("freshness") == "volatile"
                            and result.get("requires_live_provider") is not True
                        )
                        topic_is_supported = bool(
                            not live_topic
                            or (
                                isinstance(supported_topics, list)
                                and live_topic in supported_topics
                            )
                        )
                        trace["grounding"]["live_provider"] = bool(
                            trace["grounding"]["live_provider"]
                            or (result_is_current and topic_is_supported)
                        )
                    else:
                        trace["grounding"]["guardian_account"] = True
                        trace["number_of_guardian_tools_used"] += 1
                    self._record_knowledge_category(trace, category)
                    if category in {"manifest", "guide", LIVE_PROVIDER_CATEGORY}:
                        self._merge_sources(
                            knowledge_sources, self._extract_knowledge_sources(result)
                        )
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": json.dumps(result, separators=(",", ":")),
                        }
                    )
                if stream_status is not None and mode_context.mode in {
                    "recommendation",
                    "comparison",
                    "build_advice",
                }:
                    await stream_status.emit(
                        "build" if mode_context.mode == "build_advice" else "comparison"
                    )
                    trace["emitted_status_categories"] = list(stream_status.emitted_categories)
            # Unreachable in practice: the last round is forced to answer and returns above.
            trace["tool_budget"]["limit_reached"] = "rounds"
            trace["tool_loop"]["final_answer_succeeded"] = False
            raise RuntimeError("The recommendation did not finish its account lookup.")
        except OpenAIError as exc:
            logger.exception("OpenAI recommendation failed")
            raise RuntimeError("The recommendation service is temporarily unavailable.") from exc
        finally:
            if stream_status is not None:
                trace["emitted_status_categories"] = list(stream_status.emitted_categories)
            if trace_id not in self._traces:
                self._store_trace(trace_id, trace)

    @staticmethod
    def _safe_tool_trace(
        tool_name: str,
        category: str,
        arguments_json: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        trace: dict[str, Any] = {"name": tool_name, "category": category}
        if category != "guardian" or tool_name not in TRACEABLE_GUARDIAN_TOOLS:
            return trace

        request = RecommendationService._safe_guardian_trace_request(arguments_json)
        if request:
            trace["request"] = request

        error = result.get("error")
        if error is not None:
            trace["success"] = False
            error_type = error.get("code") if isinstance(error, dict) else None
            trace["error_type"] = (
                error_type
                if isinstance(error_type, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", error_type)
                else "tool_error"
            )
            return trace

        trace["success"] = True
        summary = RecommendationService._safe_guardian_result_summary(tool_name, result)
        if summary:
            trace["result_summary"] = summary
        return trace

    @staticmethod
    def _safe_guardian_trace_request(arguments_json: str) -> dict[str, Any]:
        try:
            arguments = json.loads(arguments_json)
        except (json.JSONDecodeError, TypeError):
            return {}
        if not isinstance(arguments, dict):
            return {}

        safe: dict[str, Any] = {}
        for key in SAFE_GUARDIAN_TRACE_ARGUMENTS:
            if key not in arguments:
                continue
            value = arguments[key]
            scalar_is_safe = bool(
                value is None
                or isinstance(value, bool)
                or (isinstance(value, str) and len(value) <= 80)
                or (key == "limit" and isinstance(value, int) and 1 <= value <= 100)
            )
            if scalar_is_safe:
                safe[key] = value
            elif key == "locations" and isinstance(value, list):
                locations = [
                    location
                    for location in value[:4]
                    if isinstance(location, str) and len(location) <= 20
                ]
                safe[key] = locations
        return dict(sorted(safe.items()))

    @staticmethod
    def _safe_guardian_result_summary(
        tool_name: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        character_class = result.get("character_class")
        character = result.get("character")
        if not isinstance(character_class, str) and isinstance(character, dict):
            character_class = character.get("class")
        if isinstance(character_class, str) and len(character_class) <= 20:
            summary["character_class"] = character_class

        item_groups: list[dict[str, Any]] = []
        for key in ("candidates", "items", "equipped_weapons", "weapons"):
            values = result.get(key)
            if isinstance(values, list):
                item_groups.extend(value for value in values if isinstance(value, dict))
        item_names = RecommendationService._bounded_trace_item_names(item_groups)
        if item_names:
            summary["item_names"] = item_names

        returned = result.get("returned")
        if not isinstance(returned, int) and tool_name == "search_inventory":
            items = result.get("items")
            returned = len(items) if isinstance(items, list) else None
        if isinstance(returned, int) and returned >= 0:
            summary["returned"] = returned

        total_matching = result.get("total_matching")
        if not isinstance(total_matching, int):
            total_matching = result.get("total_matching_owned_copies")
        if isinstance(total_matching, int) and total_matching >= 0:
            summary["total_matching"] = total_matching
        if isinstance(result.get("truncated"), bool):
            summary["truncated"] = result["truncated"]

        if tool_name == "analyze_current_build":
            weapons = result.get("equipped_weapons")
            if isinstance(weapons, list):
                summary["equipped_weapon_names"] = RecommendationService._bounded_trace_item_names(
                    [value for value in weapons if isinstance(value, dict)]
                )
            exotics = result.get("equipped_exotics")
            if isinstance(exotics, dict):
                exotic_items = [
                    {"name": value}
                    for key in ("weapon", "armor")
                    for value in (exotics.get(key) or [])
                    if isinstance(value, str)
                ]
                summary["equipped_exotic_names"] = RecommendationService._bounded_trace_item_names(
                    exotic_items
                )
            observations = result.get("observations")
            if isinstance(observations, dict) and isinstance(
                observations.get("weapons_missing_roll_data"), list
            ):
                summary["roll_data_complete"] = not bool(observations["weapons_missing_roll_data"])
        elif tool_name == "get_equipped_loadout":
            exotics = [
                value
                for key in ("weapons", "armor")
                for value in (result.get(key) or [])
                if isinstance(value, dict) and value.get("tier") == "Exotic"
            ]
            exotic_names = RecommendationService._bounded_trace_item_names(exotics)
            if exotic_names:
                summary["equipped_exotic_names"] = exotic_names
        return summary

    @staticmethod
    def _bounded_trace_item_names(items: list[dict[str, Any]]) -> list[str]:
        names: list[str] = []
        for item in items:
            name = item.get("name")
            if not isinstance(name, str) or not name or len(name) > 120 or name in names:
                continue
            names.append(name)
            if len(names) == MAX_TRACE_ITEM_NAMES:
                break
        return names

    @staticmethod
    def _update_build_validation_context(
        context: BuildResponseValidationContext,
        trace: dict[str, Any],
        tool_name: str,
        result: dict[str, Any],
        character_classes: dict[str, str] | None = None,
    ) -> None:
        if tool_name not in {
            "analyze_current_build",
            "find_build_alternatives",
            "get_build_details",
            "get_equipped_loadout",
            "search_inventory",
        }:
            return
        if "error" in result:
            return
        # Equipped, analyzed, and inventory results each prove possession of what they list.
        record_ownership_evidence(context.ownership, tool_name, result, character_classes)
        if tool_name in {"analyze_current_build", "get_build_details", "get_equipped_loadout"}:
            context.guardian_build_data_used = True
        if tool_name == "analyze_current_build":
            context.current_build_analysis_used = True
        if tool_name in {"find_build_alternatives", "search_inventory"}:
            context.inventory_ownership_checked = True

        item_groups: list[dict[str, Any]] = []
        for key in ("equipped_weapons", "equipped_armor", "weapons", "armor"):
            for value in result.get(key) or []:
                name = value.get("name") if isinstance(value, dict) else None
                if isinstance(name, str) and name not in context.equipped_item_names:
                    context.equipped_item_names.append(name)
        for key in (
            "equipped_weapons",
            "equipped_armor",
            "weapons",
            "armor",
            "items",
            "candidates",
        ):
            values = result.get(key)
            if isinstance(values, list):
                item_groups.extend(value for value in values if isinstance(value, dict))
        for item in item_groups:
            name = item.get("name")
            if not isinstance(name, str) or not name:
                continue
            if name not in context.owned_item_names:
                context.owned_item_names.append(name)
            tier = item.get("rarity") or item.get("tier")
            item_type = item.get("item_type")
            if tier == "Exotic" or item.get("exotic") is True:
                target = (
                    context.owned_exotic_weapons
                    if item_type == "Weapon"
                    else context.owned_exotic_armor
                    if item_type == "Armor"
                    else None
                )
                if target is not None and name not in target:
                    target.append(name)

        exotic_data = result.get("equipped_exotics")
        if isinstance(exotic_data, dict):
            for key, target in (
                ("weapon", context.equipped_exotic_weapons),
                ("armor", context.equipped_exotic_armor),
            ):
                values = exotic_data.get(key)
                if isinstance(values, list):
                    for value in values:
                        if isinstance(value, str) and value not in target:
                            target.append(value)
                            owned_target = (
                                context.owned_exotic_weapons
                                if key == "weapon"
                                else context.owned_exotic_armor
                            )
                            if value not in owned_target:
                                owned_target.append(value)
                            if (
                                context.preserve_equipped_exotics
                                and value not in context.locked_item_names
                            ):
                                context.locked_item_names.append(value)
        preservation = result.get("preservation")
        if isinstance(preservation, dict):
            for key, target in (
                ("equipped_exotic_weapon", context.equipped_exotic_weapons),
                ("equipped_exotic_armor", context.equipped_exotic_armor),
                ("locked_items", context.locked_item_names),
            ):
                values = preservation.get(key)
                if isinstance(values, list):
                    for value in values:
                        if isinstance(value, str) and value not in target:
                            target.append(value)
        locked = result.get("locked_items")
        if isinstance(locked, list):
            for value in locked:
                name = value.get("name") if isinstance(value, dict) else value
                if isinstance(name, str) and name not in context.locked_item_names:
                    context.locked_item_names.append(name)

        if tool_name == "analyze_current_build":
            trace["build_analysis"]["analysis_used"] = True
        elif tool_name == "find_build_alternatives":
            trace["build_analysis"]["alternative_search_used"] = True
            trace["build_analysis"]["owned_candidates_returned"] += int(
                result.get("returned", 0) or 0
            )

    def _store_trace(self, trace_id: str, trace: dict[str, Any]) -> None:
        if not self.settings.enable_debug_tools:
            return
        self._traces[trace_id] = trace
        self._traces.move_to_end(trace_id)
        while len(self._traces) > 100:
            self._traces.popitem(last=False)

    def mark_stream_delivery(
        self,
        trace_id: str | None,
        *,
        final_response_streamed: bool,
        stream_completed: bool,
        emitted_status_categories: list[str] | None = None,
    ) -> None:
        if not trace_id or trace_id not in self._traces:
            return
        trace = self._traces[trace_id]
        trace["final_response_streamed"] = final_response_streamed
        trace["stream_completed"] = stream_completed
        if emitted_status_categories is not None:
            trace["emitted_status_categories"] = list(emitted_status_categories)

    def trace(self, trace_id: str) -> dict[str, Any] | None:
        value = self._traces.get(trace_id)
        return dict(value) if value is not None else None

    def latest_trace(self) -> dict[str, Any] | None:
        if not self._traces:
            return None
        return dict(next(reversed(self._traces.values())))

    def recent_traces(self, limit: int = 20) -> list[dict[str, Any]]:
        values = list(reversed(self._traces.values()))[:limit]
        return [dict(value) for value in values]

    def _knowledge_handles(self, name: str) -> bool:
        if self.knowledge is None:
            return False
        handles = getattr(self.knowledge, "handles", None)
        if callable(handles):
            return bool(handles(name))
        return name in {value["name"] for value in self.knowledge.definitions()}

    def _knowledge_category(self, name: str) -> str | None:
        if not self._knowledge_handles(name):
            return None
        category = getattr(self.knowledge, "category_for_tool", None)
        if callable(category):
            return category(name)
        return "guide" if "guide" in name else "manifest" if self._knowledge_handles(name) else None

    def _has_live_provider(self) -> bool:
        if self.knowledge is None:
            return False
        has_category = getattr(self.knowledge, "has_category", None)
        if callable(has_category):
            return bool(has_category(LIVE_PROVIDER_CATEGORY))
        providers = getattr(self.knowledge, "providers", [])
        return any(
            getattr(provider, "knowledge_category", None) == LIVE_PROVIDER_CATEGORY
            for provider in providers
        )

    @staticmethod
    def _record_repair(
        trace: dict[str, Any], stage: str, validator: str, action: str, result: str
    ) -> None:
        """One validator decision: which finding, what was done about it, and the outcome."""

        trace.setdefault("local_repairs", []).append(
            {"stage": stage, "validator": validator, "action": action, "result": result}
        )

    @staticmethod
    def _flagged_sentences(
        text: str,
        violations: list[Any],
        quality_validator: ResponseQualityValidator,
        mode_context: ResponseModeContext,
        user_message: str,
        planning_context: SessionPlanningContext | None,
        build_context: BuildResponseValidationContext,
        read_only_context: ReadOnlyActionValidationContext,
    ) -> set[str]:
        """Sentences that raise one of the findings when validated on their own."""

        def sentence_codes(piece: str) -> list[str]:
            return [
                value.code
                for value in quality_validator.validate(
                    piece,
                    mode_context,
                    user_message,
                    planning_context,
                    build_context,
                    read_only_context,
                )
            ]

        return offending_sentences(text, {v.code for v in violations}, sentence_codes)

    @staticmethod
    def _remove_flagged(
        message: str,
        violations: list[Any],
        quality_validator: ResponseQualityValidator,
        mode_context: ResponseModeContext,
        user_message: str,
        planning_context: SessionPlanningContext | None,
        build_context: BuildResponseValidationContext,
        read_only_context: ReadOnlyActionValidationContext,
    ) -> str | None:
        """The draft without the sentences that raise the given findings, or None."""

        flagged = RecommendationService._flagged_sentences(
            message,
            violations,
            quality_validator,
            mode_context,
            user_message,
            planning_context,
            build_context,
            read_only_context,
        )
        wanted = {v.code for v in violations}

        def acceptable_pick(body: str) -> bool:
            # A promoted backup becomes the recommendation, so it is checked as one.
            checked = RecommendationService._flagged_sentences(
                f"Recommended: {body}",
                violations,
                quality_validator,
                mode_context,
                user_message,
                planning_context,
                build_context,
                read_only_context,
            )
            codes = {
                value.code
                for value in quality_validator.validate(
                    f"Recommended: {body}",
                    mode_context,
                    user_message,
                    planning_context,
                    build_context,
                    read_only_context,
                )
            }
            return not checked and not (codes & wanted)

        if not flagged:
            return None
        return remove_flagged_sentences(message, flagged, acceptable_pick)

    @staticmethod
    def _rewrite_damage_reason(
        draft: str,
        rewrite: str,
        violations: list[Any],
        quality_validator: ResponseQualityValidator,
        mode_context: ResponseModeContext,
        user_message: str,
        planning_context: SessionPlanningContext | None,
        build_context: BuildResponseValidationContext,
        read_only_context: ReadOnlyActionValidationContext,
    ) -> str | None:
        """Why the rewrite is worse than the draft (see rewrite_damage), or None."""

        flagged = RecommendationService._flagged_sentences(
            draft,
            violations,
            quality_validator,
            mode_context,
            user_message,
            planning_context,
            build_context,
            read_only_context,
        )
        names = {
            *read_only_context.retrieved_item_names,
            *(value.casefold() for value in read_only_context.account_quest_names),
            *(
                value.name.casefold()
                for value in (planning_context.candidates if planning_context else [])
            ),
        }
        return rewrite_damage(draft, rewrite, dependents_of(draft, flagged), names)

    @staticmethod
    def _check_answer(
        quality_validator: ResponseQualityValidator,
        message: str,
        stage: str,
        mode_context: ResponseModeContext,
        user_message: str,
        planning_context: SessionPlanningContext | None,
        build_context: BuildResponseValidationContext,
        read_only_context: ReadOnlyActionValidationContext,
        trace: dict[str, Any],
    ) -> tuple[str, list[Any]]:
        """Validate an answer; repair removable unsupported possession wording in build turns.

        A possession or "equipped" sentence that no Guardian result supports is removed on its
        own when it is not part of a recommendation, so a grounded build answer is not
        replaced by a generic refusal. A closing hand-back (see trim_trailing_handback) is
        removed the same way. Everything else keeps its fail-closed handling. This never calls
        tools or the model.
        """

        # A closing hand-back costs a model round to correct and adds nothing: an offer of more
        # read-only work once lookups ran (or once the answer declined an impossible change),
        # an offer of an ability the app lacks, or a "which do you want?" after a ranked
        # recommendation. Before any lookup, a lookup offer is the missing work itself, so it
        # is left for the tool-enabled retry.
        trimmed = quality_validator.trim_trailing_handback(
            message,
            offers=read_only_context.lookups_performed
            or (not read_only_context.research_requested and declines_change(message))
            # A planning turn already carries the Guardian data its recommendation used.
            or (planning_context is not None and not read_only_context.research_requested),
            choices=planning_context is not None,
            account_checked=read_only_context.guardian_lookup_performed,
        )
        if trimmed is not None:
            message = trimmed
            trace["trailing_handback_removed"] = True
        opener_removed = strip_filler_opener(message)
        if opener_removed is not None:
            # The label is the only problem; the cited facts and the rest stay as written.
            message = opener_removed
            trace["filler_opener_removed"] = True
        violations = quality_validator.validate(
            message,
            mode_context,
            user_message,
            planning_context,
            build_context,
            read_only_context,
        )
        if {"unsupported_loot_claim", "invented_duration"} & {v.code for v in violations}:
            # Cut only the unsupported loot ranking or stated duration; the recommendation,
            # its steps, and the player's constraints stay, so no model round is spent.
            repaired = quality_validator.repair_planning_claims(
                message,
                quality_validator.allowed_minutes(user_message, planning_context)
                if planning_context is not None
                else None,
                # The "not established" note answers a loot question; unasked, the cut is silent.
                note_loot=bool(LOOT_QUESTION.search(user_message)),
            )
            if repaired is not None:
                message = repaired
                trace["planning_claims_repaired"] = True
                violations = quality_validator.validate(
                    message,
                    mode_context,
                    user_message,
                    planning_context,
                    build_context,
                    read_only_context,
                )
        if planning_context is not None and "activity_record_conflation" in {
            v.code for v in violations
        }:
            # Drop only the sentence that applies one activity's record to another; the rest
            # of the plan stays instead of being replaced by a generic answer.
            repaired = quality_validator.repair_activity_conflations(message, planning_context)
            if repaired is not None:
                message = repaired
                trace["activity_conflation_repaired"] = True
                violations = quality_validator.validate(
                    message,
                    mode_context,
                    user_message,
                    planning_context,
                    build_context,
                    read_only_context,
                )
        if "absence_reported_as_value" in {v.code for v in violations}:
            # Deterministic and faithful: the sentence becomes a plain statement that the
            # measure is unavailable, so no model round is needed to say what is already known.
            repaired = replace_absent_values(message, read_only_context.evidence_terms)
            if repaired is not None:
                message = repaired
                trace["absent_values_replaced"] = True
                violations = quality_validator.validate(
                    message,
                    mode_context,
                    user_message,
                    planning_context,
                    build_context,
                    read_only_context,
                )
        if "contradicted_item_fact" in {v.code for v in violations}:
            # The retrieved value is known, so state it: no model round is needed.
            repaired = correct_item_facts(message, read_only_context.item_definitions)
            if repaired is not None:
                message = repaired
                trace["item_facts_corrected"] = True
                violations = quality_validator.validate(
                    message,
                    mode_context,
                    user_message,
                    planning_context,
                    build_context,
                    read_only_context,
                )
        if "unverified_item_fact" in {v.code for v in violations} and (
            stage == "after_correction" or trace["initiative_retry"]["attempted"]
        ):
            # The lookup (or the one correction) did not settle it: keep the rest of the answer
            # and drop only the definition facts nothing retrieved supports.
            repaired = remove_unverified_item_claims(
                message, read_only_context.retrieved_item_names
            )
            if repaired is not None:
                message = repaired
                trace["unverified_item_claims_removed"] = True
                violations = quality_validator.validate(
                    message,
                    mode_context,
                    user_message,
                    planning_context,
                    build_context,
                    read_only_context,
                )
        if "unsupported_capability" in {v.code for v in violations}:
            # An offer of something no tool can do (screenshots, item changes, an unlocked-Facet
            # list, K/D from matches) is cut with the existing cleanup: no model round needed.
            cleaned = remove_unsupported_capability_offers(message)
            # Only a closing offer is cut here (the cleaned answer is the draft's opening part),
            # and only when a real statement (10+ words) remains. An offer in the middle of a
            # menu, or a draft that is mostly the offer, needs the rewrite.
            kept_words = len(re.findall(r"\w+", cleaned))
            if (
                cleaned != message
                and cleaned != UNSUPPORTED_CAPABILITY_FALLBACK
                and message.startswith(cleaned)
                and kept_words >= 10
            ):
                message = cleaned
                RecommendationService._record_repair(
                    trace, stage, "unsupported_capability", "remove_capability_offers", "applied"
                )
                violations = quality_validator.validate(
                    message,
                    mode_context,
                    user_message,
                    planning_context,
                    build_context,
                    read_only_context,
                )
        codes = {v.code for v in violations}
        if codes & LOCAL_STYLE_CODES:
            repaired, done = repair_style(
                message,
                codes,
                mode=mode_context,
                # Before any lookup an offer is work for the tool-enabled retry, not a cut.
                permission_allowed=read_only_context.lookups_performed
                or trace["initiative_retry"]["attempted"]
                or "unsupported_capability" in codes,
            )
            for entry in done:
                RecommendationService._record_repair(
                    trace, stage, entry["validator"], entry["action"], entry["result"]
                )
            if repaired != message:
                message = repaired
                violations = quality_validator.validate(
                    message,
                    mode_context,
                    user_message,
                    planning_context,
                    build_context,
                    read_only_context,
                )
        constraint = {v.code for v in violations if is_constraint_code(v.code)}
        if constraint:
            repaired_text = RecommendationService._remove_flagged(
                message,
                [v for v in violations if v.code in constraint],
                quality_validator,
                mode_context,
                user_message,
                planning_context,
                build_context,
                read_only_context,
            )
            label = ",".join(sorted(constraint))
            if repaired_text is None:
                RecommendationService._record_repair(
                    trace, stage, label, "remove_flagged_sentences", "no_recommendation_remains"
                )
            else:
                after = quality_validator.validate(
                    repaired_text,
                    mode_context,
                    user_message,
                    planning_context,
                    build_context,
                    read_only_context,
                )
                if constraint & {v.code for v in after}:
                    RecommendationService._record_repair(
                        trace, stage, label, "remove_flagged_sentences", "constraint_remains"
                    )
                else:
                    message, violations = repaired_text, after
                    RecommendationService._record_repair(
                        trace, stage, label, "remove_flagged_sentences", "applied"
                    )
        if not build_context.is_build_request:
            return message, violations
        claims = quality_validator.ownership_claims(message, build_context)
        removed: list[Any] = []
        if any(value.code == "unsupported_ownership_claim" for value in violations):
            repaired, removed = quality_validator.repair_ownership(message, build_context)
            if removed:
                message = repaired
                violations = quality_validator.validate(
                    message,
                    mode_context,
                    user_message,
                    planning_context,
                    build_context,
                    read_only_context,
                )
        # Which characters the recorded "equipped" evidence belongs to (classes only).
        trace["ownership_validation"]["equipped_evidence_characters"] = (
            build_context.ownership.equipped_characters()
        )
        trace["ownership_validation"]["passes"].append(
            {
                "stage": stage,
                "claims": [
                    value.trace(
                        "removed"
                        if value in removed
                        else "kept"
                        if value.supported
                        else "unsupported"
                    )
                    for value in claims[:MAX_TRACED_OWNERSHIP_CLAIMS]
                ],
                "claims_total": len(claims),
                "active_character": build_context.active_character,
                "repaired": bool(removed),
                "removed_claims": len(removed),
                "remaining_violation_codes": [value.code for value in violations],
            }
        )
        return message, violations

    @staticmethod
    def _scoped_history(
        history: list[ChatTurn], scope: ConversationScope
    ) -> tuple[list[dict[str, Any]], int]:
        """History for the model with other characters' earlier replies removed.

        An assistant reply was written for whichever character the conversation was scoped to
        when it was sent. If that is not the active character, its quests, builds, and gear
        claims are stale for this conversation, so the reply is replaced by a placeholder and
        cannot be reused. The user's own turns are kept: they carry the constraints.
        """

        items: list[dict[str, Any]] = []
        removed = 0
        for index, turn in enumerate(history):
            if turn.role == "assistant" and scope.active:
                stated = derive_session_preferences(history[:index], "").preferences.character
                written_for = stated if stated != "unspecified" else scope.bungie_current
                if written_for and written_for != scope.active:
                    items.append(
                        {
                            "role": "assistant",
                            "content": (
                                f"(An earlier reply about the player's {written_for} is omitted: "
                                f"this conversation is now about the {scope.active}.)"
                            ),
                        }
                    )
                    removed += 1
                    continue
            items.append({"role": turn.role, "content": turn.content})
        return items, removed

    @staticmethod
    def _conversation_scope_trace(
        scope: ConversationScope,
        *,
        planning_continuation: bool,
        planning_rebuilt: bool,
        stale_turns_removed: int,
    ) -> dict[str, Any]:
        return {
            "bungie_current": scope.bungie_current,
            "active": scope.active,
            "source": scope.source,
            "previous_active": scope.previous_active,
            "switched": scope.switched,
            "planning_continuation": planning_continuation,
            "planning_context_rebuilt": planning_rebuilt,
            "stale_context_removed": {"assistant_turns": stale_turns_removed},
        }

    @staticmethod
    def _current_character_trace(
        current: CurrentCharacter | None, explicit_scope: str | None
    ) -> dict[str, Any]:
        """Why the current character is what it is; never alters tool behavior."""

        if current is None:
            return {"resolved": False, "explicit_user_scope": explicit_scope}
        return {
            "resolved": True,
            "character_ref": hashlib.sha256(current.character_id.encode()).hexdigest()[:10],
            "class": current.class_name,
            "reason": current.reason,
            "last_played": current.last_played.isoformat() if current.last_played else None,
            "candidates": current.candidates,
            "explicit_user_scope": explicit_scope,
        }

    @staticmethod
    def _character_selection_trace(
        tool_name: str,
        arguments_json: str,
        explicit_scope: str | None,
        current: CurrentCharacter | None,
        conversation_class: str | None = None,
    ) -> dict[str, Any] | None:
        """How one character-scoped Guardian call chose its character."""

        try:
            arguments = json.loads(arguments_json)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(arguments, dict) or not (
            "character_id" in arguments or "character_class" in arguments
        ):
            return None
        source = selector_source(arguments.get("character_id"), arguments.get("character_class"))
        unspecified = source == "unspecified"
        conversation_default = bool(
            unspecified
            and conversation_class
            and tool_name in (CURRENT_DEFAULT_TOOLS | CONVERSATION_SCOPED_TOOLS)
        )
        defaulted = unspecified and tool_name in CURRENT_DEFAULT_TOOLS and not conversation_default
        selected_class = arguments.get("character_class")
        return {
            "tool": tool_name,
            "selector": (
                "conversation_default"
                if conversation_default
                else "current_default"
                if defaulted
                else "all_characters"
                if source == "unspecified"
                else source
            ),
            "class": selected_class if isinstance(selected_class, str) else None,
            "used_current_class": current.class_name if defaulted and current else None,
            "used_conversation_class": conversation_class if conversation_default else None,
            # A class chosen with no class named by the user is a model guess.
            "class_without_user_scope": bool(source == "explicit_class" and explicit_scope is None),
        }

    def _core_engine(self) -> DestinyKnowledgeEngine | None:
        providers = getattr(self.knowledge, "providers", None) or []
        provider = next(
            (
                value
                for value in providers
                if getattr(value, "knowledge_category", None) == CORE_PROVIDER_CATEGORY
            ),
            None,
        )
        engine = getattr(provider, "engine", None)
        return engine if isinstance(engine, DestinyKnowledgeEngine) else None

    def _core_grounding(
        self, message: str, planning_context: SessionPlanningContext | None
    ) -> CoreGrounding:
        """Build the turn's grounded-facts block once, from local deterministic inputs only.

        Inputs are the user's message and planning candidate names, in candidate order. The
        engine is in memory: no network, disk, fuzzy search, or model call happens here.
        """

        engine = self._core_engine()
        if engine is None:
            return CoreGrounding()
        candidates = (
            tuple(value.name for value in planning_context.candidates) if planning_context else ()
        )
        trace: dict[str, Any] = {
            "available": True,
            "used": False,
            "corpus_version": engine.corpus_version,
            "candidate_names_considered": len(candidates),
            "bundle_attached": False,
            "matched_entity_ids": [],
            "resolutions": [],
            "ambiguity_groups": [],
            "bundle_chars": 0,
            "bundle_token_estimate": 0,
            "instruction_chars_added": 0,
            "omitted": {},
            "truncated": False,
            "tool_used": False,
            "manifest_conflicts": [],
            "destination_conflicts": [],
        }
        try:
            bundle = engine.bundle(message, candidates)
            mentioned = [
                engine.entity_brief(value.resolution.entity_id)
                for value in engine.mentions(message)
                if value.resolution.entity_id is not None
            ]
            archetypes = tuple(
                dict.fromkeys(
                    value["name"] for value in mentioned if value["kind"] == "weapon_archetype"
                )
            )
        except Exception:
            # Grounding is additive; a failure must never break the chat turn.
            logger.exception("Core knowledge bundle could not be built")
            trace["error"] = True
            return CoreGrounding(trace=trace)
        if bundle is None:
            return CoreGrounding(trace=trace, weapon_archetypes=archetypes)
        block = f"{CORE_FACTS_HEADER}\n{bundle.text}"
        trace.update(
            used=True,
            bundle_attached=True,
            matched_entity_ids=[value.id for value in bundle.matches if value.via != "ambiguity"],
            resolutions=[
                {"id": value.id, "via": value.via, "origin": value.origin}
                for value in bundle.matches
            ],
            ambiguity_groups=[value.id for value in bundle.matches if value.via == "ambiguity"],
            bundle_chars=bundle.char_count,
            bundle_token_estimate=-(-bundle.char_count // 4),
            instruction_chars_added=len(block) + 2,
            omitted=dict(bundle.payload.get("omitted", {})),
            truncated=bundle.truncated,
        )
        return CoreGrounding(block=block, trace=trace, weapon_archetypes=archetypes)

    @staticmethod
    def _with_clock(instructions: str, clock_block: str) -> str:
        # Last of all, after the core facts, so everything before it is an unchanged prefix.
        return f"{instructions}\n\n{clock_block}"

    @staticmethod
    def _with_core_facts(instructions: str, grounding: CoreGrounding) -> str:
        # Appended last, so the static system prefix and routing stay a cacheable prefix.
        return f"{instructions}\n\n{grounding.block}" if grounding.block else instructions

    @staticmethod
    def _safe_core_tool_summary(result: dict[str, Any]) -> dict[str, Any]:
        summary: dict[str, Any] = {"resolution": result.get("status")}
        if isinstance(result.get("matched_via"), str):
            summary["matched_via"] = result["matched_via"]
        entity = result.get("entity")
        if isinstance(entity, dict) and isinstance(entity.get("kind"), str):
            summary["entity_kind"] = entity["kind"]
        if result.get("status") == "ambiguous":
            summary["candidate_count"] = len(result.get("candidates") or [])
        return summary

    def _record_manifest_conflict(
        self, grounding: CoreGrounding, tool_name: str, result: dict[str, Any]
    ) -> None:
        """Trace-only: an exact Manifest activity whose destination contradicts curation."""

        engine = self._core_engine()
        if engine is None or tool_name != "get_activity_details":
            return
        try:
            resolution = result.get("resolution") or {}
            activity = result.get("activity") or {}
            destination = (activity.get("destination") or {}).get("name")
            if not resolution.get("exact") or not destination:
                return
            resolved = engine.resolve(str(resolution.get("matched_name") or ""))
            grounded = engine.located_on(resolved.entity_id) if resolved.entity_id else None
            if grounded and normalize_text(grounded) != normalize_text(str(destination)):
                grounding.trace["manifest_conflicts"].append(
                    {
                        "entity_id": resolved.entity_id,
                        "grounded_located_on": grounded,
                        "manifest_destination": str(destination)[:80],
                    }
                )
        except Exception:
            logger.warning("Core/Manifest conflict check failed")

    def _destination_names(self) -> frozenset[str]:
        names: set[str] = set()
        for provider in getattr(self.knowledge, "providers", None) or []:
            accessor = getattr(provider, "destination_names", None)
            if callable(accessor):
                names.update(accessor())
        return frozenset(names)

    def _record_destination_conflicts(self, grounding: CoreGrounding, answer: str) -> None:
        """Trace-only: a final answer that places a grounded entity somewhere else.

        Emitted only when one sentence names the entity and a different known destination
        with locational wording, and omits the grounded destination. Never alters answers.
        """

        engine = self._core_engine()
        ids = grounding.trace.get("matched_entity_ids") or []
        if engine is None or not ids:
            return
        try:
            located = {
                entity_id: normalize_text(value)
                for entity_id in ids
                if (value := engine.located_on(entity_id))
            }
            if not located:
                return
            vocabulary = {
                normalize_text(value): value
                for value in (*self._destination_names(), *located.values())
                if normalize_text(value)
            }
            for sentence in re.split(r"[.!?\n]+", answer):
                padded = f" {normalize_text(sentence)} "
                if not _LOCATION_WORDS.search(padded):
                    continue
                for entity_id, grounded in located.items():
                    name = normalize_text(engine.entity_brief(entity_id)["name"])
                    if f" {name} " not in padded or f" {grounded} " in padded:
                        continue
                    others = sorted(
                        original
                        for key, original in vocabulary.items()
                        if key != grounded and key not in name and f" {key} " in padded
                    )
                    if others:
                        grounding.trace["destination_conflicts"].append(
                            {
                                "entity_id": entity_id,
                                "grounded_located_on": engine.located_on(entity_id),
                                "answer_destinations": others[:3],
                            }
                        )
        except Exception:
            logger.warning("Core destination conflict check failed")

    @staticmethod
    def _is_session_planning(message: str) -> bool:
        return is_standalone_planning(message)

    @staticmethod
    def _intent_category(message: str) -> str:
        normalized = message.casefold()
        if TIME_INTENT_PATTERN.search(message):
            return "time_limited"
        if any(value in normalized for value in ("better gear", "loot", "reward", "weapon")):
            return "gear"
        if any(value in normalized for value in ("chill", "casual", "relax", "low stress")):
            return "casual"
        if any(value in normalized for value in ("challenge", "challenging", "hard")):
            return "challenge"
        if any(value in normalized for value in ("build", "loadout", "subclass")):
            return "build"
        if any(value in normalized for value in ("story", "campaign", "dlc", "expansion")) or (
            mentioned_content_names(message)
        ):
            return "story"
        if any(value in normalized for value in ("quest", "bounty", "objective")):
            return "quest_completion"
        if any(value in normalized for value in ("progress", "unlock", "advance")):
            return "progression"
        return "general"

    @classmethod
    def _effective_intent_category(
        cls, message: str, derivation: SessionPreferenceDerivation
    ) -> str:
        goal_categories = {
            "story_progression": "story",
            "gear_rewards": "gear",
            "quest_completion": "quest_completion",
            "build_improvement": "build",
            "casual_chill": "casual",
            "challenge": "challenge",
            "exploration": "exploration",
        }
        goal = derivation.preferences.primary_goal
        if goal != "unspecified":
            return goal_categories[goal]
        if derivation.preferences.time_minutes is not None:
            return "time_limited"
        if "primary_goal" in derivation.current_turn_updates:
            return "general"
        return cls._intent_category(message)

    @staticmethod
    def _requested_character_scope(message: str) -> str | None:
        match = CHARACTER_SCOPE_PATTERN.search(message)
        if not match:
            return None
        return next(value for value in match.groups() if value).title()

    @staticmethod
    def _forced_alternatives_relevance(
        message: str,
        context: GuardianContext,
        arguments_json: str,
        result: dict[str, Any],
        alias_terms: tuple[str, ...] = (),
    ) -> str:
        try:
            arguments = json.loads(arguments_json)
            return forced_alternatives_relevance(
                message, context, arguments, result, alias_terms=alias_terms
            )
        except Exception:
            # A trace signal must never be able to break a chat turn.
            logger.warning("Forced alternatives relevance could not be computed")
            return "not_applicable"

    @staticmethod
    def _round_tool_choice(
        round_number: int,
        title_request_context: TitleRequestContext,
        build_context: BuildResponseValidationContext,
        already_forced: set[str],
        item_lookup: str | None = None,
    ) -> tuple[str | dict[str, Any], tuple[str, ...]]:
        """Select this round's tool_choice and the build evidence tools it forces.

        Title retrieval keeps first-round priority, then a required item-definition lookup.
        Required build evidence is acquired up front instead of being requested later by a
        tool-less correction. Each evidence tool is forced at most once, and the final round
        allows no tools so the model must answer.
        """

        if round_number == 1 and title_request_context.retrieval_required:
            return {"type": "function", "name": "get_title_progress"}, ()
        if round_number == 1 and item_lookup:
            return {"type": "function", "name": item_lookup}, ()
        if round_number >= MAX_TOOL_ROUNDS:
            return "none", ()
        missing = tuple(
            name
            for name, required in (
                (
                    "analyze_current_build",
                    build_context.requires_current_build_analysis
                    and not build_context.current_build_analysis_used,
                ),
                (
                    "find_build_alternatives",
                    build_context.requires_owned_inventory
                    and not build_context.inventory_ownership_checked,
                ),
            )
            if required and name not in already_forced
        )
        if not missing:
            return "auto", ()
        if len(missing) == 1:
            return {"type": "function", "name": missing[0]}, missing
        # Require at least one of them while still allowing both in one parallel round.
        return (
            {
                "type": "allowed_tools",
                "mode": "required",
                "tools": [{"type": "function", "name": name} for name in missing],
            },
            missing,
        )

    @staticmethod
    def _turn_instructions(
        session_planning: bool,
        intent_category: str | None,
        character_scope: str | None,
        planning_context: SessionPlanningContext | None,
        mode_context: ResponseModeContext,
        build_request_context: BuildRequestContext,
        title_request_context: TitleRequestContext,
        current_character: CurrentCharacter | None = None,
        *,
        research_requested: bool = False,
        live_state_research: bool = False,
    ) -> str:
        routing: list[str] = [response_mode_instruction(mode_context)]
        if live_state_research:
            routing.append(
                "This asks about current or rotating game state, which Manifest, guide, and "
                "account data cannot establish. Use web research now. Answer from what the "
                "source says, name the source, and give its date or say it is undated. If the "
                "source does not establish the state right now, say what it does show and that "
                "the current state is unconfirmed."
            )
        if research_requested:
            routing.append(
                "The user explicitly asked you to verify, confirm, or look this up. Run the "
                "lookups now without asking which source to use: Manifest or Guide for canonical "
                "item, activity, and source facts; web research for current or changed "
                "information; Guardian tools for anything about this player's account. If the "
                "first source cannot settle it, try the next, then say in a sentence what you "
                "found and where. If nothing settles it, say so briefly."
            )
        if session_planning:
            routing.append(
                "Turn routing: handle this turn as session planning. "
                f"The safe intent category is {intent_category or 'general'}. "
                "Use the bounded effective preferences in the planning context; do not reconstruct "
                "or override them from assistant messages or Guardian activity, and do not mention "
                "this routing metadata."
            )
        if character_scope:
            routing.append(
                f"Explicit character scope: {character_scope}. Keep account inspection and the "
                f"recommendation scoped to the user's {character_scope}; do not offer another "
                "class as the default or backup unless switching is necessary to answer the "
                f"request. Pass character_class={character_scope} and character_id=null to "
                "character-scoped Guardian tools; do not ask the user for an ID."
            )
        elif current_character is not None:
            routing.append(
                f"Current character (most recently played): {current_character.class_name}. "
                "When the user does not name a character, pass null for both character_id "
                "and character_class to the tools that default to it (loadout, build details, "
                "build analysis, build alternatives); never guess a class. get_content_progression "
                f"has no default and needs one selector: pass character_class="
                f"{current_character.class_name}. If the user names a character, that explicit "
                "choice wins."
            )
        if planning_context is not None:
            routing.append(
                "Application-generated session planning context follows as bounded JSON. Treat "
                "its evidence states literally, use only candidates within its character scope, "
                "and do not infer additional account facts from missing values:\n"
                f"{planning_context.model_dump_json()}"
            )
        if build_request_context.is_build_request:
            routing.append(
                "Application-derived build constraints follow as bounded JSON. Treat explicit "
                "locked items, preserved Exotics, activity mode, fireteam, and change limit as "
                "hard constraints. For a review of the current build, keep the equipped Exotic and "
                "the weapon the player said they like unless they asked to change them: give "
                "one smaller improvement you can verify from the returned data (an open "
                "socket, a mod, a fragment, a stat), or say plainly that the evidence does not "
                "justify a particular change. Never ask permission to swap an Exotic. "
                "goal_items are items the player said they want to get or use: "
                "build around them and never recommend replacing one unless the player asks about "
                "it directly. A null goal means this message names no build goal: use the "
                "activity mode and goals from the conversation, and otherwise do not assume boss "
                "DPS or another niche goal; judge changes for general play. A normal_change_limit "
                "of 1 means exactly one change. Prefer a change grounded in returned data (an "
                "owned item, an open socket, a mod, fragment, or stat observation) over a generic "
                "weapon archetype. If requires_current_build_analysis is true, call "
                "analyze_current_build before judging or recommending changes to the current "
                "setup. If requires_owned_inventory is true, use find_build_alternatives or an "
                "appropriate Guardian inventory lookup before naming owned choices or exact "
                "instance perks. Do not mention this routing metadata:\n"
                f"{build_request_context.model_dump_json()}"
            )
        if title_request_context.retrieval_required:
            routing.append(
                "Resolved read-only Title/Seal request: call get_title_progress now without "
                "asking permission or confirming a format. "
                f"Request at most {title_request_context.requested_limit} title result(s). "
                "For a closest/easiest request, report observable completion proximity and say "
                "that actual time or effort may differ. Do not estimate effort or expose raw "
                "Record identifiers."
            )
        elif title_request_context.title_intent_present:
            routing.append(
                "The word 'title' may still have materially different meanings. One targeted "
                "clarification about whether the user means Destiny Triumph/Record Titles or "
                "Seals is allowed; do not ask permission to retrieve data."
            )
        return f"{SYSTEM_INSTRUCTIONS}\n\n" + "\n".join(routing) if routing else SYSTEM_INSTRUCTIONS

    async def _create_response(
        self,
        *,
        stream_status: StreamStatusReporter | None,
        **request_options: Any,
    ) -> Any:
        if stream_status is None:
            response = await self.client.responses.create(**request_options)
        else:
            async with self.client.responses.stream(**request_options) as stream:
                async for event in stream:
                    if is_web_search_stream_event(self._field(event, "type")):
                        await stream_status.emit("web")
                response = await stream.get_final_response()
        self.limits.record_response(response)
        return response

    @classmethod
    def _normalize_replay_output_items(cls, output_items: list[Any]) -> list[dict[str, Any]]:
        """Convert SDK response output models to strict Responses input wire objects."""

        return [cls._normalize_replay_output_item(item) for item in output_items]

    @classmethod
    def _normalize_replay_output_item(cls, item: Any) -> dict[str, Any]:
        item_type = cls._field(item, "type")
        if item_type == "function_call":
            return {
                "type": "function_call",
                "call_id": cls._required_replay_string(item, "call_id", item_type),
                "name": cls._required_replay_string(item, "name", item_type),
                "arguments": cls._required_replay_string(item, "arguments", item_type),
            }
        if item_type == "reasoning":
            normalized: dict[str, Any] = {
                "type": "reasoning",
                "id": cls._required_replay_string(item, "id", item_type),
                "summary": cls._normalize_replay_text_parts(
                    cls._field(item, "summary"), "summary_text", item_type
                ),
            }
            encrypted_content = cls._field(item, "encrypted_content")
            if isinstance(encrypted_content, str):
                normalized["encrypted_content"] = encrypted_content
            content = cls._field(item, "content")
            if content is not None:
                normalized["content"] = cls._normalize_replay_text_parts(
                    content, "reasoning_text", item_type
                )
            cls._copy_replay_enum(
                item,
                normalized,
                "status",
                {"in_progress", "completed", "incomplete"},
            )
            return normalized
        if item_type == "message":
            content = cls._field(item, "content")
            if not isinstance(content, (list, tuple)):
                raise RuntimeError("Responses message output is missing replayable content.")
            normalized_content = [cls._normalize_replay_message_content(value) for value in content]
            normalized = {
                "type": "message",
                "id": cls._required_replay_string(item, "id", item_type),
                "role": cls._required_replay_string(item, "role", item_type),
                "status": cls._required_replay_string(item, "status", item_type),
                "content": normalized_content,
            }
            if normalized["role"] != "assistant":
                raise RuntimeError("Responses output message has an unsupported replay role.")
            if normalized["status"] not in {"in_progress", "completed", "incomplete"}:
                raise RuntimeError("Responses output message has an unsupported replay status.")
            cls._copy_replay_enum(item, normalized, "phase", {"commentary", "final_answer"})
            return normalized
        if item_type == "web_search_call":
            return cls._normalize_replay_web_search_call(item)
        raise RuntimeError(
            "Unsupported Responses output item type for stateless replay: "
            f"{item_type or type(item).__name__}."
        )

    @classmethod
    def _normalize_replay_web_search_call(cls, item: Any) -> dict[str, Any]:
        item_type = "web_search_call"
        action = cls._field(item, "action")
        action_type = cls._field(action, "type")
        normalized_action: dict[str, Any] = {"type": action_type}
        if action_type == "search":
            query = cls._field(action, "query")
            if isinstance(query, str):
                normalized_action["query"] = query
            queries = cls._field(action, "queries")
            if queries is not None:
                if not isinstance(queries, (list, tuple)) or not all(
                    isinstance(value, str) for value in queries
                ):
                    raise RuntimeError("Responses web-search output has malformed replay queries.")
                normalized_action["queries"] = list(queries)
            sources = cls._field(action, "sources")
            if sources is not None:
                if not isinstance(sources, (list, tuple)):
                    raise RuntimeError("Responses web-search output has malformed replay sources.")
                normalized_sources: list[dict[str, str]] = []
                for source in sources:
                    if cls._field(source, "type") != "url":
                        raise RuntimeError(
                            "Responses web-search output has an unsupported replay source type."
                        )
                    normalized_sources.append(
                        {
                            "type": "url",
                            "url": cls._required_replay_string(source, "url", "web search source"),
                        }
                    )
                normalized_action["sources"] = normalized_sources
        elif action_type == "open_page":
            url = cls._field(action, "url")
            if url is not None:
                if not isinstance(url, str):
                    raise RuntimeError("Responses web-search output has a malformed replay URL.")
                normalized_action["url"] = url
        elif action_type == "find_in_page":
            normalized_action.update(
                {
                    "pattern": cls._required_replay_string(action, "pattern", action_type),
                    "url": cls._required_replay_string(action, "url", action_type),
                }
            )
        else:
            raise RuntimeError(
                "Unsupported Responses web-search action type for stateless replay: "
                f"{action_type or type(action).__name__}."
            )
        normalized = {
            "type": item_type,
            "id": cls._required_replay_string(item, "id", item_type),
            "status": cls._required_replay_string(item, "status", item_type),
            "action": normalized_action,
        }
        if normalized["status"] not in {"in_progress", "searching", "completed", "failed"}:
            raise RuntimeError("Responses web-search output has an unsupported replay status.")
        return normalized

    @classmethod
    def _normalize_replay_message_content(cls, content: Any) -> dict[str, Any]:
        content_type = cls._field(content, "type")
        if content_type == "refusal":
            return {
                "type": "refusal",
                "refusal": cls._required_replay_string(content, "refusal", content_type),
            }
        if content_type != "output_text":
            raise RuntimeError(
                "Unsupported Responses message content type for stateless replay: "
                f"{content_type or type(content).__name__}."
            )
        annotations = cls._field(content, "annotations", []) or []
        if not isinstance(annotations, (list, tuple)):
            raise RuntimeError("Responses output text has malformed replay annotations.")
        return {
            "type": "output_text",
            "text": cls._required_replay_string(content, "text", content_type),
            "annotations": [cls._normalize_replay_annotation(value) for value in annotations],
        }

    @classmethod
    def _normalize_replay_annotation(cls, annotation: Any) -> dict[str, Any]:
        annotation_type = cls._field(annotation, "type")
        required_fields = {
            "url_citation": ("start_index", "end_index", "title", "url"),
            "file_citation": ("file_id", "filename", "index"),
            "container_file_citation": (
                "container_id",
                "end_index",
                "file_id",
                "filename",
                "start_index",
            ),
            "file_path": ("file_id", "index"),
        }
        fields = required_fields.get(annotation_type)
        if fields is None:
            raise RuntimeError(
                "Unsupported Responses annotation type for stateless replay: "
                f"{annotation_type or type(annotation).__name__}."
            )
        normalized = {"type": annotation_type}
        for field in fields:
            value = cls._field(annotation, field)
            expected_type = int if field in {"start_index", "end_index", "index"} else str
            if not isinstance(value, expected_type):
                raise RuntimeError(
                    f"Responses {annotation_type} annotation is missing replay field {field}."
                )
            normalized[field] = value
        return normalized

    @classmethod
    def _normalize_replay_text_parts(
        cls, values: Any, expected_type: str, parent_type: str
    ) -> list[dict[str, str]]:
        if not isinstance(values, (list, tuple)):
            raise RuntimeError(f"Responses {parent_type} output is missing replay text parts.")
        normalized: list[dict[str, str]] = []
        for value in values:
            if cls._field(value, "type") != expected_type:
                raise RuntimeError(
                    f"Responses {parent_type} output has unsupported replay content."
                )
            normalized.append(
                {
                    "type": expected_type,
                    "text": cls._required_replay_string(value, "text", expected_type),
                }
            )
        return normalized

    @classmethod
    def _required_replay_string(cls, value: Any, field: str, item_type: str) -> str:
        result = cls._field(value, field)
        if not isinstance(result, str):
            raise RuntimeError(f"Responses {item_type} output is missing replay field {field}.")
        return result

    @classmethod
    def _copy_replay_enum(
        cls,
        source: Any,
        target: dict[str, Any],
        field: str,
        allowed: set[str],
    ) -> None:
        value = cls._field(source, field)
        if value is None:
            return
        if value not in allowed:
            raise RuntimeError(f"Responses output has unsupported replay {field}.")
        target[field] = value

    async def _request_planning_correction(
        self,
        input_items: list[Any],
        response_output: list[Any],
        instructions: str,
        tool_definitions: list[dict[str, Any]],
        correction_instruction: str,
        stream_status: StreamStatusReporter | None,
    ) -> Any:
        correction_input = [
            *input_items,
            *self._normalize_replay_output_items(response_output),
            {"role": "developer", "content": correction_instruction},
        ]
        self._log_request_metadata(correction_input, MAX_TOOL_ROUNDS + 1)
        return await self._create_response(
            stream_status=stream_status,
            model=self.settings.openai_model,
            instructions=instructions,
            input=correction_input,
            tools=tool_definitions,
            include=RESPONSE_INCLUDE,
            tool_choice="none",
            parallel_tool_calls=False,
            reasoning={"effort": self.settings.openai_reasoning_effort},
            max_output_tokens=self.settings.openai_max_output_tokens,
            truncation="disabled",
            store=False,
        )

    @staticmethod
    def _record_mechanics_evidence(
        ledger: MechanicsEvidenceLedger, tool_name: str, category: str, result: Any
    ) -> None:
        """Observation only: reads a tool result into the trace-only mechanics ledger."""

        try:
            ledger.record(tool_name, category, result)
        except Exception:
            logger.warning("Mechanics evidence ledger could not record a tool result")

    @staticmethod
    def _record_planning_requirements(
        ledger: MechanicsEvidenceLedger, planning_context: SessionPlanningContext | None
    ) -> None:
        """Observation only: lets the trace-only ledger see requirements shown to the model."""

        if planning_context is None:
            return
        try:
            ledger.record_planning(planning_context)
        except Exception:
            logger.warning("Mechanics evidence ledger could not record planning requirements")

    def _mechanics_claim_trace(
        self, answer: str, ledger: MechanicsEvidenceLedger
    ) -> dict[str, Any]:
        """Trace-only mechanics-claim summary; it never alters or blocks the answer."""

        try:
            report = detect_mechanics_claims(
                answer, ledger, entity_resolver=self._manifest_name_resolver()
            )
            return report.to_trace()
        except Exception:
            logger.warning("Mechanics claim detection failed; trace field omitted")
            return {"mode": "trace_only", "error": "detector_failed"}

    def _manifest_name_resolver(self) -> Callable[[str], bool] | None:
        """Local index lookup: does the Manifest know this exact item name? No I/O."""

        providers = getattr(self.knowledge, "providers", None) or []
        indexed = [value for value in providers if hasattr(value, "exact_entries")]
        if not indexed:
            return None

        def known(name: str) -> bool:
            try:
                return any(
                    provider.exact_entries(name, "DestinyInventoryItemDefinition")
                    for provider in indexed
                )
            except Exception:
                return True  # An unavailable index must never read as "unknown item".

        return known

    @staticmethod
    def _record_knowledge_category(trace: dict[str, Any], category: str) -> None:
        categories = trace["knowledge_categories_used"]
        if category not in categories:
            categories.append(category)

    @staticmethod
    def _enforce_power_safety(message: str) -> str:
        if not UNSUPPORTED_POWER_PATTERN.search(message):
            return message
        logger.warning("Blocked an unsupported Power-level comparison in an AI response")
        limit = (
            "I can’t reliably compare the raw Power values returned by the current APIs, so your "
            "Power eligibility is unknown."
        )
        # Replace only the offending sentence(s); the rest of the answer, and the sources it
        # relies on, stay. A claim spread across sentences replaces the whole answer.
        surgical = replace_flagged_sentences(
            message, lambda sentence: limit if UNSUPPORTED_POWER_PATTERN.search(sentence) else None
        )
        if surgical is not None and not UNSUPPORTED_POWER_PATTERN.search(surgical):
            return surgical
        return (
            "I can’t reliably compare the raw Power values returned by the current APIs. "
            "Their meanings may not be compatible with current activity requirements, so your "
            "Power eligibility is unknown. I can still help with the activity’s mechanics, "
            "prerequisites, and a practical loadout based on grounded information."
        )

    @staticmethod
    def _enforce_campaign_completion_safety(message: str, user_message: str = "") -> str:
        premise = states_completion_premise(user_message)
        premise_note = (
            " (going on your word that it's finished; I can't see that in your account data)"
        )

        def replacement(sentence: str) -> str | None:
            normalized = sentence.casefold()
            absence_inference = (
                any(value in normalized for value in CAMPAIGN_MARKERS)
                and any(
                    value in normalized
                    for value in (
                        "no active",
                        "not active",
                        "isn't active",
                        "is not active",
                        "only active on",
                        "doesn't have",
                        "does not have",
                        "wasn't returned",
                        "was not returned",
                    )
                )
                and any(
                    value in normalized
                    for value in (
                        "complete",
                        "finished",
                        "not started",
                        "unavailable",
                        "irrelevant",
                        "owned",
                        "unowned",
                    )
                )
                and any(
                    value in normalized
                    for value in (
                        " so ",
                        "therefore",
                        "means",
                        "must be",
                        "indicates",
                        "proves",
                        "because",
                        "since",
                    )
                )
                and not any(value in normalized for value in COMPLETION_DENIAL_MARKERS)
            )
            if absence_inference and premise:
                return (
                    "No matching active quest was returned, which doesn't show whether it's "
                    "completed; I'm going on your word."
                )
            if absence_inference:
                logger.warning("Blocked an inference from active-quest absence")
                return (
                    "No matching active quest was returned, but that does not establish whether "
                    "the campaign is completed, not started, available, or owned. Campaign "
                    "completion is unknown until relevant Guardian progression explicitly "
                    "establishes it."
                )
            if (
                any(value in normalized for value in CAMPAIGN_MARKERS)
                and any(value in normalized for value in INDIRECT_COMPLETION_MARKERS)
                and any(value in normalized for value in COMPLETION_MARKERS)
                and any(value in normalized for value in CAUSAL_COMPLETION_MARKERS)
                and not any(value in normalized for value in COMPLETION_DENIAL_MARKERS)
            ):
                if premise:
                    # The player said it; the how-to stands on that premise. One clause marks
                    # it as theirs, not as account-verified.
                    stem = sentence.rstrip()
                    end = stem[-1] if stem[-1] in ".!?" else ""
                    return f"{stem.rstrip('.!?')}{premise_note}{end or '.'}"
                logger.warning("Blocked an unsupported campaign-completion inference")
                return (
                    "I can't verify campaign completion from an equipped subclass, item, "
                    "destination, reward, or related unlock. Based on the available account "
                    "data, completion is unknown. I'd verify the campaign quest or progression "
                    "state before choosing between them; once that is known, I can recommend "
                    "the better next step."
                )
            return None

        # Only a flagged sentence is replaced; the rest of the answer and its sources stay.
        return replace_flagged_sentences(message, replacement) or message

    @staticmethod
    def _enforce_campaign_comparison_safety(
        message: str,
        request_message: str,
        character_scope: str | None,
    ) -> str:
        request = request_message.casefold()
        if not ("the final shape" in request and "edge of fate" in request and " or " in request):
            return message

        normalized = message.casefold()
        denial_markers = (
            "not enough",
            "not by itself",
            "alone is not",
            "wouldn't use",
            "would not use",
            "can't choose",
            "cannot choose",
            "unknown",
        )
        active_only_recommendation = any(
            "active" in sentence
            and any(value in sentence for value in ("because", "since", "therefore", "so "))
            and any(
                value in sentence
                for value in (
                    "recommend",
                    "should",
                    "choose",
                    "pick",
                    "continue",
                    "start",
                    "best next",
                )
            )
            and not any(value in sentence for value in denial_markers)
            for sentence in re.split(r"(?<=[.!?])\s+|[\r\n]+", normalized)
        )

        cross_character_recommendation = False
        if character_scope:
            other_classes = {"titan", "hunter", "warlock"} - {character_scope.casefold()}
            cross_character_recommendation = any(
                re.search(
                    rf"\b(?:switch(?:ing)?\s+to|play(?:ing)?\s+(?:on\s+)?|"
                    rf"try(?:ing)?\s+|use\s+|using\s+)(?:your\s+)?{other_class}\b",
                    normalized,
                )
                for other_class in other_classes
            )

        if not active_only_recommendation and not cross_character_recommendation:
            return message

        logger.warning("Blocked an unsupported or out-of-scope campaign recommendation")
        scope = f"your {character_scope}" if character_scope else "the requested character"
        return (
            f"I'd keep this comparison scoped to {scope}. An active campaign quest is only one "
            "signal, and no matching active quest being returned for the other campaign does not "
            "establish its completion, availability, or relevance. That completion state is "
            f"unknown. I can't make a grounded choice until {scope}'s relevant progression and "
            "the significance of both campaigns are established."
        )

    @staticmethod
    def _hide_internal_identifiers(
        message: str,
        context: GuardianContext,
        *,
        allow_identifiers: bool,
    ) -> str:
        if allow_identifiers:
            return message
        safe_message = message
        replacements = [
            (character.character_id, character.class_name) for character in context.characters
        ]
        replacements.append((context.membership_id, "your Bungie account"))
        for identifier, replacement in sorted(
            replacements, key=lambda value: len(value[0]), reverse=True
        ):
            if identifier:
                safe_message = safe_message.replace(identifier, replacement)
        return safe_message

    @staticmethod
    def _field(value: Any, name: str, default: Any = None) -> Any:
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)

    @staticmethod
    def _safe_live_trace(result: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in result.items() if key != "query"}

    @classmethod
    def _source(cls, url: Any, title: Any = None) -> ChatSource | None:
        if not isinstance(url, str):
            return None
        try:
            parsed = urlsplit(url.strip())
            domain = parsed.hostname.casefold() if parsed.hostname else None
            resolved_title = title.strip() if isinstance(title, str) and title.strip() else domain
            if not resolved_title:
                return None
            return ChatSource(title=resolved_title, url=url, domain=domain)
        except (TypeError, ValueError, ValidationError):
            return None

    @staticmethod
    def _canonical_source_key(url: str) -> str:
        """Collapse cosmetic URL differences without changing the URL shown to the user."""

        try:
            parsed = urlsplit(url.strip())
            host = (parsed.hostname or "").casefold()
            if host.startswith("www."):
                host = host[4:]
            port = parsed.port
            if port and not (
                (parsed.scheme.casefold() == "http" and port == 80)
                or (parsed.scheme.casefold() == "https" and port == 443)
            ):
                host = f"{host}:{port}"
            path = parsed.path.rstrip("/") or "/"
            query = urlencode(
                sorted(
                    (key, value)
                    for key, value in parse_qsl(parsed.query, keep_blank_values=True)
                    if not key.casefold().startswith("utm_")
                    and key.casefold() not in {"fbclid", "gclid"}
                )
            )
            return f"{host}{path}?{query}" if query else f"{host}{path}"
        except ValueError:
            return url.strip().casefold()

    @staticmethod
    def _source_title_quality(source: ChatSource) -> tuple[int, int]:
        title = source.title.casefold()
        domain = (source.domain or "").casefold()
        return (int(bool(title and title != domain and title != f"www.{domain}")), len(title))

    @classmethod
    def _merge_sources(
        cls, target: list[ChatSource], candidates: list[ChatSource], limit: int = MAX_CHAT_SOURCES
    ) -> None:
        existing = {
            cls._canonical_source_key(source.url): index for index, source in enumerate(target)
        }
        for source in candidates:
            key = cls._canonical_source_key(source.url)
            if key in existing:
                index = existing[key]
                if cls._source_title_quality(source) > cls._source_title_quality(target[index]):
                    target[index] = source
                continue
            if len(target) >= limit:
                continue
            target.append(source)
            existing[key] = len(target) - 1

    @classmethod
    def _combined_sources(
        cls,
        web_sources: list[ChatSource],
        knowledge_sources: list[ChatSource],
        *,
        limit: int = MAX_CHAT_SOURCES,
    ) -> list[ChatSource]:
        combined: list[ChatSource] = []
        cls._merge_sources(combined, web_sources, limit)
        cls._merge_sources(combined, knowledge_sources, limit)
        return combined

    @classmethod
    def _cited_sources(cls, output_items: list[Any]) -> list[ChatSource]:
        """Only the url_citation sources in assistant messages, in citation order."""

        messages = [item for item in output_items if cls._field(item, "type") == "message"]
        return cls._extract_web_sources(messages)[0]

    @classmethod
    def _extract_web_sources(cls, output_items: list[Any]) -> tuple[list[ChatSource], bool]:
        citation_sources: list[ChatSource] = []
        metadata_sources: list[ChatSource] = []
        web_research_occurred = False
        for item in output_items:
            item_type = cls._field(item, "type")
            if item_type == "web_search_call":
                web_research_occurred = True
                action = cls._field(item, "action")
                for source in cls._field(action, "sources", []) or []:
                    candidate = cls._source(cls._field(source, "url"))
                    if candidate:
                        metadata_sources.append(candidate)
            if item_type != "message":
                continue
            for content in cls._field(item, "content", []) or []:
                for annotation in cls._field(content, "annotations", []) or []:
                    if cls._field(annotation, "type") != "url_citation":
                        continue
                    web_research_occurred = True
                    nested = cls._field(annotation, "url_citation", annotation)
                    candidate = cls._source(cls._field(nested, "url"), cls._field(nested, "title"))
                    if candidate:
                        citation_sources.append(candidate)
        sources: list[ChatSource] = []
        cls._merge_sources(sources, citation_sources)
        cls._merge_sources(sources, metadata_sources)
        return sources, web_research_occurred

    @classmethod
    def _extract_knowledge_sources(cls, result: Any) -> list[ChatSource]:
        candidates: list[ChatSource] = []

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                if "source_url" in value:
                    candidate = cls._source(
                        value.get("source_url"), value.get("title") or value.get("provider")
                    )
                    if candidate:
                        candidates.append(candidate)
                for nested in value.values():
                    visit(nested)
            elif isinstance(value, list):
                for nested in value:
                    visit(nested)

        visit(result)
        sources: list[ChatSource] = []
        cls._merge_sources(sources, candidates)
        return sources

    def _log_request_metadata(self, input_items: list[Any], round_number: int) -> None:
        sequence: list[str] = []
        function_call_positions: dict[str, int] = {}
        function_output_positions: list[tuple[str, int]] = []
        for index, item in enumerate(input_items):
            role = self._field(item, "role")
            item_type = self._field(item, "type")
            call_id = self._field(item, "call_id")
            if role:
                sequence.append(f"{role}_message")
            elif item_type:
                sequence.append(str(item_type))
            else:
                sequence.append(type(item).__name__)
            if item_type == "function_call" and call_id:
                function_call_positions[str(call_id)] = index
            elif item_type == "function_call_output" and call_id:
                function_output_positions.append((str(call_id), index))

        correlation_valid = all(
            call_id in function_call_positions
            and function_call_positions[call_id] < output_position
            for call_id, output_position in function_output_positions
        )
        logger.info(
            "OpenAI request round=%d continuation=stateless_input_replay model=%s "
            "max_output_tokens=%d reasoning_effort=%s truncation=disabled "
            "input_sequence=%s function_call_ids=%s function_call_output_ids=%s "
            "tool_call_correlation_valid=%s",
            round_number,
            self.settings.openai_model,
            self.settings.openai_max_output_tokens,
            self.settings.openai_reasoning_effort,
            sequence,
            list(function_call_positions),
            [call_id for call_id, _ in function_output_positions],
            correlation_valid,
        )
        if not correlation_valid:
            raise RuntimeError("OpenAI tool-call input contains an unmatched function result.")

    def _log_response_metadata(self, response: Any, phase: str) -> None:
        output_items = self._field(response, "output", []) or []
        output_types = [self._field(item, "type", type(item).__name__) for item in output_items]
        usage = self._field(response, "usage")
        output_details = self._field(usage, "output_tokens_details")
        reasoning = self._field(response, "reasoning")
        incomplete = self._field(response, "incomplete_details")
        error = self._field(response, "error")
        logger.info(
            "OpenAI response phase=%s status=%s incomplete_details=%s error=%s "
            "output_types=%s usage_input_tokens=%s usage_output_tokens=%s "
            "reasoning_tokens=%s model=%s max_output_tokens=%s "
            "request_reasoning_effort=%s response_reasoning_effort=%s truncation=%s",
            phase,
            self._field(response, "status"),
            ({"reason": self._field(incomplete, "reason")} if incomplete else None),
            (
                {
                    "code": self._field(error, "code"),
                    "message": self._field(error, "message"),
                }
                if error
                else None
            ),
            output_types,
            self._field(usage, "input_tokens"),
            self._field(usage, "output_tokens"),
            self._field(output_details, "reasoning_tokens"),
            self._field(response, "model", self.settings.openai_model),
            self._field(response, "max_output_tokens", self.settings.openai_max_output_tokens),
            self.settings.openai_reasoning_effort,
            self._field(reasoning, "effort"),
            self._field(response, "truncation", "disabled"),
        )

    def _require_completed_response(self, response: Any) -> None:
        status = self._field(response, "status")
        incomplete = self._field(response, "incomplete_details")
        error = self._field(response, "error")
        if status == "completed" and incomplete is None and error is None:
            return
        reason = self._field(incomplete, "reason") or self._field(error, "code") or status
        logger.error("OpenAI response did not complete reason=%s", reason)
        raise RuntimeError(f"OpenAI response did not complete ({reason or 'unknown reason'}).")

    @staticmethod
    def _extract_assistant_text(response: Any) -> str:
        """Extract final text from the heterogeneous Responses API output list."""

        output_items = getattr(response, "output", None) or []
        output_types = [
            (
                item.get("type", type(item).__name__)
                if isinstance(item, dict)
                else getattr(item, "type", type(item).__name__)
            )
            for item in output_items
        ]

        missing = object()
        sdk_output_text = getattr(response, "output_text", missing)
        accessor_present = sdk_output_text is not missing
        text = sdk_output_text if isinstance(sdk_output_text, str) else ""

        if not text.strip():
            text_parts: list[str] = []
            for item in output_items:
                item_type = (
                    item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
                )
                if item_type != "message":
                    continue
                content_items = (
                    item.get("content", [])
                    if isinstance(item, dict)
                    else getattr(item, "content", [])
                ) or []
                for content in content_items:
                    content_type = (
                        content.get("type")
                        if isinstance(content, dict)
                        else getattr(content, "type", None)
                    )
                    if content_type not in {"output_text", "text"}:
                        continue
                    value = (
                        content.get("text")
                        if isinstance(content, dict)
                        else getattr(content, "text", None)
                    )
                    if isinstance(value, str):
                        text_parts.append(value)
            text = "".join(text_parts)

        text = text.strip()
        logger.info(
            "OpenAI final response output_types=%s output_text_accessor_present=%s "
            "output_text_nonempty=%s extracted_text_length=%d",
            output_types,
            accessor_present,
            isinstance(sdk_output_text, str) and bool(sdk_output_text.strip()),
            len(text),
        )
        return text

    async def _execute_tool_call(
        self, tools: GuardianToolService, name: str, arguments_json: str
    ) -> dict[str, Any]:
        try:
            arguments = json.loads(arguments_json)
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be an object.")
            if self._knowledge_handles(name):
                assert self.knowledge is not None
                result = await self.knowledge.execute(name, arguments)
            else:
                result = tools.execute(name, arguments)
                result["data_limitations"] = tools.context.data_availability.unavailable_components
                result["data_freshness_notes"] = tools.context.data_availability.notes
            return result
        except (
            json.JSONDecodeError,
            ValidationError,
            GuardianToolError,
            KnowledgeToolError,
            ValueError,
        ) as exc:
            logger.warning("Guardian tool call failed tool=%s error=%s", name, exc)
            if isinstance(exc, CharacterResolutionError):
                return {
                    "error": {
                        "code": exc.code,
                        "message": str(exc),
                        "authentication_required": False,
                    }
                }
            return {"error": str(exc)}
