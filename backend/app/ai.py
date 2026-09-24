import json
import logging
import re
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit
from uuid import uuid4

from openai import AsyncOpenAI, OpenAIError
from pydantic import ValidationError

from app.build_analysis import BuildRequestContext, derive_build_request_context
from app.character_selection import CharacterResolutionError
from app.chat_stream import (
    StreamStatusReporter,
    is_web_search_stream_event,
    status_for_tool,
)
from app.config import Settings
from app.content_progression import mentioned_content_names, supported_content_aliases
from app.destiny_knowledge import DestinyKnowledgeService, KnowledgeToolError
from app.guardian_tools import GuardianToolError, GuardianToolService
from app.live_knowledge import (
    LIVE_PROVIDER_CATEGORY,
    unavailable_live_data,
    unavailable_live_message,
    volatile_topic,
)
from app.models import ChatRequest, ChatResponse, ChatSource, GuardianContext
from app.response_quality import (
    BuildResponseValidationContext,
    ResponseModeContext,
    ResponseQualityValidator,
    is_account_fact_only,
    response_mode_context,
    response_mode_instruction,
)
from app.session_planning import (
    SessionPlanningContext,
    build_session_planning_context,
)
from app.session_preferences import (
    SessionPreferenceDerivation,
    derive_session_preferences,
)

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTIONS = """You are Guardian Copilot, a concise and practical Destiny 2 companion.
You have five read-only information categories. Guardian tools retrieve facts about this
authenticated player. Manifest tools retrieve canonical Destiny definitions and metadata. Guide
tools retrieve source-backed walkthroughs, acquisition instructions, and encounter mechanics.
Live tools retrieve explicitly current rotations, vendors, activities, and reset-bounded state.
Web research retrieves current public information and broader sourced Destiny knowledge.
Decide which categories each question needs. You do not receive the player's full account state.
Retrieve only the facts needed for the question.

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

Quest steps, exotic acquisition methods, drop sources, encounter mechanics, prerequisites, and
practical walkthroughs must come from retrieved guide or web research results, never model memory.
Never infer missing ownership, player progression, or missing game facts. Treat every limitation,
freshness marker, conflict, and warning returned by a tool as meaningful. If retrieved knowledge is
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
- If missing preference materially changes the choice, give the best default and a short backup,
  then ask at most ONE targeted follow-up such as "Story or loot?" or "How much time do you have?"
  Do not ask generic clarification or closing questions.
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
- Questions about recent play need get_recent_activities.
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
MAX_CHAT_SOURCES = 8
MAX_TRACE_ITEM_NAMES = 8
TRACEABLE_GUARDIAN_TOOLS = {
    "analyze_current_build",
    "find_build_alternatives",
    "get_equipped_loadout",
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
SESSION_PLANNING_PATTERNS = (
    re.compile(r"\bwhat should i (?:do|play|start)(?: next| on my \w+)?\b", re.IGNORECASE),
    re.compile(
        r"\bi (?:have|have got|got)\b.{0,24}\b(?:minutes?|mins?|hours?|hrs?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bi (?:just )?bought\b.{0,80}\b(?:dlcs?|expansions?|campaigns?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bshould i\b.{0,160}\bor\b", re.IGNORECASE),
    re.compile(
        r"\bi want\b.{0,80}\b(?:better gear|loot|rewards?|something chill|challenge|challenging)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bgive me something\b", re.IGNORECASE),
    re.compile(r"\b(?:what(?:'s| is) )?worth doing\b", re.IGNORECASE),
    re.compile(r"\brecommend\b.{0,80}\b(?:activity|quest|campaign|play|do)\b", re.IGNORECASE),
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
        preference_derivation = derive_session_preferences(request.history, request.message)
        preference_context = preference_derivation.preferences
        build_request_context = derive_build_request_context(
            request.message, preference_context, request.history
        )
        build_validation_context = BuildResponseValidationContext(
            is_build_request=build_request_context.is_build_request,
            requires_current_build_analysis=(build_request_context.requires_current_build_analysis),
            requires_owned_inventory=build_request_context.requires_owned_inventory,
            preserve_equipped_exotics=build_request_context.preserve_equipped_exotics,
            locked_item_names=build_request_context.locked_items,
            activity_mode=build_request_context.activity_mode,
            fireteam=build_request_context.fireteam,
            normal_change_limit=build_request_context.normal_change_limit,
        )
        session_planning = bool(
            self._is_session_planning(request.message) or preference_derivation.is_planning_followup
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
        quality_validator = ResponseQualityValidator()
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
                "analysis_used": False,
                "alternative_search_used": False,
                "owned_candidates_returned": 0,
                "locked_item_count": len(build_request_context.locked_items),
            },
            "number_of_guardian_tools_used": 0,
            "knowledge_categories_used": [],
            "web_research_occurred": False,
            "tools": [],
            "grounding": {
                "guardian_account": False,
                "manifest": False,
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
            },
            "planning_correction": {
                "attempted": False,
                "violation_codes": [],
                "succeeded": None,
            },
            "answer": "not_generated",
            "tool_trace": [],
        }
        live_topic = volatile_topic(request.message)
        if live_topic:
            trace["required_live_topic"] = live_topic
        if live_topic and not self._has_live_provider():
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

        guardian_tools = GuardianToolService(context)
        tool_definitions = list(guardian_tools.definitions())
        account_fact_only = is_account_fact_only(request.message)
        if self.knowledge is not None and not account_fact_only:
            tool_definitions.extend(self.knowledge.definitions())
        if not account_fact_only:
            tool_definitions.append(WEB_SEARCH_TOOL)
        input_items: list[Any] = [
            {"role": turn.role, "content": turn.content} for turn in request.history
        ]
        input_items.append({"role": "user", "content": request.message})
        total_tool_calls = 0
        web_sources: list[ChatSource] = []
        knowledge_sources: list[ChatSource] = []
        turn_instructions = self._turn_instructions(
            session_planning,
            intent_category,
            character_scope,
            planning_context,
            mode_context,
            build_request_context,
        )

        try:
            for round_number in range(1, MAX_TOOL_ROUNDS + 1):
                self._log_request_metadata(input_items, round_number)
                response = await self._create_response(
                    stream_status=stream_status,
                    model=self.settings.openai_model,
                    instructions=turn_instructions,
                    input=input_items,
                    tools=tool_definitions,
                    include=RESPONSE_INCLUDE,
                    tool_choice="auto",
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
                    trace["web_research_occurred"] = True
                    self._record_knowledge_category(trace, "web")
                trace["web_research"]["source_count"] = len(web_sources)
                function_calls = [
                    item for item in response_output if self._field(item, "type") == "function_call"
                ]
                phase = "tool_calls" if function_calls else "final"
                self._log_response_metadata(response, phase)
                self._require_completed_response(response)
                if not function_calls:
                    if live_topic and not trace["grounding"]["live_provider"]:
                        live_result = unavailable_live_data(request.message, live_topic)
                        trace["live_data"] = self._safe_live_trace(live_result)
                        trace["answer"] = "unavailable_live_data"
                        self._store_trace(trace_id, trace)
                        return ChatResponse(
                            source="local",
                            message=unavailable_live_message(live_topic, provider_connected=True),
                        )
                    message = self._extract_assistant_text(response)
                    if not message:
                        logger.error(
                            "OpenAI final response contained no assistant text; "
                            "refusing to return an empty chat message"
                        )
                        raise RuntimeError(
                            "OpenAI returned a final response without assistant text."
                        )
                    if stream_status is not None:
                        if mode_context.mode in {"recommendation", "comparison"}:
                            await stream_status.emit("comparison")
                        elif mode_context.mode == "build_advice":
                            await stream_status.emit("build")
                        await stream_status.emit("final")
                        trace["emitted_status_categories"] = list(stream_status.emitted_categories)
                    build_validation_context.current_external_grounding = bool(
                        web_sources or trace["grounding"]["live_provider"]
                    )
                    safe_fallback = False
                    violations = quality_validator.validate(
                        message,
                        mode_context,
                        request.message,
                        planning_context,
                        build_validation_context,
                    )
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
                            quality_validator.correction_instruction(violations),
                            stream_status,
                        )
                        correction_output = self._field(correction_response, "output", []) or []
                        self._log_response_metadata(correction_response, "planning_correction")
                        self._require_completed_response(correction_response)
                        corrected_message = self._extract_assistant_text(correction_response)
                        corrected_violations = (
                            quality_validator.validate(
                                corrected_message,
                                mode_context,
                                request.message,
                                planning_context,
                                build_validation_context,
                            )
                            if corrected_message
                            else violations
                        )
                        if corrected_message and not corrected_violations:
                            message = corrected_message
                            trace["planning_correction"]["succeeded"] = True
                            correction_sources, _ = self._extract_web_sources(correction_output)
                            self._merge_sources(web_sources, correction_sources)
                            trace["web_research"]["source_count"] = len(web_sources)
                        elif build_validation_context.is_build_request:
                            message = quality_validator.safe_build_answer(build_validation_context)
                            safe_fallback = True
                            trace["planning_correction"]["fallback"] = "build_evidence"
                        elif planning_context is not None:
                            message = quality_validator.planning.safe_answer(planning_context)
                            safe_fallback = True
                    safe_message = self._enforce_power_safety(message)
                    safe_message = self._enforce_campaign_completion_safety(safe_message)
                    safe_message = self._enforce_campaign_comparison_safety(
                        safe_message,
                        request.message,
                        character_scope,
                    )
                    sources = self._combined_sources(
                        web_sources,
                        knowledge_sources,
                        limit=mode_context.source_limit,
                    )
                    trace["web_research"]["source_count"] = len(
                        [source for source in sources if source in web_sources]
                    )
                    if safe_message != message or safe_fallback:
                        sources = []
                    message = self._hide_internal_identifiers(
                        safe_message,
                        context,
                        allow_identifiers=bool(DEBUG_IDENTIFIER_PATTERN.search(request.message)),
                    )
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
                    total_tool_calls += 1
                    if total_tool_calls > MAX_TOOL_CALLS:
                        raise RuntimeError("The recommendation requested too many account tools.")
                    category = self._knowledge_category(call.name) or "guardian"
                    if stream_status is not None:
                        stage, label = status_for_tool(call.name, category)
                        await stream_status.emit(stage, label)
                        trace["emitted_status_categories"] = list(stream_status.emitted_categories)
                    if category == "guardian" and refresh_guardian is not None:
                        context = await refresh_guardian(call.name)
                        guardian_tools = GuardianToolService(context)
                        if planning_context is not None:
                            planning_context = build_session_planning_context(
                                context,
                                request.message,
                                intent_category or "general",
                                character_scope,
                                preference_context,
                            )
                            turn_instructions = self._turn_instructions(
                                session_planning,
                                intent_category,
                                character_scope,
                                planning_context,
                                mode_context,
                                build_request_context,
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
                    self._update_build_validation_context(
                        build_validation_context,
                        trace,
                        call.name,
                        result,
                    )
                    trace["tools"].append(call.name)
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
        if tool_name in {"analyze_current_build", "get_build_details", "get_equipped_loadout"}:
            context.guardian_build_data_used = True
        if tool_name == "analyze_current_build":
            context.current_build_analysis_used = True
        if tool_name in {"find_build_alternatives", "search_inventory"}:
            context.inventory_ownership_checked = True

        item_groups: list[dict[str, Any]] = []
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
    def _is_session_planning(message: str) -> bool:
        return any(pattern.search(message) for pattern in SESSION_PLANNING_PATTERNS)

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
    def _turn_instructions(
        session_planning: bool,
        intent_category: str | None,
        character_scope: str | None,
        planning_context: SessionPlanningContext | None,
        mode_context: ResponseModeContext,
        build_request_context: BuildRequestContext,
    ) -> str:
        routing: list[str] = [response_mode_instruction(mode_context)]
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
                "hard constraints. If requires_current_build_analysis is true, call "
                "analyze_current_build before judging or recommending changes to the current "
                "setup. If requires_owned_inventory is true, use find_build_alternatives or an "
                "appropriate Guardian inventory lookup before naming owned choices or exact "
                "instance perks. Do not mention this routing metadata:\n"
                f"{build_request_context.model_dump_json()}"
            )
        return f"{SYSTEM_INSTRUCTIONS}\n\n" + "\n".join(routing) if routing else SYSTEM_INSTRUCTIONS

    async def _create_response(
        self,
        *,
        stream_status: StreamStatusReporter | None,
        **request_options: Any,
    ) -> Any:
        if stream_status is None:
            return await self.client.responses.create(**request_options)
        async with self.client.responses.stream(**request_options) as stream:
            async for event in stream:
                if is_web_search_stream_event(self._field(event, "type")):
                    await stream_status.emit("web")
            return await stream.get_final_response()

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
    def _record_knowledge_category(trace: dict[str, Any], category: str) -> None:
        categories = trace["knowledge_categories_used"]
        if category not in categories:
            categories.append(category)

    @staticmethod
    def _enforce_power_safety(message: str) -> str:
        if not UNSUPPORTED_POWER_PATTERN.search(message):
            return message
        logger.warning("Blocked an unsupported Power-level comparison in an AI response")
        return (
            "I can’t reliably compare the raw Power values returned by the current APIs. "
            "Their meanings may not be compatible with current activity requirements, so your "
            "Power eligibility is unknown. I can still help with the activity’s mechanics, "
            "prerequisites, and a practical loadout based on grounded information."
        )

    @staticmethod
    def _enforce_campaign_completion_safety(message: str) -> str:
        for sentence in re.split(r"(?<=[.!?])\s+|[\r\n]+", message):
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
                logger.warning("Blocked an unsupported campaign-completion inference")
                return (
                    "I can't verify campaign completion from an equipped subclass, item, "
                    "destination, reward, or related unlock. Based on the available account "
                    "data, completion is unknown. I'd verify the campaign quest or progression "
                    "state before choosing between them; once that is known, I can recommend "
                    "the better next step."
                )
        return message

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
