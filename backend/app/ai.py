import json
import logging
import re
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

from openai import AsyncOpenAI, OpenAIError
from pydantic import ValidationError

from app.config import Settings
from app.destiny_knowledge import DestinyKnowledgeService, KnowledgeToolError
from app.guardian_tools import GuardianToolError, GuardianToolService
from app.live_knowledge import (
    LIVE_PROVIDER_CATEGORY,
    unavailable_live_data,
    unavailable_live_message,
    volatile_topic,
)
from app.models import ChatRequest, ChatResponse, GuardianContext

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTIONS = """You are Guardian Copilot, a concise and practical Destiny 2 companion.
You have four read-only information categories. Guardian tools retrieve facts about this
authenticated player. Manifest tools retrieve canonical Destiny definitions and metadata. Guide
tools retrieve source-backed walkthroughs, acquisition instructions, and encounter mechanics.
Live tools retrieve explicitly current rotations, vendors, activities, and reset-bounded state.
Decide which categories each question needs. You do not receive the player's full account state.
Retrieve only the facts needed for the question.

Account-specific claims must come from Guardian tool results. Canonical entity metadata and rewards
must come from Manifest results. Quest steps, exotic acquisition methods, drop sources, encounter
mechanics, prerequisites, and practical walkthroughs must come from retrieved guide results, never
model memory. Never infer missing ownership, player progression, or missing game facts. Treat every
limitation, freshness marker, conflict, and warning returned by a tool as meaningful. If retrieved
knowledge is incomplete or conflicting, say exactly what is known and uncertain. General gameplay
suggestions may be inferred only when clearly labeled as suggestions rather than retrieved facts.
Keep provenance internal unless a source or uncertainty note helps the user verify the answer.
Prefer one primary recommendation and one backup. Do not claim to perform actions in the game.

Power safety: raw Guardian Power and activity Power values may use incompatible or obsolete
semantics. Never compare them, infer that a player is underleveled, filter recommendations by them,
or make eligibility/difficulty claims from them unless a retrieved source explicitly establishes
that both values and their interpretation are current and compatible. Otherwise say Power
eligibility is unknown.

Tool routing guidance:
- General "what next" questions usually need character summary, quests, activities, progression,
  and sometimes recent activities.
- Build questions need get_build_details and optionally search_inventory.
- Item ownership questions need search_inventory.
- Questions about recent play need get_recent_activities.
- General item, activity, quest, destination, perk, or source questions need Manifest knowledge.
- Questions asking how to acquire, complete, walk through, farm, locate, or execute mechanics need
  guide knowledge, usually after resolving the canonical entity with Manifest knowledge.
- Personalized acquisition or quest questions usually need Guardian, Manifest, and guide tools.
- Personalized current-state questions may need Live plus Guardian tools. Keep "currently live"
  separate from "available to this Guardian" and report either unknown dimension explicitly.
- Current farming questions may need Live for what is active and Guide/Manifest for loot or
  mechanics. Do not infer farmability when the live result leaves it unknown.
- Use current Guardian objectives to skip already-completed guide steps and lead with the next
  relevant action. Do not imply that similar names alone prove a step is complete.
- Questions about weekly rotations, featured activities, vendors, the current meta, or drop rates
  are volatile. Call a live tool. Only a LiveDestinyProvider result explicitly supporting the
  requested live topic can establish those facts. Do not extend a live result beyond its stated
  supported topics or limitations. Guardian CharacterActivities mean "available to this
  Guardian," not "featured this week." Manifest, guide knowledge, and model memory can never
  establish current rotation state. A public milestone listing does not by itself mean featured,
  farmable, or generally available.
Keep the final answer under 220 words unless the user explicitly asks for detail.

After you have gathered enough account information using tools, always produce a normal assistant
response for the user. Do not end the turn with reasoning or tool calls only."""

MAX_TOOL_ROUNDS = 4
MAX_TOOL_CALLS = 10
UNSUPPORTED_POWER_PATTERN = re.compile(
    r"\bpower\b.{0,80}\b(too low|under[- ]?leveled|below (?:the )?requirement|"
    r"not high enough|must be|need(?:s)? to be|recommended power)\b",
    re.IGNORECASE | re.DOTALL,
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
    ) -> ChatResponse:
        trace_id = uuid4().hex
        trace: dict[str, Any] = {
            "trace_id": trace_id,
            "user": request.message,
            "tools": [],
            "grounding": {
                "guardian_account": False,
                "manifest": False,
                "guide_provider": False,
                "live_provider": False,
            },
            "answer": "not_generated",
            "tool_trace": [],
        }
        live_topic = volatile_topic(request.message)
        if live_topic:
            trace["required_live_topic"] = live_topic
        if live_topic and not self._has_live_provider():
            live_result = unavailable_live_data(request.message, live_topic)
            trace["live_data"] = live_result
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
        if self.knowledge is not None:
            tool_definitions.extend(self.knowledge.definitions())
        input_items: list[Any] = [
            {"role": turn.role, "content": turn.content} for turn in request.history
        ]
        input_items.append({"role": "user", "content": request.message})
        total_tool_calls = 0

        try:
            for round_number in range(1, MAX_TOOL_ROUNDS + 1):
                self._log_request_metadata(input_items, round_number)
                response = await self.client.responses.create(
                    model=self.settings.openai_model,
                    instructions=SYSTEM_INSTRUCTIONS,
                    input=input_items,
                    tools=tool_definitions,
                    tool_choice="auto",
                    parallel_tool_calls=True,
                    reasoning={"effort": self.settings.openai_reasoning_effort},
                    max_output_tokens=self.settings.openai_max_output_tokens,
                    truncation="disabled",
                    store=False,
                )
                function_calls = [item for item in response.output if item.type == "function_call"]
                phase = "tool_calls" if function_calls else "final"
                self._log_response_metadata(response, phase)
                self._require_completed_response(response)
                if not function_calls:
                    if live_topic and not trace["grounding"]["live_provider"]:
                        live_result = unavailable_live_data(request.message, live_topic)
                        trace["live_data"] = live_result
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
                    message = self._enforce_power_safety(message)
                    trace["answer"] = "generated"
                    self._store_trace(trace_id, trace)
                    return ChatResponse(
                        source="openai",
                        message=message,
                    )

                input_items.extend(response.output)
                for call in function_calls:
                    total_tool_calls += 1
                    if total_tool_calls > MAX_TOOL_CALLS:
                        raise RuntimeError("The recommendation requested too many account tools.")
                    category = self._knowledge_category(call.name) or "guardian"
                    if category == "guardian" and refresh_guardian is not None:
                        context = await refresh_guardian(call.name)
                        guardian_tools = GuardianToolService(context)
                    result = await self._execute_tool_call(
                        guardian_tools, call.name, call.arguments
                    )
                    trace["tools"].append(call.name)
                    tool_trace = {"name": call.name, "category": category}
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
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": json.dumps(result, separators=(",", ":")),
                        }
                    )
            raise RuntimeError("The recommendation did not finish its account lookup.")
        except OpenAIError as exc:
            logger.exception("OpenAI recommendation failed")
            raise RuntimeError("The recommendation service is temporarily unavailable.") from exc
        finally:
            if trace_id not in self._traces:
                self._store_trace(trace_id, trace)

    def _store_trace(self, trace_id: str, trace: dict[str, Any]) -> None:
        if not self.settings.enable_debug_tools:
            return
        self._traces[trace_id] = trace
        self._traces.move_to_end(trace_id)
        while len(self._traces) > 100:
            self._traces.popitem(last=False)

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
    def _field(value: Any, name: str, default: Any = None) -> Any:
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)

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
            "output_text_nonempty=%s extracted_text_length=%d extracted_text_preview=%r",
            output_types,
            accessor_present,
            isinstance(sdk_output_text, str) and bool(sdk_output_text.strip()),
            len(text),
            text[:200],
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
            return {"error": str(exc)}
