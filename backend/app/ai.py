import json
import logging
from typing import Any

from openai import AsyncOpenAI, OpenAIError
from pydantic import ValidationError

from app.config import Settings
from app.destiny_knowledge import (
    DESTINY_KNOWLEDGE_TOOL_DEFINITIONS,
    DestinyKnowledgeService,
    KnowledgeToolError,
)
from app.guardian_tools import GuardianToolError, GuardianToolService
from app.models import ChatRequest, ChatResponse, GuardianContext

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTIONS = """You are Guardian Copilot, a concise and practical Destiny 2 companion.
You have two read-only tool families. Guardian tools retrieve facts about this authenticated
player. Destiny knowledge tools retrieve normalized game facts from Bungie's Manifest. Decide
whether each question needs Guardian tools, Destiny knowledge tools, or both. You do not receive
the player's full account state. Retrieve only the facts needed for the question.

Account-specific claims must come from Guardian tool results. Game-specific claims, especially
acquisition steps, quest facts, activity details, and rewards, should be grounded in Destiny
knowledge results. Never infer missing ownership or missing game facts. Treat every limitation
returned by a tool as meaningful. If Manifest knowledge is insufficient, say so rather than
confidently filling gaps from memory. General conversational guidance is allowed only when clearly
separated from retrieved facts. Prefer one primary recommendation and one backup. Do not claim to
perform actions in the game.

Tool routing guidance:
- General "what next" questions usually need character summary, quests, activities, progression,
  and sometimes recent activities.
- Build questions need get_build_details and optionally search_inventory.
- Item ownership questions need search_inventory.
- Questions about recent play need get_recent_activities.
- General item, activity, quest, destination, perk, or source questions need Destiny knowledge.
- Personalized acquisition or quest questions usually need both tool families.
Keep the final answer under 220 words unless the user explicitly asks for detail.

After you have gathered enough account information using tools, always produce a normal assistant
response for the user. Do not end the turn with reasoning or tool calls only."""

MAX_TOOL_ROUNDS = 4
MAX_TOOL_CALLS = 10


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

    async def close(self) -> None:
        if self.client is not None:
            await self.client.close()

    async def chat(self, request: ChatRequest, context: GuardianContext) -> ChatResponse:
        if self.client is None:
            return ChatResponse(
                source="local",
                message=(
                    "Your Guardian is connected, but AI recommendations are not enabled yet. "
                    "Add OPENAI_API_KEY to the root .env file and restart the backend. "
                    f"The local tool layer can query {len(context.characters)} character(s) and "
                    f"{context.inventory.total_items} normalized inventory slots."
                ),
            )

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
                function_calls = [
                    item for item in response.output if item.type == "function_call"
                ]
                phase = "tool_calls" if function_calls else "final"
                self._log_response_metadata(response, phase)
                self._require_completed_response(response)
                if not function_calls:
                    message = self._extract_assistant_text(response)
                    if not message:
                        logger.error(
                            "OpenAI final response contained no assistant text; "
                            "refusing to return an empty chat message"
                        )
                        raise RuntimeError(
                            "OpenAI returned a final response without assistant text."
                        )
                    return ChatResponse(source="openai", message=message)

                input_items.extend(response.output)
                for call in function_calls:
                    total_tool_calls += 1
                    if total_tool_calls > MAX_TOOL_CALLS:
                        raise RuntimeError("The recommendation requested too many account tools.")
                    result = await self._execute_tool_call(
                        guardian_tools, call.name, call.arguments
                    )
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
            self._field(
                response, "max_output_tokens", self.settings.openai_max_output_tokens
            ),
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
            if name in {value["name"] for value in DESTINY_KNOWLEDGE_TOOL_DEFINITIONS}:
                if self.knowledge is None:
                    raise KnowledgeToolError("Destiny knowledge is not configured.")
                result = await self.knowledge.execute(name, arguments)
            else:
                result = tools.execute(name, arguments)
                result["data_limitations"] = (
                    tools.context.data_availability.unavailable_components
                )
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
