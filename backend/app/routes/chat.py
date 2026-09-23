import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.chat_stream import (
    ChatStreamEvent,
    CompletedStreamEvent,
    ErrorStreamEvent,
    MessageDeltaStreamEvent,
    StreamStatusReporter,
    initial_guardian_status,
    response_events,
    safe_stream_error_message,
    serialize_sse,
)
from app.dependencies import ServicesDep
from app.models import ChatRequest, ChatResponse, GuardianContext
from app.routes.guardian import _refresh_service, guardian_credentials, load_guardian

router = APIRouter(prefix="/api/chat", tags=["chat"])
logger = logging.getLogger(__name__)


@router.post("", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    request: Request,
    services: ServicesDep,
) -> ChatResponse:
    guardian = await load_guardian(request, services, stale_while_revalidate=False)
    try:
        refresh_callback = None
        if getattr(services, "guardian_refresh", None) is not None:
            session_id, access_token = await guardian_credentials(request, services)
            refresh = _refresh_service(services)
            guardian = (await refresh.for_prompt(session_id, access_token, body.message)).context

            async def refresh_callback(tool_name: str) -> GuardianContext:
                return (await refresh.for_tool(session_id, access_token, tool_name)).context

        return await services.recommendations.chat(
            body, guardian, refresh_guardian=refresh_callback
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/stream")
async def chat_stream(
    body: ChatRequest,
    request: Request,
    services: ServicesDep,
) -> StreamingResponse:
    # Preserve a normal HTTP authentication failure before the event stream begins.
    await guardian_credentials(request, services)

    async def event_stream() -> AsyncIterator[str]:
        queue: asyncio.Queue[ChatStreamEvent | None] = asyncio.Queue()
        reporter = StreamStatusReporter(queue.put)

        async def run_chat() -> None:
            try:
                stage, label = initial_guardian_status(body.message)
                await reporter.emit(stage, label)
                guardian = await load_guardian(request, services, stale_while_revalidate=False)
                refresh_callback = None
                if getattr(services, "guardian_refresh", None) is not None:
                    session_id, access_token = await guardian_credentials(request, services)
                    refresh = _refresh_service(services)
                    guardian = (
                        await refresh.for_prompt(session_id, access_token, body.message)
                    ).context

                    async def refresh_callback(tool_name: str) -> GuardianContext:
                        return (await refresh.for_tool(session_id, access_token, tool_name)).context

                response = await services.recommendations.chat(
                    body,
                    guardian,
                    refresh_guardian=refresh_callback,
                    stream_status=reporter,
                )
                response_mode = reporter.response_mode or "direct_fact"
                if response_mode in {"recommendation", "comparison"}:
                    await reporter.emit("comparison")
                await reporter.emit("final")
                for event in response_events(response, response_mode):
                    await queue.put(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Streaming chat failed error_type=%s", type(exc).__name__)
                await queue.put(ErrorStreamEvent(message=safe_stream_error_message(exc)))
            finally:
                await queue.put(None)

        task = asyncio.create_task(run_chat())
        final_response_streamed = False
        stream_completed = False
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield serialize_sse(event)
                if isinstance(event, MessageDeltaStreamEvent):
                    final_response_streamed = True
                if isinstance(event, CompletedStreamEvent):
                    stream_completed = True
                    break
        finally:
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            services.recommendations.mark_stream_delivery(
                reporter.trace_id,
                final_response_streamed=final_response_streamed,
                stream_completed=stream_completed,
                emitted_status_categories=list(reporter.emitted_categories),
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )
