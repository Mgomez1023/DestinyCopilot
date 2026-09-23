import type {
  AuthStatus,
  ChatMessage,
  ChatResponse,
  GuardianContext,
  GuardianRefreshStatus,
  GuardianStateResponse,
} from "./types";
import type { ChatStreamEvent } from "./types";
import { ChatSseParser } from "./chatStream";

const configuredApiBaseUrl = (import.meta.env.VITE_API_BASE_URL ?? "").trim();
const apiBaseUrl = configuredApiBaseUrl.replace(/\/+$/, "");

if (apiBaseUrl && !/^https?:\/\//i.test(apiBaseUrl)) {
  throw new Error("VITE_API_BASE_URL must be an absolute HTTP(S) URL.");
}
if (import.meta.env.PROD && apiBaseUrl.startsWith("http://")) {
  throw new Error("VITE_API_BASE_URL must use HTTPS in production.");
}

function apiUrl(path: string): string {
  return `${apiBaseUrl}${path}`;
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(apiUrl(path), {
    credentials: "include",
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) message = body.detail;
    } catch {
      // Keep the status-based message for non-JSON errors.
    }
    throw new Error(message);
  }
  return (await response.json()) as T;
}

export interface ChatStreamHandlers {
  onEvent: (event: ChatStreamEvent) => void;
}

async function chatStream(
  message: string,
  history: ChatMessage[],
  handlers: ChatStreamHandlers,
  signal: AbortSignal,
): Promise<void> {
  const response = await fetch(apiUrl("/api/chat/stream"), {
    method: "POST",
    credentials: "include",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({
      message,
      history: history.map(({ role, content }) => ({ role, content })),
    }),
    signal,
  });
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) message = body.detail;
    } catch {
      // Keep the status-based message for non-JSON errors.
    }
    throw new Error(message);
  }
  if (!response.headers.get("content-type")?.toLowerCase().includes("text/event-stream")) {
    throw new Error("The backend did not return a chat event stream.");
  }
  if (!response.body) throw new Error("The chat stream is unavailable in this browser.");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const parser = new ChatSseParser();
  let completed = false;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const events = parser.push(decoder.decode(value, { stream: true }));
      for (const event of events) {
        if (event.type === "error") throw new Error(event.message);
        handlers.onEvent(event);
        if (event.type === "completed") completed = true;
      }
    }
    const tail = decoder.decode();
    if (tail) {
      for (const event of parser.push(tail)) {
        if (event.type === "error") throw new Error(event.message);
        handlers.onEvent(event);
        if (event.type === "completed") completed = true;
      }
    }
    parser.finish();
    if (!completed) throw new Error("The chat stream ended before the answer was completed.");
  } finally {
    if (!completed) await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}

export const api = {
  authLoginUrl: apiUrl("/api/auth/login"),
  authStatus: () => request<AuthStatus>("/api/auth/status"),
  guardian: () => request<GuardianContext>("/api/guardian/me"),
  guardianResume: () =>
    request<GuardianStateResponse>("/api/guardian/resume", { method: "POST" }),
  guardianRefresh: (level: "normal" | "full" = "normal") =>
    request<GuardianStateResponse>(`/api/guardian/refresh?level=${level}`, { method: "POST" }),
  guardianRefreshStatus: () =>
    request<GuardianRefreshStatus>("/api/guardian/refresh/status"),
  debugRefreshSlice: (slice: string) =>
    request<GuardianStateResponse>(
      `/api/debug/guardian-refresh/slice/${encodeURIComponent(slice)}`,
      { method: "POST" },
    ),
  debugClearGuardianCache: () =>
    request<{ cleared: boolean }>("/api/debug/guardian-refresh/clear", { method: "POST" }),
  guardianTool: (toolName: string, arguments_: Record<string, unknown>) =>
    request<{ tool_name: string; result: Record<string, unknown> }>(
      `/api/debug/guardian-tools/${encodeURIComponent(toolName)}`,
      {
        method: "POST",
        body: JSON.stringify(arguments_),
      },
    ),
  destinyKnowledgeTool: (toolName: string, arguments_: Record<string, unknown>) =>
    request<{ tool_name: string; result: Record<string, unknown> }>(
      `/api/debug/destiny-knowledge/${encodeURIComponent(toolName)}`,
      {
        method: "POST",
        body: JSON.stringify(arguments_),
      },
    ),
  chat: (message: string, history: ChatMessage[]) =>
    request<ChatResponse>("/api/chat", {
      method: "POST",
      body: JSON.stringify({
        message,
        history: history.map(({ role, content }) => ({ role, content })),
      }),
    }),
  chatStream,
  latestChatTrace: () =>
    request<Record<string, unknown>>("/api/debug/chat-traces/latest"),
  logout: async () => {
    const response = await fetch(apiUrl("/api/auth/logout"), {
      method: "POST",
      credentials: "include",
    });
    if (!response.ok) throw new Error("Could not disconnect the account.");
  },
};
