import type {
  AuthStatus,
  ChatMessage,
  ChatResponse,
  GuardianContext,
  GuardianRefreshStatus,
  GuardianStateResponse,
} from "./types";

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
      body: JSON.stringify({ message, history }),
    }),
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
