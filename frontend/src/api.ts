import type { AuthStatus, ChatMessage, ChatResponse, GuardianContext } from "./types";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, {
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
  authStatus: () => request<AuthStatus>("/api/auth/status"),
  guardian: () => request<GuardianContext>("/api/guardian/me"),
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
  logout: async () => {
    const response = await fetch("/api/auth/logout", {
      method: "POST",
      credentials: "include",
    });
    if (!response.ok) throw new Error("Could not disconnect the account.");
  },
};
