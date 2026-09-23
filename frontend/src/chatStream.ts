import type {
  ChatResponseMode,
  ChatSource,
  ChatStreamEvent,
  ChatStreamStatusStage,
} from "./types";

const statusStages = new Set<ChatStreamStatusStage>([
  "guardian",
  "loadout",
  "inventory",
  "manifest",
  "guide",
  "live",
  "web",
  "comparison",
  "build",
  "final",
]);

const responseModes = new Set<ChatResponseMode>([
  "direct_fact",
  "recommendation",
  "comparison",
  "walkthrough",
  "build_advice",
  "account_summary",
  "troubleshooting",
]);

function objectValue(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("The chat stream returned malformed data.");
  }
  return value as Record<string, unknown>;
}

function stringValue(value: unknown): string {
  if (typeof value !== "string" || !value) {
    throw new Error("The chat stream returned malformed data.");
  }
  return value;
}

function sourceValue(value: unknown): ChatSource {
  const source = objectValue(value);
  const url = stringValue(source.url);
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    throw new Error("The chat stream returned an invalid source URL.");
  }
  if (
    !["http:", "https:"].includes(parsed.protocol) ||
    parsed.username ||
    parsed.password
  ) {
    throw new Error("The chat stream returned an invalid source URL.");
  }
  return {
    title: stringValue(source.title),
    url,
    domain: source.domain == null ? null : stringValue(source.domain),
  };
}

function typedEvent(eventName: string, rawData: string): ChatStreamEvent | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(rawData);
  } catch {
    throw new Error("The chat stream returned malformed data.");
  }
  const data = objectValue(parsed);
  if (eventName === "status") {
    const stage = stringValue(data.stage) as ChatStreamStatusStage;
    if (!statusStages.has(stage)) throw new Error("The chat stream returned an unknown status.");
    return { type: "status", stage, label: stringValue(data.label) };
  }
  if (eventName === "message_delta") {
    return { type: "message_delta", delta: stringValue(data.delta) };
  }
  if (eventName === "sources") {
    if (!Array.isArray(data.sources) || data.sources.length > 8) {
      throw new Error("The chat stream returned malformed sources.");
    }
    return { type: "sources", sources: data.sources.map(sourceValue) };
  }
  if (eventName === "completed") {
    const responseMode = stringValue(data.response_mode) as ChatResponseMode;
    const source = stringValue(data.source);
    if (!responseModes.has(responseMode) || !["openai", "local"].includes(source)) {
      throw new Error("The chat stream returned malformed completion data.");
    }
    return {
      type: "completed",
      response_mode: responseMode,
      source: source as "openai" | "local",
    };
  }
  if (eventName === "error") {
    return { type: "error", message: stringValue(data.message) };
  }
  return null;
}

export class ChatSseParser {
  private buffer = "";

  push(chunk: string): ChatStreamEvent[] {
    this.buffer += chunk;
    const events: ChatStreamEvent[] = [];
    while (true) {
      const boundary = this.buffer.match(/\r?\n\r?\n/);
      if (!boundary || boundary.index === undefined) break;
      const frame = this.buffer.slice(0, boundary.index);
      this.buffer = this.buffer.slice(boundary.index + boundary[0].length);
      const event = this.parseFrame(frame);
      if (event) events.push(event);
    }
    return events;
  }

  finish(): void {
    if (this.buffer.trim()) throw new Error("The chat stream ended unexpectedly.");
  }

  private parseFrame(frame: string): ChatStreamEvent | null {
    let eventName = "message";
    const data: string[] = [];
    for (const line of frame.split(/\r?\n/)) {
      if (!line || line.startsWith(":")) continue;
      const separator = line.indexOf(":");
      const field = separator >= 0 ? line.slice(0, separator) : line;
      const value = separator >= 0 ? line.slice(separator + 1).replace(/^ /, "") : "";
      if (field === "event") eventName = value;
      if (field === "data") data.push(value);
    }
    if (!data.length) return null;
    return typedEvent(eventName, data.join("\n"));
  }
}
