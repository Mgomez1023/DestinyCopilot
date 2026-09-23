import assert from "node:assert/strict";
import test from "node:test";

import { ChatSseParser } from "../src/chatStream.ts";

function frame(event: string, data: Record<string, unknown>): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

test("parses typed events split across arbitrary chunks", () => {
  const parser = new ChatSseParser();
  const stream =
    frame("status", { stage: "guardian", label: "Checking your Guardian…" }) +
    frame("message_delta", { delta: "Ready." }) +
    frame("completed", { response_mode: "direct_fact", source: "openai" });
  const events = [
    ...parser.push(stream.slice(0, 17)),
    ...parser.push(stream.slice(17, 63)),
    ...parser.push(stream.slice(63)),
  ];
  parser.finish();

  assert.deepEqual(events, [
    { type: "status", stage: "guardian", label: "Checking your Guardian…" },
    { type: "message_delta", delta: "Ready." },
    { type: "completed", response_mode: "direct_fact", source: "openai" },
  ]);
});

test("rejects malformed events and incomplete trailing frames", () => {
  const malformed = new ChatSseParser();
  assert.throws(
    () => malformed.push("event: status\ndata: not-json\n\n"),
    /malformed data/,
  );

  const incomplete = new ChatSseParser();
  incomplete.push('event: message_delta\ndata: {"delta":"partial"}');
  assert.throws(() => incomplete.finish(), /ended unexpectedly/);
});

test("accepts only bounded safe external source metadata", () => {
  const parser = new ChatSseParser();
  const valid = parser.push(
    frame("sources", {
      sources: [
        {
          title: "Bungie update",
          url: "https://www.bungie.net/7/en/News/article/example",
          domain: "www.bungie.net",
        },
      ],
    }),
  );
  assert.equal(valid[0].type, "sources");

  assert.throws(
    () =>
      new ChatSseParser().push(
        frame("sources", {
          sources: [{ title: "Unsafe", url: "javascript:alert(1)" }],
        }),
      ),
    /invalid source URL/,
  );
});

test("accepts safe build execution status categories", () => {
  const parser = new ChatSseParser();
  const events = parser.push(
    frame("status", { stage: "loadout", label: "Checking your loadout…" }) +
      frame("status", { stage: "inventory", label: "Searching your inventory…" }) +
      frame("status", { stage: "build", label: "Preparing your build…" }),
  );

  assert.deepEqual(
    events.map((event) => event.type === "status" && event.stage),
    ["loadout", "inventory", "build"],
  );
});
