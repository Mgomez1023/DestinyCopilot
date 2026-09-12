import { FormEvent, useEffect, useRef, useState } from "react";
import { api } from "./api";
import type { AuthStatus, ChatMessage, GuardianContext } from "./types";

const prompts = [
  "What should I do next?",
  "I have 30 minutes",
  "I want better gear",
  "Give me something chill",
];

const guardianToolPresets = [
  { name: "get_character_summary", label: "Character summary", arguments: { character_id: null } },
  { name: "get_equipped_loadout", label: "Equipped loadout", arguments: { character_id: "" } },
  { name: "get_active_quests", label: "Active quests", arguments: { character_id: null } },
  { name: "get_available_activities", label: "Available activities", arguments: { character_id: null } },
  { name: "get_recent_activities", label: "Recent activities", arguments: { character_id: null, limit: 10 } },
  { name: "get_progression", label: "Progression", arguments: { character_id: null } },
  {
    name: "search_inventory",
    label: "Search inventory",
    arguments: {
      query: null,
      character_id: null,
      item_type: null,
      subtype: null,
      bucket: null,
      equipped_only: false,
      limit: 25,
    },
  },
  { name: "get_build_details", label: "Build details", arguments: { character_id: "" } },
];

const knowledgeToolPresets = [
  {
    name: "search_destiny_entities",
    label: "Search Destiny entities",
    arguments: { query: "Wish-Ender", entity_types: null, limit: 10 },
  },
  {
    name: "get_item_details",
    label: "Item details",
    arguments: { item_name_or_hash: "Wish-Ender" },
  },
  {
    name: "get_activity_details",
    label: "Activity details",
    arguments: { activity_name_or_hash: "The Shattered Throne" },
  },
  {
    name: "get_quest_details",
    label: "Quest details",
    arguments: { quest_name_or_hash: "Hunter's Remembrance" },
  },
  {
    name: "find_item_source",
    label: "Find item source",
    arguments: { item_name_or_hash: "Wish-Ender" },
  },
];

function GhostMark() {
  return (
    <svg aria-hidden="true" viewBox="0 0 64 64" className="ghost-mark">
      <path d="M32 4 40 20 58 23 45 36 48 55 32 46 16 55 19 36 6 23 24 20Z" />
      <circle cx="32" cy="31" r="7" />
    </svg>
  );
}

function PowerIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 20 20" className="power-icon">
      <path d="m10 2 6.5 12H3.5L10 2Z" />
    </svg>
  );
}

function formatHours(minutes: number) {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(minutes / 60);
}

function relativeDate(value: string | null) {
  if (!value) return "Unknown";
  const date = new Date(value);
  const days = Math.floor((Date.now() - date.getTime()) / 86_400_000);
  if (days <= 0) return "Today";
  if (days === 1) return "Yesterday";
  if (days < 30) return `${days} days ago`;
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

export default function App() {
  const [auth, setAuth] = useState<AuthStatus | null>(null);
  const [guardian, setGuardian] = useState<GuardianContext | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [debugOpen, setDebugOpen] = useState(false);
  const [toolName, setToolName] = useState(guardianToolPresets[0].name);
  const [toolArguments, setToolArguments] = useState(
    JSON.stringify(guardianToolPresets[0].arguments, null, 2),
  );
  const [toolResult, setToolResult] = useState<Record<string, unknown> | null>(null);
  const [toolRunning, setToolRunning] = useState(false);
  const [toolError, setToolError] = useState<string | null>(null);
  const [knowledgeToolName, setKnowledgeToolName] = useState(knowledgeToolPresets[0].name);
  const [knowledgeArguments, setKnowledgeArguments] = useState(
    JSON.stringify(knowledgeToolPresets[0].arguments, null, 2),
  );
  const [knowledgeResult, setKnowledgeResult] = useState<Record<string, unknown> | null>(null);
  const [knowledgeRunning, setKnowledgeRunning] = useState(false);
  const [knowledgeError, setKnowledgeError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const messageEnd = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const query = new URLSearchParams(window.location.search);
    const authError = query.get("auth_error");
    if (authError) setError(`Bungie sign-in failed: ${authError}`);
    if (query.size) window.history.replaceState({}, "", window.location.pathname);

    async function load() {
      try {
        const status = await api.authStatus();
        setAuth(status);
        if (status.authenticated) setGuardian(await api.guardian());
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : "Could not reach the backend.");
      } finally {
        setLoading(false);
      }
    }
    void load();
  }, []);

  useEffect(() => {
    messageEnd.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, sending]);

  async function sendMessage(text: string) {
    const clean = text.trim();
    if (!clean || sending || !guardian) return;
    const userMessage: ChatMessage = { role: "user", content: clean };
    const priorHistory = messages.slice(-10);
    setMessages((current) => [...current, userMessage]);
    setInput("");
    setSending(true);
    setError(null);
    try {
      const result = await api.chat(clean, priorHistory);
      setMessages((current) => [
        ...current,
        { role: "assistant", content: result.message },
      ]);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The Copilot could not respond.");
    } finally {
      setSending(false);
    }
  }

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    await sendMessage(input);
  }

  async function disconnect() {
    try {
      await api.logout();
      setAuth((current) =>
        current ? { ...current, authenticated: false } : current,
      );
      setGuardian(null);
      setMessages([]);
      setDebugOpen(false);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not disconnect.");
    }
  }

  function selectTool(name: string) {
    const preset = guardianToolPresets.find((value) => value.name === name);
    if (!preset) return;
    const arguments_ = { ...preset.arguments } as Record<string, unknown>;
    if (arguments_.character_id === "") {
      arguments_.character_id = guardian?.characters[0]?.character_id ?? "";
    }
    setToolName(name);
    setToolArguments(JSON.stringify(arguments_, null, 2));
    setToolResult(null);
    setToolError(null);
  }

  async function runGuardianTool() {
    setToolRunning(true);
    setToolError(null);
    try {
      const parsed = JSON.parse(toolArguments) as unknown;
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        throw new Error("Tool arguments must be a JSON object.");
      }
      const response = await api.guardianTool(toolName, parsed as Record<string, unknown>);
      setToolResult(response.result);
    } catch (caught) {
      setToolError(caught instanceof Error ? caught.message : "The Guardian tool failed.");
    } finally {
      setToolRunning(false);
    }
  }

  function selectKnowledgeTool(name: string) {
    const preset = knowledgeToolPresets.find((value) => value.name === name);
    if (!preset) return;
    setKnowledgeToolName(name);
    setKnowledgeArguments(JSON.stringify(preset.arguments, null, 2));
    setKnowledgeResult(null);
    setKnowledgeError(null);
  }

  async function runKnowledgeTool() {
    setKnowledgeRunning(true);
    setKnowledgeError(null);
    try {
      const parsed = JSON.parse(knowledgeArguments) as unknown;
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        throw new Error("Tool arguments must be a JSON object.");
      }
      const response = await api.destinyKnowledgeTool(
        knowledgeToolName,
        parsed as Record<string, unknown>,
      );
      setKnowledgeResult(response.result);
    } catch (caught) {
      setKnowledgeError(
        caught instanceof Error ? caught.message : "The Destiny knowledge tool failed.",
      );
    } finally {
      setKnowledgeRunning(false);
    }
  }

  const authenticated = Boolean(auth?.authenticated);
  const connected = authenticated && guardian;

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <GhostMark />
          <div>
            <span className="eyebrow">Destiny companion</span>
            <span className="brand-name">Guardian Copilot</span>
          </div>
        </div>
        <div className="connection-state">
          <span className={`status-dot ${authenticated ? "online" : ""}`} />
          {connected
            ? guardian.bungie_display_name
            : authenticated
              ? "Bungie connected · Guardian data unavailable"
              : "Bungie not connected"}
          {authenticated && (
            <>
              {guardian && (
                <button
                  className="text-button"
                  type="button"
                  aria-expanded={debugOpen}
                  onClick={() => setDebugOpen((current) => !current)}
                >
                  {debugOpen ? "Close context" : "Inspect context"}
                </button>
              )}
              <button className="text-button" type="button" onClick={() => void disconnect()}>
                Disconnect
              </button>
            </>
          )}
        </div>
      </header>

      <main className="workspace">
        <section className="guardian-pane" aria-label="Guardian summary">
          <div className="section-heading">
            <span className="section-index">01</span>
            <span>Your Guardian</span>
          </div>

          {loading ? (
            <div className="skeleton-card" aria-label="Loading Guardian" />
          ) : connected ? (
            <>
              <div className="profile-intro">
                <p className="kicker">Welcome back</p>
                <h1>{guardian.bungie_display_name}</h1>
                <div className="profile-meta">
                  <span>{guardian.platform_name}</span>
                  <span>Last played {relativeDate(guardian.last_played)}</span>
                  <span>{formatHours(guardian.total_minutes_played)}h in orbit</span>
                </div>
              </div>

              <div className="character-list">
                {guardian.characters.map((character, index) => (
                  <article
                    className="character-card"
                    key={character.character_id}
                    style={
                      character.emblem_background_url
                        ? { backgroundImage: `url(${character.emblem_background_url})` }
                        : undefined
                    }
                  >
                    <div className="character-shade" />
                    <div className="character-content">
                      <span className="character-number">0{index + 1}</span>
                      <div>
                        <h2>{character.class_name}</h2>
                        <p>
                          {character.race_name} · {character.gender_name}
                          {character.subclass ? ` · ${character.subclass.name}` : ""}
                        </p>
                      </div>
                      <div className="power">
                        <PowerIcon />
                        <strong>{character.power}</strong>
                      </div>
                    </div>
                  </article>
                ))}
              </div>

              <p className="scope-note">
                {guardian.inventory.total_items} inventory slots · {guardian.inventory.vault_items} in
                vault · {guardian.characters.reduce(
                  (total, character) => total + character.quests.length,
                  0,
                )} active pursuits
              </p>
            </>
          ) : (
            <div className="connect-card">
              <div className="orbit-rings"><GhostMark /></div>
              <p className="kicker">{authenticated ? "Connection issue" : "Eyes up, Guardian"}</p>
              <h1>
                {authenticated ? "Your Bungie account is connected." : "Make your next session count."}
              </h1>
              <p>
                {authenticated
                  ? "Guardian data could not be prepared. Retry after the backend finishes reloading."
                  : "Connect your Bungie account to bring your characters into focus and get advice shaped around the time and energy you have."}
              </p>
              {authenticated ? (
                <button className="connect-button" type="button" onClick={() => window.location.reload()}>
                  <span>Retry Guardian data</span>
                  <span aria-hidden="true">↻</span>
                </button>
              ) : (
                <a
                  className={`connect-button ${auth && !auth.configured ? "disabled" : ""}`}
                  href={auth?.configured ? "/api/auth/login" : undefined}
                  aria-disabled={auth ? !auth.configured : true}
                >
                  <span>Connect with Bungie</span>
                  <span aria-hidden="true">↗</span>
                </a>
              )}
              {auth && !auth.configured && (
                <p className="config-note">Add Bungie credentials to the root .env file to enable sign-in.</p>
              )}
            </div>
          )}
        </section>

        <section className="copilot-pane" aria-label="Guardian Copilot chat">
          <div className="section-heading">
            <span className="section-index">02</span>
            <span>Copilot</span>
            <span className="read-only">Read-only</span>
          </div>

          <div className="chat-window">
            {messages.length === 0 ? (
              <div className="chat-empty">
                <div className="signal"><span /><span /><span /></div>
                <p className="kicker">Standing by</p>
                <h2>What do you want from this session?</h2>
                <p>
                  Give me a mood, a goal, or a time limit. I’ll work with the Guardian data you’ve shared.
                </p>
                <div className="prompt-grid">
                  {prompts.map((prompt) => (
                    <button
                      key={prompt}
                      type="button"
                      disabled={!connected || sending}
                      onClick={() => void sendMessage(prompt)}
                    >
                      <span>{prompt}</span><span aria-hidden="true">→</span>
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div className="message-list" aria-live="polite">
                {messages.map((message, index) => (
                  <div className={`message ${message.role}`} key={`${message.role}-${index}`}>
                    <span className="message-label">{message.role === "user" ? "You" : "Copilot"}</span>
                    <p>{message.content}</p>
                  </div>
                ))}
                {sending && (
                  <div className="message assistant typing">
                    <span className="message-label">Copilot</span>
                    <p><span /><span /><span /></p>
                  </div>
                )}
                <div ref={messageEnd} />
              </div>
            )}

            {error && <div className="error-banner" role="alert">{error}</div>}

            <form className="composer" onSubmit={(event) => void onSubmit(event)}>
              <label htmlFor="message" className="sr-only">Ask Guardian Copilot</label>
              <input
                id="message"
                value={input}
                onChange={(event) => setInput(event.target.value)}
                disabled={!connected || sending}
                placeholder={connected ? "Ask what you should do next…" : "Connect your Guardian to begin"}
                autoComplete="off"
              />
              <button type="submit" disabled={!connected || !input.trim() || sending} aria-label="Send">
                <span aria-hidden="true">↑</span>
              </button>
            </form>
          </div>
        </section>
      </main>
      {connected && debugOpen && (
        <aside className="debug-panel" aria-label="Guardian account tool developer view">
          <div className="debug-heading">
            <div>
              <span className="kicker">Developer view</span>
              <h2>Developer inspector</h2>
            </div>
            <button type="button" onClick={() => setDebugOpen(false)} aria-label="Close context view">
              Close
            </button>
          </div>
          <p>
            Invoke the same normalized, read-only account and Destiny knowledge functions available
            to the Copilot. No OAuth token or raw Bungie response is included.
          </p>
          <h3>Guardian account tools</h3>
          <div className="tool-inspector">
            <label htmlFor="guardian-tool">Tool</label>
            <select
              id="guardian-tool"
              value={toolName}
              onChange={(event) => selectTool(event.target.value)}
            >
              {guardianToolPresets.map((tool) => (
                <option key={tool.name} value={tool.name}>{tool.label}</option>
              ))}
            </select>
            <label htmlFor="tool-arguments">Arguments</label>
            <textarea
              id="tool-arguments"
              value={toolArguments}
              onChange={(event) => setToolArguments(event.target.value)}
              spellCheck={false}
            />
            <button type="button" disabled={toolRunning} onClick={() => void runGuardianTool()}>
              {toolRunning ? "Running…" : "Run tool"}
            </button>
            {toolError && <p className="tool-error">{toolError}</p>}
            {toolResult && <pre>{JSON.stringify(toolResult, null, 2)}</pre>}
          </div>
          <h3>Destiny Knowledge</h3>
          <p className="inspector-note">
            The first search may build a versioned local index from Bungie's English Manifest.
          </p>
          <div className="tool-inspector">
            <label htmlFor="knowledge-tool">Tool</label>
            <select
              id="knowledge-tool"
              value={knowledgeToolName}
              onChange={(event) => selectKnowledgeTool(event.target.value)}
            >
              {knowledgeToolPresets.map((tool) => (
                <option key={tool.name} value={tool.name}>{tool.label}</option>
              ))}
            </select>
            <label htmlFor="knowledge-arguments">Arguments</label>
            <textarea
              id="knowledge-arguments"
              value={knowledgeArguments}
              onChange={(event) => setKnowledgeArguments(event.target.value)}
              spellCheck={false}
            />
            <button
              type="button"
              disabled={knowledgeRunning}
              onClick={() => void runKnowledgeTool()}
            >
              {knowledgeRunning ? "Running…" : "Run tool"}
            </button>
            {knowledgeError && <p className="tool-error">{knowledgeError}</p>}
            {knowledgeResult && <pre>{JSON.stringify(knowledgeResult, null, 2)}</pre>}
          </div>
          <details className="context-details">
            <summary>Full normalized GuardianContext</summary>
            <pre>{JSON.stringify(guardian, null, 2)}</pre>
          </details>
        </aside>
      )}
      <footer>
        <span>Guardian Copilot · Local preview</span>
        <span>Not affiliated with Bungie, Inc.</span>
      </footer>
    </div>
  );
}
