import { FormEvent, useEffect, useRef, useState } from "react";
import { api } from "./api";
import type {
  AuthStatus,
  ChatMessage,
  GuardianContext,
  GuardianRefreshStatus,
  GuardianStateResponse,
} from "./types";

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

const guideToolPresets = [
  {
    name: "search_destiny_guides",
    label: "Search Destiny guides",
    arguments: {
      query: "How do I get Wish-Ender?",
      entity_name: "Wish-Ender",
      guide_type: "exotic_acquisition",
      limit: 5,
    },
  },
  {
    name: "get_destiny_guide",
    label: "Get detailed guide",
    arguments: {
      entity_or_query: "Hunter's Remembrance",
      guide_type: "quest_walkthrough",
    },
  },
];

const liveToolPresets = [
  {
    name: "get_live_destiny_status",
    label: "Live source status",
    arguments: {},
  },
  {
    name: "get_weekly_rotation",
    label: "Weekly rotation",
    arguments: { category: "dungeon" },
  },
  {
    name: "get_vendor_status",
    label: "Vendor status",
    arguments: { vendor: "Xur" },
  },
  {
    name: "get_current_activity_status",
    label: "Current activity status",
    arguments: { activity_name_or_hash: "Warlord's Ruin" },
  },
  {
    name: "search_live_destiny",
    label: "Search live Destiny",
    arguments: { query: "What's the featured dungeon this week?" },
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

function MenuIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" className="menu-icon">
      <path d="M4 7h16M4 12h16M4 17h16" />
    </svg>
  );
}

function CloseIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" className="menu-icon">
      <path d="m6 6 12 12M18 6 6 18" />
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
  const [refreshStatus, setRefreshStatus] = useState<GuardianRefreshStatus | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [mobileLayout, setMobileLayout] = useState(() =>
    window.matchMedia("(max-width: 900px)").matches,
  );
  const [accountMenuOpen, setAccountMenuOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [chatStatus, setChatStatus] = useState<string | null>(null);
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
  const [guideToolName, setGuideToolName] = useState(guideToolPresets[0].name);
  const [guideArguments, setGuideArguments] = useState(
    JSON.stringify(guideToolPresets[0].arguments, null, 2),
  );
  const [guideResult, setGuideResult] = useState<Record<string, unknown> | null>(null);
  const [guideRunning, setGuideRunning] = useState(false);
  const [guideError, setGuideError] = useState<string | null>(null);
  const [liveToolName, setLiveToolName] = useState(liveToolPresets[0].name);
  const [liveArguments, setLiveArguments] = useState(
    JSON.stringify(liveToolPresets[0].arguments, null, 2),
  );
  const [liveResult, setLiveResult] = useState<Record<string, unknown> | null>(null);
  const [liveRunning, setLiveRunning] = useState(false);
  const [liveError, setLiveError] = useState<string | null>(null);
  const [chatTrace, setChatTrace] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const messageEnd = useRef<HTMLDivElement>(null);
  const accountDrawer = useRef<HTMLElement>(null);
  const accountMenuButton = useRef<HTMLButtonElement>(null);
  const activeChat = useRef<AbortController | null>(null);
  const chatRequestId = useRef(0);

  useEffect(() => {
    return () => {
      chatRequestId.current += 1;
      activeChat.current?.abort();
    };
  }, []);

  useEffect(() => {
    const query = new URLSearchParams(window.location.search);
    const authError = query.get("auth_error");
    if (authError) setError(`Bungie sign-in failed: ${authError}`);
    if (query.size) window.history.replaceState({}, "", window.location.pathname);

    async function load() {
      try {
        const status = await api.authStatus();
        setAuth(status);
        if (status.authenticated) {
          const state = await api.guardianResume();
          setGuardian(state.guardian);
          setRefreshStatus(state.refresh);
        }
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : "Could not reach the backend.");
      } finally {
        setLoading(false);
      }
    }
    void load();
  }, []);

  useEffect(() => {
    async function resume() {
      if (document.visibilityState !== "visible" || !auth?.authenticated) return;
      try {
        const state = await api.guardianResume();
        setGuardian(state.guardian);
        setRefreshStatus(state.refresh);
      } catch {
        // Keep the last usable cached view; explicit actions surface failures.
      }
    }
    document.addEventListener("visibilitychange", resume);
    return () => document.removeEventListener("visibilitychange", resume);
  }, [auth?.authenticated]);

  useEffect(() => {
    const media = window.matchMedia("(max-width: 900px)");
    const updateLayout = () => setMobileLayout(media.matches);
    updateLayout();
    media.addEventListener("change", updateLayout);
    return () => media.removeEventListener("change", updateLayout);
  }, []);

  useEffect(() => {
    if (!mobileLayout) setAccountMenuOpen(false);
  }, [mobileLayout]);

  useEffect(() => {
    if (!mobileLayout || !accountMenuOpen) return;
    const drawerElement = accountDrawer.current;
    if (!drawerElement) return;
    const activeDrawer: HTMLElement = drawerElement;
    const menuButton = accountMenuButton.current;
    const previouslyFocused = document.activeElement as HTMLElement | null;
    const focusableSelector = [
      "button:not([disabled])",
      "a[href]",
      "input:not([disabled])",
      "select:not([disabled])",
      "textarea:not([disabled])",
      "[tabindex]:not([tabindex='-1'])",
    ].join(",");
    const focusable = () =>
      Array.from(activeDrawer.querySelectorAll<HTMLElement>(focusableSelector));

    document.body.classList.add("account-drawer-open");
    window.requestAnimationFrame(() => focusable()[0]?.focus());

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        event.preventDefault();
        setAccountMenuOpen(false);
        return;
      }
      if (event.key !== "Tab") return;
      const elements = focusable();
      if (!elements.length) {
        event.preventDefault();
        activeDrawer.focus();
        return;
      }
      const first = elements[0];
      const last = elements[elements.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }

    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      document.body.classList.remove("account-drawer-open");
      const remainsMobile = window.matchMedia("(max-width: 900px)").matches;
      (remainsMobile ? menuButton : previouslyFocused)?.focus();
    };
  }, [accountMenuOpen, mobileLayout]);

  useEffect(() => {
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    messageEnd.current?.scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth" });
  }, [messages, sending, chatStatus]);

  useEffect(() => {
    if (!auth?.authenticated || !refreshStatus?.refreshing) return;
    const timer = window.setInterval(() => {
      void (async () => {
        try {
          const status = await api.guardianRefreshStatus();
          setRefreshStatus(status);
          if (!status.refreshing) setGuardian(await api.guardian());
        } catch {
          window.clearInterval(timer);
        }
      })();
    }, 750);
    return () => window.clearInterval(timer);
  }, [auth?.authenticated, refreshStatus?.refreshing]);

  async function sendMessage(text: string) {
    const clean = text.trim();
    if (!clean || sending || !guardian) return;
    const userMessage: ChatMessage = { role: "user", content: clean };
    const assistantMessage: ChatMessage = { role: "assistant", content: "" };
    const priorHistory = messages.slice(-12);
    activeChat.current?.abort();
    const controller = new AbortController();
    activeChat.current = controller;
    const requestId = ++chatRequestId.current;
    setMessages((current) => [...current, userMessage, assistantMessage]);
    setInput("");
    setSending(true);
    setChatStatus("Checking your Guardian…");
    setError(null);
    let completed = false;
    try {
      await api.chatStream(
        clean,
        priorHistory,
        {
          onEvent: (event) => {
            if (chatRequestId.current !== requestId) return;
            if (event.type === "status") {
              setChatStatus(event.label);
              return;
            }
            if (event.type === "message_delta") {
              setChatStatus(null);
              setMessages((current) => {
                const next = [...current];
                const last = next[next.length - 1];
                if (last?.role !== "assistant") return current;
                next[next.length - 1] = { ...last, content: last.content + event.delta };
                return next;
              });
              return;
            }
            if (event.type === "sources") {
              setMessages((current) => {
                const next = [...current];
                const last = next[next.length - 1];
                if (last?.role !== "assistant") return current;
                next[next.length - 1] = { ...last, sources: event.sources };
                return next;
              });
              return;
            }
            if (event.type === "completed") {
              completed = true;
              setChatStatus(null);
            }
          },
        },
        controller.signal,
      );
      if (auth?.debug_enabled) {
        try {
          setChatTrace(await api.latestChatTrace());
        } catch {
          setChatTrace(null);
        }
      }
    } catch (caught) {
      if (chatRequestId.current !== requestId) return;
      setMessages((current) =>
        current[current.length - 1]?.role === "assistant" ? current.slice(0, -1) : current,
      );
      if (!(caught instanceof DOMException && caught.name === "AbortError")) {
        setError(caught instanceof Error ? caught.message : "The Copilot could not respond.");
      }
    } finally {
      if (chatRequestId.current === requestId) {
        if (!completed) setChatStatus(null);
        activeChat.current = null;
        setSending(false);
      }
    }
  }

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    await sendMessage(input);
  }

  async function disconnect() {
    chatRequestId.current += 1;
    activeChat.current?.abort();
    activeChat.current = null;
    setSending(false);
    setChatStatus(null);
    try {
      await api.logout();
      setAuth((current) =>
        current ? { ...current, authenticated: false } : current,
      );
      setGuardian(null);
      setRefreshStatus(null);
      setMessages([]);
      setDebugOpen(false);
      setAccountMenuOpen(false);
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

  function openInspector() {
    setAccountMenuOpen(false);
    setDebugOpen(true);
  }

  async function refreshGuardian(level: "normal" | "full" = "normal") {
    setRefreshing(true);
    setError(null);
    try {
      const state: GuardianStateResponse = await api.guardianRefresh(level);
      setGuardian(state.guardian);
      setRefreshStatus(state.refresh);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Guardian refresh failed.");
    } finally {
      setRefreshing(false);
    }
  }

  async function refreshSlice(slice: string) {
    setRefreshing(true);
    try {
      const state = await api.debugRefreshSlice(slice);
      setGuardian(state.guardian);
      setRefreshStatus(state.refresh);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Guardian slice refresh failed.");
    } finally {
      setRefreshing(false);
    }
  }

  async function clearGuardianCache() {
    setRefreshing(true);
    try {
      await api.debugClearGuardianCache();
      setRefreshStatus({ cached: false, refreshing: false, slices: {} });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not clear Guardian cache.");
    } finally {
      setRefreshing(false);
    }
  }

  function selectGuideTool(name: string) {
    const preset = guideToolPresets.find((value) => value.name === name);
    if (!preset) return;
    setGuideToolName(name);
    setGuideArguments(JSON.stringify(preset.arguments, null, 2));
    setGuideResult(null);
    setGuideError(null);
  }

  async function runGuideTool() {
    setGuideRunning(true);
    setGuideError(null);
    try {
      const parsed = JSON.parse(guideArguments) as unknown;
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        throw new Error("Tool arguments must be a JSON object.");
      }
      const response = await api.destinyKnowledgeTool(
        guideToolName,
        parsed as Record<string, unknown>,
      );
      setGuideResult(response.result);
    } catch (caught) {
      setGuideError(caught instanceof Error ? caught.message : "The guide tool failed.");
    } finally {
      setGuideRunning(false);
    }
  }

  function selectLiveTool(name: string) {
    const preset = liveToolPresets.find((value) => value.name === name);
    if (!preset) return;
    setLiveToolName(name);
    setLiveArguments(JSON.stringify(preset.arguments, null, 2));
    setLiveResult(null);
    setLiveError(null);
  }

  async function runLiveTool() {
    setLiveRunning(true);
    setLiveError(null);
    try {
      const parsed = JSON.parse(liveArguments) as unknown;
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        throw new Error("Tool arguments must be a JSON object.");
      }
      const response = await api.destinyKnowledgeTool(
        liveToolName,
        parsed as Record<string, unknown>,
      );
      setLiveResult(response.result);
    } catch (caught) {
      setLiveError(caught instanceof Error ? caught.message : "The live Destiny tool failed.");
    } finally {
      setLiveRunning(false);
    }
  }

  const authenticated = Boolean(auth?.authenticated);
  const connected = authenticated && guardian;
  const freshnessText = refreshing || refreshStatus?.refreshing
    ? "Refreshing Guardian…"
    : Object.values(refreshStatus?.slices ?? {}).some((slice) => slice.last_error)
      ? "Refresh failed · cached data"
      : refreshStatus && !refreshStatus.cached
        ? "Guardian cache cleared"
        : Object.values(refreshStatus?.slices ?? {}).some((slice) => !slice.fresh)
          ? "Using cached data"
          : "Guardian data current";

  return (
    <div
      className={`app-shell ${authenticated ? "is-authenticated" : ""} ${
        connected ? "is-connected" : ""
      } ${
        messages.length ? "has-conversation" : "is-empty-chat"
      }`}
    >
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
                <>
                  <span className="freshness-state">
                    {freshnessText}
                  </span>
                  <button
                    className="text-button"
                    type="button"
                    disabled={refreshing}
                    onClick={() => void refreshGuardian("normal")}
                  >
                    Refresh
                  </button>
                </>
              )}
              {guardian && auth?.debug_enabled && (
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
        {authenticated && (
          <button
            ref={accountMenuButton}
            className="mobile-menu-button"
            type="button"
            aria-label="Open Guardian account menu"
            aria-controls="guardian-account-drawer"
            aria-expanded={accountMenuOpen}
            onClick={() => {
              setDebugOpen(false);
              setAccountMenuOpen(true);
            }}
          >
            <MenuIcon />
          </button>
        )}
      </header>

      {authenticated && (
        <div
          className={`drawer-backdrop ${accountMenuOpen ? "is-open" : ""}`}
          aria-hidden="true"
          onClick={() => setAccountMenuOpen(false)}
        />
      )}

      <main className="workspace">
        <section
          ref={accountDrawer}
          id="guardian-account-drawer"
          className={`guardian-pane ${authenticated ? "account-drawer" : "mobile-connect-pane"} ${
            accountMenuOpen ? "is-open" : ""
          }`}
          aria-label={mobileLayout && authenticated ? "Guardian account menu" : "Guardian summary"}
          aria-labelledby={mobileLayout && authenticated ? "account-drawer-title" : undefined}
          aria-modal={mobileLayout && authenticated ? true : undefined}
          aria-hidden={mobileLayout && authenticated ? !accountMenuOpen : undefined}
          role={mobileLayout && authenticated ? "dialog" : undefined}
          tabIndex={mobileLayout && authenticated ? -1 : undefined}
        >
          {authenticated && (
            <div className="drawer-heading">
              <div>
                <span className="kicker">Guardian account</span>
                <h2 id="account-drawer-title">Your Guardian</h2>
              </div>
              <button
                type="button"
                aria-label="Close Guardian account menu"
                onClick={() => setAccountMenuOpen(false)}
              >
                <CloseIcon />
              </button>
            </div>
          )}
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
                  href={auth?.configured ? api.authLoginUrl : undefined}
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
          {authenticated && (
            <div className="drawer-account-actions">
              <div className="drawer-status" role="status">
                <span className={`status-dot ${connected ? "online" : ""}`} />
                <span>{connected ? guardian.bungie_display_name : "Bungie connected"}</span>
                {guardian && <small>{freshnessText}</small>}
              </div>
              {guardian && (
                <button
                  className="drawer-action primary"
                  type="button"
                  disabled={refreshing}
                  onClick={() => void refreshGuardian("normal")}
                >
                  {refreshing ? "Refreshing…" : "Refresh Guardian"}
                  <span aria-hidden="true">↻</span>
                </button>
              )}
              {guardian && auth?.debug_enabled && (
                <button className="drawer-action" type="button" onClick={openInspector}>
                  Inspect context
                  <span aria-hidden="true">↗</span>
                </button>
              )}
              <button className="drawer-action danger" type="button" onClick={() => void disconnect()}>
                Disconnect
              </button>
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
              <div className="message-list">
                {messages.map((message, index) => (
                  <div className={`message ${message.role}`} key={`${message.role}-${index}`}>
                    <span className="message-label">{message.role === "user" ? "You" : "Copilot"}</span>
                    {message.role === "assistant" && !message.content && sending ? (
                      <div
                        className="stream-status"
                        role="status"
                        aria-live="polite"
                        aria-atomic="true"
                      >
                        <span>{chatStatus ?? "Working…"}</span>
                        <span className="status-dots" aria-hidden="true"><i /><i /><i /></span>
                      </div>
                    ) : (
                      <p>{message.content}</p>
                    )}
                    {message.sources && message.sources.length > 0 && (
                      <div className="message-sources">
                        <span>Sources</span>
                        <ul>
                          {message.sources.map((source) => (
                            <li key={source.url}>
                              <a href={source.url} target="_blank" rel="noopener noreferrer">
                                <span>{source.title}</span>
                                {source.domain &&
                                  source.domain.toLowerCase() !== source.title.toLowerCase() && (
                                    <small>{source.domain}</small>
                                  )}
                              </a>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                ))}
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
      {connected && auth?.debug_enabled && debugOpen && (
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
          <h3>Guardian Refresh &amp; Cache</h3>
          <p className="inspector-note">
            Normalized per-account slices only. OAuth tokens and raw Bungie payloads are never shown.
          </p>
          <div className="refresh-controls">
            {Object.keys(refreshStatus?.slices ?? {}).map((slice) => (
              <button
                key={slice}
                type="button"
                disabled={refreshing}
                onClick={() => void refreshSlice(slice)}
              >
                Refresh {slice}
              </button>
            ))}
            <button type="button" disabled={refreshing} onClick={() => void refreshGuardian("normal")}>
              Normal refresh
            </button>
            <button type="button" disabled={refreshing} onClick={() => void refreshGuardian("full")}>
              Full refresh
            </button>
            <button type="button" disabled={refreshing} onClick={() => void clearGuardianCache()}>
              Clear cache
            </button>
          </div>
          {refreshStatus && (
            <div className="tool-inspector">
              <pre>{JSON.stringify(refreshStatus, null, 2)}</pre>
            </div>
          )}
          <h3>Guide Knowledge</h3>
          <p className="inspector-note">
            Source-backed summaries retain canonical resolution, provenance, freshness, cache,
            conflicts, and warnings. Volatile questions are rejected until live knowledge exists.
          </p>
          <div className="tool-inspector">
            <label htmlFor="guide-tool">Tool</label>
            <select
              id="guide-tool"
              value={guideToolName}
              onChange={(event) => selectGuideTool(event.target.value)}
            >
              {guideToolPresets.map((tool) => (
                <option key={tool.name} value={tool.name}>{tool.label}</option>
              ))}
            </select>
            <label htmlFor="guide-arguments">Arguments</label>
            <textarea
              id="guide-arguments"
              value={guideArguments}
              onChange={(event) => setGuideArguments(event.target.value)}
              spellCheck={false}
            />
            <button type="button" disabled={guideRunning} onClick={() => void runGuideTool()}>
              {guideRunning ? "Running…" : "Run tool"}
            </button>
            {guideError && <p className="tool-error">{guideError}</p>}
            {guideResult && <pre>{JSON.stringify(guideResult, null, 2)}</pre>}
          </div>
          <h3>Live Destiny Data</h3>
          <p className="inspector-note">
            Current source, effective window, reset cadence, cache state, confidence, conflicts,
            and limitations. Public milestones are not assumed to be featured activities.
          </p>
          <div className="tool-inspector">
            <label htmlFor="live-tool">Tool</label>
            <select
              id="live-tool"
              value={liveToolName}
              onChange={(event) => selectLiveTool(event.target.value)}
            >
              {liveToolPresets.map((tool) => (
                <option key={tool.name} value={tool.name}>{tool.label}</option>
              ))}
            </select>
            <label htmlFor="live-arguments">Arguments</label>
            <textarea
              id="live-arguments"
              value={liveArguments}
              onChange={(event) => setLiveArguments(event.target.value)}
              spellCheck={false}
            />
            <button type="button" disabled={liveRunning} onClick={() => void runLiveTool()}>
              {liveRunning ? "Running…" : "Run tool"}
            </button>
            {liveError && <p className="tool-error">{liveError}</p>}
            {liveResult && <pre>{JSON.stringify(liveResult, null, 2)}</pre>}
          </div>
          <h3>AI Tool Trace</h3>
          <p className="inspector-note">
            Compact trace only: tool names and grounding categories, never tool payloads or secrets.
          </p>
          {chatTrace ? (
            <div className="tool-inspector">
              <pre>{JSON.stringify(chatTrace, null, 2)}</pre>
            </div>
          ) : (
            <p className="inspector-note">Send a chat message to capture a trace.</p>
          )}
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
