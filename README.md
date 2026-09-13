# Guardian Copilot

Guardian Copilot is a local, read-only Destiny 2 companion built around one question: **What should I do next?** It connects to Bungie with OAuth and turns the authenticated player's profile into a typed, compact `GuardianContext`. Raw Bungie response envelopes never enter that model and are never sent to OpenAI.

## Architecture

```text
React + TypeScript + Vite
  └─ /api via the local HTTPS Vite proxy
      └─ FastAPI
          ├─ Bungie OAuth + in-memory token/session store
          ├─ read-only Bungie Platform client
          ├─ versioned on-disk SQLite Manifest resolver + bounded definition LRU
          ├─ GuardianContext normalization boundary
          ├─ bounded read-only Guardian account tools
          ├─ normalized Manifest + detailed-guide + live knowledge providers
          └─ server-only OpenAI Responses API tool loop
```

OAuth tokens and CSRF state remain server-side behind an `OAuthSessionStore` abstraction. The first
deployment uses its single-process memory implementation: restarting or scaling the backend signs
users out. There is no application database. `.cache/manifest` contains only Bungie's public,
versioned mobile SQLite Manifest and rebuildable search indexes and is gitignored.

## Setup

Prerequisites: Python 3.11+, Node.js 20.19+ (or 22.13+), and a Bungie.net application.

### 1. Create and trust the local HTTPS certificate on Windows

From PowerShell at the repository root:

```powershell
winget install --id FiloSottile.mkcert -e --accept-package-agreements --accept-source-agreements
```

Close and reopen PowerShell if `mkcert` is not yet on `PATH`, then run this as one complete command on the final line:

```powershell
mkcert -install
New-Item -ItemType Directory -Force .certs
mkcert -cert-file .certs\localhost.pem -key-file .certs\localhost-key.pem localhost 127.0.0.1 ::1
```

The generated certificate and key remain under `.certs/` and are gitignored. Restart an already-open browser if it does not trust the local CA immediately.

### 2. Configure the Bungie application

Create or edit the application in the [Bungie.net Application Portal](https://www.bungie.net/en/Application):

1. Configure a confidential OAuth client.
2. Enable `ReadBasicUserProfile` and `ReadDestinyInventoryAndVault`. The latter is Bungie's required Destiny 2 read scope for private inventory, currency, milestone, and progression data. No write scopes are needed.
3. Set the redirect URL to exactly `https://localhost:8000/api/auth/callback`.
4. If the portal requests an origin, use `https://localhost:5173`.

The callback in the browser must begin with `/api/auth/callback`. If Bungie redirects to `/auth/bungie/callback`, the application portal still has the old redirect URL; update and save it there before signing in again.

### 3. Configure environment variables

```powershell
Copy-Item .env.example .env
```

Fill in the Bungie values in the root `.env`. `OPENAI_API_KEY` is optional for account inspection and required only for generated recommendations.

```dotenv
APP_ENV=development
PORT=8000
BUNGIE_API_KEY=
BUNGIE_CLIENT_ID=
BUNGIE_CLIENT_SECRET=
BUNGIE_REDIRECT_URI=https://localhost:8000/api/auth/callback
OPENAI_API_KEY=
OPENAI_MODEL=gpt-5-mini
OPENAI_REASONING_EFFORT=low
OPENAI_MAX_OUTPUT_TOKENS=2000
FRONTEND_ORIGIN=https://localhost:5173
FRONTEND_URL=https://localhost:5173
COOKIE_SECURE=true
COOKIE_SAMESITE=lax
SESSION_BACKEND=memory
SESSION_COOKIE_MAX_AGE_SECONDS=2592000
OAUTH_STATE_TTL_SECONDS=600
TLS_CERT_FILE=.certs/localhost.pem
TLS_KEY_FILE=.certs/localhost-key.pem
MANIFEST_CACHE_DIR=.cache/manifest
MANIFEST_DEFINITION_CACHE_SIZE=1024
MANIFEST_METADATA_TTL_SECONDS=900
GUIDE_CORPUS_FILE=backend/app/data/destiny_guides.json
GUIDE_CACHE_DIR=.cache/guides
GUIDE_CACHE_STABLE_TTL_SECONDS=2592000
GUIDE_CACHE_SEMI_STABLE_TTL_SECONDS=86400
LIVE_CACHE_DIR=.cache/live
LIVE_CACHE_TTL_SECONDS=300
REQUEST_TIMEOUT_SECONDS=20
ENABLE_DEBUG_TOOLS=true
ALLOW_PRODUCTION_DEBUG=false
LOG_LEVEL=INFO
VITE_API_BASE_URL=
```

Never use a `VITE_` prefix for a secret. `.env`, `.certs`, and `.cache` are gitignored.

## Run locally

Backend:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python run_dev.py
```

Frontend, in a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `https://localhost:5173`. API docs are at `https://localhost:8000/docs`. On a cold cache, the
first definition lookup streams Bungie's English mobile Manifest to `.cache/manifest`, extracts its
SQLite database, and queries only referenced hashes. Later lookups reuse that versioned artifact.

## Production deployment: Vercel + Railway

The recommended first production topology is one persistent **Railway service** and replica for the
FastAPI API, plus one Vercel project for the static Vite frontend. Railway terminates public TLS and
forwards HTTP to Uvicorn. `backend/run_prod.py` binds to `0.0.0.0` and Railway's automatically
supplied `PORT`; it does not load the local certificate files. Vercel serves only compiled frontend
assets. Bungie credentials, OAuth tokens, and the OpenAI key never enter the Vite build.

Use one canonical Vercel production URL and one canonical Railway API URL. Platform preview URLs
are not automatically allowed by CORS. For the most reliable cookie behavior, use sibling custom
domains such as `app.example.com` and `api.example.com`; privacy modes can block third-party cookies
between unrelated `vercel.app` and `up.railway.app` sites even when the required `SameSite=None`
setting is present.

### 1. Choose the production URLs

Pick the Vercel project name and Railway service name first. The examples below use:

```text
Frontend: https://guardian-copilot.vercel.app
Backend:  https://guardian-copilot-api-production.up.railway.app
Callback: https://guardian-copilot-api-production.up.railway.app/api/auth/callback
```

Replace all three examples with the URLs actually assigned to the projects. Do not include a
trailing slash in `FRONTEND_ORIGIN`, `FRONTEND_URL`, or `VITE_API_BASE_URL`.

### 2. Create the Railway backend

1. Push this repository to GitHub.
2. In Railway, click **New Project > Empty Project**.
3. On the project canvas, create an **Empty Service**, open its settings, connect the GitHub
   repository under **Source**, and select the production branch.
4. In the service settings, set **Root Directory** to `/backend`.
5. Under **Config as Code**, set **Config File Path** to `/backend/railway.json`. Railway config
   files do not automatically follow a service's root directory, so the absolute repository path
   matters here.
6. Railpack detects Python from `requirements.txt`/`pyproject.toml` and installs the backend
   dependencies. Leave the build command on its detected default. If the Railway UI requires an
   explicit override, use `pip install -r requirements.txt`.
7. The checked-in Railway config sets **Start Command** to `python run_prod.py`, **Healthcheck Path**
   to `/api/health`, a 300-second deployment healthcheck timeout, and restart-on-failure behavior.
8. Under **Networking > Public Networking**, click **Generate Domain**. Use the assigned HTTPS
   `*.up.railway.app` domain in the variables and Bungie callback below.
9. Open the service **Variables** tab and add these values individually or through the Raw Editor:

| Variable | Production value |
| --- | --- |
| `APP_ENV` | `production` |
| `BUNGIE_API_KEY` | Production Bungie API key |
| `BUNGIE_CLIENT_ID` | Production confidential-client ID |
| `BUNGIE_CLIENT_SECRET` | Production client secret |
| `BUNGIE_REDIRECT_URI` | `https://<backend-domain>/api/auth/callback` |
| `OPENAI_API_KEY` | Server-side OpenAI API key |
| `OPENAI_MODEL` | Optional; defaults to `gpt-5-mini` |
| `OPENAI_REASONING_EFFORT` | Optional; defaults to `low` |
| `OPENAI_MAX_OUTPUT_TOKENS` | Optional; defaults to `2000` |
| `FRONTEND_ORIGIN` | Exact `https://<vercel-domain>` origin |
| `FRONTEND_URL` | Exact `https://<vercel-domain>` redirect target |
| `COOKIE_SECURE` | `true` |
| `COOKIE_SAMESITE` | `none` for separate Vercel/Railway sites |
| `SESSION_BACKEND` | `memory` |
| `SESSION_COOKIE_MAX_AGE_SECONDS` | Optional; defaults to `2592000` |
| `OAUTH_STATE_TTL_SECONDS` | Optional; defaults to `600` |
| `MANIFEST_DEFINITION_CACHE_SIZE` | Optional bounded definition LRU; defaults to `1024` |
| `MANIFEST_METADATA_TTL_SECONDS` | Optional version recheck interval; defaults to `900` |
| `ENABLE_DEBUG_TOOLS` | `false` |
| `ALLOW_PRODUCTION_DEBUG` | `false` |
| `LOG_LEVEL` | `INFO` |

Railway supplies `PORT` automatically; do not create or hard-code a `PORT` variable. Keep the
service at one replica while `SESSION_BACKEND=memory`. Do not attach a Railway Volume: local caches
are disposable and no required application state is file-backed. Deploy the service after the
domain-dependent values are set, or redeploy it from the service deployment controls if Railway
already attempted an initial build.

### 3. Configure the production Bungie application

Bungie registers a single exact redirect URL per application. Create a separate production Bungie
application so the existing localhost application can keep
`https://localhost:8000/api/auth/callback`.

1. Open the [Bungie.net Application Portal](https://www.bungie.net/en/Application).
2. Create or edit the production confidential client with the same read-only scopes used locally.
3. Set its redirect URL to exactly
   `https://<backend-domain>/api/auth/callback` (case and path included).
4. If an origin is requested, set the canonical Vercel frontend origin.
5. Copy that application's API key, client ID, and secret into Railway. The credentials and
   redirect URI must belong to the same Bungie application.
6. Save, then deploy/redeploy the Railway service so it loads the final values.
7. Before creating the Vercel deployment, open `https://<backend-domain>/api/health` and confirm it
   returns `{"status":"ok"}`. If the deployment fails, inspect Railway's build/deploy logs and
   correct the rejected configuration rather than weakening the production checks.

### 4. Create the Vercel frontend

1. In Vercel, click **Add New > Project**, import the same repository, and choose `frontend` as the
   **Root Directory**.
2. Confirm **Framework Preset: Vite**, **Build Command: `npm run build`**, and **Output Directory:
   `dist`**. The install command can remain the detected `npm install`/`npm ci` behavior.
3. Under **Environment Variables**, add only:

   ```text
   VITE_API_BASE_URL=https://<backend-domain>
   ```

   Apply it to Production. Preview deployments need their own explicitly allowed frontend origin;
   do not weaken backend CORS with `*`.
4. Click **Deploy**. `frontend/vercel.json` provides the Vite SPA fallback to `index.html`.
5. If the final Vercel domain differs from the planned value, update `FRONTEND_ORIGIN` and
   `FRONTEND_URL` on Railway, then redeploy the backend. Vercel environment-variable changes also
   require a new frontend deployment because Vite embeds public variables at build time.

Never add Bungie or OpenAI credentials to Vercel and never give them a `VITE_` prefix.

### Production cookies, sessions, and caches

- The browser receives only an opaque HttpOnly session ID. Tokens stay in the backend memory store.
- The production session cookie is host-only, `Secure`, `HttpOnly`, and `SameSite=None`; CORS allows
  credentials only from `FRONTEND_ORIGIN`. OAuth state uses a short-lived, one-time, HttpOnly
  `SameSite=Lax` cookie on the callback path.
- `run_prod.py` deliberately starts one worker and disables Uvicorn access logs so OAuth callback
  query strings are not written by the application server.
- A deploy, restart, crash, or sleeping instance clears OAuth state and sessions. Active users must
  reconnect, and an OAuth callback that lands after a restart fails closed. Do not scale beyond one
  process until a shared implementation (for example Redis) is added behind `OAuthSessionStore`.
- Guardian state is a process-local normalized cache and can be rebuilt while the OAuth session
  exists. Manifest, guide, and live caches under `.cache/` are disposable performance caches. They
  may disappear on every deploy; the app redownloads/rebuilds them and remains correct. The curated
  guide corpus under `backend/app/data/` is versioned application data, not a cache.
- Manifest definitions are queried by hash from Bungie's official mobile SQLite database. Only
  returned rows are JSON-decoded, and the process retains at most `MANIFEST_DEFINITION_CACHE_SIZE`
  hot definitions. A cold deployment may take longer while the rebuildable database downloads.
- No Railway Volume, Postgres, or Redis is required for the first personal beta. Add shared session
  storage before multi-instance or zero-sign-out requirements.

### First-deploy verification checklist

1. Open `https://<backend-domain>/api/health` and confirm `{"status":"ok"}`.
2. Confirm `/docs`, `/openapi.json`, and `/api/debug/guardian-tools` return 404 in production.
3. Open the Vercel site and confirm the browser calls the HTTPS backend URL, not localhost.
4. In the browser Network panel, confirm the API response has the exact
   `Access-Control-Allow-Origin` value and `Access-Control-Allow-Credentials: true`.
5. Click **Connect with Bungie**, approve access, and confirm the callback returns to the Vercel URL.
6. Confirm `guardian_session` is stored on the backend host with `Secure`, `HttpOnly`, and
   `SameSite=None`; no access token should appear in browser storage or responses.
7. Confirm Guardian loading, refresh, chat, Manifest/guide/live grounding, and Disconnect work.
8. Confirm the developer inspector is absent and debug URLs return 404.
9. Redeploy the one backend instance and confirm the expected beta limitation: the old session is
   rejected and reconnecting restores normal operation.
10. Test OAuth in the privacy modes/browsers you intend to support. If cross-site cookies are
    blocked, attach sibling custom domains rather than relaxing cookie or CORS security.

## Current capabilities

- Bungie authorization-code OAuth, CSRF state validation, token refresh, and secure HTTP-only local session cookie
- Cross-save primary membership selection
- Typed character, subclass, equipped gear, searchable inventory/vault, quest/objective, milestone, progression/reputation, currency, available activity, recent activity, collectible, record, and crafting summaries
- Manifest-resolved names, descriptions, icons, item/bucket types, perks, stats, activities, objectives, milestones, progressions, and season metadata
- Explicit data-availability and truncation notes in `GuardianContext`
- Temporary **Inspect context** frontend view showing the complete normalized payload
- Eight bounded read-only Guardian tools for character, loadout, quest, activity, history, progression, inventory, and build queries
- Five normalized Destiny knowledge tools for entity search, item, activity, quest, and acquisition-evidence queries
- Two detailed guide tools for compact guide search and practical normalized walkthrough retrieval
- Five live-data tools for source status, weekly rotations, public vendors, current activities,
  and live search
- Versioned local fuzzy-search index derived from public English Manifest definitions; it rebuilds only when Bungie's Manifest version changes
- A versioned, source-attributed guide corpus containing concise factual extracts rather than copied articles
- One AI tool loop that can combine Guardian account, Manifest, and guide tools
- Server-side OpenAI Responses API function-calling loop; the complete `GuardianContext` is not sent to the model

## Guardian tools and debugging

The tool service queries one normalized `GuardianContext`. Its declared slice dependencies may
soft-refresh relevant stale state before execution; it does not fetch unrelated account data.
Available tools are:

- `get_character_summary`
- `get_equipped_loadout`
- `get_active_quests`
- `get_available_activities`
- `get_recent_activities`
- `get_progression`
- `search_inventory`
- `get_build_details`

While signed in, open `https://localhost:8000/docs` to invoke a tool manually through `POST /api/debug/guardian-tools/{tool_name}`. For example, invoke `search_inventory` with:

```json
{
  "query": "void",
  "character_id": null,
  "item_type": "weapon",
  "subtype": "submachine gun",
  "bucket": null,
  "equipped_only": false,
  "limit": 25
}
```

`GET /api/debug/guardian-tools` returns the callable schemas. Set `ENABLE_DEBUG_TOOLS=false` to disable the debug routes.

### Destiny knowledge tools

The separate `DestinyKnowledgeService` routes each tool to the provider that owns it. It uses
`BungieManifestProvider` for canonical metadata, `GuideKnowledgeProvider` for practical sourced
instructions, and `LiveDestinyProvider` for explicitly current data. The callable Manifest tools
are:

- `search_destiny_entities`
- `get_item_details`
- `get_activity_details`
- `get_quest_details`
- `find_item_source`

The guide tools are:

- `search_destiny_guides`
- `get_destiny_guide`

The live tools are:

- `get_live_destiny_status`
- `get_weekly_rotation`
- `get_vendor_status`
- `get_current_activity_status`
- `search_live_destiny`

The initial live source uses Bungie's public milestones and public-vendors endpoints. Results keep
source URL, authority, retrieval time, effective/reset window, stale boundary, cache status,
supported live topics, conflicts, and limitations. Xur's public inventory is returned when Bungie
exposes it. Public milestone presence is intentionally not interpreted as weekly featured status,
farmability, or complete Director availability.

Guide records retain their source, URL, relevant section, retrieval/publication/update timestamps,
factual claims, confidence, entity associations, freshness, warnings, and detected conflicts. The
AI receives compact summaries or normalized steps/encounters—not complete article bodies. Entity
aliases and reasonable misspellings are normalized before Manifest-backed canonical resolution.

The guide cache is file-backed under `.cache/guides`, keyed primarily by canonical entity,
guide type, operation, and corpus version. Stable records default to 30 days, semi-stable records to
one day, and volatile requests bypass the guide cache. Weekly rotations, featured content, vendors,
current meta, and drop-rate questions never use guide data as current truth. The separate live cache
under `.cache/live` has a short TTL and never survives the upstream stale/reset boundary. For a
volatile chat question, the requested live topic must appear in a successful live tool result's
`supported_live_topics`; calling a live tool or retrieving a different current fact is insufficient.

Open **Inspect context** in the frontend to run these tools and inspect their normalized JSON. They can also be invoked without an authenticated Guardian through `POST /api/debug/destiny-knowledge/{tool_name}` when debug tools are enabled. `GET /api/debug/destiny-knowledge` returns their schemas and `GET /api/debug/destiny-knowledge/status` builds the index if needed and reports its coverage. For example, call `search_destiny_entities` with:

```json
{
  "query": "Wish-Ender",
  "entity_types": ["item"],
  "limit": 5
}
```

The index contains visible, non-redacted, named entries from these English Manifest tables:

- `DestinyInventoryItemDefinition`
- `DestinyActivityDefinition`
- `DestinyActivityTypeDefinition`
- `DestinyDestinationDefinition`
- `DestinyPlaceDefinition`
- `DestinyObjectiveDefinition`
- `DestinyRecordDefinition`
- `DestinyProgressionDefinition`
- `DestinyFactionDefinition`
- `DestinySeasonDefinition`
- `DestinyCollectibleDefinition`
- `DestinySandboxPerkDefinition`
- `DestinySocketTypeDefinition`
- `DestinyStatDefinition`
- `DestinyTraitDefinition`
- `DestinyActivityModifierDefinition`
- `DestinyRewardSourceDefinition` when Bungie's current table contains named entries
- `DestinyDamageTypeDefinition`
- `DestinyItemCategoryDefinition`
- `DestinyClassDefinition`

The status endpoint separates tables with indexed named entries from empty and unavailable tables. In the currently validated Manifest, `DestinyRewardSourceDefinition` exists but is empty; Bungie's own schema also warns that item source generation is generally not populated.

The production chat tool loop follows the official [OpenAI function-calling flow](https://developers.openai.com/api/docs/guides/function-calling): it sends the user request and all available tool definitions, executes requested functions locally, replays every returned output item, appends each matching `function_call_output` by `call_id`, and asks the model for a final grounded response.

When debug tools are enabled, the frontend inspector includes a **Guide Knowledge** panel showing
canonical resolution, normalized content, provenance, freshness, cache status, conflicts, and
warnings. It also shows the latest compact AI trace. The trace stores only the user question, tool
names, grounding categories, and answer status—never tool arguments, raw Guardian payloads, OAuth
tokens, or secrets. Trace JSON is available at `GET /api/debug/chat-traces/latest` and
`GET /api/debug/chat-traces/{trace_id}`; `GET /api/debug/chat-traces` lists recent traces for
development investigation.

## Bungie API usage

OAuth uses:

- `GET https://www.bungie.net/en/OAuth/Authorize`
- `POST /Platform/App/OAuth/Token/` for authorization-code exchange and refresh

All Destiny player/content operations are read-only `GET` requests:

- `GET /Platform/User/GetMembershipsForCurrentUser/`
- `GET /Platform/Destiny2/{membershipType}/Profile/{destinyMembershipId}/`
- `GET /Platform/Destiny2/{membershipType}/Account/{destinyMembershipId}/Character/{characterId}/Stats/Activities/`
- `GET /Platform/Destiny2/Manifest/`
- `GET /Platform/Destiny2/Milestones/`
- `GET /Platform/Destiny2/Vendors/?components=400,401,402`
- `GET /Platform/Destiny2/Manifest/{entityType}/{hash}/` as a small-set/fallback definition lookup
- Bungie-hosted `mobileWorldContentPaths.en` archive for indexed, on-disk Manifest definition lookup

The profile request asks for `Profiles`, `ProfileInventories`, `ProfileCurrencies`, `ProfileProgression`, `Characters`, `CharacterInventories`, `CharacterProgressions`, `CharacterActivities`, `CharacterEquipment`, `ItemInstances`, `ItemObjectives`, `ItemSockets`, `ItemStats`, `Collectibles`, `Records`, and `Craftables`.

## Guardian refresh and cache

`GuardianRefreshService` is the single policy boundary for account refresh. It stores only
normalized data in isolated, process-local entries keyed by the authenticated session. The
replaceable `GuardianStateCache` interface keeps a future shared production cache possible without
introducing Redis for local development. OAuth token refresh remains an independent concern.

State is split into profile, equipment, inventory, quests/progress, activity history, and
collections slices. Each slice tracks fetch/stale/access timestamps, source components, reason,
generation, cache counters, in-flight state, and its last success/error. Concurrent requests are
coalesced. A failed refresh retains usable stale data and observes a short retry backoff; cold-load
failure still surfaces normally. Logout, account identity change, and cache-schema change clear or
replace affected state.

The app uses stale-while-revalidate on foreground resume: under five minutes idle it reuses cache,
between five minutes and two hours it targets dynamic slices, and after two hours it schedules a
full refresh. Guardian tool and prompt intents have per-slice minimum refresh intervals. A recent
completed activity can invalidate older equipment, inventory, and quest/progress slices. The
frontend shows compact freshness state, a normal refresh action, and detailed per-slice debug
controls (including full refresh and clear). All refresh logs contain metadata only, never OAuth
tokens or normalized payloads.

## Known limitations and next milestone

- Bungie may omit private components because of the OAuth scope, player privacy settings, or profile state. `data_availability.unavailable_components` reports those omissions.
- Active quests are combined from `CharacterProgressions.quests` and quest/bounty items in character inventories because Bungie documents the former as incompletely populated.
- `CharacterActivities` is useful but is not treated as a complete or perfectly reliable Director listing.
- The complete normalized inventory is kept locally for search. Weapons and armor retain Bungie-returned stats and socketed perks; other item records stay compact. Individual search results remain bounded and the full inventory is never sent to the model by default.
- Recent history retains up to 25 activities per character. Bungie may return fewer entries, and activity history is not a complete lifetime record.
- Collectibles and records are summarized. Invisible entries are honored and excluded; only near-complete record objectives are expanded.
- Crafting reports visible craftables and whether Bungie returned no failed requirement indexes. It does not yet model every pattern socket requirement.
- Vendor inventories, postmaster-specific recommendations, loadouts, full socket choices, and per-character collectible ownership are not yet normalized.
- Manifest definitions are static content metadata. They do not reliably provide full quest walkthroughs, dungeon/raid mechanics, authoritative loot tables, current rotations, live modifiers, weekly availability, drop rates, or patch-sensitive strategy.
- Item and collectible source strings are often broad hints. `find_item_source` reports the exact available evidence and says when the Manifest has no reliable source instead of inventing acquisition steps.
- Activity rewards and modifiers are labeled as possible static metadata; they are not treated as current or guaranteed.
- Raw Guardian and activity Power values remain available to the local UI/context but are removed from AI-facing eligibility inputs. The AI must treat Power eligibility as unknown unless a future source establishes compatible, current semantics for both values.
- Duplicate and legacy definitions exist. Detail tools prefer usable item/quest definitions, while results include hashes so ambiguous searches can be disambiguated explicitly.
- Detailed guide coverage is currently curated and intentionally narrow. A missing guide returns explicit uncertainty; runtime search or brittle site-specific scraping is not used. Adding a maintained search/retrieval API later only requires another source implementation behind `GuideKnowledgeProvider`.

The current official public endpoints do not reliably identify every weekly raid, dungeon,
Nightfall, exotic mission, loot rotation, drop-rate, or meta claim. Those categories return an
explicit unsupported-live-data state until a trustworthy structured current source is connected.
Full recommendation scoring, cross-process cache infrastructure, and scheduled background polling
remain intentionally out of scope.
