# Guardian Copilot

Guardian Copilot is a local, read-only Destiny 2 companion built around one question: **What should I do next?** It connects to Bungie with OAuth and turns the authenticated player's profile into a typed, compact `GuardianContext`. Raw Bungie response envelopes never enter that model and are never sent to OpenAI.

## Architecture

```text
React + TypeScript + Vite
  └─ /api via the local HTTPS Vite proxy
      └─ FastAPI
          ├─ Bungie OAuth + in-memory token/session store
          ├─ read-only Bungie Platform client
          ├─ versioned JSON Manifest resolver + local content cache
          ├─ GuardianContext normalization boundary
          ├─ bounded read-only Guardian account tools
          └─ server-only OpenAI Responses API tool loop
```

OAuth tokens and CSRF state exist only in backend memory. Restarting the backend signs the user out. There is no application database. `.cache/manifest` contains only Bungie's public, versioned definition files and is gitignored.

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
TLS_CERT_FILE=.certs/localhost.pem
TLS_KEY_FILE=.certs/localhost-key.pem
MANIFEST_CACHE_DIR=.cache/manifest
REQUEST_TIMEOUT_SECONDS=20
ENABLE_DEBUG_TOOLS=true
LOG_LEVEL=INFO
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

Open `https://localhost:5173`. API docs are at `https://localhost:8000/docs`. The first normalized profile request downloads the necessary English JSON Manifest component files; later runs use `.cache/manifest`.

## Current capabilities

- Bungie authorization-code OAuth, CSRF state validation, token refresh, and secure HTTP-only local session cookie
- Cross-save primary membership selection
- Typed character, subclass, equipped gear, searchable inventory/vault, quest/objective, milestone, progression/reputation, currency, available activity, recent activity, collectible, record, and crafting summaries
- Manifest-resolved names, descriptions, icons, item/bucket types, perks, stats, activities, objectives, milestones, progressions, and season metadata
- Explicit data-availability and truncation notes in `GuardianContext`
- Temporary **Inspect context** frontend view showing the complete normalized payload
- Eight bounded read-only Guardian tools for character, loadout, quest, activity, history, progression, inventory, and build queries
- Five normalized Destiny knowledge tools for entity search, item, activity, quest, and acquisition-evidence queries
- Versioned local fuzzy-search index derived from public English Manifest definitions; it rebuilds only when Bungie's Manifest version changes
- One AI tool loop that can call Guardian account tools, Destiny knowledge tools, or both
- Server-side OpenAI Responses API function-calling loop; the complete `GuardianContext` is not sent to the model

## Guardian tools and debugging

The tool service queries one normalized `GuardianContext`; individual tool calls do not make Bungie requests. Available tools are:

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

The separate `DestinyKnowledgeService` currently uses a `ManifestKnowledgeProvider`. Its provider interface is the extension point for future curated guides, rotations, mechanics, or patch-sensitive sources; none are included yet. The callable tools are:

- `search_destiny_entities`
- `get_item_details`
- `get_activity_details`
- `get_quest_details`
- `find_item_source`

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

The production chat tool loop follows the official [OpenAI function-calling flow](https://developers.openai.com/api/docs/guides/function-calling): it sends the user request and both tool families, executes requested functions locally, replays every returned output item, appends each matching `function_call_output` by `call_id`, and asks the model for a final grounded response.

## Bungie API usage

OAuth uses:

- `GET https://www.bungie.net/en/OAuth/Authorize`
- `POST /Platform/App/OAuth/Token/` for authorization-code exchange and refresh

All Destiny player/content operations are read-only `GET` requests:

- `GET /Platform/User/GetMembershipsForCurrentUser/`
- `GET /Platform/Destiny2/{membershipType}/Profile/{destinyMembershipId}/`
- `GET /Platform/Destiny2/{membershipType}/Account/{destinyMembershipId}/Character/{characterId}/Stats/Activities/`
- `GET /Platform/Destiny2/Manifest/`
- `GET /Platform/Destiny2/Manifest/{entityType}/{hash}/` as a small-set/fallback definition lookup
- Bungie-hosted paths from `jsonWorldComponentContentPaths.en` for bulk Manifest definition resolution

The profile request asks for `Profiles`, `ProfileInventories`, `ProfileCurrencies`, `ProfileProgression`, `Characters`, `CharacterInventories`, `CharacterProgressions`, `CharacterActivities`, `CharacterEquipment`, `ItemInstances`, `ItemObjectives`, `ItemSockets`, `ItemStats`, `Collectibles`, `Records`, and `Craftables`.

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
- Duplicate and legacy definitions exist. Detail tools prefer usable item/quest definitions, while results include hashes so ambiguous searches can be disambiguated explicitly.

The next milestone is a second, curated knowledge provider for walkthroughs, mechanics, loot tables, rotations, and other facts the Manifest cannot reliably answer. Recommendation scoring remains intentionally out of scope for this milestone.
