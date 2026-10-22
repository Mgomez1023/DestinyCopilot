# Claude Code — Destiny Copilot Project Instructions

Read this at the start of **every coding session** and after compaction. Read [ROADMAP.md](ROADMAP.md) for the current milestone, latest evaluation, open issues, and release gates. This is a project contract, not a script to optimize one benchmark.

## Project overview and current state

- Destiny 2 account-aware assistant; React/TypeScript/Vite frontend and Python/FastAPI backend.
- Bungie OAuth/API, normalized character and inventory state, Destiny Manifest, core-knowledge corpus, curated Destiny guides, targeted current-source web research.
- Local development branch at handoff: `feat/guardian-product-sprint`. Last report: no commit/push/deploy of this feature work. **Check `git status` instead of assuming this is still true.**
- The latest ten-scenario *real-model / simulated-account* eval: `started_at=2026-10-22T03:46:12+00:00`, `9/10` without deterministic serious flags, mean `15.6/16`, `judge=false`, `live_provider=false`. **Do not equate those scores with 90% player satisfaction or launch readiness.**
- **Current task after handoff:** design and run 15 new held-out scenarios (5+ multi-turn) as described in `ROADMAP.md`, then triage only generalizable beta-blocking findings.

## Non-negotiable engineering invariants

1. **NEVER BRANCH ON GOLDEN PROMPTS.** Never add production conditions keyed to evaluation scenario IDs, profile names, exact user question text, golden answers, expected tool names, fixture paths, or test-specific phrasing. No canned answers or hidden evaluation detection. Write domain-general behavior that also works for unfamiliar prompts and real Guardian accounts.
2. **Evidence before specificity.** A recommendation is only as strong as the tools' actual return values. Do not claim to have looked up, compared, equipped, moved, unlocked, verified, or sourced something without doing so.
3. **Do not invent capabilities.** The app cannot directly move an item to the vault merely because the player asks. Current tools do not expose a complete list of a Guardian's unlocked Prismatic Facets. Read-only analysis is not an account mutation.
4. **Honor user intent and constraints.** Preserve named item and equipped Exotic preferences; keep prior solo/time/no-PvP/no-dungeon constraints through follow-ups. A player saying “I finished the campaign” gives a usable conversational premise for how-to guidance, **not** account-side proof of completion.
5. **No unsupported optimization.** “Single biggest improvement” requires an evidenced comparison and identifiable action. If owned alternatives cannot be established, keep the build and say what is unknown; do not manufacture a superior swap.
6. **Prefer clear answers to defensive boilerplate.** Correct a claim rather than discarding a helpful recommendation. Do not return a generic safe fallback when a legitimate bounded answer is available.
7. **Do not modify source solely to turn the score green.** After a failure, inspect prompts, tool traces, evidence, validator corrections, and the final player experience. Tests must assess real behavior, not simply one exact sentence.
8. **Preserve working behavior.** Inventory/vault location, Sunshot fixed barrel, source lookup, and inability to see catalyst completion are regression-sensitive. Avoid broad rewrites when a scoped repair suffices.

## Evidence contract (what each source establishes)

| Evidence | Appropriate claims | Never infer |
| --- | --- | --- |
| **Guardian API / normalized tools** | Items returned as owned; exact equipment, location, class, quest/objective state, perks or stats **actually returned** | Unseen unlocked items, absent inventory as definite non-ownership if data are partial, a missing stat as low, completed quests from vague related evidence |
| **Bungie Manifest** | Static definitions, sockets, item types, source *hints*, activity definitions | Current activity availability, active weekly rewards, guaranteed drop rates, fresh patch behavior, exact player ownership |
| **Curated guide provider** | Acquisition steps, stable mechanics, known paths with source/freshness context | Current live rotation, every edge case, or personal completion state |
| **Current web/live evidence** | A dated and relevant claim **as supported by the retrieved page/data** | That a page is current without date/freshness verification; facts not contained in the source |
| **Player's words** | Stated goals, preferences, subjective experiences, conversational assumptions (“I finished The Final Shape”) | Confirmation that the Guardian API independently verified an account fact |
| **Absent/limited data** | “I can't determine this from the available data,” plus a useful next step | A fabricated inventory item, Facet checklist, reward ranking, timer, or ability to perform an action |

**Freshness:** Requests such as “currently,” “this week,” “latest,” “is The Coil dropping X now,” and “verify with a current source” need actual relevant, time-aware web/live evidence, or an explicit limitation. Manifest candidate mappings alone cannot prove present-day drops. Carry source URLs and dates through final-answer editing when the user requests verification.

**Grounding:** Bind every ownership or equipment assertion to the right character and storage location. `bucket` may describe a weapon slot while `location` identifies *vault / equipped / character*; never mistake a slot label for vault state. Quote/trace tool values rather than hallucinating fields.

**Citations and provenance:** When making source-based claims, show appropriate source names/links and any caveats. A source's authority does not extend beyond what it actually says. Do not claim a source was read if only an unresolved hint was available.

## Destiny-specific reasoning safeguards

### Gear/builds

- In current Armor 3.0 use the game's actual stat names. Do not recommend raising deprecated **Recovery** as a currently available armor stat or attribute grenade cooldown to it. Do not treat an unreturned stat as low.
- Treat rarity, archetype, slots, weapon perks, Exotic restrictions, current subclass, and items as data that must be resolved, not inferred from their names alone.
- Preserve an equipped Exotic by default when suggesting a small optimization; preserve the user's favored weapon (e.g. Rose) unless they explicitly ask to change it or a supported trade-off is explained.
- Before replacing an equipped item, inspect the existing item and search **owned, unequipped alternatives** (`find_build_alternatives` / `search_inventory` as appropriate). Name the specific verified candidate and its meaningful trade-off; if none, say “no justified swap established.”
- Kills with another weapon do **not** stack or maintain **Rose's Rampage**. Validate perk triggers for the actual weapon; never assume a second Hand Cannon improves range coverage just because both are Hand Cannons.
- Compare across *actual current* subclass, armor, weapons, perks, and content goals. Avoid general “best” claims that ignore the player's desired activity.

### Activities, loot and progression

- “Solo” may mean **alone without a premade fireteam** or **strictly without teammates**. Do not infer matchmade activity is soloable, nor automatically exclude all matchmade activities without understanding context. If uncertainty matters, offer a bounded recommendation or concise assumption.
- Distinguish **Haunted Altars of Sorrow** (seasonal) from **Altars of Sorrow** (Moon patrol). Do not transfer the ordinary “Waves of Altars of Sorrow” Triumph to the Haunted activity. Similarly distinguish activity variants, rewards, and record scopes.
- Never invent duration, reward quality, comparative drop frequency, guaranteed drops, live weekly activity availability, or source-specific reward chances. “Generally rewarding” is not proof of “best loot”; provide a grounded reason and uncertainty.
- Preserve prior user constraints across follow-ups. Recommend one primary action and, if useful, one short backup. Do not ask the player to choose when a safe and useful recommendation is possible.
- Prismatic **Facets** are not Solar **Embers**. Equipped Solar fragments (`Ember of Ashes`, `Ember of Empyrean`) do not establish the player's unlocked Prismatic Facets.
- Guide-backed Prismatic paths include Micah-10's *Alone in the Dark* / Cysts for **Light Vestiges**, Pale Heart timed Darkness chest cleanses for **Darkness Vestiges**, five same-kind Vestiges to a Memory, and certain named/campaign chest Facets. Avoid declaring one route universally fastest without knowing which Facet is missing.
- Do not use `get_title_progress` or `analyze_current_build` as stand-ins for a nonexistent complete Facet unlock inventory. Tool calls must answer the question, not merely fulfill an evaluation property.

### Capability and action boundaries

- Do not assert that a read-only OAuth tool changes game state, equips items, transfers items, dismantles gear, or manages inventory.
- If asked for unsupported account modification, say it cannot be performed here; offer accurate instructions or a supported external handoff without implying the action succeeded.
- Protect player-specific data, OAuth secrets, and logs; never put tokens in tests, comments, committed fixtures, or exported traces.

## Validator and evaluation design

- Validators should detect **unsafe or unsupported claims**, not punish negation (“does not progress the record”), qualified uncertainty, player quotes, or explanations distinguishing two activities.
- Validate that a *correction* truly removed the issue. If the correction still fails, either try evidence-based repair within the configured budget or return the **best supported useful answer**; do not erase the entire answer by default.
- Surgical trimming must leave a grammatical answer. Never begin a final answer with a dangling “Why:”, “Backup:”, or bare fragment after deleting its headline.
- Do not overfit regex to saved turns; include counterexamples/negative assertions and test unusual phrasings. A regex passing on fixtures is not proof of live-model reliability.
- Do not require a specific tool if it cannot supply the requested data. Evaluate claim correctness, applicability, constraints, and usefulness rather than tool-call mimicry.
- Inspect each model run's **final answer**, intermediate drafts, validator decisions, tool arguments/results, use of sources, and latency/tokens; do not rely on the aggregate score alone.
- Keep the **held-out evaluation set separate** from implementation tuning. When a held-out failure becomes a regression test, replace or replenish the held-out case so future measurement stays independent.

## Commands: Windows / PowerShell

The developer has worked from `C:\Users\bella\Projects\DestinyCopilot\backend`. Paths and CLI options should be verified against the actual checked-out repository. **Avoid bare `python`** if it points to a system interpreter missing `pydantic`.

```powershell
# From the repository root, inspect state before any edits:
git status --short --branch

# Backend (from ./backend)
cd backend
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .

# Frontend (from repo root, if frontend/ is the package directory):
cd ..\frontend
npm test
```

If the venv is elsewhere, use its actual interpreter; verify `python -m pytest` and Ruff availability instead of inventing environment setup. `ruff format --check` previously flagged three files; **do not label them baseline until names and status have been checked**. Formatting deviations are not the same as clean lint.

### Evaluation commands

```powershell
# From ./backend, using the configured development API key:
.\.venv\Scripts\python.exe -m evals.runner --help

# Ten known playtest scenarios, one real-model run, judge disabled:
.\.venv\Scripts\python.exe -m evals.runner --subset playtest --no-judge --concurrency 1

# Only when explicitly approved to spend more calls / assess variability:
.\.venv\Scripts\python.exe -m evals.runner --subset playtest --repeat 3
```

The `--subset playtest --no-judge --concurrency 1` invocation was used successfully in prior full evaluations. **Do not invent the command or subset name for the future held-out tests:** inspect `evals.runner --help` and the actual scenario registration first. Confirm `--repeat` defaults and judge/live-provider flags before a multi-run evaluation.

**Cost/scope precautions:** a test suite is free/offline; real-model evals spend API budget. Run paid model calls, Bungie requests, web queries, live-provider tests, or deployment only with an explicit user request/authorization. `judge=false` means deterministic scoring only; `live_provider=false` and simulated profiles mean the eval does **not** test the actual current Guardian account or real-time availability. Some model responses may still invoke web research when enabled; inspect trace flags.

**Results log:** record `started_at`, model, manifest snapshot, scenario IDs, score/flags (separate deterministic vs judge), tool usage, validator fallback counts, source freshness, input/output tokens, and latency. Compare with the latest canonical run in [ROADMAP.md](ROADMAP.md). A scenario score can be 16/16 and still contain a factual or usability error.

## Before/after any coding work

1. Read `ROADMAP.md`, inspect local diff and current code; verify what is already fixed. **Never assign or repeat the same repair only because a previous report was not noticed.**
2. Identify user-facing failure and evidence, ideally by the **exact run `started_at`**, scenario and turn. Determine whether root cause is tool availability, selection, planning, validator false positive/negative, stale data, prompt wording, or the eval itself.
3. Make the **smallest generalizable change**; add passing and failing tests, including semantic counterexamples where appropriate. No golden-prompt branches.
4. Run focused offline tests; then full backend tests and Ruff; frontend `npm test` when relevant. Report skips, lint/format findings, and the exact commands used.
5. Separate **offline replay passed** from **real-model behavior verified**. Only real-model runs validate the latter.
6. Update `ROADMAP.md` when milestones/status or a release gate changes. Summarize changed behavior, new risks, outstanding uncertainty, and next action.
7. Do **not** commit, push, merge, deploy, or run paid evals without explicit instruction. Do not claim the deployed version includes local uncommitted changes.

## Collaboration style

The user sets product direction and priorities; you are an engineering partner: investigate, reason, run tests, and give a candid assessment. Be concise and concrete. Favor reliable player utility over impressive-looking automated scores. Explain trade-offs when something remains unverified. Do not send the user into another fix/re-eval loop unless you can name a new, material, grounded issue.
