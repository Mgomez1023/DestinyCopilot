# Destiny Copilot — Product & Engineering Roadmap

> Source of truth for milestones, release gates, and remaining work. Keep this file up to date whenever a milestone changes. See [CLAUDE.md](CLAUDE.md) for evidence rules, coding conventions, and commands.

**Last updated:** 2026-10-22 (America/Chicago: evening of 2026-10-21)  
**Working branch at handoff:** `feat/guardian-product-sprint`  
**Release state:** Pre-beta, undergoing evaluation; latest feature work reported **uncommitted, unpushed, and undeployed**. Existing Railway/Vercel infrastructure does not mean this branch is live.  
**Next action:** Create and execute 15 **unseen** evaluation scenarios; triage serious player-facing failures before a controlled beta.

## North star

A useful Destiny 2 copilot that combines a Guardian's actual account state with trustworthy game knowledge to answer questions and recommend **one clear next action**. Prefer a short, accurate answer with verifiable evidence over a confident, elaborate guess. The experience should feel account-aware, helpful to new and experienced players, and honest about what cannot be inspected or changed.

### Working product scope

- **Frontend:** React + TypeScript + Vite; **backend:** FastAPI/Python.
- **Integrations:** Bungie OAuth/API, Destiny Manifest, normalized account snapshots, curated Destiny guides, core-knowledge corpus, external/current-source research when available.
- **App hosting:** Railway backend / Vercel frontend have been used; verify the deployed revision independently.
- **Present capabilities:** character/build analysis, equipped loadouts, quests/progression, activities, inventory lookup, item details/sources, guided answers. Tool capabilities are **not** interchangeable; a tool must actually return evidence for the claim.
- **Not established:** authoritative per-Facet unlocked/locked inventory, executing vault moves/loadout changes, guaranteed reward rankings, or up-to-the-minute availability without current evidence.
- **Later ideas, not current commitments:** DIM links and handoffs, vault cleanup, wishlist awareness, one-tap loadouts if legitimately supported, detailed build explanations.

## Milestone board

| ID | Milestone | Status | Evidence / exit requirement |
| --- | --- | --- | --- |
| M0 | Core app, account context, Destiny Manifest | **Implemented; maintain** | FastAPI/React app, OAuth and account tools exist. Confirm runtime/deployment health before beta. |
| M1 | Knowledge Engine + retrieval integration | **Implemented; improve as needed** | Real Manifest, core entities, guide search/full-guide fetching; the Prismatic acquisition guide can be retrieved. |
| M2 | Evidence-first tools, validators, offline regressions | **Implemented; not infallible** | Vault lookup and source tools work in eval; validator/repair framework has extensive tests. Guard against false positives and false negatives. |
| M3 | Full ten-scenario real-model qualification | **Completed for baseline, not an unconditional pass** | Latest: 9/10 without deterministic serious flags, mean 15.6/16. Manual review still found important deficiencies below. |
| M4 | New, held-out player scenarios | **NEXT / NOT STARTED at handoff** | Create 15 brand-new scenarios, **at least five multi-turn**; run once with real model only after offline validation and authorized API budget. |
| M5 | Focused beta hardening | **Pending** | Triage held-out failures; verify auth/privacy, errors, source links, latency/cost, real-account flows, and unsupported-action boundaries. |
| M6 | Small, controlled beta | **Pending** | Deploy an explicitly approved revision, test on consenting real accounts, collect qualitative reports and corrections, monitor spend and reliability, maintain rollback plan. |

**Status vocabulary:** `NEXT`, `IN PROGRESS`, `BLOCKED`, `COMPLETED`, `PENDING`. “Passed deterministic eval” and “human-reviewed product ready” are separate claims.

## Most recent confirmed evaluation snapshot

**Full playtest run:** `started_at=2026-10-22T03:46:12+00:00` (the exported files in the conversation were `results(7).json` and `summary(7).md`). This is the run identifier to use when comparing future results; do not assume the files have these exact names in the local repo.

| Metric | Latest | Previous full run |
| --- | ---: | ---: |
| Scenarios with no *deterministic serious flags* | **9/10** | 7/10 |
| Mean *deterministic* score | **15.6/16** | 14.1/16 |
| Total player turns | 14 | 14 |
| Mean latency | **9.72 s** | 9.84 s |
| Approx. model requests / turn | **2.36** | 2.14 |
| Input tokens / turn | **31,327** | 27,288 |
| Serious deterministic flags | **1 (`ungrounded_advice`)** | 4 flags across 3 scenarios |

Run conditions: `gpt-5-mini`, `judge=false`, `live_provider=false`, simulated Guardian accounts; web research **did** occur for the current-source case. These are neither a judge-reviewed score nor proof of real-account production readiness. Inspect the actual final answers and tool traces.

**Offline test status at the last code-change report (before this real-model run):** backend **1,581 passed, 1 skipped**, Ruff lint clean, frontend **4/4**. `ruff format --check` reported 3 files, presumed baseline but **not verified**; inspect names before calling them pre-existing. No further local code changes have been reported after that test result.

## Manual-review findings / open quality register

These are **observations from the latest real-model run**, not instructions to automatically launch another repair sprint. Triage by real impact; prefer generalizable improvements that stand up to unseen tests.

| ID | Area | Evidence from latest run | Priority / next decision |
| --- | --- | --- | --- |
| Q1 | **Prismatic user premise** | User says *“I've finished The Final Shape”*; assistant refuses to explain how to unlock Facets because account completion is unknown, despite successfully retrieving a relevant guide. | **High**: respect a user's stated premise when giving conditional how-to guidance; do not pretend it was account-verified. |
| Q2 | **Prismatic follow-up routing** | Follow-up used `analyze_current_build` even though it cannot establish all unlocked Prismatic Facets, plus guide retrieval; 153-word answer exceeded the targeted 120-word check. | **Medium**: avoid irrelevant account tool calls, concise next step; explicitly state unavailable exact Facet inventory. |
| Q3 | **Build recommendation grounding** | `find_build_alternatives` was invoked, but final answer still suggested swapping Calus Mini-Tool for an unnamed long-range Energy weapon without identifying a verified owned, superior candidate. | **High**: compare actual candidates or say no justified swap has been established. Do not promote a speculative category to “single biggest improvement.” |
| Q4 | **Solo loot planning** | A planning turn used `planning_safe_answer`; follow-up promoted Haunted Altars with comparative loot / 30-minute solo claims without current evidence, while acknowledging that evidence was unavailable. | **High**: distinguish solo/no premade team vs strictly alone; preserve constraints and avoid contradictory claims; prevent blanket fallback where safe choices exist. |
| Q5 | **Sunshot detector** | `exotic_replacement_grounded` flagged the answer “Sunshot's barrel is fixed; to get another barrel effect use a different hand cannon” as an unsupported replacement. | **Medium / likely evaluator false positive**: confirm no actual owned-item recommendation was made before changing the detector. |
| Q6 | **Live-source citations** | The Coil / Mint Retrograde answer ran web research but the final answer did not preserve a clickable source and date; a current-source request warrants visible source provenance. | **Medium**: make citations survive answer-shortening, with freshness and certainty clearly stated. |
| Q7 | **Broader answer quality** | Successful deterministic properties missed questionable details such as generic item-transfer UI instructions or unsupported item-source characterizations. | **Medium**: apply player-style manual review and held-out eval; avoid tuning to fixture wording. |

**Known wins to preserve:** vault vs equipped hand-cannon answer correctly identifies vaulted Fatebringer; item lookup and source retrieval; fixed Exotic-barrel explanation; bounded honesty when catalyst completion cannot be read; legitimate full-guide retrieval; improved mean latency.

## Immediate next deliverable: held-out evaluation

1. **Author 15 unseen scenarios** in a separate evaluation subset/fixture, distinct from existing playtest scenarios and replayed drafts. At least five should be multi-turn. Include informal player wording.
2. Cover account lookup, gear/build and perk mechanics, quest or Prismatic guidance, activity selection, source freshness, unsupported actions, unavailable data, and changing constraints. Include cases where the best answer is “I cannot determine that from this data.”
3. Evaluate **behavior**, not preferred vocabulary or arbitrary tool calls. Ensure expected tools can actually provide the information the property demands.
4. **Offline first:** validate fixture/profile integrity, inspect scenario goals, run existing backend/frontend checks. Do not alter implementation while constructing the held-out set.
5. **Real-model run once** with an approved development key; save run metadata, raw responses, tool traces, flags, token usage, and latency. Perform human review of each answer.
6. Only then label failures `BLOCKER`, `IMPORTANT`, `POLISH`, or `TEST_FALSE_POSITIVE`. Fix blockers using a new regression that generalizes beyond the triggering wording; no golden-prompt branching.

### Beta-release gates (qualitative, not just a target score)

- No known severe hallucinations about owned gear, subclass mechanics, loot guarantees, progression, or executable account actions in a reviewed representative sample.
- No silent use of stale Manifest strings to substantiate “current” facts. Evidence links shown where source verification is requested.
- Player constraints carried through follow-ups; recommendations are actionable or limitations are explicit and useful.
- All required tests/lints pass or deviations are documented and understood.
- No OAuth tokens, user data, or private profile details disclosed in logs/eval exports; appropriate error/timeout and rate-limit handling.
- Beta access, usage budget, monitoring, deployment revision, and rollback explicitly approved.

## Decision log / ways of working

- **2026-10-17–19:** Manifest normalization, Core Knowledge, guide retrieval, and tool orchestration expanded; extensive backend regression coverage established.
- **2026-10-22 02:50 UTC:** Full 10-scenario eval: 7/10, 14.1/16; grounded serious failures in planning/build/Prismatic.
- **2026-10-22 03:46 UTC:** Fresh full eval after stabilization: 9/10, 15.6/16 deterministic; manual review showed remaining recommendation-quality gaps.
- **Current decision:** **Do not automatically loop on the same golden prompts.** Create held-out scenarios next, then fix *generalizable beta-blocking* failures. No paid evals, commits, pushes, merges, or deployment without explicit instruction.

## Handoff ritual (any supervisor or coding agent)

1. Read **this file** and **[CLAUDE.md](CLAUDE.md)** before proposing work.
2. Verify repository facts (`git status`, current branch, actual tools and CLI `--help`); treat this document as a snapshot, not a substitute for inspecting code.
3. Cite the **run `started_at`** and actual result path when discussing a failure. Separate code fixes, offline replays, real-model outcomes, and beta outcomes.
4. Maintain a one-paragraph session update: changed behavior, tests, remaining limitation, and next milestone. Update this file when status or scope materially changes.
5. Respect the user's collaboration preference: user sets direction; assistants research, debug, test, and critique. Prioritize durable correctness and a good player experience over score-chasing.
