# AI Pipeline Automation System — Implementation Plan

## Guiding Principles

Every design decision in this system is evaluated against five principles:

- **Scalability** — components scale independently; no single bottleneck owns the whole pipeline.
- **Modularity** — each agent, runner, and storage layer is swappable without touching the rest of the system.
- **Continuous learning** — agents carry in-context learning through structured feedback (judge scores, user corrections, revision briefs) so each iteration is informed by everything that came before it.
- **Resilience** — failures are isolated, retried, and logged; no silent data loss; every state transition is durable.
- **Future-proof / open standards** — interfaces are built on open standards (OpenAI tool-call spec, JSON Schema, OpenAPI, OAuth2) so the underlying model, storage, or cloud provider can be swapped without rewriting business logic.

---

## Goal

Automate the process by which analysts turn a plain-language question into a validated, structured analysis of call transcripts. The system generalises an existing pattern (Jinja2 prompt templates + Pydantic structured outputs + Databricks/Azure OpenAI batch processing) so any analyst can define a new question without a human building the extraction pipeline by hand each time.

Two human-approval gates and two automated judge passes catch a bad prompt or schema on a small sample before it ever runs against the full transcript volume.

---

## End-to-end flow

1. **Generate prompt.** `PromptGeneratorAgent` takes the analyst's question and any prior `RevisionBrief` (which carries all previous judge scores, user corrections, and failure reasoning) and drafts a Jinja2 extraction prompt. The revision brief is the in-context learning signal — the agent sees exactly what went wrong before and why.
2. **User approves prompt.** Human gate. Flow does not advance until approved.
3. **Generate schema + tools.** `SchemaGeneratorAgent` takes the approved prompt and produces a JSON Schema output definition plus tool definitions expressed in the OpenAI tool-call format (open standard — portable across providers). Can send feedback back to step 1 if the prompt is insufficient.
4. **User approves schema + tools.** Second human gate.
5. **Run on a defined sample.** Approved prompt + schema run against a small, fixed sample of transcripts.
6. **Judge evaluates the sample, row by row.** `JudgeAgent` scores each row (fit, correctness, completeness). User can leave row-level feedback in parallel. Both signals are stored and will feed the next revision cycle if needed.
7. **User approves or rejects the sample.**
   - **Approve** → proceed to step 8.
   - **Reject** → all judge feedback + user feedback are aggregated into a `RevisionBrief` and sent back to step 1. The agent receives the full history of every revision cycle, not just the most recent one, so learning compounds across iterations.
8. **Run on the full defined set** via the existing Databricks/Azure OpenAI batch pipeline.
9. **Judge validates the full-set output.** Problems flag the run for manual review — no automatic re-run (full-batch re-runs are expensive). A human decides next steps.

---

## Component breakdown

### 1. Data model layer

Defined as Pydantic v2 models — serialisable to JSON, validated at the boundary, and independent of any storage backend (modularity).

- `RunSpec` — the central record for a pipeline run: `question`, `prompt_template` (Jinja2 str), `schema_definition` (JSON Schema dict), `tools` (OpenAI tool-call format list), `version` (int), `status` (state machine enum), `lineage` (parent version UUID — full revision history is reconstructable), `lob`, `dataset_parameters`, `validation_set` reference, `scores`.
- `JudgeFeedback` — per-row: `score` (0–1), `reasoning`, `flags` (typed issue list). Stored against a specific `RunSpec` version so scores are comparable across revisions.
- `UserFeedback` — per-row: free-text correction tied to a row ID and `RunSpec` version.
- `RevisionBrief` — aggregates *all* `JudgeFeedback` and `UserFeedback` from a rejected cycle into a single object. This is the primary in-context learning payload handed back to `PromptGeneratorAgent`. It accumulates across revision cycles so the agent always has the full picture.

### 2. State machine / orchestrator

```
draft_prompt
  -> prompt_approved
  -> draft_schema
  -> schema_approved
  -> sample_running
  -> sample_judged
  -> sample_approved | sample_rejected
       (approved)  -> full_running -> full_judged -> complete | flagged_for_review
       (rejected)  -> draft_prompt  [carries RevisionBrief]
```

Design notes:
- Explicit transitions map — no heavyweight framework needed. Simple, auditable, easy to extend with new states.
- Every transition is persisted atomically before the next step begins (resilience — a crash mid-pipeline resumes from the last committed state, not from scratch).
- The `RevisionBrief` is a required argument on the `sample_rejected → draft_prompt` transition, enforced at the state machine level — you cannot start a revision cycle without the learning signal.
- State history is append-only; old versions are never deleted (auditability + lineage).

### 3. Agent layer

Agents are plain Python classes with a single `run()` method. They have no dependency on Celery, FastAPI, or any specific LLM provider — they accept a client interface and call it (modularity + future-proof).

- `PromptGeneratorAgent` — inputs: `question`, optional `metadata`, optional `answers`, optional `RevisionBrief`. The revision brief is injected directly into the system prompt so the model sees every prior failure and correction as in-context examples. Output: Jinja2 prompt template or clarifying questions.
- `SchemaGeneratorAgent` — input: approved prompt. Output: JSON Schema dict + OpenAI-format tool definitions. Uses open standards so the schema works with any OpenAI-compatible endpoint.
- `JudgeAgent` — input: `(question, prompt, row_output)`. Output: `JudgeFeedback`. Stateless and parallelisable — can run N rows concurrently without coordination (scalability).

**In-context learning pattern:** Each agent receives a structured context object rather than raw strings. The `RevisionBrief` is designed to grow richer with each cycle — it carries not just the latest failure but the full chain of what was tried, what the judge said, and what users corrected. Agents are prompted to explicitly reference prior attempts when generating new outputs.

### 4. Execution layer

- **Sample runner** — synchronous or small-batch, sized for the sample. Runs inside a Celery task so it doesn't block the API.
- **Full runner** — submits to the existing Databricks/Azure OpenAI batch pipeline via Delta control tables. Celery Beat polls for completion. The runner is a thin adapter — the batch infrastructure is unchanged (modularity).
- Both runners write results incrementally so a partial failure doesn't lose completed rows (resilience).

### 5. Persistence / versioning

- **Postgres** for orchestration metadata (`run_specs`, `judge_feedback`, `user_feedback`, `revision_briefs`). Chosen for ACID guarantees on state transitions and rich query support for revision history.
- **Storage backend is abstracted** behind a repository interface. A CSV backend exists for local dev/testing — swap to Postgres by setting `STORAGE_BACKEND=postgres`. Adding a new backend (e.g. DynamoDB, SQLite) requires only a new repository class, nothing else changes (modularity).
- Every `RunSpec` write is an upsert keyed on `(id, version)` — versions are immutable once written. The full revision lineage is always queryable.
- Raw transcript-scale output stays in Databricks Delta; Postgres holds pointers, not the raw data (scalability — don't move data you don't need to move).

### 6. Review interface

Needs to support: prompt approval (step 2), schema/tools approval (step 4), row-level judge output + user feedback (step 6), sample approve/reject (step 7), flagged-run dashboard (step 9).

Built as a React + Vite + shadcn/ui frontend talking to the FastAPI backend over a versioned REST API (OpenAPI spec auto-generated). The API is the contract — the frontend is replaceable (modularity + open standards).

---

## Tech stack

| Layer | Tool | Notes |
|---|---|---|
| API | **FastAPI** | OpenAPI spec auto-generated. OAuth2 for auth. Drives state machine transitions. |
| Background jobs | **Celery + Redis** | Agent calls, sample runner, full-run submission + polling. Redis as broker is swappable (RabbitMQ, SQS). |
| Database | **Postgres** | ACID state transitions. Repository pattern abstracts the backend. |
| Frontend | **React + Vite + shadcn/ui** | Talks to FastAPI over the OpenAPI contract. |
| LLM client | **OpenAI-compatible interface** | Any endpoint that speaks the OpenAI tool-call spec works — Azure OpenAI today, any other provider tomorrow. |
| Infra | **Docker + docker-compose** | One container per concern. Same images local → prod. |

---

## Build order

| Step | What | Status |
|---|---|---|
| 1 | Data models (`RunSpec`, `JudgeFeedback`, `UserFeedback`, `RevisionBrief`) | ⬜ |
| 2 | Postgres schema + repository interface + CSV backend | ⬜ |
| 3 | State machine (transitions map, durable persistence, `RevisionBrief` enforcement) | ⬜ |
| 4 | Agent layer (`PromptGeneratorAgent` with revision brief injection, `SchemaGeneratorAgent`, `JudgeAgent`) | ⬜ |
| 5 | Celery wiring (agent tasks, sample runner task, full-run submit + poll tasks) | ⬜ |
| 6 | FastAPI layer (endpoints, state transitions, OpenAPI spec) | ⬜ |
| 7 | Full runner integration (Databricks/Azure OpenAI batch pipeline adapter) | ⬜ |
| 8 | React frontend (approval screens, sample review grid, flagged-run dashboard) | ⬜ |
| 9 | Docker (containerise, docker-compose, promote to prod) | ⬜ |

---

## Resolved decisions

- **Edits require re-review.** Editing the prompt or schema does not auto-advance — it must go through the approval gate again.
- **Schema agent can push back to prompt agent.** If `SchemaGeneratorAgent` determines the prompt won't produce a usable schema, it returns `prompt_feedback` and the flow routes back to step 1 before proceeding.
- **Revision briefs accumulate.** Each rejected cycle appends to the brief rather than replacing it. The agent always sees the full history of what was tried and why it failed — this is the core continuous learning mechanism.
- **Open tool-call standard.** Tool definitions are expressed in the OpenAI tool-call JSON format throughout. This is the de-facto open standard for agentic tool use and is supported by every major provider.
- **Repository pattern is mandatory.** No component outside `db/` is allowed to write SQL or touch a file path directly. All storage goes through the repository interface so backends are truly swappable.
- **Agents are provider-agnostic.** Agents accept a client object that implements a minimal interface (`chat()`, `run_agent()`). Swapping from Azure OpenAI to another provider means swapping the client, not the agents.
