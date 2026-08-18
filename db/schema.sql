-- db/schema.sql
--
-- Postgres DDL for the Transcript Analysis Automation pipeline.
-- Tables mirror the Pydantic models in models/ exactly.
-- Run once against a fresh database to initialise the schema.
--
-- Tables:
--   run_specs        — one row per RunSpec version (central pipeline record)
--   judge_feedback   — per-row JudgeAgent scores tied to a run_spec version
--   user_feedback    — per-row analyst comments tied to a run_spec version
--   revision_briefs  — aggregated feedback handed back to PromptGeneratorAgent

CREATE EXTENSION IF NOT EXISTS "pgcrypto";  -- for gen_random_uuid()

-- -------------------------------------------------------------------------
-- run_specs
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS run_specs (
    id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    version             INTEGER     NOT NULL DEFAULT 1,
    parent_version_id   UUID        REFERENCES run_specs(id),

    question            TEXT        NOT NULL,
    lob                 TEXT        NOT NULL,

    prompt_template     TEXT        NOT NULL DEFAULT '',
    schema_definition   JSONB       NOT NULL DEFAULT '{}',
    tools               JSONB       NOT NULL DEFAULT '[]',

    status              TEXT        NOT NULL DEFAULT 'draft_prompt',

    dataset_parameters  JSONB       NOT NULL DEFAULT '{}',
    validation_set      TEXT,
    scores              JSONB       NOT NULL DEFAULT '{}',

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- -------------------------------------------------------------------------
-- judge_feedback
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS judge_feedback (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    run_spec_id     UUID        NOT NULL REFERENCES run_specs(id) ON DELETE CASCADE,
    row_id          TEXT        NOT NULL,
    score           NUMERIC(4,3) NOT NULL CHECK (score >= 0 AND score <= 1),
    reasoning       TEXT        NOT NULL,
    flags           JSONB       NOT NULL DEFAULT '[]',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- -------------------------------------------------------------------------
-- user_feedback
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS user_feedback (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    run_spec_id     UUID        NOT NULL REFERENCES run_specs(id) ON DELETE CASCADE,
    row_id          TEXT        NOT NULL,
    comment         TEXT        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- -------------------------------------------------------------------------
-- revision_briefs
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS revision_briefs (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    run_spec_id     UUID        NOT NULL REFERENCES run_specs(id) ON DELETE CASCADE,
    judge_feedback  JSONB       NOT NULL DEFAULT '[]',
    user_feedback   JSONB       NOT NULL DEFAULT '[]',
    summary         TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
