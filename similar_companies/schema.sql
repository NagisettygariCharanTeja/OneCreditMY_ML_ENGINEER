-- Schema for the similar-companies service.
-- Safe to run on every startup: every statement is idempotent.

CREATE EXTENSION IF NOT EXISTS vector;

-- One row per company, always holding its latest known state.
CREATE TABLE IF NOT EXISTS companies (
    registration_number text PRIMARY KEY,                -- normalized, e.g. SYN-001550
    status              text NOT NULL
                        CHECK (status IN ('active', 'struck_off')),
    updated_at          timestamptz NOT NULL,            -- source time in UTC; newer-wins compares this
    company_name        text,
    nature_of_business  text,
    state               text,
    industry            text,
    embedded_text       text,                            -- exact text sent to the model
    embedding           vector(384),
    embedding_model     text,                            -- which model produced the vector
    -- "No lost companies": every active company must be displayable and searchable.
    CONSTRAINT active_company_is_complete
        CHECK (status = 'struck_off' OR (company_name IS NOT NULL AND embedding IS NOT NULL))
);

-- Quarantine: one row per distinct message the consumer could not use.
CREATE TABLE IF NOT EXISTS rejected_events (
    id          bigserial PRIMARY KEY,
    message_id  text UNIQUE,                             -- redelivery of the same bad message is stored once
    raw_body    text NOT NULL,                           -- text, not jsonb: the body may not be valid JSON
    reason      text NOT NULL,                           -- short code, e.g. invalid_json
    received_at timestamptz NOT NULL DEFAULT now()
);

-- No vector index on purpose: at ~2,000 rows an exact scan takes milliseconds and
-- always returns the true nearest neighbours. See DESIGN.md (10x volume) for HNSW.
