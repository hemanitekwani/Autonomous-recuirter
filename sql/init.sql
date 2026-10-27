-- =====================================================================
-- Autonomous AI Recruiter & Auto-Scheduler - PostgreSQL schema
--
-- Business tables (source of truth): jobs, applications, interviews
-- Auxiliary table: audit_log (history of HR overrides / corrections;
--   a single "latest value" column cannot preserve who changed what).
--
-- The script is idempotent: safe to run more than once.
-- Run it via:  python -m scripts.init_db
-- =====================================================================

-- ---------------------------------------------------------------------
-- updated_at helper
-- ---------------------------------------------------------------------
CREATE OR REPLACE FUNCTION set_updated_at() RETURNS trigger AS $$
BEGIN
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ---------------------------------------------------------------------
-- jobs
-- skills = {"mandatory": [SkillRequirement...], "preferred": [...]}
-- (an OBJECT, not an array - validated again by Pydantic before insert)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS jobs (
    job_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    title VARCHAR(200) NOT NULL
        CHECK (length(btrim(title)) > 0),
    min_experience_years INTEGER NOT NULL
        CHECK (min_experience_years BETWEEN 0 AND 50),
    skills JSONB NOT NULL DEFAULT '{"mandatory": [], "preferred": []}'::jsonb
        CHECK (
            jsonb_typeof(skills) = 'object'
            AND jsonb_typeof(skills -> 'mandatory') = 'array'
            AND jsonb_typeof(skills -> 'preferred') = 'array'
        ),
    interview_duration_minutes INTEGER NOT NULL DEFAULT 45
        CHECK (interview_duration_minutes BETWEEN 15 AND 240),
    status VARCHAR(20) NOT NULL DEFAULT 'OPEN'
        CHECK (status IN ('OPEN', 'CLOSED')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_jobs_status ON jobs (status);

-- ---------------------------------------------------------------------
-- applications
--   name / email      : entered by the candidate (authoritative contact)
--   skills            : validated, normalized skill objects (array)
--   experience        : COMPLETED whole years (floor of months / 12);
--                       NULL = could not be calculated reliably
--   profile           : validated parsed profile incl. evidence,
--                       employment history, experience calculation
--   screening_status  : machine screening result
--   pipeline_status   : HR / interview stage of the candidate
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS applications (
    application_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    job_id INTEGER NOT NULL REFERENCES jobs (job_id),
    email VARCHAR(254) NOT NULL,
    name VARCHAR(200) NOT NULL,
    skills JSONB NOT NULL DEFAULT '[]'::jsonb
        CHECK (jsonb_typeof(skills) = 'array'),
    experience INTEGER
        CHECK (experience IS NULL OR experience >= 0),
    resume_path TEXT,
    profile JSONB NOT NULL DEFAULT '{}'::jsonb
        CHECK (jsonb_typeof(profile) = 'object'),
    screening_status VARCHAR(30) NOT NULL DEFAULT 'PENDING'
        CHECK (
            screening_status IN (
                'PENDING', 'PROCESSING', 'ELIGIBLE',
                'NOT_ELIGIBLE', 'REVIEW_REQUIRED', 'ERROR'
            )
        ),
    screening_result JSONB NOT NULL DEFAULT '{}'::jsonb
        CHECK (jsonb_typeof(screening_result) = 'object'),
    -- allowed values: see ck_applications_pipeline_status below
    pipeline_status VARCHAR(30) NOT NULL DEFAULT 'SCREENING',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- target of the composite FK from interviews
    CONSTRAINT uq_applications_id_job UNIQUE (application_id, job_id)
);

-- Pipeline stages. SUPERSEDED = an older attempt replaced by a newer application
-- from the same candidate for the same job. (Drop + add keeps this idempotent and
-- migrates databases created with the earlier inline check.)
ALTER TABLE applications DROP CONSTRAINT IF EXISTS applications_pipeline_status_check;
ALTER TABLE applications DROP CONSTRAINT IF EXISTS ck_applications_pipeline_status;
ALTER TABLE applications ADD CONSTRAINT ck_applications_pipeline_status CHECK (
    pipeline_status IN (
        'SCREENING', 'APPROVED', 'INTERVIEW_SCHEDULED',
        'INTERVIEWED', 'SELECTED', 'REJECTED', 'SUPERSEDED'
    )
);

-- Re-application policy: every attempt is its own application (new ID, history kept),
-- but only ONE ACTIVE application per (job, case-insensitive email).
-- Closed attempts (REJECTED / SUPERSEDED) do not block a new one.
DROP INDEX IF EXISTS ux_applications_job_email;
CREATE UNIQUE INDEX IF NOT EXISTS ux_applications_job_email_active
    ON applications (job_id, lower(email))
    WHERE pipeline_status NOT IN ('REJECTED', 'SUPERSEDED');
CREATE INDEX IF NOT EXISTS ix_applications_job_email
    ON applications (job_id, lower(email), created_at);
CREATE INDEX IF NOT EXISTS ix_applications_job_screening
    ON applications (job_id, screening_status);
CREATE INDEX IF NOT EXISTS ix_applications_pipeline
    ON applications (pipeline_status);

DROP TRIGGER IF EXISTS trg_applications_updated_at ON applications;
CREATE TRIGGER trg_applications_updated_at
    BEFORE UPDATE ON applications
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------------------
-- interviews
-- TIMESTAMPTZ start/end (not separate DATE + TIME): unambiguous across
-- time zones / DST / midnight and usable in a range exclusion constraint.
-- `timezone` is kept for displaying the slot in the panel's local time.
--
-- Conflict protection: there is a single interview panel (configured
-- calendars), so no two ACTIVE interviews may overlap in time. The
-- database enforces this even under concurrent booking attempts.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS interviews (
    interview_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    job_id INTEGER NOT NULL REFERENCES jobs (job_id),
    application_id INTEGER NOT NULL,
    starts_at TIMESTAMPTZ NOT NULL,
    ends_at TIMESTAMPTZ NOT NULL,
    timezone VARCHAR(100) NOT NULL DEFAULT 'Asia/Kolkata',
    status VARCHAR(30) NOT NULL DEFAULT 'PENDING_CONFIRMATION'
        CHECK (
            status IN (
                'PROPOSED', 'PENDING_CONFIRMATION', 'CONFIRMED',
                'CANCELLED', 'COMPLETED'
            )
        ),
    -- Chosen BEFORE calling Google so retries reuse the same event id
    -- (Google rejects a duplicate id -> no duplicate calendar events).
    google_event_id TEXT UNIQUE,
    meeting_link TEXT,
    email_status VARCHAR(20) NOT NULL DEFAULT 'NOT_SENT'
        CHECK (email_status IN ('NOT_SENT', 'SENDING', 'SENT', 'FAILED')),
    email_attempts INTEGER NOT NULL DEFAULT 0 CHECK (email_attempts >= 0),
    email_sent_at TIMESTAMPTZ,
    last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ck_interviews_time_order CHECK (ends_at > starts_at),
    -- interview.job_id must equal the job of the referenced application
    CONSTRAINT fk_interviews_application_job
        FOREIGN KEY (application_id, job_id)
        REFERENCES applications (application_id, job_id),
    -- no two active interviews overlap (plain GiST on a range; no extension needed)
    CONSTRAINT ex_interviews_no_overlap
        EXCLUDE USING gist (tstzrange(starts_at, ends_at, '[)') WITH &&)
        WHERE (status IN ('PENDING_CONFIRMATION', 'CONFIRMED'))
);

-- at most one active interview per application
CREATE UNIQUE INDEX IF NOT EXISTS ux_interviews_one_active
    ON interviews (application_id)
    WHERE status IN ('PENDING_CONFIRMATION', 'CONFIRMED');
CREATE INDEX IF NOT EXISTS ix_interviews_status_start
    ON interviews (status, starts_at);
CREATE INDEX IF NOT EXISTS ix_interviews_job ON interviews (job_id);

DROP TRIGGER IF EXISTS trg_interviews_updated_at ON interviews;
CREATE TRIGGER trg_interviews_updated_at
    BEFORE UPDATE ON interviews
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------------------
-- audit_log (auxiliary)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS audit_log (
    audit_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    entity_type VARCHAR(30) NOT NULL,
    entity_id INTEGER NOT NULL,
    action VARCHAR(60) NOT NULL,
    actor VARCHAR(100) NOT NULL,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_audit_entity ON audit_log (entity_type, entity_id);
