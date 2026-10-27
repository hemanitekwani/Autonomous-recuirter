# Autonomous AI Recruiter & Auto-Scheduler

A Streamlit + PostgreSQL application that automates the first stage of hiring:

1. HR posts a job with hard constraints: minimum experience, mandatory skills and preferred skills.
2. A candidate uploads a PDF or DOCX resume.
3. **Agent 1 (two LLM calls)** extracts facts, and Python normalizes the skills. The second LLM call validates the result, then Python grounds every skill in the resume text.
4. **Python** calculates the authoritative experience from employment dates.
5. A **deterministic SQL/Python engine** returns ELIGIBLE, NOT_ELIGIBLE or REVIEW_REQUIRED, with an explanation for each requirement.
6. HR reviews and corrects uncertain cases, then approves.
7. **Agent 2** finds real panel availability in Google Calendar and books the interview without conflicts. It sends the invitation only after the booking is confirmed.

Everything runs in **mock mode** without any paid API or Google/SMTP credentials. Only PostgreSQL is required.

---

## 1. Architecture

```
 Streamlit UI (app.py)
   ├─ Candidate Portal ── upload ─► resume_extractor (validate type/size/signature, private storage)
   │                                     │
   │                        LangGraph INTAKE graph (recruitment_workflow.py)
   │   extract_resume_text → parse_resume (LLM call 1) → normalize_skills (Python)
   │   → validate_profile (LLM call 2 + Python guard-rails) → calculate_experience (Python)
   │   → persist_application → screen_candidate (SQL + Python) → human_review | finish
   │                                     │ errors → handle_error / parse_failed_review
   │
   ├─ HR Dashboard / Candidate Review ── approve, override (audited), correct & re-screen
   │
   └─ Interview Scheduler ── LangGraph SCHEDULING graph
       find_available_slots (Google free/busy ∩ DB bookings) → confirm_booking
       (reserve in DB → create Google event → confirm in DB) → send_invitation → finish
                                      │
                         PostgreSQL: jobs · applications · interviews (+ audit_log)
```

**Design rules**

* The LLM extracts and checks facts. It never decides eligibility and never calculates experience.
* The agents exchange only Pydantic-validated JSON through the workflow state. They never exchange free-form conversation.
* The three business tables are the source of truth. Human review is a database state, not a paused graph. A review can take days, and every node is idempotent and keyed by `application_id`, so a run can always be re-invoked. This is why no LangGraph checkpointer tables are needed.
* PostgreSQL enforces correctness under concurrency. Two active interviews cannot overlap (exclusion constraint), an application can have only one active interview (partial unique index), and an interview's job must match its application's job (composite FK).

## 2. Project structure

```
app.py                         Streamlit UI (5 pages + candidate status lookup)
sql/init.sql                   Schema: jobs, applications, interviews, audit_log
src/config.py                  .env loading + validation
src/database.py                engine, per-operation session_scope, ORM mappings, init_db
src/schemas.py                 all Pydantic models / JSON contracts
src/repository.py              all DB access (parameterized), status transitions, audit
src/llm_client.py              LLM adapter (Gemini + Mock)
src/mock_llm.py                deterministic stand-ins for the LLM tasks
src/resume_extractor.py        upload validation, private storage, PDF/DOCX text
src/resume_parser_agent.py     LLM call 1 - extraction prompt + schema + repair loop
src/skill_normalizer.py        taxonomy, alias dictionary, normalization, text scan
src/data/skill_taxonomy.json   editable canonical skills + aliases
src/profile_validator_agent.py LLM call 2 - validation prompt + Python guard-rails
src/experience_calculator.py   authoritative experience policy
src/screening_service.py       deterministic SQL + decision policy
src/calendar_service.py        Google Calendar (OAuth, free/busy, events) + mock + slot generation
src/scheduling_agent.py        booking lifecycle, compensation, reconciliation, email sending
src/email_service.py           invitation template + ICS, SMTP + mock outbox
src/recruitment_workflow.py    LangGraph intake + scheduling graphs, HR corrections
scripts/init_db.py             create schema (controlled setup command)
scripts/seed_demo.py           demo jobs + sample PDF/DOCX resumes (+ --submit)
scripts/google_auth.py         one-time Google OAuth consent
scripts/maintenance.py         reconcile bookings, complete past interviews, retry emails
tests/                         pytest suite (unit + integration)
sample_data/                   demo jobs and resumes
```

## 3. Database (`sql/init.sql`)

| Table | Purpose | Key constraints |
|---|---|---|
| `jobs` | Requisitions. `skills` JSONB = `{"mandatory": [...], "preferred": [...]}` | CHECK that JSON is an object with two arrays; `interview_duration_minutes` 15-240 |
| `applications` | Candidate, validated `skills` (array), `experience` (completed years or NULL), `profile` (evidence, employment, experience calculation), `screening_status`, `screening_result`, `pipeline_status` | unique `(job_id, lower(email))` (duplicate policy); `UNIQUE(application_id, job_id)` as the composite-FK target |
| `interviews` | `starts_at` / `ends_at` TIMESTAMPTZ + `timezone`, `status`, `google_event_id`, email delivery state | composite FK `(application_id, job_id)`; **EXCLUDE USING gist (tstzrange(...) WITH &&)** for active rows; one active interview per application |
| `audit_log` (auxiliary) | Who changed what: HR overrides, corrections, bookings, emails | Needed because a "latest value" column cannot preserve history |

**Why TIMESTAMPTZ instead of DATE + TIME:** the time is unambiguous across timezones, DST and midnight, and it can be used directly in the range exclusion constraint. `timezone` is stored for display.

**`experience` integer policy:** it stores *completed* years (floor of months / 12). Job minimums are whole years, so `floor(years) >= min` is equivalent to `exact years >= min` and no precision is lost. Exact months are kept in `profile.experience`.

**Re-application policy:** every upload creates a **new application ID**, and earlier attempts are kept with their own resume and result. Only **one active** application per job and email (case-insensitive) is allowed. The partial unique index `ux_applications_job_email_active` enforces this, ignoring `REJECTED` / `SUPERSEDED` rows.
* An earlier attempt still in `SCREENING` becomes `SUPERSEDED`.
* A `REJECTED` candidate can apply again, after `REAPPLY_COOLDOWN_DAYS` (default 0).
* Re-applying is refused while an attempt is approved, interviewing or selected.
* HR sees all attempts in *Candidate Review*. Candidates see them on the status page.
* Superseded attempts are hidden from dashboard counts and lists.

**Pipeline status** (`applications.pipeline_status`):
`SCREENING → APPROVED → INTERVIEW_SCHEDULED → INTERVIEWED → SELECTED | REJECTED`. Allowed transitions are enforced in `repository.PIPELINE_TRANSITIONS`. The interview outcome is recorded on the application row.

## 4. The two-call LLM pipeline

| Step | Who | What |
|---|---|---|
| Call 1 | LLM (`resume_parser_agent.py`) | Extracts name, email, raw skill mentions + verbatim evidence, and employment periods (`YYYY-MM`), each with an employment type. The resume is wrapped in `<resume>` tags and treated as untrusted data. Output uses structured outputs (JSON schema), followed by Pydantic validation and up to `LLM_MAX_REPAIR_ATTEMPTS` repairs. |
| Normalize | Python | Casefold, trim, separator normalization, exact canonical / alias / UNKNOWN, dedupe, raw name and evidence preserved. TensorFlow ≠ PyTorch. |
| Call 2 | LLM (`profile_validator_agent.py`) | Checks evidence, judges proposed mappings, flags employment contradictions. |
| Guard-rails | Python | The LLM cannot add skills or change deterministic mappings. Contextual mappings are always REVIEW_REQUIRED. Canonical names must be in the taxonomy. Evidence must appear in the resume text, otherwise the skill is REJECTED. The LLM can only add review flags. |

A regex backstop flags instruction-like resume text (prompt injection) for review. Since the LLM never decides eligibility, injected text cannot make a candidate eligible.

## 5. Experience policy (`experience_calculator.py`)

* Completed calendar months; overlapping periods are merged.
* Current roles end in the current month (`DEFAULT_TIMEZONE`).
* Counted: full-time, part-time, contract, freelance, unknown type. Internships only count with `COUNT_INTERNSHIPS_AS_EXPERIENCE=true`. Academic work and personal projects never count.
* A missing start date, a missing end date (when the role isn't current), an end date before the start date, a future start date, or no employment history at all → **unknown (NULL)** → REVIEW_REQUIRED. Missing data is never treated as zero.

## 6. Screening policy (`screening_service.py`)

`REQUIREMENT_MATCH_SQL` joins `applications` and `jobs`, expands the JSONB requirements with `jsonb_array_elements`, and picks the best candidate skill for each requirement with a `LATERAL` subquery. `decide()` then applies these rules:

| Requirement | PASS | UNKNOWN | FAIL |
|---|---|---|---|
| experience | ≥ minimum | not calculable | < minimum |
| skill | validated skill (exact/alias, or HR-confirmed) | contextual/disputed mapping, **or** name appears in the resume but wasn't validated | not mentioned anywhere |

Overall: any mandatory FAIL → **NOT_ELIGIBLE**. Otherwise any UNKNOWN or blocking profile flag → **REVIEW_REQUIRED**. Otherwise → **ELIGIBLE**. Preferred skills are reported but never change the status. The result and status are saved in one transaction. ELIGIBLE candidates still need an HR click unless `AUTO_APPROVE_ELIGIBLE=true`.

## 7. Booking lifecycle (calendar + DB without a shared transaction)

1. Re-check live free/busy for the exact slot.
2. **Reserve:** insert the interview as `PENDING_CONFIRMATION` with a pre-generated `google_event_id`. The exclusion constraint makes this the atomic lock.
3. **Create the event** with that ID. A retry gets HTTP 409, which the code treats as "already created", so there are never duplicates. If creation fails, the reservation is cancelled to release the slot.
4. **Confirm** in the DB. If this fails, the row stays pending and *Reconcile pending bookings* (UI button or `scripts/maintenance.py`) confirms it if the event exists, or releases it.
5. **Email** only for CONFIRMED rows, with an atomic `UPDATE ... WHERE email_status IN ('NOT_SENT','FAILED') AND attempts < max` claim. A failed email never affects the booking and is retried separately. If the process crashes mid-send, the status stays `SENDING`. HR checks the outbox or SMTP log and clicks *Allow resend*, so the system never auto-sends a possible duplicate.

The email is rendered from a fixed template using only the confirmed booking details, with an `.ics` invite attached.

## 8. Setup

### Local setup (Python 3.10+, no Docker or PostgreSQL install needed)

```bash
python -m venv .venv
.venv\Scriptsctivate         # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
cp .env.example .env           # DATABASE_URL=pgserver:private/pgdata = embedded local PostgreSQL
python -m scripts.init_db      # create tables (first time / after schema changes)
python -m scripts.seed_demo    # optional demo jobs + sample resumes
streamlit run app.py           # http://localhost:8501
```

To use your own PostgreSQL server instead, set `DATABASE_URL=postgresql+psycopg://user:password@host:5432/db`.

`HR_PASSWORD` protects the HR pages. Candidates only see their own application, using application ID + email.

### Using the real LLM (Gemini)

```dotenv
MOCK_LLM=false
LLM_PROVIDER=gemini
GEMINI_API_KEY=...                  # https://aistudio.google.com/apikey
GEMINI_MODEL=gemini-3.5-flash-lite
```

Check it on one resume with `python -m scripts.check_llm`. The adapter uses JSON-schema structured output and a hard timeout (`LLM_TIMEOUT_SECONDS`). To add another provider, implement `generate_json` in `src/llm_client.py`.

### Google Calendar OAuth

1. In Google Cloud Console: create a project → enable **Google Calendar API**.
2. Set up the OAuth consent screen (internal, or external with yourself as a test user).
3. Go to **Credentials → Create credentials → OAuth client ID → Desktop app** and download the JSON as `credentials.json` into the project root.
4. In `.env`:
   ```dotenv
   MOCK_CALENDAR=false
   GOOGLE_CALENDAR_IDS=panel.lead@yourco.com,engineer2@yourco.com
   ```
   The authorizing account must be able to read free/busy on every listed calendar, and to create events on the **first** one.
5. Run `python -m scripts.google_auth` once. A browser opens and `token.json` is written. It is refreshed automatically and git-ignored.

Then check it with `python -m scripts.check_calendar --write`.

### Email (SMTP)

```dotenv
MOCK_EMAIL=false
SMTP_HOST=smtp.gmail.com     # or your provider
SMTP_PORT=587
SMTP_USE_TLS=true
SMTP_USERNAME=recruiting@yourco.com
SMTP_PASSWORD=<app password>  # Gmail: Google Account → Security → App passwords
EMAIL_FROM=recruiting@yourco.com
RECRUITER_CONTACT_EMAIL=recruiting@yourco.com
```

In mock mode, emails are written as `.eml` files to `private/mock/outbox/`.

### Maintenance job

Schedule `python -m scripts.maintenance` every ~10 minutes (cron or Windows Task Scheduler). It reconciles pending bookings, marks past interviews completed, and retries failed emails.

## 9. Tests

```bash
pip install pytest pgserver                 # pgserver = throw-away local PostgreSQL for integration tests
pytest -q                                   # unit + integration
pytest -q -m "not integration"              # unit only, no database needed
# or an existing PostgreSQL test database (it is WIPED):
TEST_DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/recruiter_test pytest -q
```

| Area | Tests |
|---|---|
| Schemas / jobs | valid job, malformed skill JSON, both mandatory+preferred, invalid LLM output, bounded repair |
| Extraction | PDF, DOCX, bad extension, oversized, wrong signature/MIME, empty PDF, path traversal |
| Normalization | ML → Machine Learning, torch → PyTorch, unknown, dedupe, TensorFlow ≠ PyTorch, validator guard-rails |
| Experience | overlap, current, floor, missing dates, end < start, internships/projects, no history |
| Screening | match, missing, unknown vs fail, min experience, review routing, SQL-vs-Python parity, SQL injection |
| Scheduling | slot generation, conflicts, concurrent reservations, calendar failure, DB failure + reconcile, email failure/retry/bound, cancel/complete |
| Workflow | eligible / not eligible / review, prompt injection, duplicates, idempotent rerun, invalid parser output, LLM outage, HR correction + audit, end-to-end booking + email |

## 10. Demo walkthrough (mock mode)

1. `python -m scripts.seed_demo` creates 2 jobs and writes sample resumes to `sample_data/resumes/generated/`.
2. **Candidate Portal:** pick *Machine Learning Engineer*, upload `priya_sharma.pdf`. Result: **ELIGIBLE**, with "torch" → PyTorch (alias) and "ML" → Machine Learning.
3. Upload `anita_rao.pdf`. Result: **REVIEW_REQUIRED** (year-only dates, so experience is unknown).
4. Upload `rahul_verma.docx`. Result: **NOT_ELIGIBLE**, with the reasons shown.
5. **Candidate Review:** open Anita, set `start_date`/`end_date`, tick *resolved*, then *Save corrections and re-screen*. She becomes ELIGIBLE, and the change is in the audit history.
6. **HR Dashboard:** approve Priya.
7. **Interview Scheduler:** *Find available slots* (synthetic mock availability: lunch and Tue/Thu meetings are blocked), then pick one and *Confirm booking*. The interview is CONFIRMED, the email is SENT, and an `.eml` + `.ics` appears in `private/mock/outbox/`.
8. Try booking another candidate into the same time. You get a **CONFLICT** with alternative slots.
9. After the interview, use *Mark completed* and then *Record decision*. The application ends as SELECTED or REJECTED.

## 11. Mock-mode vs credentials checklist

| Feature | Mock mode | Needs credentials |
|---|---|---|
| Job creation, taxonomy, JSONB validation | ✅ | - |
| Resume upload, PDF/DOCX extraction | ✅ | - |
| Call 1 / Call 2 | ✅ rule-based stand-in | Real Gemini: `GEMINI_API_KEY`, `GEMINI_MODEL` |
| Normalization, grounding, experience, screening | ✅ (identical code) | - |
| HR review, corrections, audit | ✅ | - |
| Slot finding / booking / conflicts / reconcile | ✅ synthetic calendar (labelled) | Real availability: Google OAuth |
| Invitation email | ✅ written to outbox | Delivery: SMTP |
| PostgreSQL | **required** in every mode | - |

## 12. Security, privacy and retention

* Uploads are checked by extension, MIME type, size and file signature (`%PDF-` / DOCX zip structure), with a zip-bomb guard and a page limit. Files are stored under random server-generated names in `private/` and never served publicly.
* Resume text is untrusted. It is delimited in prompts, injection attempts are flagged, the UI shows it only as text (never as HTML), and it is not logged.
* All SQL is parameterized. Secrets come from `.env` and are never logged. Error messages are sanitized.
* Screening uses only skills and experience. Name, gender, age and photo are not inputs. LLM "confidence" is shown only as extraction metadata, never as a measure of candidate ability.
* **Retention:** delete resumes and the `profile` JSON for rejected or inactive candidates after your policy period (e.g. 6-12 months). Delete the file in `private/resumes/` and null `resume_path`/`profile`. Keep `audit_log` entries for compliance, without resume content.

## 13. Developer: inspecting the database

| Way | Command / settings |
|---|---|
| **Browser viewer (dev only)** | `streamlit run tools/db_viewer.py --server.port 8502`, then open http://localhost:8502 (read-only: rows, schema, SQL) |
| Quick look from the terminal | `python -m scripts.show_db schema` · `jobs` · `apps` · `match <id>` · `sql` |
| Any read-only SQL | `python -m scripts.show_db query "SELECT * FROM applications"` |
| Interactive SQL shell (psql) | `python -m scripts.show_db shell` (full access: be careful) |
| GUI (DBeaver / pgAdmin) | host `127.0.0.1`, port `5433` (`LOCAL_DB_PORT`), database `postgres`, user `postgres`, no password |

The embedded local database (`DATABASE_URL=pgserver:private/pgdata`) always listens on `LOCAL_DB_PORT` (default 5433). It starts with the app or any script; `python -m scripts.stop_db` stops it.

## 14. Troubleshooting

| Symptom | Fix |
|---|---|
| "Configuration error" on start | Check `.env`. `DATABASE_URL` is required. Non-mock integrations need their variables. |
| "Check that PostgreSQL is running and initialized" | Start the DB and run `python -m scripts.init_db`. |
| Resume "No readable text found" | The file is a scanned image PDF. OCR is not supported, so ask for a text PDF or DOCX. |
| Google "token not found / refresh failed" | Run `python -m scripts.google_auth` again. |
| Google free/busy error for a calendar | The authorizing account can't see that calendar. Share it (at least free/busy) with that account. |
| Booking shows DB_ERROR | Click *Reconcile pending bookings*. It confirms or releases without duplicating the event. |
| Email stuck in SENDING | The process crashed mid-send. Check the outbox or SMTP logs, then *Allow resend*. |
| LLM output fails validation repeatedly | The application goes to REVIEW_REQUIRED for manual entry. Check the model ID and `LLM_MAX_REPAIR_ATTEMPTS`. |
| pip very slow / NameResolutionError | A machine-wide pip `extra-index-url` is unreachable (see `pip config debug`, e.g. `C:\ProgramData\pip\pip.ini`). Remove it there, or install with `PIP_CONFIG_FILE=NUL` (Windows) / `PIP_CONFIG_FILE=/dev/null`. |
