# Autonomous AI Recruiter & Auto-Scheduler

An AI-assisted hiring pipeline. Candidates upload a resume. An LLM reads it, and a **deterministic SQL/Python engine** decides eligibility against the job's hard constraints. HR reviews the uncertain cases. Approved candidates are booked into a **real Google Calendar slot**, with no double-booking.

> **Core principle - AI reads, rules decide.** The LLM only extracts and verifies facts from resumes. Experience calculation, screening, booking and every status change are plain Python and SQL, so results are reproducible, explainable and auditable.

---

## Table of contents

1. [Features](#1-features)
2. [Tech stack](#2-tech-stack)
3. [Architecture](#3-architecture)
4. [Quick start](#4-quick-start)
5. [Configuration](#5-configuration)
6. [Using the app](#6-using-the-app)
7. [How it works](#7-how-it-works)
8. [Database](#8-database)
9. [Project structure](#9-project-structure)
10. [Testing](#10-testing)
11. [Developer tools](#11-developer-tools)
12. [Security and privacy](#12-security-and-privacy)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Features

**For candidates**
- Upload a PDF or DOCX resume. Name and email are read from the resume automatically.
- Instant result: *Eligible*, *Needs review* or *Not eligible*, with the unmet requirements.
- Track status with application ID + email. Re-apply after a rejection (each attempt gets a new ID).

**For HR**
- Post jobs with minimum experience plus mandatory and preferred skills. Aliases are resolved (`ML` → Machine Learning, `torch` → PyTorch).
- Dashboard with pipeline counts and every applicant's status.
- Candidate review: evidence for each requirement, correction of extracted data, re-screening. Everything is audited.
- Approve, reject, or override with a recorded reason.

**Scheduling**
- Real panel availability from Google Calendar (free/busy across all panel calendars).
- Conflict-safe booking enforced by the database, with a Google Meet link.
- Invitation email with an `.ics` calendar file, sent only after the booking is confirmed and never twice.

**Engineering**
- Two-call LLM pipeline (extract → validate) with strict JSON schemas and Python grounding checks.
- LangGraph workflows with explicit states, error routing and human-review checkpoints.
- **Mock mode** for every integration (LLM, calendar, email), so it runs and is tested without any credentials.
- 109 automated tests (unit + integration against real PostgreSQL).

---

## 2. Tech stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Workflow orchestration | LangGraph |
| LLM | Google Gemini (`google-genai`, JSON-schema structured output) |
| Validation | Pydantic 2 |
| Database | PostgreSQL · SQLAlchemy 2 · psycopg 3 |
| Local database runtime | `pgserver` (portable PostgreSQL inside the virtualenv, no install needed) |
| Resume parsing | PyMuPDF (PDF), python-docx (DOCX) |
| Calendar | Google Calendar API (OAuth) |
| Email | SMTP + `.ics` invite (mock outbox by default) |
| Tests | pytest |

---

## 3. Architecture

```
 Candidate ─► Candidate Portal ─┐                       ┌─► Google Gemini (resume reading)
                                │   LangGraph           │
 HR ───────► HR pages ──────────┼─► INTAKE graph ───────┤
             (password)         │   SCHEDULING graph ───┼─► Google Calendar (availability, events)
                                │                       └─► Email (SMTP / mock outbox)
                                ▼
                 PostgreSQL: jobs · applications · interviews · audit_log
```

**Intake graph** (runs when a resume is submitted)

```
extract_resume_text → parse_resume (LLM call 1) → normalize_skills → validate_profile (LLM call 2 + guard-rails)
  → calculate_experience → persist_application → screen_candidate → human_review | finish
                                              (any failure → handle_error / manual-review path)
```

**Scheduling graph** (runs when HR schedules)

```
find_available_slots → confirm_booking (reserve in DB → create Google event → confirm in DB)
  → send_invitation → finish            (conflict → new slots · calendar error → reservation released)
```

Design rules:
- Agents exchange only **Pydantic-validated JSON** through the workflow state.
- **Human review is a database state**, not a paused graph. Reviews can take days, and every node is idempotent, so any run can be re-invoked.
- **The database enforces correctness**: no overlapping interviews, one active application per job and email, and an interview's job always matches its application's job.

---

## 4. Quick start

Requirements: **Python 3.10+**. No Docker or PostgreSQL installation is needed.

```powershell
# 1. install
python -m venv .venv
.venv\Scripts\activate              # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# 2. configure (works as-is in mock mode)
copy .env.example .env              # macOS/Linux: cp .env.example .env

# 3. create the database tables (first time, and after schema changes)
python -m scripts.init_db

# 4. optional: demo jobs + sample PDF/DOCX resumes
python -m scripts.seed_demo

# 5. run
streamlit run app.py
```

Open **http://localhost:8501**. HR password: the `HR_PASSWORD` value in `.env` (`change-me` by default).

---

## 5. Configuration

All settings live in `.env` (see `.env.example`). Never commit `.env`, `credentials.json` or `token.json`; they are git-ignored.

### 5.1 Main settings

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | `pgserver:private/pgdata` | Embedded local PostgreSQL, or any `postgresql+psycopg://user:pass@host:5432/db` |
| `LOCAL_DB_PORT` | `5433` | Port of the embedded database |
| `MOCK_MODE` | `true` | Mock every integration. Override per integration with `MOCK_LLM`, `MOCK_CALENDAR`, `MOCK_EMAIL` |
| `HR_PASSWORD` | `change-me` | Shared password for HR pages |
| `DEFAULT_TIMEZONE` | `Asia/Kolkata` | Timezone for slots and "current month" |
| `WORKING_HOURS_START` / `_END` | `10` / `18` | Interview hours |
| `MIN_NOTICE_HOURS` | `24` | Earliest bookable slot from now |
| `COUNT_INTERNSHIPS_AS_EXPERIENCE` | `false` | Experience policy |
| `AUTO_APPROVE_ELIGIBLE` | `false` | Move ELIGIBLE candidates straight to APPROVED |
| `REAPPLY_COOLDOWN_DAYS` | `0` | Wait time before a rejected candidate can re-apply |

### 5.2 Gemini (resume reading)

```dotenv
MOCK_LLM=false
LLM_PROVIDER=gemini
GEMINI_API_KEY=...                 # https://aistudio.google.com/apikey
GEMINI_MODEL=gemini-3.5-flash-lite
LLM_TIMEOUT_SECONDS=90
```

Check it on one resume: `python -m scripts.check_llm` (or `python -m scripts.check_llm path\to\resume.pdf`).

### 5.3 Google Calendar (interview panel)

1. In Google Cloud Console, create a project and enable the **Google Calendar API**.
2. Set up the **OAuth consent screen** and add your Google account under **Audience → Test users**.
3. Go to **Credentials → Create credentials → OAuth client ID → Desktop app**, download the JSON, and save it as `credentials.json` in the project root.
4. In `.env`:
   ```dotenv
   MOCK_CALENDAR=false
   GOOGLE_CALENDAR_IDS=primary          # or comma-separated panel calendar ids/emails
   GOOGLE_CREATE_MEET_LINK=true
   ```
5. Sign in once: `python -m scripts.google_auth`. This writes `token.json`, which is refreshed automatically.
6. Check it: `python -m scripts.check_calendar --write`. This creates and deletes a test event.

### 5.4 Email (SMTP)

```dotenv
MOCK_EMAIL=false
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USE_TLS=true
SMTP_USERNAME=recruiting@yourco.com
SMTP_PASSWORD=<app password>       # Gmail: Google Account → Security → App passwords
EMAIL_FROM=recruiting@yourco.com
RECRUITER_CONTACT_EMAIL=recruiting@yourco.com
```

In mock mode, invitations are written as `.eml` files to `private/mock/outbox/`.

### 5.5 What works without credentials

| Feature | Mock mode | Needs |
|---|---|---|
| Jobs, uploads, PDF/DOCX extraction | ✅ | - |
| Resume reading (Call 1 + Call 2) | ✅ rule-based stand-in | Gemini key |
| Skill normalization, experience, screening | ✅ (same code) | - |
| HR review, corrections, audit log | ✅ | - |
| Slots, booking, conflicts, reconciliation | ✅ synthetic calendar | Google OAuth |
| Invitation email | ✅ written to outbox | SMTP |

---

## 6. Using the app

### 6.1 Pages

| Page | Who | Purpose |
|---|---|---|
| **Candidate Portal** | Candidates | Choose a job, upload a resume, see the result |
| **Application Status** | Candidates | Look up an application by ID + email |
| **HR Dashboard** | HR | Pipeline counts, jobs, applicants; approve, reject, re-run, reopen; job settings |
| **Create Job** | HR | Post a job with experience and skill requirements |
| **Candidate Review** | HR | Evidence per requirement, fix extracted data, re-screen |
| **Interview Scheduler** | HR | Find slots, book, cancel, complete, record the final decision |

Direct links work too, e.g. `http://localhost:8501/?page=Create Job`.

**HR sign-in:** enter any name (recorded in the audit log) plus `HR_PASSWORD`.

### 6.2 Demo walkthrough (mock mode)

1. `python -m scripts.seed_demo` creates 2 jobs and writes sample resumes to `sample_data/resumes/generated/`.
2. **Candidate Portal:** apply to *Machine Learning Engineer* with `priya_sharma.pdf`. Result: **Eligible** (`torch` → PyTorch, `ML` → Machine Learning).
3. Apply with `anita_rao.pdf`. Result: **Needs review** (year-only dates, so experience is unknown).
4. Apply with `rahul_verma.docx`. Result: **Not eligible**, with the unmet requirements listed.
5. **Candidate Review:** open Anita, set the missing dates, tick *resolved*, then *Save corrections and re-screen*. She becomes Eligible, and the change is audited.
6. **HR Dashboard:** approve Priya.
7. **Interview Scheduler:** *Find available slots*, then *Confirm booking*. The interview is confirmed and the invitation (`.eml` + `.ics`) appears in the outbox.
8. Book another candidate into the same time. You get a **conflict** with alternative slots.
9. After the interview: *Mark completed*, then *Record decision* (Selected / Rejected).

---

## 7. How it works

### 7.1 Resume pipeline: two LLM calls + Python checks

| Step | Done by | What happens |
|---|---|---|
| Text extraction | Python | PDF/DOCX → text; scanned or empty files are rejected |
| **Call 1: extract** | Gemini | Name, email, skills with verbatim evidence, jobs with `YYYY-MM` dates and type. The resume is wrapped in `<resume>` tags as untrusted data. Output follows a JSON schema, is validated with Pydantic, and gets up to 2 repair attempts |
| Normalize | Python | Lowercase, alias map, dedupe; unknown skills are flagged, never guessed (TensorFlow ≠ PyTorch) |
| **Call 2: validate** | Gemini | Checks that each skill has supporting evidence and flags date contradictions |
| Guard-rails | Python | Evidence must appear in the resume text (otherwise REJECTED). The LLM can't add skills or change alias mappings. LLM-suggested "related" skills always need HR confirmation |

Both calls are told today's date and follow the resume's own section headings (Experience vs Projects). A regex backstop flags instruction-like text (prompt injection), which can't help anyway because the LLM never decides eligibility.

### 7.2 Experience calculation (`experience_calculator.py`)

- Counts **completed calendar months**; overlapping jobs are **merged**, so they're never double-counted.
- Current roles run up to the current month.
- Counted: full-time, part-time, contract, freelance, unspecified. Internships only if enabled. Academic work and personal projects are never counted.
- Missing or contradictory dates give **unknown**, never zero, which sends the application to review.
- Stored as completed whole years; exact months are kept in the profile.

### 7.3 Deterministic screening (`screening_service.py`)

A SQL query joins the application with its job, expands the JSON requirements (`jsonb_array_elements`), and picks the best matching skill for each one (`LATERAL`). The decision rules:

| Check | PASS | UNKNOWN | FAIL |
|---|---|---|---|
| Experience | ≥ minimum | can't be calculated | < minimum |
| Skill | validated match | mapping needs confirmation, or mentioned but not validated | not in the resume |

**Result:** any mandatory FAIL → **NOT_ELIGIBLE**; otherwise any UNKNOWN or open review flag → **REVIEW_REQUIRED**; otherwise → **ELIGIBLE**. Preferred skills are reported but never change the result.

### 7.4 Conflict-safe booking

Google Calendar and PostgreSQL can't share a transaction, so the booking runs as a sequence of steps:

1. Re-check live availability for the exact slot.
2. **Reserve** in the database (`PENDING_CONFIRMATION`) with a pre-generated event ID. The exclusion constraint is the atomic lock.
3. **Create the Google event** with that ID. A retry reuses it, so there are no duplicate events. If this fails, the reservation is released.
4. **Confirm** in the database. If this fails, *Reconcile pending bookings* (or `scripts/maintenance.py`) finishes the booking later.
5. **Send the invitation** through an atomic claim, so it's never sent twice. An email failure never cancels the booking.

### 7.5 Application lifecycle

```
SCREENING ─► APPROVED ─► INTERVIEW_SCHEDULED ─► INTERVIEWED ─► SELECTED / REJECTED
    │            └──────────────► REJECTED (HR)
    └─► SUPERSEDED (candidate uploaded a newer resume for the same job)
```

- **Re-applying:** every upload gets a **new application ID**, and old attempts are kept as history.
- Only **one active** application per job and email is allowed.
- A rejected candidate can apply again (optional cooldown).
- Re-applying is blocked while an attempt is approved or interviewing.

---

## 8. Database

Schema file: `sql/init.sql` (idempotent; apply with `python -m scripts.init_db`).

| Table | Holds | Key rules |
|---|---|---|
| `jobs` | Title, minimum experience, `skills` JSONB `{mandatory, preferred}`, interview length, status | JSON shape enforced by CHECK |
| `applications` | Candidate, validated `skills` (JSONB), `experience`, `profile` (evidence, history), screening result, pipeline stage | One **active** application per job + email (partial unique index) |
| `interviews` | Start/end (`TIMESTAMPTZ`) + timezone, status, Google event ID, Meet link, email status | **Exclusion constraint:** active interviews never overlap; composite FK keeps job and application consistent |
| `audit_log` | Who did what and when | History of HR actions, re-screens, bookings, emails |

**Where the data lives (local mode):** `private/pgdata/`, served by a portable PostgreSQL from the `pgserver` package on `127.0.0.1:5433`. Uploaded resume files are in `private/resumes/`. Stop the server with `python -m scripts.stop_db`.

---

## 9. Project structure

```
app.py                          Streamlit UI (candidate + HR pages)
sql/init.sql                    Database schema
src/
  config.py                     .env loading and validation
  database.py                   engine, sessions, ORM mappings, embedded PostgreSQL startup
  schemas.py                    Pydantic models (all JSON contracts)
  repository.py                 all database access, status transitions, audit log
  llm_client.py                 LLM adapter (Gemini + mock)
  mock_llm.py                   rule-based stand-ins for the LLM tasks
  resume_extractor.py           upload validation, private storage, PDF/DOCX text
  resume_parser_agent.py        LLM call 1: extraction prompt, schema, repair loop
  profile_validator_agent.py    LLM call 2: validation prompt + Python guard-rails
  skill_normalizer.py           taxonomy, aliases, normalization
  data/skill_taxonomy.json      approved skills + aliases (editable)
  experience_calculator.py      experience policy
  screening_service.py          screening SQL + decision rules
  calendar_service.py           Google Calendar + mock + slot generation
  scheduling_agent.py           booking lifecycle, reconciliation, invitations
  email_service.py              invitation template, .ics, SMTP + mock outbox
  recruitment_workflow.py       LangGraph intake + scheduling graphs, HR corrections
  db_explorer.py                read-only database helpers for developer tools
  ui_theme.py                   UI styling and components
scripts/
  init_db.py  seed_demo.py  google_auth.py  maintenance.py
  check_llm.py  check_calendar.py  show_db.py  stop_db.py
tools/db_viewer.py              developer database viewer (browser, read-only)
tests/                          pytest suite
sample_data/                    demo jobs and fictional resumes
```

---

## 10. Testing

```powershell
pytest -q                          # all 109 tests (integration tests start a throw-away PostgreSQL)
pytest -q -m "not integration"     # unit tests only, no database
```

To use an existing PostgreSQL test database instead (it is **wiped**), set `TEST_DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/recruiter_test`.

| Area | Covered |
|---|---|
| Schemas & LLM output | Valid/invalid jobs, malformed JSON, bounded repair, Gemini adapter (timeouts, retries, refusals) |
| Uploads | PDF, DOCX, bad extension, oversized, fake file signature, empty PDF, path traversal |
| Skills | Aliases, unknown skills, dedupe, TensorFlow ≠ PyTorch, validator guard-rails |
| Experience | Overlaps, current role, missing or contradictory dates, internships/projects |
| Screening | Pass/fail/unknown rules, SQL-vs-Python parity, SQL injection |
| Scheduling | Slot generation, conflicts, concurrent booking, calendar/database failures + reconciliation, email retry and dedup |
| Workflow | End-to-end intake and booking, prompt injection, re-application rules, HR corrections + audit |

---

## 11. Developer tools

| Task | Command |
|---|---|
| Check the LLM on one resume | `python -m scripts.check_llm [resume.pdf]` |
| Check Google Calendar | `python -m scripts.check_calendar --write` |
| Database viewer in the browser | `streamlit run tools/db_viewer.py --server.port 8502` → http://localhost:8502 |
| Tables, jobs, applications | `python -m scripts.show_db schema` · `jobs` · `apps` |
| How one application was screened | `python -m scripts.show_db match <id>` |
| Read-only SQL | `python -m scripts.show_db query "SELECT * FROM applications"` |
| SQL shell (psql) | `python -m scripts.show_db shell` |
| GUI (DBeaver / pgAdmin) | host `127.0.0.1`, port `5433`, database `postgres`, user `postgres`, no password |
| Periodic maintenance | `python -m scripts.maintenance` (reconcile bookings, complete past interviews, retry emails) |

---

## 12. Security and privacy

- **Uploads** are checked by extension, MIME type, size and file signature (zip-bomb guard, page limit). They're stored under random names in `private/` and never served publicly.
- **Resume text is untrusted:** it's delimited in prompts, injection attempts are flagged, it's never rendered as HTML, and it isn't logged.
- **SQL** is always parameterized. **Secrets** come only from `.env` and are never logged.
- **Access:** HR pages need the HR password. Candidates only see their own application (ID + email).
- **Fairness:** screening uses only skills and experience; name, gender, age and photo are not inputs.
- **Retention:** delete resumes and parsed profiles of inactive candidates after your policy period. Keep `audit_log` entries without resume content.

---

## 13. Troubleshooting

| Problem | Fix |
|---|---|
| "The database has no tables yet" | `python -m scripts.init_db` |
| "Configuration error" on start | Check `.env`. Integrations that aren't mocked need their variables |
| HR sign-in "Wrong HR password" | Use the `HR_PASSWORD` from `.env`. The browser may have auto-filled a generated password |
| Resume "No readable text found" | Scanned image PDF; upload a text PDF or DOCX |
| Resume marked ERROR "did not respond" | The LLM timed out. Use *Re-run processing* on the HR Dashboard |
| Google "Access blocked … not completed verification" | Add your account under OAuth consent screen → Audience → Test users |
| Google "token not found / refresh failed" | `python -m scripts.google_auth` |
| Booking shows `DB_ERROR` | Interview Scheduler → Maintenance → *Reconcile pending bookings* |
| Email stuck in `SENDING` | Check the outbox/SMTP log, then *Allow resend* |
| GUI can't connect to the database | Run any project command first (starts the DB); port `5433` |
| `pip install` very slow / NameResolutionError | A machine-wide pip `extra-index-url` is unreachable; remove it (see `pip config debug`) |
