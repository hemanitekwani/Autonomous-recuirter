"""Streamlit UI - Autonomous AI Recruiter & Auto-Scheduler.

Run:  streamlit run app.py
(Initialize the database first:  python -m scripts.init_db)
"""
from __future__ import annotations

import hmac
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from src import repository as repo
from src.config import ConfigError, get_settings
from src.database import session_scope
from src.recruitment_workflow import IntakeWorkflow, SchedulingWorkflow, apply_hr_corrections
from src.resume_extractor import ResumeError, extract_text, find_email, save_resume, validate_upload
from src.repository import NAME_FROM_RESUME
from src.schemas import (
    CandidateSubmission,
    EmploymentEntry,
    EmploymentType,
    JobCreate,
    JobSkills,
    MatchType,
    PipelineStatus,
    SkillValidationStatus,
    TimeSlot,
    ValidatedSkill,
)
from src.skill_normalizer import add_custom_skills, approved_skills, normalize_job_skills, resolve, split_skill_input
from src.ui_theme import (
    badge,
    hero,
    html,
    inject_css,
    job_card,
    kpis,
    notice,
    page_header,
    sidebar_brand,
    sidebar_integrations,
    sidebar_label,
    sidebar_note,
    status_icon,
    table,
)

logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="AI Recruiter", page_icon="🧭", layout="wide", initial_sidebar_state="expanded")
inject_css()

try:
    settings = get_settings()
except ConfigError as exc:
    st.error(f"Configuration error: {exc}")
    st.stop()

TZ = ZoneInfo(settings.default_timezone)

try:
    from streamlit.runtime.scriptrunner_utils.exceptions import ScriptControlException as _SCRIPT_CONTROL
except ImportError:  # older Streamlit layout
    from streamlit.runtime.scriptrunner.exceptions import ScriptControlException as _SCRIPT_CONTROL


@st.cache_resource
def intake_workflow() -> IntakeWorkflow:
    return IntakeWorkflow(settings)


@st.cache_resource
def scheduling_workflow() -> SchedulingWorkflow:
    return SchedulingWorkflow(settings)


def fmt_dt(dt: datetime | None) -> str:
    return dt.astimezone(TZ).strftime("%a %d %b %Y, %H:%M") if dt else "-"


def fmt_slot(slot: TimeSlot) -> str:
    return f"{slot.starts_at.astimezone(TZ):%a %d %b %Y  %H:%M}-{slot.ends_at.astimezone(TZ):%H:%M} ({settings.default_timezone})"


def db_error_guard(fn):
    """Show a friendly message instead of a stack trace when the DB is unreachable."""
    try:
        return fn()
    except _SCRIPT_CONTROL:  # st.stop() / st.rerun() must propagate
        raise
    except Exception as exc:  # noqa: BLE001 - UI boundary
        if "UndefinedTable" in repr(getattr(exc, "orig", exc)):
            st.error("The database has no tables yet. Stop the app and run:  python -m scripts.init_db  "
                     "(and optionally  python -m scripts.seed_demo), then start the app again.")
            st.stop()
        logging.exception("UI action failed")
        st.error(f"Something went wrong: {type(exc).__name__}. Check that PostgreSQL is running and initialized.")
        st.stop()


# =============================================================================
# Sidebar / auth
# =============================================================================
sidebar_brand("Screening & auto-scheduling")
sidebar_label("Menu")

CANDIDATE_PAGES = ["Candidate Portal", "Application Status"]
HR_PAGES = ["HR Dashboard", "Create Job", "Candidate Review", "Interview Scheduler"]
PAGE_ICONS = {"Candidate Portal": "📄", "Application Status": "🔎", "HR Dashboard": "📊", "Create Job": "📝",
              "Candidate Review": "🧾", "Interview Scheduler": "📅"}
ALL_PAGES = CANDIDATE_PAGES + HR_PAGES
_requested = st.query_params.get("page", "")  # deep links, e.g. ?page=HR Dashboard
page = st.sidebar.radio("Navigate", ALL_PAGES, label_visibility="collapsed",
                        index=ALL_PAGES.index(_requested) if _requested in ALL_PAGES else 0,
                        format_func=lambda p: f"{PAGE_ICONS[p]}   {p}")

sidebar_label("Integrations")
sidebar_integrations([("Resume AI (Gemini)", settings.mock_llm), ("Google Calendar", settings.mock_calendar),
                      ("Email", settings.mock_email)])
for w in settings.warnings:
    sidebar_note(f"⚠ {w}")


def require_hr() -> str:
    """Simple shared-password gate for HR pages (set HR_PASSWORD)."""
    if not settings.hr_password:
        return "hr"
    if st.session_state.get("hr_authed"):
        return st.session_state.get("hr_name", "hr")
    page_header("HR sign-in", "Sign in to review candidates, approve them and schedule interviews.", "HR team")
    with st.form("hr_login"):
        name = st.text_input("Your name (recorded in the audit log)")
        # "current-password" stops browsers from offering a newly generated password here
        pw = st.text_input("HR password", type="password", autocomplete="current-password")
        if st.form_submit_button("Sign in", type="primary"):
            if not name.strip():
                st.error("Please enter your name - it is recorded in the audit log.")
            elif not hmac.compare_digest(pw.encode(), settings.hr_password.encode()):
                st.error("Wrong HR password. (It is the HR_PASSWORD value in the .env file.)")
            else:
                st.session_state.update(hr_authed=True, hr_name=f"hr:{name.strip()[:60]}")
                st.rerun()
    st.stop()


if st.session_state.get("hr_authed") and st.sidebar.button("Sign out (HR)"):
    st.session_state.pop("hr_authed", None)
    st.rerun()


def requirement_text(skills: dict) -> tuple[str, str]:
    mand = ", ".join(s["canonical_name"] for s in skills.get("mandatory", []))
    pref = ", ".join(s["canonical_name"] for s in skills.get("preferred", []))
    return mand or "-", pref or "-"


# =============================================================================
# Candidate portal
# =============================================================================
def page_candidate_portal() -> None:
    hero("Find your next role", "Pick a position and upload your resume - we read it instantly and tell you "
         "where you stand.", "Careers")
    with session_scope() as s:
        jobs = [(j.job_id, j.title, j.min_experience_years, j.skills) for j in repo.list_jobs(s, status="OPEN")]
    if not jobs:
        st.info("There are no open positions right now.")
        return
    labels: dict[str, tuple] = {}
    for jid, title, exp, skills in jobs:
        labels[title if title not in labels else f"{title} (#{jid})"] = (jid, title, exp, skills)
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown("### Open positions")
        choice = st.radio("Position", list(labels), label_visibility="collapsed", horizontal=True)
        job_id, title, min_exp, skills = labels[choice]
        job_card(title, min_exp, [s["canonical_name"] for s in skills.get("mandatory", [])],
                 [s["canonical_name"] for s in skills.get("preferred", [])], selected=True)
    with right:
        st.markdown("### Apply")
        apply_form(job_id)


def apply_form(job_id: int) -> None:
    with st.form("apply", clear_on_submit=False):
        upload = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])
        with st.expander("Contact details (optional - taken from your resume if left empty)"):
            name = st.text_input("Full name", max_chars=200)
            email = st.text_input("Email", max_chars=254)
        consent = st.checkbox("I agree that my resume is processed (including by an AI model) to assess this "
                              "application and is retained only for the recruitment process.")
        submitted = st.form_submit_button("Submit application", type="primary")
    if not submitted:
        return
    if upload is None:
        st.error("Please upload your resume.")
        return
    if not consent:
        st.error("Please accept the processing notice to apply.")
        return
    data = upload.getvalue()
    try:
        ext = validate_upload(upload.name, data, settings.max_resume_size_mb, upload.type)
        resume_text = extract_text(data, ext)
    except ResumeError as exc:
        st.error(str(exc))
        return

    # Contact details: typed values win; otherwise the email comes from the resume text (deterministic)
    # and the name is filled in by the extraction step of the workflow.
    email = email.strip() or (find_email(resume_text) or "")
    if not email:
        st.error("We couldn't find an email address in your resume. Please open 'Contact details' and enter it.")
        return
    try:
        submission = CandidateSubmission(job_id=job_id, name=name.strip() or NAME_FROM_RESUME, email=email)
    except ValidationError:
        st.error(f"'{email}' is not a valid email address. Please open 'Contact details' and correct it.")
        return
    # Policy: every upload is a NEW application (new ID); earlier attempts are kept as history.
    # An earlier attempt still in screening becomes SUPERSEDED; rejected attempts may re-apply
    # (after REAPPLY_COOLDOWN_DAYS). Blocked while an attempt is approved / interviewing / selected.
    path = save_resume(data, ext, settings.resume_storage_dir)
    try:
        with session_scope() as s:
            application_id, previous = repo.reapply(s, submission, path.name, settings.reapply_cooldown_days)
    except repo.ReapplyNotAllowedError as exc:
        path.unlink(missing_ok=True)
        st.warning(str(exc))
        return
    except repo.DuplicateApplicationError:  # rare race with a parallel submission
        path.unlink(missing_ok=True)
        st.warning("An application with this email was just submitted - please try again.")
        return
    except repo.RepositoryError as exc:
        path.unlink(missing_ok=True)
        st.error(str(exc))
        return

    with st.spinner("Reading and screening your resume..."):
        state = intake_workflow().run(application_id)
    st.success(f"Application received. Your application ID is **{application_id}**. We'll contact you at "
               f"**{submission.email}** - use both to check your status.")
    if previous:
        st.caption("Previous attempts for this job: " + ", ".join(f"#{p}" for p in previous) +
                   ". Your new application replaces them.")
    show_candidate_status(application_id, state.get("status"))


def show_candidate_status(application_id: int, status: str | None = None) -> None:
    with session_scope() as s:
        app = repo.get_application(s, application_id)
        interview = repo.get_active_interview(s, application_id)
        screening, pipeline = app.screening_status, app.pipeline_status
        result = app.screening_result or {}
        slot = (interview.starts_at, interview.ends_at, interview.status, interview.meeting_link) if interview else None
        attempts = [(a.application_id, a.pipeline_status, a.screening_status, a.created_at)
                    for a in repo.list_attempts(s, app.job_id, app.email)]
    messages = {
        "PENDING": "Your application is queued for processing.",
        "PROCESSING": "Your resume is being processed.",
        "ELIGIBLE": "You meet the requirements. The hiring team will review your application.",
        "REVIEW_REQUIRED": "Your application needs a manual review by the hiring team.",
        "NOT_ELIGIBLE": "Your profile does not meet the minimum requirements for this role.",
        "ERROR": "We could not process your resume automatically; the hiring team has been notified.",
    }
    from html import escape

    unmet = [c["requirement"] for c in result.get("checks", [])
             if c["result"] == "FAIL" and c["kind"] != "PREFERRED_SKILL"] if screening == "NOT_ELIGIBLE" else []
    unmet_html = (f'<div class="section-title">Requirements not met</div>'
                  + "".join(f"<span class='chip pref'>{escape(u)}</span>" for u in unmet)) if unmet else ""
    interview_html = ""
    if slot and slot[2] == "CONFIRMED":
        interview_html = (f'<div class="section-title">Interview</div><div>📅 <b>{escape(fmt_dt(slot[0]))} - '
                          f'{slot[1].astimezone(TZ):%H:%M}</b> ({escape(settings.default_timezone)}). '
                          'The invitation was sent to your email.</div>')
    message = messages.get(screening, screening)
    if pipeline == "SUPERSEDED":
        newer = [a[0] for a in attempts if a[0] > application_id]
        message = (f"This application was replaced by your newer application #{newer[-1]}." if newer
                   else "This application was replaced by a newer one.")
    history_html = ""
    if len(attempts) > 1:
        rows = "".join(
            f"<div style='margin:.15rem 0'>{'➡️' if aid == application_id else '•'} #{aid} · "
            f"{escape(fmt_dt(created))} · {badge(stage)} {badge(scr) if stage != 'SUPERSEDED' else ''}</div>"
            for aid, stage, scr, created in attempts)
        history_html = f'<div class="section-title">Your applications for this job</div>{rows}'
    html(f"""<div class="card">
        <div class="meta">Application #{int(application_id)}</div>
        <div style="display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:.6rem">{badge(screening)} {badge(pipeline)}</div>
        <div>{escape(message)}</div>
        {unmet_html}{interview_html}{history_html}
        </div>""")


def page_status() -> None:
    page_header("Track your application", "Enter your application ID and the email address from your resume.",
                "Candidates")
    with st.form("lookup"):
        app_id = st.number_input("Application ID", min_value=1, step=1)
        email = st.text_input("Email used to apply")
        go = st.form_submit_button("Check status")
    if go:
        with session_scope() as s:
            found = repo.candidate_lookup(s, int(app_id), email)
            found_id = found.application_id if found else None
        if found_id is None:
            st.error("No application found for that ID and email.")
        else:
            show_candidate_status(found_id)


# =============================================================================
# HR dashboard
# =============================================================================
def page_hr_dashboard(actor: str) -> None:
    page_header("Dashboard", "Pipeline overview, screening results and decisions.", "HR team")
    with session_scope() as s:
        counts = repo.dashboard_counts(s)
        jobs = repo.list_jobs(s)
        job_rows = [{"ID": j.job_id, "Title": j.title, "Min exp": j.min_experience_years,
                     "Mandatory": requirement_text(j.skills)[0], "Preferred": requirement_text(j.skills)[1],
                     "Interview (min)": j.interview_duration_minutes, "Status": j.status} for j in jobs]
    kpis([("Open jobs", counts.get("open_jobs", 0), "indigo"), ("Applications", counts.get("applications", 0), "blue"),
          ("Eligible", counts.get("ELIGIBLE", 0), "green"), ("Needs review", counts.get("REVIEW_REQUIRED", 0), "amber"),
          ("Not eligible", counts.get("NOT_ELIGIBLE", 0), "red"),
          ("Interviews", counts.get("interviews_confirmed", 0), "indigo")])

    st.markdown("### Jobs")
    if not job_rows:
        st.info("No jobs yet - create one on the 'Create Job' page.")
        return
    table(job_rows)

    with session_scope() as s:
        per_job = {r["ID"]: len(repo.list_applications(s, job_id=r["ID"])) for r in job_rows}
    job_ids = {f"{r['Title']}  ·  {per_job[r['ID']]} applicant(s)": r["ID"] for r in job_rows}
    busiest = max(range(len(job_rows)), key=lambda i: per_job[job_rows[i]["ID"]])
    st.markdown("### Applicants")
    selected = st.selectbox("Job", list(job_ids), index=busiest)
    job_id = job_ids[selected]

    with st.expander("Job settings (status, interview duration)"):
        with session_scope() as s:
            job = repo.get_job(s, job_id)
            duration, status = job.interview_duration_minutes, job.status
        c1, c2 = st.columns(2)
        new_duration = c1.number_input("Interview duration (minutes)", 15, 240, duration, step=15)
        if c1.button("Save duration"):
            with session_scope() as s:
                repo.set_job_interview_duration(s, job_id, int(new_duration), actor)
            st.success("Saved.")
        if c2.button("Close job" if status == "OPEN" else "Re-open job"):
            with session_scope() as s:
                repo.set_job_status(s, job_id, "CLOSED" if status == "OPEN" else "OPEN", actor)
            st.rerun()
        st.caption(f"Panel calendars: {', '.join(settings.google_calendar_ids) or '(mock calendar)'} · "
                   f"Working hours {settings.working_hours_start}:00-{settings.working_hours_end}:00 "
                   f"{settings.default_timezone} · minimum notice {settings.min_notice_hours} h "
                   "(set via .env)")

    with session_scope() as s:
        apps = [{"ID": a.application_id, "Name": a.name, "Email": a.email, "Experience (yrs)": a.experience,
                 "Screening": a.screening_status, "Stage": a.pipeline_status, "Applied": fmt_dt(a.created_at)}
                for a in repo.list_applications(s, job_id=job_id)]
    if not apps:
        st.info("No applications for this job yet.")
        return
    table([{**a, "Screening": status_icon(a["Screening"]), "Stage": status_icon(a["Stage"])} for a in apps])

    st.markdown("### Decision")
    d1, d2 = st.columns([3, 2])
    app_choice = d1.selectbox("Application", [f"#{a['ID']}  {a['Name']}  ·  {status_icon(a['Screening'])}  ·  "
                                               f"{status_icon(a['Stage'])}" for a in apps])
    app_id = int(app_choice.split()[0][1:])
    current = next(a for a in apps if a["ID"] == app_id)
    reason = d2.text_input("Reason / note (required for overrides and rejections)")
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Approve for interview", type="primary", width="stretch", disabled=current["Stage"] != "SCREENING"):
        if current["Screening"] != "ELIGIBLE" and not reason.strip():
            st.error("This candidate is not ELIGIBLE - enter a reason to override.")
        else:
            try:
                with session_scope() as s:
                    repo.set_pipeline_status(s, app_id, PipelineStatus.APPROVED, actor,
                                             reason or "Eligible by screening")
                    if current["Screening"] != "ELIGIBLE":
                        repo.audit(s, "application", app_id, "HR_OVERRIDE_APPROVE", actor,
                                   {"screening_status": current["Screening"], "reason": reason})
                st.success("Approved - schedule the interview on the 'Interview Scheduler' page.")
            except repo.InvalidTransitionError as exc:
                st.error(str(exc))
    if c2.button("Reject", width="stretch"):
        if not reason.strip():
            st.error("Enter a reason for the rejection.")
        else:
            try:
                with session_scope() as s:
                    repo.set_pipeline_status(s, app_id, PipelineStatus.REJECTED, actor, reason)
                st.success("Rejected.")
            except repo.InvalidTransitionError as exc:
                st.error(str(exc))
    if c3.button("Re-run processing", width="stretch"):
        with session_scope() as s:
            edited = (repo.get_application(s, app_id).profile or {}).get("human_edited")
        if edited and not reason.strip():
            st.error("This profile has HR corrections that re-processing would overwrite. Enter a reason to proceed.")
        else:
            with st.spinner("Processing..."):
                state = intake_workflow().run(app_id)
            with session_scope() as s:
                repo.audit(s, "application", app_id, "REPROCESSED", actor, {"reason": reason})
            st.success(f"Done: {state.get('status')} - {state.get('message')}")
    if c4.button("Reopen (back to screening)", width="stretch", disabled=current["Stage"] not in ("REJECTED", "APPROVED")):
        try:
            with session_scope() as s:
                repo.set_pipeline_status(s, app_id, PipelineStatus.SCREENING, actor, reason or "Reopened")
            st.success("Reopened.")
        except repo.InvalidTransitionError as exc:
            st.error(str(exc))


# =============================================================================
# Create job
# =============================================================================
def page_create_job(actor: str) -> None:
    page_header("Create a job", "Define the hard constraints - screening against them is fully deterministic.",
                "HR team")
    with st.form("create_job"):
        title = st.text_input("Job title", max_chars=200, placeholder="e.g. Machine Learning Engineer")
        n1, n2, n3 = st.columns(3)
        min_exp = n1.number_input("Minimum experience (years)", 0, 50, 2)
        duration = n2.number_input("Interview length (minutes)", 15, 240, settings.interview_duration_minutes, 15)
        status = n3.selectbox("Status", ["OPEN", "CLOSED"])
        k1, k2 = st.columns(2)
        mandatory = k1.text_area("Mandatory skills", placeholder="Python, PyTorch, ML", height=110,
                                 help="Comma separated. Aliases like 'ML' or 'torch' are resolved automatically.")
        preferred = k2.text_area("Preferred skills", placeholder="SQL, Docker", height=110,
                                 help="Comma separated. Shown on results but never change eligibility.")
        approve_new = st.checkbox("Add unrecognized skills to the approved taxonomy")
        submitted = st.form_submit_button("Create job", type="primary")
    with st.expander("Approved skill taxonomy"):
        st.write(", ".join(approved_skills()))
    if not submitted:
        return
    mand_list, pref_list = split_skill_input(mandatory), split_skill_input(preferred)
    _, unknown = normalize_job_skills(mand_list, pref_list)
    if unknown:
        if not approve_new:
            st.error("Not in the approved taxonomy: " + ", ".join(unknown) +
                     ". Fix the spelling or tick 'Add unrecognized skills to the approved taxonomy'.")
            return
        added = add_custom_skills(unknown)
        with session_scope() as s:
            repo.audit(s, "taxonomy", 0, "SKILLS_APPROVED", actor, {"skills": added})
    skills_dict, unknown = normalize_job_skills(mand_list, pref_list)
    try:
        job = JobCreate(title=title, min_experience_years=int(min_exp), skills=JobSkills.model_validate(skills_dict),
                        interview_duration_minutes=int(duration), status=status)
    except ValidationError as exc:
        st.error("Invalid job: " + "; ".join(e["msg"] for e in exc.errors()))
        return
    with session_scope() as s:
        job_id = repo.create_job(s, job, actor)
    st.success(f"Job #{job_id} created.")
    table([{"Type": "Mandatory", "Entered": r.name, "Approved skill": r.canonical_name} for r in job.skills.mandatory] +
          [{"Type": "Preferred", "Entered": r.name, "Approved skill": r.canonical_name} for r in job.skills.preferred])


# =============================================================================
# Candidate review
# =============================================================================
def page_candidate_review(actor: str) -> None:
    page_header("Candidate review", "Evidence for every requirement - correct extraction errors and re-screen.",
                "HR team")
    filters = {"Needs review": "REVIEW_REQUIRED", "Eligible": "ELIGIBLE", "Not eligible": "NOT_ELIGIBLE",
               "Processing error": "ERROR", "All": "All"}
    f1, f2 = st.columns([1, 2])
    status_filter = filters[f1.selectbox("Show", list(filters))]
    with session_scope() as s:
        apps = repo.list_applications(s, screening_status=None if status_filter == "All" else status_filter,
                                      include_superseded=status_filter == "All")
        options = {f"#{a.application_id}  {a.name}  ·  {status_icon(a.screening_status)}": a.application_id
                   for a in apps}
    if not options:
        f2.selectbox("Application", ["No applications in this state"], disabled=True)
        return
    app_id = options[f2.selectbox("Application", list(options))]
    with session_scope() as s:
        app = repo.get_application(s, app_id)
        job = repo.get_job(s, app.job_id)
        profile, result = dict(app.profile or {}), dict(app.screening_result or {})
        header = (app.name, app.email, job.title, job.min_experience_years, app.experience, app.screening_status,
                  app.pipeline_status)
        history = [(fmt_dt(h.created_at), h.action, h.actor, h.details) for h in
                   repo.list_audit(s, "application", app_id)]
        other_attempts = [{"Application": f"#{a.application_id}", "Submitted": fmt_dt(a.created_at),
                           "Screening": status_icon(a.screening_status), "Stage": status_icon(a.pipeline_status),
                           "Experience (yrs)": a.experience}
                          for a in repo.list_attempts(s, app.job_id, app.email) if a.application_id != app_id]

    name, email, job_title, min_exp, exp, screening, pipeline = header
    from html import escape

    checks = result.get("checks", [])
    mandatory = [c for c in checks if c["kind"] != "PREFERRED_SKILL"]
    edited_note = " · <i>contains HR corrections</i>" if profile.get("human_edited") else ""
    html(f"""<div class="card" style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem">
        <div><h3>{escape(name)}</h3><div class="meta" style="margin:0">{escape(email)} · applying for
        <b>{escape(job_title)}</b>{edited_note}</div></div>
        <div style="display:flex;gap:.5rem">{badge(screening)} {badge(pipeline)}</div></div>""")
    kpis([("Experience", "unknown" if exp is None else f"{exp} yrs", "amber" if exp is None else "indigo"),
          ("Required", f"{min_exp} yrs", "gray"),
          ("Mandatory passed", f"{sum(1 for c in mandatory if c['result'] == 'PASS')} / {len(mandatory)}", "green"),
          ("Preferred matched", len(result.get("preferred_matched", [])), "blue")])
    if other_attempts:
        with st.expander(f"Other applications by this candidate for this job ({len(other_attempts)})"):
            table(other_attempts)

    if result.get("error"):
        st.error(f"Processing error: {result['error']}")
    reasons = list(dict.fromkeys(profile.get("review_reasons", []) + result.get("review_reasons", [])))
    if reasons:
        notice("Needs attention", reasons, "warn")
    if profile.get("notes"):
        notice("Notes from parsing", profile["notes"], "info")

    if checks:
        st.markdown("### Requirement checks")
        kinds = {"EXPERIENCE": "Experience", "MANDATORY_SKILL": "Mandatory", "PREFERRED_SKILL": "Preferred"}
        results = {"PASS": "✅ Pass", "FAIL": "❌ Fail", "UNKNOWN": "🟠 Unknown"}

        def detail(c):
            if c["kind"] == "EXPERIENCE":
                actual = c.get("actual_value")
                return f"{'unknown' if actual is None else actual} yrs (needs {c.get('required_value')})"
            return (c.get("match_type") or "").title() or None

        table([{"Requirement": "Minimum experience" if c["kind"] == "EXPERIENCE" else c["requirement"],
                "Type": kinds.get(c["kind"], c["kind"]), "Result": results.get(c["result"], c["result"]),
                "Detail": detail(c), "Evidence / reason": c.get("evidence") or c.get("reason")} for c in checks])

    exp_calc = profile.get("experience") or {}
    with st.expander("How experience was calculated", expanded=exp is None):
        st.caption(exp_calc.get("method", "not calculated"))
        if exp_calc.get("counted_periods"):
            table(exp_calc["counted_periods"])
        for line in exp_calc.get("excluded_entries", []) + exp_calc.get("reasons", []):
            st.text(f"• {line}")

    st.markdown("### Correct the extracted profile")
    st.caption("Edit skills or employment, then re-screen. Changes are recorded in the audit log. To confirm a "
               "skill mapping, pick the approved skill and set status to VALID.")
    skill_df = pd.DataFrame(profile.get("skills") or [], columns=[
        "raw_name", "canonical_name", "match_type", "validation_status", "evidence", "confidence", "note", "grounded"])
    st.markdown("#### Skills")
    edited_skills = st.data_editor(
        skill_df, num_rows="dynamic", width="stretch", key=f"skills_{app_id}",
        column_config={
            "raw_name": st.column_config.TextColumn("In resume"),
            "canonical_name": st.column_config.SelectboxColumn("Approved skill", options=approved_skills()),
            "match_type": st.column_config.SelectboxColumn("Match", options=[m.value for m in MatchType]),
            "validation_status": st.column_config.SelectboxColumn("Status",
                                                                  options=[v.value for v in SkillValidationStatus]),
            "evidence": st.column_config.TextColumn("Evidence", width="large"),
            "confidence": st.column_config.NumberColumn("Confidence", disabled=True, format="%.2f"),
            "note": st.column_config.TextColumn("Note"),
            "grounded": st.column_config.CheckboxColumn("Found in text", disabled=True),
        })
    emp_df = pd.DataFrame(profile.get("employment_history") or [], columns=[
        "company", "title", "start_date", "end_date", "is_current", "employment_type", "evidence"])
    st.markdown("#### Employment history")
    edited_emp = st.data_editor(
        emp_df, num_rows="dynamic", width="stretch", key=f"emp_{app_id}",
        column_config={
            "company": st.column_config.TextColumn("Company"),
            "title": st.column_config.TextColumn("Title"),
            "start_date": st.column_config.TextColumn("Start (YYYY-MM)", help="YYYY-MM"),
            "end_date": st.column_config.TextColumn("End", help="YYYY-MM or 'present'"),
            "is_current": st.column_config.CheckboxColumn("Current"),
            "employment_type": st.column_config.SelectboxColumn("Type", options=[t.value for t in EmploymentType]),
            "evidence": st.column_config.TextColumn("Evidence", width="large"),
        })
    clear = st.checkbox("I have resolved the listed review reasons")
    if st.button("Save corrections and re-screen", type="primary"):
        try:
            skills = [_row_to_skill(r) for r in edited_skills.to_dict("records") if str(r.get("raw_name") or "").strip()]
            employment = [EmploymentEntry.model_validate(_clean_row(r)) for r in edited_emp.to_dict("records")
                          if any(str(v or "").strip() for k, v in r.items() if k != "is_current")]
        except (ValidationError, ValueError) as exc:
            st.error(f"Invalid correction: {exc}")
            return
        result = apply_hr_corrections(app_id, skills, employment, actor, clear, settings)
        st.success(f"Re-screened: {result['status']}")
        st.rerun()

    with st.expander("Audit history"):
        for when, action, who, details in history:
            st.text(f"{when}  {action}  by {who}  {details}")


def _clean_row(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if isinstance(v, float) and pd.isna(v):
            v = None
        if isinstance(v, str):
            v = v.strip() or None
        out[k] = v
    out["is_current"] = bool(out.get("is_current"))
    out["evidence"] = out.get("evidence") or "Entered by HR"
    out["employment_type"] = out.get("employment_type") or "UNKNOWN"
    return out


def _row_to_skill(row: dict) -> ValidatedSkill:
    r = _clean_row(row)
    canonical = r.get("canonical_name")
    if canonical:
        resolved, _ = resolve(canonical)
        if resolved is None:
            raise ValueError(f"'{canonical}' is not in the approved taxonomy")
        canonical = resolved
    match_type = r.get("match_type") or "UNKNOWN"
    if canonical is None:
        match_type = "UNKNOWN"
    elif match_type == "UNKNOWN":
        match_type = "CONTEXTUAL"  # HR mapped an unknown skill to a canonical one
    return ValidatedSkill(
        raw_name=r["raw_name"], canonical_name=canonical, match_type=match_type,
        validation_status=r.get("validation_status") or "REVIEW_REQUIRED",
        evidence=r["evidence"],
        confidence=float(r.get("confidence") or 0.0), note=r.get("note") or "Edited by HR",
        grounded=bool(r.get("grounded")),
    )


# =============================================================================
# Interview scheduler
# =============================================================================
def page_scheduler(actor: str) -> None:
    page_header("Interview scheduler", "Real panel availability, conflict-safe booking and invitations.", "HR team")
    wf = scheduling_workflow()
    agent = wf.agent
    if settings.mock_calendar:
        notice("Mock calendar", ["Availability shown here is synthetic demo data, not a real calendar."], "info")

    with st.expander("Maintenance"):
        c1, c2, _ = st.columns([1, 1, 1])
        if c1.button("Reconcile pending bookings", width="stretch"):
            report = agent.reconcile_pending(older_than_minutes=0)
            st.info(f"Reconciliation: {report}")
        if c2.button("Mark past interviews completed", width="stretch"):
            with session_scope() as s:
                n = repo.complete_past_interviews(s)
            st.info(f"{n} interview(s) marked completed.")

    st.markdown("### Schedule an approved candidate")
    with session_scope() as s:
        approved = {f"#{a.application_id} {a.name} - {repo.get_job(s, a.job_id).title}": a.application_id
                    for a in repo.list_applications(s, pipeline_statuses=["APPROVED"])}
    if not approved:
        st.info("No approved candidates waiting for an interview.")
    else:
        a1, a2 = st.columns([3, 1], vertical_alignment="bottom")
        app_id = approved[a1.selectbox("Candidate", list(approved))]
        if a2.button("Find available slots", type="primary", width="stretch"):
            with st.spinner("Checking panel availability..."):
                state = wf.propose(app_id)
            st.session_state["slots"] = {"app": app_id, "slots": state.get("available_slots", []),
                                         "msg": state.get("message")}
        cached = st.session_state.get("slots")
        if cached and cached["app"] == app_id:
            if not cached["slots"]:
                st.warning(cached["msg"] or "No available slots.")
            else:
                slots = [TimeSlot.model_validate(x) for x in cached["slots"]]
                choice = st.radio("Proposed slots", range(len(slots)), format_func=lambda i: fmt_slot(slots[i]))
                if st.button("Confirm booking", type="primary"):
                    with st.spinner("Booking..."):
                        state = wf.book(app_id, slots[choice], actor)
                    booking = state.get("booking") or {}
                    outcome = booking.get("outcome")
                    if outcome == "CONFIRMED":
                        st.success(f"Interview confirmed (#{booking.get('interview_id')}). "
                                   f"Invitation email: {state.get('email_status')}")
                        st.session_state.pop("slots", None)
                    elif outcome == "CONFLICT":
                        st.warning(booking.get("message"))
                        st.session_state["slots"] = {"app": app_id, "slots": state.get("available_slots", []),
                                                     "msg": "No alternative slots"}
                    else:
                        st.error(f"{outcome}: {state.get('message') or booking.get('message')}")

    st.markdown("### Interviews")
    with session_scope() as s:
        rows = []
        for i in repo.list_interviews(s):
            a = repo.get_application(s, i.application_id)
            rows.append({"ID": i.interview_id, "Candidate": a.name, "Stage": a.pipeline_status,
                         "Starts": fmt_dt(i.starts_at), "Ends": fmt_dt(i.ends_at), "Status": i.status,
                         "Email": i.email_status, "Attempts": i.email_attempts, "Meeting": i.meeting_link,
                         "Last error": i.last_error})
    if not rows:
        st.info("No interviews yet.")
        return
    tabs = st.tabs(["Confirmed / pending", "Completed", "Cancelled & failed"])
    groups = [("CONFIRMED", "PENDING_CONFIRMATION"), ("COMPLETED",), ("CANCELLED",)]
    for tab, statuses in zip(tabs, groups):
        with tab:
            subset = [r for r in rows if r["Status"] in statuses]
            if subset:
                table([{**r, "Status": status_icon(r["Status"]), "Email": status_icon(r["Email"]),
                        "Stage": status_icon(r["Stage"])} for r in subset],
                      column_config={"Meeting": st.column_config.LinkColumn("Meeting", display_text="Join")})
            else:
                st.caption("Nothing here yet.")

    st.markdown("### Manage an interview")
    m1, m2 = st.columns([2, 3])
    pick = m1.selectbox("Interview", [f"#{r['ID']}  {r['Candidate']}  ·  {r['Starts']}" for r in rows])
    row = next(r for r in rows if r["ID"] == int(pick.split()[0][1:]))
    reason = m2.text_input("Reason (for cancellation / decision)")
    b1, b2, b3, b4 = st.columns(4)
    if b4.button("Cancel interview", width="stretch",
                 disabled=row["Status"] not in ("CONFIRMED", "PENDING_CONFIRMATION")):
        try:
            agent.cancel(row["ID"], actor, reason or "Cancelled by HR")
            st.success("Cancelled - the candidate is back in 'Approved candidates' for rescheduling.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Cancellation failed: {exc}")
    if b2.button("Retry invitation email", width="stretch", disabled=not (row["Status"] == "CONFIRMED" and row["Email"] in
                                                         ("FAILED", "NOT_SENT"))):
        st.info(f"Email status: {agent.send_invitation(row['ID'])}")
    if b3.button("Allow resend", width="stretch", disabled=row["Email"] != "SENDING"):
        with session_scope() as s:
            repo.reset_stuck_email(s, row["ID"], actor)
        st.info("Marked FAILED - use 'Retry invitation email'. Check the outbox/SMTP logs first to avoid a duplicate.")
    if b1.button("Mark completed", type="primary", width="stretch", disabled=row["Status"] != "CONFIRMED"):
        with session_scope() as s:
            repo.complete_interview(s, row["ID"], actor)
        st.success("Marked completed.")

    if row["Stage"] == "INTERVIEWED":
        d1, d2 = st.columns(2)
        with session_scope() as s:
            app_id = repo.get_interview(s, row["ID"]).application_id
        if d1.button("Record decision: SELECTED", type="primary", width="stretch"):
            with session_scope() as s:
                repo.set_pipeline_status(s, app_id, PipelineStatus.SELECTED, actor, reason or "Selected after interview")
            st.success("Recorded.")
        if d2.button("Record decision: REJECTED", width="stretch"):
            with session_scope() as s:
                repo.set_pipeline_status(s, app_id, PipelineStatus.REJECTED, actor, reason or "Rejected after interview")
            st.success("Recorded.")


# =============================================================================
# Router
# =============================================================================
if page == "Candidate Portal":
    db_error_guard(page_candidate_portal)
elif page == "Application Status":
    db_error_guard(page_status)
else:
    hr_actor = require_hr()
    handlers = {"HR Dashboard": page_hr_dashboard, "Create Job": page_create_job,
                "Candidate Review": page_candidate_review, "Interview Scheduler": page_scheduler}
    db_error_guard(lambda: handlers[page](hr_actor))
