"""LangGraph orchestration.

Two small graphs share one explicit state schema:

  INTAKE graph (runs when a resume is submitted / re-processed)
    extract_resume_text -> parse_resume -> normalize_skills -> validate_profile
      -> calculate_experience -> persist_application -> screen_candidate
      -> human_review (REVIEW_REQUIRED) | finish
    any failure -> handle_error

  SCHEDULING graph (runs when HR asks for slots / confirms a slot)
    find_available_slots -> [no slot selected] finish
                         -> confirm_booking -> CONFIRMED -> send_invitation -> finish
                                            -> CONFLICT  -> find_available_slots (bounded) -> finish
                                            -> error     -> handle_error

Human review is a durable DATABASE state (screening_status=REVIEW_REQUIRED,
pipeline_status), not a paused graph. Reviews can take days, so the three
business tables stay the single source of truth and no checkpoint tables are
needed; every node is idempotent and keyed by application_id, so any run can
simply be re-invoked.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from src import repository as repo
from src.calendar_service import CalendarError, get_calendar_service
from src.config import Settings, get_settings
from src.database import session_scope
from src.email_service import get_email_sender
from src.experience_calculator import calculate_experience, reference_month
from src.llm_client import LLMClient, LLMError, get_llm_client
from src.profile_validator_agent import PROMPT_VERSION as VALIDATE_PROMPT_VERSION
from src.profile_validator_agent import ValidatorOutputError, call_validator, reconcile_skills
from src.resume_extractor import ResumeError, extract_text_from_path, resolve_stored_path
from src.resume_parser_agent import PROMPT_VERSION as EXTRACT_PROMPT_VERSION
from src.resume_parser_agent import ParserOutputError, extract_resume
from src.scheduling_agent import SchedulingAgent, SchedulingError
from src.schemas import (
    EmploymentEntry,
    ExperienceCalculation,
    NormalizedSkill,
    ParsedProfile,
    ParserInfo,
    ResumeExtraction,
    ScreeningStatus,
    TimeSlot,
    ValidatedSkill,
)
from src.screening_service import screen_application
from src.skill_normalizer import normalize_skills, scan_text_for_skills

log = logging.getLogger(__name__)
MAX_BOOKING_ATTEMPTS = 2


class WorkflowState(TypedDict, total=False):
    application_id: int
    job_id: Optional[int]
    resume_path: Optional[str]
    resume_text: Optional[str]
    parsed_profile: Optional[dict]       # ResumeExtraction (Call 1)
    normalized_profile: Optional[list]   # list[NormalizedSkill]
    validated_profile: Optional[dict]    # ParsedProfile (after Call 2 + Python guards)
    screening_result: Optional[dict]
    available_slots: list
    selected_slot: Optional[dict]
    booking: Optional[dict]
    booking_attempts: int
    email_status: Optional[str]
    status: str
    actor: Optional[str]
    message: Optional[str]
    errors: list
    repair_attempts: int


# =============================================================================
# Intake graph
# =============================================================================
class IntakeWorkflow:
    def __init__(self, settings: Settings | None = None, llm: LLMClient | None = None):
        self.settings = settings or get_settings()
        self.llm = llm or get_llm_client(self.settings)
        self.graph = self._build()

    # ---------------------------------------------------------------- nodes
    def extract_resume_text(self, state: WorkflowState) -> dict[str, Any]:
        with session_scope() as s:
            app = repo.get_application(s, state["application_id"])
            repo.set_screening_status(s, app.application_id, ScreeningStatus.PROCESSING)
            job_id, stored = app.job_id, app.resume_path
        if not stored:
            raise ResumeError("No resume stored for this application")
        path = resolve_stored_path(stored, self.settings.resume_storage_dir)
        return {"job_id": job_id, "resume_path": str(path), "resume_text": extract_text_from_path(path)}

    def parse_resume(self, state: WorkflowState) -> dict[str, Any]:
        extraction, repairs = extract_resume(state["resume_text"], self.llm, self.settings.llm_max_repair_attempts)
        return {"parsed_profile": extraction.model_dump(mode="json"), "repair_attempts": repairs}

    def normalize_skills_node(self, state: WorkflowState) -> dict[str, Any]:
        extraction = ResumeExtraction.model_validate(state["parsed_profile"])
        return {"normalized_profile": [n.model_dump(mode="json") for n in normalize_skills(extraction.skills)]}

    def validate_profile(self, state: WorkflowState) -> dict[str, Any]:
        text = state["resume_text"]
        extraction = ResumeExtraction.model_validate(state["parsed_profile"])
        normalized = [NormalizedSkill.model_validate(n) for n in state["normalized_profile"]]
        output, repairs = call_validator(text, extraction, normalized, self.llm, self.settings.llm_max_repair_attempts)
        skills = reconcile_skills(text, normalized, output)

        review_reasons: list[str] = list(output.employment_issues)
        if output.requires_review:
            review_reasons += output.review_reasons
        notes: list[str] = list(extraction.ambiguities)
        if extraction.suspected_prompt_injection:
            review_reasons.append("Resume contains instruction-like text (possible prompt injection)")
        rejected = [s.raw_name for s in skills if s.validation_status.value == "REJECTED"]
        if rejected:
            notes.append("Unsupported skills rejected (no evidence in resume): " + ", ".join(rejected[:15]))

        profile = ParsedProfile(
            extracted_name=extraction.full_name,
            extracted_email=extraction.email,
            phone=extraction.phone,
            skills=skills,
            employment_history=extraction.employment_history,
            text_skill_mentions=scan_text_for_skills(text),
            review_reasons=_dedupe(review_reasons),
            notes=_dedupe(notes),
            parser=ParserInfo(mode=self.llm.mode, model=self.llm.model,
                              prompt_version=f"{EXTRACT_PROMPT_VERSION}+{VALIDATE_PROMPT_VERSION}",
                              repair_attempts=state.get("repair_attempts", 0) + repairs),
        )
        return {"validated_profile": profile.model_dump(mode="json")}

    def calculate_experience_node(self, state: WorkflowState) -> dict[str, Any]:
        profile = ParsedProfile.model_validate(state["validated_profile"])
        calc = calculate_experience(profile.employment_history, reference_month(self.settings.default_timezone),
                                    self.settings.count_internships)
        profile.experience = calc
        if calc.completed_years is None:
            profile.review_reasons = _dedupe(profile.review_reasons + ["Experience could not be calculated reliably"])
        return {"validated_profile": profile.model_dump(mode="json")}

    def persist_application(self, state: WorkflowState) -> dict[str, Any]:
        profile = ParsedProfile.model_validate(state["validated_profile"])
        with session_scope() as s:
            app = repo.get_application(s, state["application_id"])
            if profile.extracted_email and profile.extracted_email.lower() != app.email.lower():
                profile.notes = _dedupe(profile.notes + [
                    f"Resume email ({profile.extracted_email}) differs from the email entered in the form"])
            if app.name == repo.NAME_FROM_RESUME and profile.extracted_name:
                app.name = profile.extracted_name[:200]  # upload-only application: name comes from the resume
            profile.requires_review = bool(profile.review_reasons)
            repo.save_profile(s, app.application_id, profile)
        return {"validated_profile": profile.model_dump(mode="json")}

    def screen_candidate(self, state: WorkflowState) -> dict[str, Any]:
        with session_scope() as s:
            result = screen_application(s, state["application_id"], auto_approve=self.settings.auto_approve_eligible)
        return {"screening_result": result.model_dump(mode="json"), "status": result.status}

    def human_review(self, state: WorkflowState) -> dict[str, Any]:
        with session_scope() as s:
            repo.audit(s, "application", state["application_id"], "AWAITING_HR_REVIEW", "workflow",
                       {"reasons": (state.get("screening_result") or {}).get("review_reasons", [])[:10]})
        return {"message": "Queued for HR review"}

    def finish(self, state: WorkflowState) -> dict[str, Any]:
        return {"message": state.get("message") or f"Screening complete: {state.get('status')}"}

    def handle_error(self, state: WorkflowState) -> dict[str, Any]:
        error = (state.get("errors") or ["Unknown error"])[-1]
        with session_scope() as s:
            repo.set_screening_status(s, state["application_id"], ScreeningStatus.ERROR, error=error)
            repo.audit(s, "application", state["application_id"], "WORKFLOW_ERROR", "workflow", {"error": error})
        return {"status": "ERROR", "message": error}

    def parse_failed_review(self, state: WorkflowState) -> dict[str, Any]:
        """Parser/validator output invalid after bounded repairs -> manual data entry by HR."""
        error = (state.get("errors") or ["Automatic parsing failed"])[-1]
        profile = ParsedProfile(review_reasons=[f"Automatic parsing failed: {error}"],
                                experience=None, requires_review=True,
                                text_skill_mentions=scan_text_for_skills(state.get("resume_text") or ""))
        with session_scope() as s:
            repo.save_profile(s, state["application_id"], profile)
            result = screen_application(s, state["application_id"])
        return {"validated_profile": profile.model_dump(mode="json"),
                "screening_result": result.model_dump(mode="json"), "status": result.status}

    # --------------------------------------------------------------- wiring
    def _guard(self, fn, recoverable_to_review: bool = False):
        """Wrap a node: known failures are recorded in state and routed, never raised through the graph."""
        def wrapped(state: WorkflowState) -> dict[str, Any]:
            try:
                update = fn(state)
                update.setdefault("errors", state.get("errors", []))
                return update
            except (ParserOutputError, ValidatorOutputError) as exc:
                detail = f"{exc} ({'; '.join(exc.errors[:3])})"
                return {"errors": state.get("errors", []) + [detail],
                        "status": "PARSE_FAILED" if recoverable_to_review else "ERROR"}
            except (ResumeError, LLMError, repo.RepositoryError) as exc:
                return {"errors": state.get("errors", []) + [str(exc)], "status": "ERROR"}
            except Exception as exc:  # unexpected - keep the app usable, log details server-side
                log.exception("Intake node %s failed", getattr(fn, "__name__", fn))
                return {"errors": state.get("errors", []) + [f"Internal error: {type(exc).__name__}"],
                        "status": "ERROR"}
        return wrapped

    @staticmethod
    def _route(next_node: str):
        def router(state: WorkflowState) -> str:
            status = state.get("status")
            if status == "PARSE_FAILED":
                return "parse_failed_review"
            if status == "ERROR":
                return "handle_error"
            return next_node
        return router

    def _build(self):
        g = StateGraph(WorkflowState)
        g.add_node("extract_resume_text", self._guard(self.extract_resume_text))
        g.add_node("parse_resume", self._guard(self.parse_resume, recoverable_to_review=True))
        g.add_node("normalize_skills", self._guard(self.normalize_skills_node))
        g.add_node("validate_profile", self._guard(self.validate_profile, recoverable_to_review=True))
        g.add_node("calculate_experience", self._guard(self.calculate_experience_node))
        g.add_node("persist_application", self._guard(self.persist_application))
        g.add_node("screen_candidate", self._guard(self.screen_candidate))
        g.add_node("human_review", self.human_review)
        g.add_node("parse_failed_review", self._guard(self.parse_failed_review))
        g.add_node("finish", self.finish)
        g.add_node("handle_error", self.handle_error)

        g.add_edge(START, "extract_resume_text")
        chain = ["extract_resume_text", "parse_resume", "normalize_skills", "validate_profile",
                 "calculate_experience", "persist_application", "screen_candidate"]
        for current, nxt in zip(chain, chain[1:]):
            g.add_conditional_edges(current, self._route(nxt),
                                    {nxt: nxt, "handle_error": "handle_error",
                                     "parse_failed_review": "parse_failed_review"})

        def after_screen(state: WorkflowState) -> str:
            if state.get("status") == "ERROR":
                return "handle_error"
            return "human_review" if state.get("status") == "REVIEW_REQUIRED" else "finish"

        g.add_conditional_edges("screen_candidate", after_screen,
                                {"human_review": "human_review", "finish": "finish", "handle_error": "handle_error"})
        g.add_conditional_edges("parse_failed_review", after_screen,
                                {"human_review": "human_review", "finish": "finish", "handle_error": "handle_error"})
        g.add_edge("human_review", "finish")
        g.add_edge("finish", END)
        g.add_edge("handle_error", END)
        return g.compile()

    def run(self, application_id: int) -> WorkflowState:
        """Idempotent: re-running recomputes the profile and screening from the stored resume."""
        return self.graph.invoke({"application_id": application_id, "status": "PENDING", "errors": [],
                                  "available_slots": [], "booking_attempts": 0, "repair_attempts": 0})


# =============================================================================
# Re-screen after HR corrections (no LLM involved)
# =============================================================================
def apply_hr_corrections(application_id: int, skills: list[ValidatedSkill], employment: list[EmploymentEntry],
                         actor: str, clear_review_reasons: bool, settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    with session_scope() as s:
        app = repo.get_application(s, application_id, for_update=True)
        profile = ParsedProfile.model_validate(app.profile or {})
        profile.skills = skills
        profile.employment_history = employment
        calc: ExperienceCalculation = calculate_experience(employment, reference_month(settings.default_timezone),
                                                           settings.count_internships)
        profile.experience = calc
        if clear_review_reasons:
            profile.review_reasons = []
        if calc.completed_years is None:
            profile.review_reasons = _dedupe(profile.review_reasons + ["Experience could not be calculated reliably"])
        profile.requires_review = bool(profile.review_reasons)
        profile.human_edited = True
        repo.save_profile(s, application_id, profile, actor=actor)
        result = screen_application(s, application_id, auto_approve=settings.auto_approve_eligible)
        repo.audit(s, "application", application_id, "HR_RESCREEN", actor,
                   {"status": result.status, "cleared_review_reasons": clear_review_reasons})
    return result.model_dump(mode="json")


# =============================================================================
# Scheduling graph
# =============================================================================
class SchedulingWorkflow:
    def __init__(self, settings: Settings | None = None, agent: SchedulingAgent | None = None):
        self.settings = settings or get_settings()
        if agent is None:
            agent = SchedulingAgent(self.settings, get_calendar_service(self.settings),
                                    get_email_sender(self.settings))
        self.agent = agent
        self.graph = self._build()

    def find_available_slots(self, state: WorkflowState) -> dict[str, Any]:
        try:
            slots = self.agent.find_available_slots(state["application_id"])
        except (SchedulingError, CalendarError, repo.RepositoryError) as exc:
            return {"status": "ERROR", "errors": state.get("errors", []) + [str(exc)]}
        update: dict[str, Any] = {"available_slots": [x.model_dump(mode="json") for x in slots]}
        if not slots:
            update.update(status="NO_SLOTS", message="No free slots in the scheduling window - return to HR")
        return update

    def confirm_booking(self, state: WorkflowState) -> dict[str, Any]:
        slot = TimeSlot.model_validate(state["selected_slot"])
        result = self.agent.book_slot(state["application_id"], slot, actor=state.get("actor") or "hr")
        update: dict[str, Any] = {"booking": result.model_dump(mode="json"), "status": result.outcome,
                                  "message": result.message,
                                  "booking_attempts": state.get("booking_attempts", 0) + 1}
        if result.outcome == "CONFLICT":
            update["selected_slot"] = None
            update["available_slots"] = [x.model_dump(mode="json") for x in result.alternative_slots]
        return update

    def send_invitation(self, state: WorkflowState) -> dict[str, Any]:
        interview_id = (state.get("booking") or {}).get("interview_id")
        email_status = self.agent.send_invitation(interview_id)
        return {"email_status": email_status}

    def finish(self, state: WorkflowState) -> dict[str, Any]:
        return {}

    def handle_error(self, state: WorkflowState) -> dict[str, Any]:
        error = (state.get("errors") or [state.get("message") or "Scheduling failed"])[-1]
        return {"message": error}

    def _build(self):
        g = StateGraph(WorkflowState)
        g.add_node("find_available_slots", self.find_available_slots)
        g.add_node("confirm_booking", self.confirm_booking)
        g.add_node("send_invitation", self.send_invitation)
        g.add_node("finish", self.finish)
        g.add_node("handle_error", self.handle_error)

        def start_router(state: WorkflowState) -> str:
            return "confirm_booking" if state.get("selected_slot") else "find_available_slots"

        def after_find(state: WorkflowState) -> str:
            if state.get("status") == "ERROR":
                return "handle_error"
            return "finish"

        def after_confirm(state: WorkflowState) -> str:
            outcome = state.get("status")
            if outcome == "CONFIRMED":
                return "send_invitation"
            if outcome == "CONFLICT" and state.get("booking_attempts", 0) <= MAX_BOOKING_ATTEMPTS:
                return "find_available_slots" if not state.get("available_slots") else "finish"
            if outcome in ("CALENDAR_ERROR", "DB_ERROR", "INVALID"):
                return "handle_error"
            return "finish"

        g.add_conditional_edges(START, start_router,
                                {"confirm_booking": "confirm_booking", "find_available_slots": "find_available_slots"})
        g.add_conditional_edges("find_available_slots", after_find, {"finish": "finish", "handle_error": "handle_error"})
        g.add_conditional_edges("confirm_booking", after_confirm,
                                {"send_invitation": "send_invitation", "find_available_slots": "find_available_slots",
                                 "finish": "finish", "handle_error": "handle_error"})
        g.add_edge("send_invitation", "finish")
        g.add_edge("finish", END)
        g.add_edge("handle_error", END)
        return g.compile()

    def propose(self, application_id: int) -> WorkflowState:
        return self.graph.invoke({"application_id": application_id, "status": "PENDING", "errors": [],
                                  "available_slots": [], "booking_attempts": 0})

    def book(self, application_id: int, slot: TimeSlot, actor: str = "hr") -> WorkflowState:
        return self.graph.invoke({"application_id": application_id, "status": "PENDING", "errors": [],
                                  "selected_slot": slot.model_dump(mode="json"), "available_slots": [],
                                  "booking_attempts": 0, "actor": actor})


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    return [x for x in items if not (x in seen or seen.add(x))]
