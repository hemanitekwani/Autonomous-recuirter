"""Pydantic models: the JSON contracts exchanged between agents, services and the DB."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator

YEAR_MONTH_PATTERN = r"^\d{4}-(0[1-9]|1[0-2])$"
END_DATE_PATTERN = r"^(\d{4}-(0[1-9]|1[0-2])|present)$"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class MatchType(str, Enum):
    EXACT = "EXACT"            # raw text is the canonical name
    ALIAS = "ALIAS"            # known alias from the editable alias dictionary
    CONTEXTUAL = "CONTEXTUAL"  # related wording proposed by the LLM -> needs HR confirmation
    UNKNOWN = "UNKNOWN"        # not in the approved taxonomy


class SkillValidationStatus(str, Enum):
    VALID = "VALID"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    REJECTED = "REJECTED"


class EmploymentType(str, Enum):
    FULL_TIME = "FULL_TIME"
    PART_TIME = "PART_TIME"
    CONTRACT = "CONTRACT"
    FREELANCE = "FREELANCE"
    INTERNSHIP = "INTERNSHIP"
    ACADEMIC = "ACADEMIC"
    PERSONAL_PROJECT = "PERSONAL_PROJECT"
    UNKNOWN = "UNKNOWN"


class CheckResult(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


class ScreeningStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    ELIGIBLE = "ELIGIBLE"
    NOT_ELIGIBLE = "NOT_ELIGIBLE"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    ERROR = "ERROR"


class PipelineStatus(str, Enum):
    SCREENING = "SCREENING"
    APPROVED = "APPROVED"
    INTERVIEW_SCHEDULED = "INTERVIEW_SCHEDULED"
    INTERVIEWED = "INTERVIEWED"
    SELECTED = "SELECTED"
    REJECTED = "REJECTED"
    SUPERSEDED = "SUPERSEDED"  # older attempt replaced by a newer application for the same job


class InterviewStatus(str, Enum):
    PROPOSED = "PROPOSED"
    PENDING_CONFIRMATION = "PENDING_CONFIRMATION"
    CONFIRMED = "CONFIRMED"
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"


class EmailStatus(str, Enum):
    NOT_SENT = "NOT_SENT"
    SENDING = "SENDING"
    SENT = "SENT"
    FAILED = "FAILED"


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------
class SkillRequirement(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    canonical_name: str = Field(min_length=1, max_length=100)
    minimum_proficiency: Optional[str] = Field(default=None, max_length=50)


class JobSkills(StrictModel):
    mandatory: list[SkillRequirement] = Field(default_factory=list, max_length=50)
    preferred: list[SkillRequirement] = Field(default_factory=list, max_length=50)

    @model_validator(mode="after")
    def _dedupe_and_check(self) -> "JobSkills":
        from src.skill_normalizer import is_canonical

        def dedupe(items: list[SkillRequirement]) -> list[SkillRequirement]:
            seen: set[str] = set()
            out: list[SkillRequirement] = []
            for item in items:
                key = item.canonical_name.casefold()
                if key not in seen:
                    seen.add(key)
                    out.append(item)
            return out

        self.mandatory = dedupe(self.mandatory)
        self.preferred = dedupe(self.preferred)
        unknown = [s.canonical_name for s in self.mandatory + self.preferred if not is_canonical(s.canonical_name)]
        if unknown:
            raise ValueError(f"Skills not in the approved taxonomy (add them first): {', '.join(unknown)}")
        both = {s.canonical_name.casefold() for s in self.mandatory} & {s.canonical_name.casefold() for s in self.preferred}
        if both:
            raise ValueError(f"Skill(s) listed as both mandatory and preferred: {', '.join(sorted(both))}")
        if not self.mandatory:
            raise ValueError("At least one mandatory skill is required")
        return self


class JobCreate(StrictModel):
    title: str = Field(min_length=1, max_length=200)
    min_experience_years: int = Field(ge=0, le=50)
    skills: JobSkills
    interview_duration_minutes: int = Field(default=45, ge=15, le=240)
    status: Literal["OPEN", "CLOSED"] = "OPEN"


# ---------------------------------------------------------------------------
# Candidate submission
# ---------------------------------------------------------------------------
class CandidateSubmission(StrictModel):
    job_id: int = Field(gt=0)
    name: str = Field(min_length=1, max_length=200)
    email: EmailStr

    @field_validator("email")
    @classmethod
    def _lower_email(cls, v: str) -> str:
        return v.lower()


# ---------------------------------------------------------------------------
# LLM Call 1 - resume extraction (facts only)
# ---------------------------------------------------------------------------
class ExtractedSkill(StrictModel):
    raw_name: str = Field(min_length=1, max_length=100)
    evidence: str = Field(min_length=1, max_length=500)


class EmploymentEntry(StrictModel):
    company: Optional[str] = Field(default=None, max_length=200)
    title: Optional[str] = Field(default=None, max_length=200)
    start_date: Optional[str] = Field(default=None, pattern=YEAR_MONTH_PATTERN)
    end_date: Optional[str] = Field(default=None, pattern=END_DATE_PATTERN)
    is_current: bool = False
    employment_type: EmploymentType = EmploymentType.UNKNOWN
    evidence: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def _current_consistency(self) -> "EmploymentEntry":
        if self.end_date == "present":
            self.is_current = True
        return self


class ResumeExtraction(StrictModel):
    full_name: Optional[str] = Field(default=None, max_length=200)
    email: Optional[str] = Field(default=None, max_length=254)
    phone: Optional[str] = Field(default=None, max_length=50)
    skills: list[ExtractedSkill] = Field(default_factory=list, max_length=200)
    employment_history: list[EmploymentEntry] = Field(default_factory=list, max_length=40)
    ambiguities: list[str] = Field(default_factory=list, max_length=50)
    suspected_prompt_injection: bool = False


# ---------------------------------------------------------------------------
# Python normalization output
# ---------------------------------------------------------------------------
class NormalizedSkill(StrictModel):
    raw_name: str = Field(max_length=100)
    canonical_name: Optional[str] = Field(default=None, max_length=100)
    match_type: MatchType
    evidence: str = Field(default="", max_length=500)


# ---------------------------------------------------------------------------
# LLM Call 2 - profile / skill validation
# ---------------------------------------------------------------------------
class SkillValidation(StrictModel):
    raw_name: str = Field(min_length=1, max_length=100)
    canonical_name: Optional[str] = Field(default=None, max_length=100)
    match_type: MatchType
    validation_status: SkillValidationStatus
    evidence: str = Field(default="", max_length=500)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    note: Optional[str] = Field(default=None, max_length=300)


class ProfileValidationOutput(StrictModel):
    validated_skills: list[SkillValidation] = Field(default_factory=list, max_length=200)
    employment_issues: list[str] = Field(default_factory=list, max_length=50)
    requires_review: bool = False
    review_reasons: list[str] = Field(default_factory=list, max_length=50)


# ---------------------------------------------------------------------------
# Final validated profile (applications.profile / applications.skills)
# ---------------------------------------------------------------------------
class ValidatedSkill(SkillValidation):
    grounded: bool = Field(description="Evidence was found verbatim in the resume text (Python check)")


class CountedPeriod(StrictModel):
    company: Optional[str] = None
    start: str
    end: str
    months: int


class ExperienceCalculation(StrictModel):
    total_months: Optional[int] = Field(default=None, ge=0)
    completed_years: Optional[int] = Field(default=None, ge=0)
    method: str
    reference_month: str
    counted_periods: list[CountedPeriod] = Field(default_factory=list)
    excluded_entries: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class ParserInfo(StrictModel):
    mode: Literal["mock", "llm"]
    model: Optional[str] = None
    prompt_version: str
    repair_attempts: int = 0


class ParsedProfile(StrictModel):
    extracted_name: Optional[str] = None
    extracted_email: Optional[str] = None
    phone: Optional[str] = None
    skills: list[ValidatedSkill] = Field(default_factory=list)
    employment_history: list[EmploymentEntry] = Field(default_factory=list)
    experience: Optional[ExperienceCalculation] = None
    text_skill_mentions: list[str] = Field(
        default_factory=list,
        description="Canonical skills whose name/alias appears anywhere in the resume text (deterministic scan)",
    )
    requires_review: bool = False
    review_reasons: list[str] = Field(default_factory=list, description="Blocking: prevent an ELIGIBLE decision")
    notes: list[str] = Field(default_factory=list, description="Informational only")
    parser: Optional[ParserInfo] = None
    human_edited: bool = False


# ---------------------------------------------------------------------------
# Screening
# ---------------------------------------------------------------------------
class RequirementCheck(StrictModel):
    requirement: str
    kind: Literal["EXPERIENCE", "MANDATORY_SKILL", "PREFERRED_SKILL"]
    result: CheckResult
    required_value: Optional[int] = None
    actual_value: Optional[int] = None
    match_type: Optional[MatchType] = None
    evidence: Optional[str] = None
    reason: Optional[str] = None


class ScreeningResult(StrictModel):
    application_id: int
    job_id: int
    status: Literal["ELIGIBLE", "NOT_ELIGIBLE", "REVIEW_REQUIRED"]
    checks: list[RequirementCheck]
    preferred_matched: list[str] = Field(default_factory=list)
    review_reasons: list[str] = Field(default_factory=list)
    policy_version: str
    evaluated_at: datetime

    @field_validator("evaluated_at")
    @classmethod
    def _aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("evaluated_at must be timezone-aware")
        return v


# ---------------------------------------------------------------------------
# Scheduling / email
# ---------------------------------------------------------------------------
class TimeSlot(StrictModel):
    starts_at: datetime
    ends_at: datetime
    timezone: str

    @model_validator(mode="after")
    def _check(self) -> "TimeSlot":
        if self.starts_at.tzinfo is None or self.ends_at.tzinfo is None:
            raise ValueError("slot datetimes must be timezone-aware")
        if self.ends_at <= self.starts_at:
            raise ValueError("slot must end after it starts")
        return self


class BookingResult(StrictModel):
    outcome: Literal["CONFIRMED", "CONFLICT", "CALENDAR_ERROR", "DB_ERROR", "INVALID"]
    interview_id: Optional[int] = None
    message: str
    alternative_slots: list[TimeSlot] = Field(default_factory=list)


class EmailMessageContent(StrictModel):
    to: EmailStr
    subject: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=10000)
    ics: Optional[str] = None
