"""Deterministic screening engine - SQL + Python. No LLM is involved in eligibility.

Step 1 (SQL, `REQUIREMENT_MATCH_SQL`): join the application with its job, expand
the job's JSONB requirements and find, for each requirement, the best matching
validated candidate skill (VALID before REVIEW_REQUIRED before REJECTED;
EXACT before ALIAS before CONTEXTUAL).

Step 2 (Python, `decide`): apply the decision policy below.

Per requirement:
  * experience  : NULL -> UNKNOWN, >= minimum -> PASS, else FAIL
  * skill       : best match VALID                -> PASS
                  best match REVIEW_REQUIRED      -> UNKNOWN (contextual / disputed mapping)
                  no usable match but the skill name/alias occurs in the resume text
                                                  -> UNKNOWN (extraction miss is possible)
                  otherwise                       -> FAIL (not mentioned anywhere in the resume)
Overall:
  * any mandatory requirement FAIL                -> NOT_ELIGIBLE
  * else any UNKNOWN or blocking profile review   -> REVIEW_REQUIRED
  * else                                          -> ELIGIBLE
Preferred skills are recorded but never change the status.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.orm import Session

from src import repository
from src.schemas import (
    CheckResult,
    MatchType,
    PipelineStatus,
    RequirementCheck,
    ScreeningResult,
    SkillValidationStatus,
)

POLICY_VERSION = "screen-v1"

REQUIREMENT_MATCH_SQL = text(
    """
    WITH app AS (
        SELECT a.application_id, a.skills, a.profile, j.skills AS job_skills
          FROM applications a
          JOIN jobs j ON j.job_id = a.job_id
         WHERE a.application_id = :application_id
    ),
    requirements AS (
        SELECT 'MANDATORY_SKILL' AS kind, r.value ->> 'canonical_name' AS skill, r.ord
          FROM app, jsonb_array_elements(app.job_skills -> 'mandatory') WITH ORDINALITY AS r(value, ord)
        UNION ALL
        SELECT 'PREFERRED_SKILL' AS kind, r.value ->> 'canonical_name' AS skill, r.ord
          FROM app, jsonb_array_elements(app.job_skills -> 'preferred') WITH ORDINALITY AS r(value, ord)
    )
    SELECT req.kind,
           req.skill,
           best.match_type,
           best.validation_status,
           best.evidence,
           EXISTS (
               SELECT 1
                 FROM jsonb_array_elements_text(COALESCE(app.profile -> 'text_skill_mentions', '[]'::jsonb)) AS m(v)
                WHERE lower(m.v) = lower(req.skill)
           ) AS mentioned_in_text
      FROM requirements req
     CROSS JOIN app
      LEFT JOIN LATERAL (
           SELECT s.value ->> 'match_type'        AS match_type,
                  s.value ->> 'validation_status' AS validation_status,
                  s.value ->> 'evidence'          AS evidence
             FROM jsonb_array_elements(app.skills) AS s(value)
            WHERE lower(s.value ->> 'canonical_name') = lower(req.skill)
            ORDER BY CASE s.value ->> 'validation_status'
                         WHEN 'VALID' THEN 0 WHEN 'REVIEW_REQUIRED' THEN 1 ELSE 2 END,
                     CASE s.value ->> 'match_type'
                         WHEN 'EXACT' THEN 0 WHEN 'ALIAS' THEN 1 WHEN 'CONTEXTUAL' THEN 2 ELSE 3 END
            LIMIT 1
      ) AS best ON TRUE
     ORDER BY req.kind, req.ord
    """
)

APPLICATION_FACTS_SQL = text(
    """
    SELECT a.application_id, a.job_id, a.experience, j.min_experience_years,
           COALESCE(a.profile -> 'review_reasons', '[]'::jsonb) AS review_reasons
      FROM applications a
      JOIN jobs j ON j.job_id = a.job_id
     WHERE a.application_id = :application_id
    """
)

_STATUS_RANK = {"VALID": 0, "REVIEW_REQUIRED": 1, "REJECTED": 2}
_MATCH_RANK = {"EXACT": 0, "ALIAS": 1, "CONTEXTUAL": 2, "UNKNOWN": 3}


@dataclass(frozen=True)
class RequirementMatch:
    kind: str  # MANDATORY_SKILL | PREFERRED_SKILL
    skill: str
    match_type: str | None
    validation_status: str | None
    evidence: str | None
    mentioned_in_text: bool


def match_requirements_py(job_skills: dict, app_skills: list[dict], text_mentions: list[str]) -> list[RequirementMatch]:
    """Pure-Python equivalent of REQUIREMENT_MATCH_SQL (used in unit tests and for parity checks)."""
    mentions = {m.casefold() for m in text_mentions}
    out: list[RequirementMatch] = []
    for kind, key in (("MANDATORY_SKILL", "mandatory"), ("PREFERRED_SKILL", "preferred")):
        for req in job_skills.get(key, []):
            name = req["canonical_name"]
            candidates = [s for s in app_skills if (s.get("canonical_name") or "").casefold() == name.casefold()]
            candidates.sort(key=lambda s: (_STATUS_RANK.get(s.get("validation_status"), 3),
                                           _MATCH_RANK.get(s.get("match_type"), 4)))
            best = candidates[0] if candidates else None
            out.append(RequirementMatch(
                kind=kind, skill=name,
                match_type=best.get("match_type") if best else None,
                validation_status=best.get("validation_status") if best else None,
                evidence=best.get("evidence") if best else None,
                mentioned_in_text=name.casefold() in mentions,
            ))
    return out


def _skill_check(m: RequirementMatch) -> RequirementCheck:
    match_type = MatchType(m.match_type) if m.match_type else None
    if m.validation_status == SkillValidationStatus.VALID.value:
        return RequirementCheck(requirement=m.skill, kind=m.kind, result=CheckResult.PASS, match_type=match_type,
                                evidence=m.evidence)
    if m.validation_status == SkillValidationStatus.REVIEW_REQUIRED.value:
        return RequirementCheck(requirement=m.skill, kind=m.kind, result=CheckResult.UNKNOWN, match_type=match_type,
                                evidence=m.evidence, reason="Skill mapping needs HR confirmation")
    if m.mentioned_in_text:
        return RequirementCheck(requirement=m.skill, kind=m.kind, result=CheckResult.UNKNOWN, match_type=match_type,
                                reason="Mentioned in the resume text but not validated as a skill")
    return RequirementCheck(requirement=m.skill, kind=m.kind, result=CheckResult.FAIL,
                            reason="Not found in the resume")


def decide(application_id: int, job_id: int, min_experience_years: int, experience: int | None,
           matches: list[RequirementMatch], profile_review_reasons: list[str],
           now: datetime | None = None) -> ScreeningResult:
    checks: list[RequirementCheck] = []
    if experience is None:
        checks.append(RequirementCheck(requirement="minimum_experience", kind="EXPERIENCE", result=CheckResult.UNKNOWN,
                                       required_value=min_experience_years, reason="Experience could not be calculated"))
    else:
        checks.append(RequirementCheck(
            requirement="minimum_experience", kind="EXPERIENCE",
            result=CheckResult.PASS if experience >= min_experience_years else CheckResult.FAIL,
            required_value=min_experience_years, actual_value=experience,
        ))
    checks.extend(_skill_check(m) for m in matches)

    blocking = [c for c in checks if c.kind != "PREFERRED_SKILL"]
    review_reasons = list(profile_review_reasons)
    if any(c.result == CheckResult.FAIL for c in blocking):
        status = "NOT_ELIGIBLE"
    elif any(c.result == CheckResult.UNKNOWN for c in blocking) or review_reasons:
        status = "REVIEW_REQUIRED"
        review_reasons += [f"{c.requirement}: {c.reason}" for c in blocking if c.result == CheckResult.UNKNOWN]
    else:
        status = "ELIGIBLE"

    return ScreeningResult(
        application_id=application_id,
        job_id=job_id,
        status=status,
        checks=checks,
        preferred_matched=[c.requirement for c in checks if c.kind == "PREFERRED_SKILL" and c.result == CheckResult.PASS],
        review_reasons=review_reasons,
        policy_version=POLICY_VERSION,
        evaluated_at=now or datetime.now(timezone.utc),
    )


def fetch_requirement_matches(session: Session, application_id: int) -> list[RequirementMatch]:
    rows = session.execute(REQUIREMENT_MATCH_SQL, {"application_id": application_id}).mappings().all()
    return [RequirementMatch(kind=r["kind"], skill=r["skill"], match_type=r["match_type"],
                             validation_status=r["validation_status"], evidence=r["evidence"],
                             mentioned_in_text=bool(r["mentioned_in_text"])) for r in rows]


def screen_application(session: Session, application_id: int, auto_approve: bool = False) -> ScreeningResult:
    """Evaluate one application and persist result + status in the caller's transaction."""
    facts = session.execute(APPLICATION_FACTS_SQL, {"application_id": application_id}).mappings().first()
    if facts is None:
        raise repository.NotFoundError(f"Application {application_id} not found")
    result = decide(
        application_id=facts["application_id"],
        job_id=facts["job_id"],
        min_experience_years=facts["min_experience_years"],
        experience=facts["experience"],
        matches=fetch_requirement_matches(session, application_id),
        profile_review_reasons=[str(r) for r in facts["review_reasons"]],
    )
    repository.save_screening(session, application_id, result)
    repository.audit(session, "application", application_id, "SCREENED", "screening_engine",
                     {"status": result.status, "policy": POLICY_VERSION})
    if auto_approve and result.status == "ELIGIBLE":
        app = repository.get_application(session, application_id)
        if app.pipeline_status == PipelineStatus.SCREENING.value:
            repository.set_pipeline_status(session, application_id, PipelineStatus.APPROVED, "auto-approval",
                                           "AUTO_APPROVE_ELIGIBLE policy")
    return result
