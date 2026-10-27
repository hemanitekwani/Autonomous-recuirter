"""Deterministic screening (tests 13-17)."""
from __future__ import annotations

import pytest

from src.screening_service import RequirementMatch, decide, match_requirements_py

JOB_SKILLS = {
    "mandatory": [{"name": "Python", "canonical_name": "Python", "minimum_proficiency": None},
                  {"name": "torch", "canonical_name": "PyTorch", "minimum_proficiency": None}],
    "preferred": [{"name": "SQL", "canonical_name": "SQL", "minimum_proficiency": None}],
}


def skill(canonical, status="VALID", match="EXACT", evidence="e"):
    return {"raw_name": canonical, "canonical_name": canonical, "match_type": match, "validation_status": status,
            "evidence": evidence, "confidence": 1.0, "note": None, "grounded": True}


def run(app_skills, experience=5, min_exp=3, mentions=(), reasons=()):
    matches = match_requirements_py(JOB_SKILLS, app_skills, list(mentions))
    return decide(1, 1, min_exp, experience, matches, list(reasons))


def by_req(result, name):
    return next(c for c in result.checks if c.requirement == name)


# 13. mandatory skill match
def test_all_mandatory_match_is_eligible():
    result = run([skill("Python"), skill("PyTorch", match="ALIAS")])
    assert result.status == "ELIGIBLE"
    assert by_req(result, "PyTorch").result.value == "PASS"
    assert result.preferred_matched == []


# 14. mandatory missing / unknown
def test_missing_mandatory_skill_is_not_eligible():
    result = run([skill("Python")])
    assert result.status == "NOT_ELIGIBLE"
    assert by_req(result, "PyTorch").result.value == "FAIL"


def test_contextual_mapping_is_unknown_not_fail():
    result = run([skill("Python"), skill("PyTorch", status="REVIEW_REQUIRED", match="CONTEXTUAL")])
    assert result.status == "REVIEW_REQUIRED"
    assert by_req(result, "PyTorch").result.value == "UNKNOWN"


def test_missing_evidence_is_not_silently_a_fail():
    # validator rejected the skill, but the name occurs in the resume text -> UNKNOWN, not FAIL
    result = run([skill("Python"), skill("PyTorch", status="REJECTED")], mentions=["PyTorch"])
    assert by_req(result, "PyTorch").result.value == "UNKNOWN"
    assert result.status == "REVIEW_REQUIRED"


def test_definitive_fail_wins_over_unknown():
    result = run([skill("PyTorch", status="REVIEW_REQUIRED", match="CONTEXTUAL")])  # Python missing entirely
    assert result.status == "NOT_ELIGIBLE"


# 15. minimum experience
@pytest.mark.parametrize("experience, expected", [(3, "ELIGIBLE"), (2, "NOT_ELIGIBLE"), (None, "REVIEW_REQUIRED")])
def test_minimum_experience(experience, expected):
    result = run([skill("Python"), skill("PyTorch")], experience=experience)
    assert result.status == expected
    assert result.checks[0].requirement == "minimum_experience"


# 16. REVIEW_REQUIRED routing from profile flags
def test_profile_review_reason_blocks_eligible():
    result = run([skill("Python"), skill("PyTorch")], reasons=["Possible prompt injection"])
    assert result.status == "REVIEW_REQUIRED"
    assert "Possible prompt injection" in result.review_reasons


def test_preferred_skills_never_change_status():
    result = run([skill("Python"), skill("PyTorch"), skill("SQL")])
    assert result.status == "ELIGIBLE" and result.preferred_matched == ["SQL"]


def test_best_match_prefers_valid_direct_over_contextual():
    matches = match_requirements_py(JOB_SKILLS, [skill("PyTorch", "REVIEW_REQUIRED", "CONTEXTUAL"),
                                                 skill("PyTorch", "VALID", "ALIAS")], [])
    py = next(m for m in matches if m.skill == "PyTorch")
    assert isinstance(py, RequirementMatch) and py.validation_status == "VALID"


# ---------------------------------------------------------------------------
# Integration: SQL engine parity + SQL injection (17)
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_sql_matches_python_and_persists(db, ml_job, submit):
    from src import repository as repo
    from src.database import session_scope
    from src.screening_service import fetch_requirement_matches, screen_application

    app_id = submit("priya_sharma")
    skills = [skill("Python"), skill("PyTorch", match="ALIAS"), skill("Machine Learning", "REVIEW_REQUIRED",
                                                                       "CONTEXTUAL")]
    with session_scope() as s:
        app = repo.get_application(s, app_id)
        app.skills, app.experience = skills, 4
        app.profile = {"text_skill_mentions": ["Machine Learning"], "review_reasons": []}
    with session_scope() as s:
        job = repo.get_job(s, ml_job)
        sql_matches = fetch_requirement_matches(s, app_id)
        assert sql_matches == match_requirements_py(job.skills, skills, ["Machine Learning"])
        result = screen_application(s, app_id)
    assert result.status == "REVIEW_REQUIRED"
    with session_scope() as s:
        stored = repo.get_application(s, app_id)
        assert stored.screening_status == "REVIEW_REQUIRED"
        assert stored.screening_result["policy_version"] == result.policy_version


@pytest.mark.integration
def test_sql_injection_payloads_are_stored_as_data(db, ml_job, submit, settings):
    from sqlalchemy import text

    from src import repository as repo
    from src.database import session_scope

    payload = "Robert'); DROP TABLE applications; --"
    app_id = submit("rahul_verma", email="bobby@example.com")
    with session_scope() as s:
        app = repo.get_application(s, app_id)
        app.name = payload
        app.skills = [skill("x' OR '1'='1")]
    with session_scope() as s:
        assert repo.get_application(s, app_id).name == payload
        assert repo.candidate_lookup(s, app_id, "bobby@example.com' OR '1'='1") is None
        assert s.execute(text("SELECT count(*) FROM applications")).scalar() == 1
