"""Pydantic contracts + LLM output validation / bounded repair."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.llm_client import MockLLMClient
from src.resume_parser_agent import ParserOutputError, extract_resume
from src.schemas import CandidateSubmission, EmploymentEntry, JobCreate, JobSkills, ResumeExtraction, ScreeningResult
from src.skill_normalizer import normalize_job_skills


def _skills(mandatory, preferred=()):
    data, unknown = normalize_job_skills(list(mandatory), list(preferred))
    assert not unknown
    return data


# 1. valid job creation
def test_valid_job_creation():
    job = JobCreate(title="ML Engineer", min_experience_years=2,
                    skills=JobSkills.model_validate(_skills(["Python", "ML", "torch"], ["SQL"])))
    assert [s.canonical_name for s in job.skills.mandatory] == ["Python", "Machine Learning", "PyTorch"]
    assert job.skills.preferred[0].canonical_name == "SQL"
    assert job.interview_duration_minutes == 45


# 2. invalid / malformed job skill JSON
@pytest.mark.parametrize(
    "payload, message",
    [
        ({"mandatory": "Python", "preferred": []}, "list"),
        ({"mandatory": [{"name": "", "canonical_name": "Python"}], "preferred": []}, "at least 1"),
        ({"mandatory": [{"name": "Foo", "canonical_name": "Foo"}], "preferred": []}, "approved taxonomy"),
        ({"mandatory": [{"name": "Python", "canonical_name": "Python"}],
          "preferred": [{"name": "py", "canonical_name": "Python"}]}, "both mandatory and preferred"),
        ({"mandatory": [], "preferred": []}, "At least one mandatory"),
        ({"mandatory": [{"name": "Python", "canonical_name": "Python", "extra": 1}], "preferred": []}, "Extra"),
    ],
)
def test_invalid_job_skills(payload, message):
    with pytest.raises(ValidationError) as exc:
        JobSkills.model_validate(payload)
    assert message.lower() in str(exc.value).lower()


def test_job_skills_dedupes_canonical():
    js = JobSkills.model_validate({"mandatory": [{"name": "ML", "canonical_name": "Machine Learning"},
                                                 {"name": "machine learning", "canonical_name": "Machine Learning"}],
                                   "preferred": []})
    assert len(js.mandatory) == 1


def test_job_rejects_negative_experience():
    with pytest.raises(ValidationError):
        JobCreate(title="X", min_experience_years=-1, skills=JobSkills.model_validate(_skills(["Python"])))


def test_candidate_email_validation():
    assert CandidateSubmission(job_id=1, name="A B", email="A.B@Example.com").email == "a.b@example.com"
    with pytest.raises(ValidationError):
        CandidateSubmission(job_id=1, name="A B", email="not-an-email")


# 6. invalid Pydantic output
def test_resume_extraction_rejects_bad_dates_and_extra_fields():
    with pytest.raises(ValidationError):
        ResumeExtraction.model_validate({"employment_history": [{"start_date": "June 2020", "evidence": "x"}]})
    with pytest.raises(ValidationError):
        ResumeExtraction.model_validate({"skills": [], "total_experience": 5})
    entry = EmploymentEntry(start_date="2020-01", end_date="present")
    assert entry.is_current is True


def test_screening_result_requires_aware_timestamp():
    from datetime import datetime

    with pytest.raises(ValidationError):
        ScreeningResult(application_id=1, job_id=1, status="ELIGIBLE", checks=[], policy_version="x",
                        evaluated_at=datetime(2026, 1, 1))


class _ScriptedLLM:
    mode, model = "llm", "fake"

    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), []

    def generate_json(self, **kwargs):
        self.calls.append(kwargs)
        return self.outputs.pop(0)


def test_extraction_repairs_invalid_output_then_succeeds():
    bad = {"full_name": "X", "skills": [{"raw_name": "Python"}]}  # evidence missing
    good = {"full_name": "X", "email": None, "phone": None, "skills": [{"raw_name": "Python", "evidence": "Python"}],
            "employment_history": [], "ambiguities": [], "suspected_prompt_injection": False}
    llm = _ScriptedLLM([bad, good])
    extraction, repairs = extract_resume("Python developer resume text", llm, max_repair_attempts=2)
    assert repairs == 1 and extraction.skills[0].raw_name == "Python"
    assert "failed validation" in llm.calls[1]["user_content"]  # validation errors fed back


def test_extraction_gives_up_after_bounded_attempts():
    llm = _ScriptedLLM([{"skills": "nope"}] * 3)
    with pytest.raises(ParserOutputError):
        extract_resume("text", llm, max_repair_attempts=2)
    assert len(llm.calls) == 3


def test_mock_extraction_matches_contract():
    from tests.conftest import sample_text

    extraction, _ = extract_resume(sample_text("priya_sharma"), MockLLMClient())
    assert extraction.full_name == "Priya Sharma"
    assert extraction.email == "priya.sharma@example.com"
    titles = [e.title for e in extraction.employment_history]
    assert "Senior ML Engineer" in titles
