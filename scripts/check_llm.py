"""Check that the configured LLM works: runs both resume calls on ONE resume (no database needed).

Usage:
  python -m scripts.check_llm                                   # default sample resume
  python -m scripts.check_llm path/to/resume.pdf                # your own PDF / DOCX / TXT
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from src.config import PROJECT_ROOT, get_settings
from src.experience_calculator import calculate_experience, reference_month
from src.llm_client import get_llm_client
from src.profile_validator_agent import call_validator, reconcile_skills
from src.resume_extractor import extract_text, validate_upload
from src.resume_parser_agent import extract_resume
from src.skill_normalizer import normalize_skills


def load_text(path: Path, max_mb: int) -> str:
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8")
    data = path.read_bytes()
    return extract_text(data, validate_upload(path.name, data, max_mb))


def main() -> None:
    settings = get_settings()
    llm = get_llm_client(settings)
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "sample_data" / "resumes" / "priya_sharma.txt"
    print(f"LLM: {settings.llm_provider if llm.mode == 'llm' else 'MOCK'} / {llm.model or 'rule-based'}")
    print(f"Resume: {path}\n")
    text = load_text(path, settings.max_resume_size_mb)

    start = time.time()
    extraction, repairs1 = extract_resume(text, llm, settings.llm_max_repair_attempts)
    print(f"[Call 1 - extraction] {time.time() - start:.1f}s, repairs={repairs1}")
    print(f"  name : {extraction.full_name}\n  email: {extraction.email}")
    for job in extraction.employment_history:
        print(f"  job  : {job.title} | {job.company} | {job.start_date} -> {job.end_date} | {job.employment_type.value}")

    normalized = normalize_skills(extraction.skills)
    start = time.time()
    output, repairs2 = call_validator(text, extraction, normalized, llm, settings.llm_max_repair_attempts)
    print(f"\n[Call 2 - validation] {time.time() - start:.1f}s, repairs={repairs2}")
    for skill in reconcile_skills(text, normalized, output):
        print(f"  {skill.raw_name:<22} -> {str(skill.canonical_name):<28} {skill.match_type.value:<10} "
              f"{skill.validation_status.value}")
    for issue in output.employment_issues + output.review_reasons:
        print(f"  flag : {issue}")

    calc = calculate_experience(extraction.employment_history, reference_month(settings.default_timezone),
                                settings.count_internships)
    years = "unknown" if calc.completed_years is None else f"{calc.completed_years} years ({calc.total_months} months)"
    print(f"\n[Experience - Python] {years}")
    for reason in calc.reasons:
        print(f"  note : {reason}")
    print("\nOK - LLM pipeline is working.")


if __name__ == "__main__":
    main()
