"""Seed demo data.

  python -m scripts.seed_demo                 # create sample jobs + build sample PDF/DOCX resumes
  python -m scripts.seed_demo --submit        # also submit every sample resume to the ML job and screen it

Sample resumes are generated into sample_data/resumes/generated/ from the .txt sources.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sqlalchemy import select

from src import repository as repo
from src.config import PROJECT_ROOT, get_settings
from src.database import Job, session_scope
from src.recruitment_workflow import IntakeWorkflow
from src.resume_extractor import save_resume, validate_upload
from src.schemas import CandidateSubmission, JobCreate, JobSkills
from src.skill_normalizer import normalize_job_skills

SAMPLE_DIR = PROJECT_ROOT / "sample_data"
GENERATED = SAMPLE_DIR / "resumes" / "generated"


def build_pdf(text: str, path: Path) -> None:
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(50, 50, 545, 800), text, fontsize=10, fontname="helv")
    doc.save(str(path))
    doc.close()


def build_docx(text: str, path: Path) -> None:
    import docx

    document = docx.Document()
    for line in text.splitlines():
        document.add_paragraph(line)
    document.save(str(path))


def generate_resumes() -> list[Path]:
    GENERATED.mkdir(parents=True, exist_ok=True)
    out = []
    for i, src in enumerate(sorted((SAMPLE_DIR / "resumes").glob("*.txt"))):
        text = src.read_text(encoding="utf-8")
        target = GENERATED / (src.stem + (".pdf" if i % 2 == 0 else ".docx"))
        (build_pdf if target.suffix == ".pdf" else build_docx)(text, target)
        out.append(target)
    return out


def seed_jobs() -> dict[str, int]:
    ids = {}
    for spec in json.loads((SAMPLE_DIR / "jobs.json").read_text(encoding="utf-8")):
        with session_scope() as s:
            existing = s.scalars(select(Job).where(Job.title == spec["title"])).first()
            if existing:
                ids[spec["title"]] = existing.job_id
                continue
            skills, unknown = normalize_job_skills(spec["mandatory_skills"], spec["preferred_skills"])
            if unknown:
                raise SystemExit(f"Unknown skills in sample job: {unknown}")
            job = JobCreate(title=spec["title"], min_experience_years=spec["min_experience_years"],
                            skills=JobSkills.model_validate(skills),
                            interview_duration_minutes=spec["interview_duration_minutes"], status=spec["status"])
            ids[spec["title"]] = repo.create_job(s, job, "seed")
    return ids


def submit_samples(job_id: int, resumes: list[Path]) -> None:
    settings = get_settings()
    workflow = IntakeWorkflow(settings)
    for path in resumes:
        source = (SAMPLE_DIR / "resumes" / (path.stem + ".txt")).read_text(encoding="utf-8").splitlines()
        name, email = source[0].strip(), source[1].split("|")[0].strip()
        data = path.read_bytes()
        ext = validate_upload(path.name, data, settings.max_resume_size_mb)
        try:
            with session_scope() as s:
                stored = save_resume(data, ext, settings.resume_storage_dir)
                app_id = repo.create_application(s, CandidateSubmission(job_id=job_id, name=name, email=email),
                                                 stored.name)
        except repo.DuplicateApplicationError:
            stored.unlink(missing_ok=True)
            print(f"  {name}: already applied - skipped")
            continue
        state = workflow.run(app_id)
        print(f"  #{app_id} {name}: {state.get('status')} - {state.get('message')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true", help="submit sample resumes to the ML job")
    args = parser.parse_args()
    job_ids = seed_jobs()
    print("Jobs:", job_ids)
    resumes = generate_resumes()
    print("Sample resumes:", ", ".join(str(p.relative_to(PROJECT_ROOT)) for p in resumes))
    if args.submit:
        print("Submitting samples to 'Machine Learning Engineer':")
        submit_samples(job_ids["Machine Learning Engineer"], resumes)


if __name__ == "__main__":
    main()
