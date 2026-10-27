"""Deterministic rule-based stand-ins for the LLM tasks (MOCK_MODE / MOCK_LLM).

They are intentionally simple heuristics - good enough to demo and test the
full workflow with the bundled sample resumes, not a production parser.
They obey the same JSON contracts as the real LLM calls.
"""
from __future__ import annotations

import re
from typing import Any

from src.resume_parser_agent import detect_prompt_injection
from src.skill_normalizer import _scan_patterns

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
_MONTH_RE = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_DATE_RE = rf"(?:{_MONTH_RE}\s+\d{{4}}|\d{{4}}-\d{{1,2}}|\d{{1,2}}/\d{{4}}|\d{{4}})"
_RANGE_RE = re.compile(
    rf"(?P<start>{_DATE_RE})\s*(?:-|–|—|to|until)\s*(?P<end>{_DATE_RE}|present|current|now|till date)",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"\+?\d[\d\s\-()]{8,}\d")
_SKILL_LINE_RE = re.compile(r"^(?:technical\s+|key\s+|core\s+)?skills?\s*[:\-–]\s*(?P<items>.+)$", re.IGNORECASE)
_HEADINGS = {
    "experience": "work", "work experience": "work", "professional experience": "work", "employment": "work",
    "employment history": "work", "work history": "work",
    "education": "education", "academic background": "education", "academics": "education",
    "projects": "projects", "personal projects": "projects", "academic projects": "projects",
    "skills": "skills", "technical skills": "skills", "key skills": "skills", "core skills": "skills",
    "certifications": "other", "summary": "other", "profile": "other", "achievements": "other",
}


def _parse_date(token: str) -> tuple[str | None, bool]:
    """Return (YYYY-MM or None, year_only)."""
    t = token.strip().lower().rstrip(".")
    m = re.match(rf"^({_MONTH_RE})\s+(\d{{4}})$", t)
    if m:
        return f"{int(m.group(2)):04d}-{_MONTHS[m.group(1)[:3]]:02d}", False
    m = re.match(r"^(\d{4})-(\d{1,2})$", t)
    if m and 1 <= int(m.group(2)) <= 12:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}", False
    m = re.match(r"^(\d{1,2})/(\d{4})$", t)
    if m and 1 <= int(m.group(1)) <= 12:
        return f"{int(m.group(2)):04d}-{int(m.group(1)):02d}", False
    if re.match(r"^\d{4}$", t):
        return None, True
    return None, False


def _heading(line: str) -> str | None:
    key = line.strip().strip(":").strip().lower()
    return _HEADINGS.get(key)


def _employment_type(line: str, section: str) -> str:
    low = line.lower()
    if re.search(r"\bintern(ship)?\b", low):
        return "INTERNSHIP"
    if section == "education" or re.search(r"teaching assistant|research assistant|thesis", low):
        return "ACADEMIC"
    if section == "projects" or re.search(r"\bproject\b", low):
        return "PERSONAL_PROJECT"
    if re.search(r"part[\s-]time", low):
        return "PART_TIME"
    if re.search(r"\bcontract(or)?\b", low):
        return "CONTRACT"
    if re.search(r"\bfreelanc", low):
        return "FREELANCE"
    if re.search(r"full[\s-]time", low):
        return "FULL_TIME"
    return "UNKNOWN"


def _title_company(text: str) -> tuple[str | None, str | None]:
    text = re.sub(r"[()\[\]]", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" ,|-–—:")
    if not text:
        return None, None
    for sep in [" at ", " @ ", " | ", ", ", " - ", " – "]:
        if sep in text:
            title, company = text.split(sep, 1)
            return title.strip(" ,|-–—") or None, company.strip(" ,|-–—") or None
    return text, None


def mock_extract_resume(payload: dict[str, Any]) -> dict[str, Any]:
    text: str = payload["resume_text"]
    raw_lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in raw_lines if ln]

    email_match = _EMAIL_RE.search(text)
    email = email_match.group(0) if email_match else None
    phone_match = _PHONE_RE.search(text)
    name = None
    for ln in lines[:6]:
        if "@" in ln or re.search(r"\d", ln) or _heading(ln):
            continue
        words = ln.split()
        if 2 <= len(words) <= 5 and all(re.match(r"^[A-Za-z][A-Za-z.'\-]*$", w) for w in words):
            if ln.lower() not in {"curriculum vitae", "resume"}:
                name = ln
                break

    skills: list[dict[str, str]] = []
    seen: set[str] = set()

    def add_skill(raw: str, evidence: str) -> None:
        raw = raw.strip(" .•*-")
        if not raw or len(raw) > 60:
            return
        key = raw.casefold()
        if key not in seen:
            seen.add(key)
            skills.append({"raw_name": raw, "evidence": evidence[:200]})

    employment: list[dict[str, Any]] = []
    ambiguities: list[str] = []
    section = "other"
    previous = ""
    for ln in lines:
        head = _heading(ln)
        if head:
            section = head
            previous = ""
            continue
        m_skill = _SKILL_LINE_RE.match(ln)
        if m_skill or section == "skills":
            items = m_skill.group("items") if m_skill else ln
            for item in re.split(r"[,;|•]", items):
                add_skill(item, ln)
        # free-text skill mentions anywhere (e.g. "trained models using torch")
        low = re.sub(r"\s+", " ", ln.casefold())
        for pattern, _canonical in _scan_patterns():
            m = pattern.search(low)
            if m:
                add_skill(ln[m.start():m.end()] if len(ln) == len(low) else m.group(0), ln)

        rng = _RANGE_RE.search(ln)
        if rng and section in {"work", "projects", "education", "other"}:
            start, start_year_only = _parse_date(rng.group("start"))
            end_token = rng.group("end").lower()
            is_current = end_token in {"present", "current", "now", "till date"}
            end, end_year_only = (None, False) if is_current else _parse_date(rng.group("end"))
            if start_year_only or end_year_only:
                ambiguities.append(f"Year-only dates (month unknown): '{ln[:120]}'")
            remainder = (ln[: rng.start()] + " " + ln[rng.end():]).strip()
            if len(remainder.strip(" ,|-–—()")) < 3:
                remainder = previous
            title, company = _title_company(remainder)
            employment.append(
                {
                    "company": company,
                    "title": title,
                    "start_date": start,
                    "end_date": "present" if is_current else end,
                    "is_current": is_current,
                    "employment_type": _employment_type(ln + " " + previous, section),
                    "evidence": ln[:200],
                }
            )
        previous = ln

    return {
        "full_name": name,
        "email": email,
        "phone": phone_match.group(0).strip() if phone_match else None,
        "skills": skills,
        "employment_history": employment,
        "ambiguities": ambiguities,
        "suspected_prompt_injection": detect_prompt_injection(text),
    }


def mock_validate_profile(payload: dict[str, Any]) -> dict[str, Any]:
    validated = []
    for skill in payload["normalized_skills"]:
        known = skill["match_type"] in ("EXACT", "ALIAS")
        validated.append(
            {
                "raw_name": skill["raw_name"],
                "canonical_name": skill["canonical_name"] if known else None,
                "match_type": skill["match_type"] if known else "UNKNOWN",
                "validation_status": "VALID" if known else "REVIEW_REQUIRED",
                "evidence": skill["evidence"],
                "confidence": 0.95 if known else 0.3,
                "note": None if known else "Not in approved taxonomy",
            }
        )
    issues = []
    for entry in payload["extraction"]["employment_history"]:
        s, e = entry.get("start_date"), entry.get("end_date")
        if s and e and e != "present" and e < s:
            issues.append(f"End date before start date for {entry.get('title') or 'a role'}")
    return {
        "validated_skills": validated,
        "employment_issues": issues,
        "requires_review": bool(issues),
        "review_reasons": issues,
    }
