"""Authoritative professional-experience calculation (pure Python, no LLM).

Policy (POLICY_VERSION):
* Granularity is calendar months. A period "2022-06 .. 2025-06" counts the
  COMPLETED months between the start month and the end month = 36 months
  (end month exclusive). Partial months are therefore not counted.
* Current employment ("present" / is_current) ends at the reference month
  (the current month in DEFAULT_TIMEZONE unless a reference is given).
* Counted types: FULL_TIME, PART_TIME, CONTRACT, FREELANCE, UNKNOWN.
  INTERNSHIP is counted only when COUNT_INTERNSHIPS_AS_EXPERIENCE=true.
  ACADEMIC and PERSONAL_PROJECT entries are never counted.
* Overlapping periods are merged, so concurrent jobs are not double-counted.
* A counted entry with a missing start date, a missing end date (and not
  current), an end before its start, or a start in the future makes the
  total UNKNOWN (None) - missing data is never treated as zero. A resume
  with no employment entries at all is also UNKNOWN (fresher vs parse miss).
* `completed_years` = floor(total_months / 12). This is what is stored in
  applications.experience (INTEGER). Since job minimums are whole years,
  floor(years) >= min  <=>  exact years >= min, so no precision is lost.
"""
from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

from src.schemas import CountedPeriod, EmploymentEntry, EmploymentType, ExperienceCalculation

POLICY_VERSION = "months-merged-v1"

_ALWAYS_COUNTED = {
    EmploymentType.FULL_TIME,
    EmploymentType.PART_TIME,
    EmploymentType.CONTRACT,
    EmploymentType.FREELANCE,
    EmploymentType.UNKNOWN,
}


def _month_index(ym: str) -> int:
    year, month = ym.split("-")
    return int(year) * 12 + (int(month) - 1)


def _ym(index: int) -> str:
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def reference_month(tz_name: str = "Asia/Kolkata", today: date | None = None) -> str:
    today = today or datetime.now(ZoneInfo(tz_name)).date()
    return f"{today.year:04d}-{today.month:02d}"


def calculate_experience(
    entries: list[EmploymentEntry],
    ref_month: str,
    count_internships: bool = False,
) -> ExperienceCalculation:
    counted_types = set(_ALWAYS_COUNTED)
    if count_internships:
        counted_types.add(EmploymentType.INTERNSHIP)

    ref_idx = _month_index(ref_month)
    reasons: list[str] = []
    excluded: list[str] = []
    intervals: list[tuple[int, int, str | None]] = []
    unknown = False
    if not entries:
        # Nothing extracted at all: could be a fresher or a parsing miss - don't guess zero.
        reasons.append("No employment history found in the resume")
        unknown = True

    for i, entry in enumerate(entries, start=1):
        label = f"#{i} {entry.title or '?'} @ {entry.company or '?'}"
        if entry.employment_type not in counted_types:
            excluded.append(f"{label}: {entry.employment_type.value} not counted as professional experience")
            continue
        if not entry.start_date:
            reasons.append(f"{label}: missing start date")
            unknown = True
            continue
        start = _month_index(entry.start_date)
        if entry.is_current or entry.end_date == "present":
            end = ref_idx
        elif entry.end_date:
            end = _month_index(entry.end_date)
        else:
            reasons.append(f"{label}: missing end date and not marked as current")
            unknown = True
            continue
        if start > ref_idx:
            reasons.append(f"{label}: start date {entry.start_date} is in the future")
            unknown = True
            continue
        if end < start:
            reasons.append(f"{label}: end date is before start date")
            unknown = True
            continue
        if end > ref_idx:
            reasons.append(f"{label}: end date in the future; capped at {ref_month}")
            end = ref_idx
        intervals.append((start, end, entry.company))

    # merge overlapping / touching intervals
    merged: list[list] = []
    for start, end, company in sorted(intervals, key=lambda t: (t[0], t[1])):
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1][1] = end
            if company and company not in merged[-1][2]:
                merged[-1][2].append(company)
        else:
            merged.append([start, end, [company] if company else []])

    if len(intervals) > len(merged):
        reasons.append("Overlapping employment periods were merged (not double-counted)")

    periods = [
        CountedPeriod(company=", ".join(c) or None, start=_ym(s), end=_ym(e), months=e - s) for s, e, c in merged
    ]
    method = (
        f"{POLICY_VERSION}: completed calendar months, overlaps merged, "
        f"internships {'counted' if count_internships else 'excluded'}, reference {ref_month}"
    )
    if unknown:
        return ExperienceCalculation(
            total_months=None,
            completed_years=None,
            method=method,
            reference_month=ref_month,
            counted_periods=periods,
            excluded_entries=excluded,
            reasons=reasons + ["Experience cannot be calculated reliably - review required"],
        )
    total = sum(p.months for p in periods)
    return ExperienceCalculation(
        total_months=total,
        completed_years=total // 12,
        method=method,
        reference_month=ref_month,
        counted_periods=periods,
        excluded_entries=excluded,
        reasons=reasons,
    )
