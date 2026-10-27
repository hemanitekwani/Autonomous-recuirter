"""Experience calculation policy (tests 11-12)."""
from __future__ import annotations

from src.experience_calculator import calculate_experience
from src.schemas import EmploymentEntry, EmploymentType

REF = "2026-10"


def job(start, end, kind=EmploymentType.FULL_TIME, company="Acme"):
    return EmploymentEntry(company=company, title="Engineer", start_date=start, end_date=end, employment_type=kind,
                           evidence="x")


def test_simple_period_counts_completed_months():
    calc = calculate_experience([job("2022-06", "2025-06")], REF)
    assert calc.total_months == 36 and calc.completed_years == 3


def test_overlapping_employment_is_not_double_counted():
    calc = calculate_experience([job("2020-01", "2022-01"), job("2021-01", "2023-01", company="Beta")], REF)
    assert calc.total_months == 36  # 2020-01..2023-01, not 24 + 24
    assert calc.completed_years == 3
    assert any("merged" in r for r in calc.reasons)


def test_current_employment_uses_reference_month():
    calc = calculate_experience([job("2024-10", "present")], REF)
    assert calc.total_months == 24 and calc.completed_years == 2


def test_partial_years_are_floored():
    calc = calculate_experience([job("2023-01", "2025-12")], REF)
    assert calc.total_months == 35 and calc.completed_years == 2


def test_missing_start_date_makes_experience_unknown():
    calc = calculate_experience([job(None, "2024-01"), job("2015-01", "2020-01")], REF)
    assert calc.total_months is None and calc.completed_years is None
    assert any("missing start date" in r for r in calc.reasons)


def test_missing_end_date_not_current_is_unknown():
    calc = calculate_experience([job("2020-01", None)], REF)
    assert calc.completed_years is None


def test_end_before_start_is_flagged():
    calc = calculate_experience([job("2023-05", "2022-01")], REF)
    assert calc.completed_years is None
    assert any("before start" in r for r in calc.reasons)


def test_internships_and_projects_excluded_by_default():
    entries = [job("2019-01", "2020-01", EmploymentType.INTERNSHIP), job("2018-01", "2019-01", EmploymentType.ACADEMIC),
               job("2017-01", "2018-01", EmploymentType.PERSONAL_PROJECT), job("2021-01", "2023-01")]
    calc = calculate_experience(entries, REF)
    assert calc.total_months == 24 and len(calc.excluded_entries) == 3
    with_interns = calculate_experience(entries, REF, count_internships=True)
    assert with_interns.total_months == 36


def test_no_employment_history_is_unknown_not_zero():
    calc = calculate_experience([], REF)
    assert calc.completed_years is None


def test_future_end_is_capped():
    calc = calculate_experience([job("2025-10", "2027-01")], REF)
    assert calc.total_months == 12
