"""Application configuration loaded from environment variables (.env)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{name} must be <= {maximum}")
    return value


@dataclass(frozen=True)
class Settings:
    database_url: str
    mock_mode: bool
    # per-integration switches; each defaults to MOCK_MODE
    mock_llm: bool
    mock_calendar: bool
    mock_email: bool

    llm_provider: str
    llm_max_repair_attempts: int
    gemini_api_key: str
    gemini_model: str
    llm_timeout_seconds: int

    google_calendar_ids: tuple[str, ...]
    google_client_secrets_file: str
    google_token_file: str
    google_create_meet_link: bool

    smtp_host: str
    smtp_port: int
    smtp_username: str
    smtp_password: str
    smtp_use_tls: bool
    email_from: str
    email_max_attempts: int
    recruiter_contact_email: str

    hr_password: str
    resume_storage_dir: Path
    mock_data_dir: Path
    max_resume_size_mb: int

    default_timezone: str
    interview_duration_minutes: int
    working_hours_start: int
    working_hours_end: int
    slot_step_minutes: int
    scheduling_horizon_days: int
    min_notice_hours: int
    max_proposed_slots: int
    count_internships: bool
    auto_approve_eligible: bool
    reapply_cooldown_days: int

    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.default_timezone)


def load_settings(validate: bool = True) -> Settings:
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    mock_mode = _bool(os.getenv("MOCK_MODE"), default=True)
    tz_name = os.getenv("DEFAULT_TIMEZONE", "Asia/Kolkata").strip() or "Asia/Kolkata"
    try:
        ZoneInfo(tz_name)
    except ZoneInfoNotFoundError as exc:
        raise ConfigError(f"DEFAULT_TIMEZONE '{tz_name}' is not a valid IANA timezone") from exc

    resume_dir = Path(os.getenv("RESUME_STORAGE_DIR", "private/resumes"))
    if not resume_dir.is_absolute():
        resume_dir = PROJECT_ROOT / resume_dir
    mock_dir = Path(os.getenv("MOCK_DATA_DIR", "private/mock"))
    if not mock_dir.is_absolute():
        mock_dir = PROJECT_ROOT / mock_dir

    calendar_ids = tuple(
        c.strip() for c in os.getenv("GOOGLE_CALENDAR_IDS", os.getenv("GOOGLE_CALENDAR_ID", "")).split(",") if c.strip()
    )

    settings = Settings(
        database_url=os.getenv("DATABASE_URL", "").strip(),
        mock_mode=mock_mode,
        mock_llm=_bool(os.getenv("MOCK_LLM"), default=mock_mode),
        mock_calendar=_bool(os.getenv("MOCK_CALENDAR"), default=mock_mode),
        mock_email=_bool(os.getenv("MOCK_EMAIL"), default=mock_mode),
        llm_provider=os.getenv("LLM_PROVIDER", "gemini").strip().lower(),
        llm_max_repair_attempts=_int("LLM_MAX_REPAIR_ATTEMPTS", 2, 0, 5),
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        gemini_model=os.getenv("GEMINI_MODEL", "").strip(),
        llm_timeout_seconds=_int("LLM_TIMEOUT_SECONDS", 90, 10, 900),
        google_calendar_ids=calendar_ids,
        google_client_secrets_file=os.getenv("GOOGLE_CLIENT_SECRETS_FILE", "credentials.json"),
        google_token_file=os.getenv("GOOGLE_TOKEN_FILE", "token.json"),
        google_create_meet_link=_bool(os.getenv("GOOGLE_CREATE_MEET_LINK"), default=True),
        smtp_host=os.getenv("SMTP_HOST", "").strip(),
        smtp_port=_int("SMTP_PORT", 587, 1, 65535),
        smtp_username=os.getenv("SMTP_USERNAME", "").strip(),
        smtp_password=os.getenv("SMTP_PASSWORD", ""),
        smtp_use_tls=_bool(os.getenv("SMTP_USE_TLS"), default=True),
        email_from=os.getenv("EMAIL_FROM", "").strip(),
        email_max_attempts=_int("EMAIL_MAX_ATTEMPTS", 3, 1, 10),
        recruiter_contact_email=os.getenv("RECRUITER_CONTACT_EMAIL", "").strip(),
        hr_password=os.getenv("HR_PASSWORD", ""),
        resume_storage_dir=resume_dir,
        mock_data_dir=mock_dir,
        max_resume_size_mb=_int("MAX_RESUME_SIZE_MB", 5, 1, 50),
        default_timezone=tz_name,
        interview_duration_minutes=_int("INTERVIEW_DURATION_MINUTES", 45, 15, 240),
        working_hours_start=_int("WORKING_HOURS_START", 10, 0, 23),
        working_hours_end=_int("WORKING_HOURS_END", 18, 1, 24),
        slot_step_minutes=_int("SLOT_STEP_MINUTES", 30, 5, 120),
        scheduling_horizon_days=_int("SCHEDULING_HORIZON_DAYS", 10, 1, 60),
        min_notice_hours=_int("MIN_NOTICE_HOURS", 24, 0, 336),
        max_proposed_slots=_int("MAX_PROPOSED_SLOTS", 8, 1, 50),
        count_internships=_bool(os.getenv("COUNT_INTERNSHIPS_AS_EXPERIENCE"), default=False),
        auto_approve_eligible=_bool(os.getenv("AUTO_APPROVE_ELIGIBLE"), default=False),
        reapply_cooldown_days=_int("REAPPLY_COOLDOWN_DAYS", 0, 0, 3650),
    )
    if validate:
        settings = _validate(settings)
    return settings


def _validate(s: Settings) -> Settings:
    errors: list[str] = []
    warnings: list[str] = []
    if not s.database_url:
        errors.append("DATABASE_URL is required")
    if s.working_hours_end <= s.working_hours_start:
        errors.append("WORKING_HOURS_END must be after WORKING_HOURS_START")
    if not s.hr_password:
        warnings.append("HR_PASSWORD is not set - HR pages are unprotected (acceptable only for local demos).")

    if not s.mock_llm:
        if s.llm_provider == "gemini":
            if not s.gemini_model:
                errors.append("GEMINI_MODEL is required when LLM_PROVIDER=gemini")
            if not (s.gemini_api_key or os.getenv("GOOGLE_API_KEY")):
                errors.append("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")
        else:
            errors.append(f"Unsupported LLM_PROVIDER '{s.llm_provider}' (supported: gemini)")
    if not s.mock_calendar and not s.google_calendar_ids:
        errors.append("GOOGLE_CALENDAR_IDS is required when the calendar is not mocked")
    if not s.mock_email and not (s.smtp_host and s.email_from):
        errors.append("SMTP_HOST and EMAIL_FROM are required when email is not mocked")
    if errors:
        raise ConfigError("Invalid configuration: " + "; ".join(errors))
    return Settings(**{**s.__dict__, "warnings": tuple(warnings)})


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
