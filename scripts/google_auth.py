"""One-time Google Calendar OAuth consent. Opens a browser and writes GOOGLE_TOKEN_FILE.

Usage:  python -m scripts.google_auth
"""
from __future__ import annotations

from src.calendar_service import authorize_interactively
from src.config import load_settings


def main() -> None:
    settings = load_settings(validate=False)
    path = authorize_interactively(settings)
    print(f"Token saved to {path}. Keep this file private (it is git-ignored).")


if __name__ == "__main__":
    main()
