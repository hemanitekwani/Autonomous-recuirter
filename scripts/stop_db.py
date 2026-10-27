"""Stop the embedded local PostgreSQL (only for DATABASE_URL=pgserver:<folder>). Data is kept.

Usage:  python -m scripts.stop_db
"""
from __future__ import annotations

from pathlib import Path

from src.config import PROJECT_ROOT, get_settings


def main() -> None:
    url = get_settings().database_url
    if not url.startswith("pgserver:"):
        raise SystemExit("DATABASE_URL is not a pgserver: URL - nothing to stop here.")
    import pgserver

    folder = Path(url.split(":", 1)[1] or "private/pgdata")
    folder = folder if folder.is_absolute() else PROJECT_ROOT / folder
    pgserver.get_server(str(folder), cleanup_mode="stop").cleanup()
    print(f"Local PostgreSQL stopped (data kept in {folder}).")


if __name__ == "__main__":
    main()
