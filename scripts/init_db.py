"""Create / migrate the schema by running sql/init.sql (idempotent).

Usage:  python -m scripts.init_db
"""
from __future__ import annotations

from src.config import get_settings
from src.database import get_engine, init_db


def main() -> None:
    settings = get_settings()
    engine = get_engine()
    init_db(engine)
    where = settings.database_url if settings.database_url.startswith("pgserver:") else \
        engine.url.render_as_string(hide_password=True)
    print(f"Schema initialized on {where}")


if __name__ == "__main__":
    main()
