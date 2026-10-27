"""Developer-only database viewer (separate from the recruiter app). Read-only.

Run:   streamlit run tools/db_viewer.py --server.port 8502
Open:  http://localhost:8502
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `import src...` when run by streamlit

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.database import get_engine  # noqa: E402
from src.db_explorer import QueryRejected, describe_table, fetch_rows, list_tables, run_readonly_query  # noqa: E402

st.set_page_config(page_title="DB Viewer (dev)", page_icon="🗄️", layout="wide")

EXAMPLES = {
    "Jobs and their requirements": "SELECT job_id, title, min_experience_years, skills, status FROM jobs",
    "Applications": "SELECT application_id, name, email, job_id, experience, screening_status, pipeline_status, "
                    "created_at FROM applications ORDER BY application_id DESC",
    "Requirement checks of application 1": "SELECT c->>'requirement' AS requirement, c->>'kind' AS kind, "
                                           "c->>'result' AS result, c->>'reason' AS reason FROM applications, "
                                           "jsonb_array_elements(screening_result->'checks') c "
                                           "WHERE application_id = 1",
    "Interviews": "SELECT interview_id, application_id, starts_at, ends_at, status, email_status, meeting_link "
                  "FROM interviews ORDER BY starts_at",
    "Audit log (latest 50)": "SELECT created_at, entity_type, entity_id, action, actor, details "
                             "FROM audit_log ORDER BY created_at DESC LIMIT 50",
}


@st.cache_resource
def engine():
    return get_engine()


st.title("🗄️ Database viewer (developer)")
eng = engine()
st.caption(f"Connected to `{eng.url.render_as_string(hide_password=True)}` - read-only. "
           "Use the browser refresh (F5) to see newly inserted rows.")
tables = list_tables(eng)
if not tables:
    st.warning("No tables yet - run  python -m scripts.init_db")
    st.stop()

counts = {t: describe_table(eng, t)[2] for t in tables}
cols = st.columns(len(tables))
for col, t in zip(cols, tables):
    col.metric(t, f"{counts[t]} rows")

tab_data, tab_schema, tab_sql = st.tabs(["📋 Rows", "🧱 Schema", "⌨️ SQL"])

with tab_data:
    c1, c2 = st.columns([4, 1])
    table = c1.radio("Table", tables, horizontal=True, key="rows_table",
                     index=tables.index("applications") if "applications" in tables else 0)
    limit = c2.number_input("Max rows", 10, 500, 100, step=10)
    rows = fetch_rows(eng, table, int(limit))
    st.caption(f"`{table}` - newest first, {len(rows)} of {counts[table]} row(s). JSON columns shown as text.")
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info("This table is empty.")

with tab_schema:
    table = st.radio("Table", tables, horizontal=True, key="schema_table")
    columns, rules, _ = describe_table(eng, table)
    st.markdown(f"**Columns of `{table}`** (defined in `sql/init.sql`)")
    st.dataframe(pd.DataFrame(columns), hide_index=True, width="stretch")
    st.markdown("**Keys, rules and indexes**")
    st.dataframe(pd.DataFrame(rules), hide_index=True, width="stretch")

with tab_sql:
    example = st.selectbox("Example", ["(write your own)"] + list(EXAMPLES))
    query = st.text_area("SQL (read-only: SELECT / WITH)", EXAMPLES.get(example, "SELECT * FROM jobs"),
                         height=120, key=f"q_{example}")
    if st.button("Run", type="primary"):
        try:
            result, truncated = run_readonly_query(eng, query)
        except QueryRejected as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001 - show the database's own message (syntax errors etc.)
            st.error(f"Query failed: {getattr(exc, 'orig', exc)}")
        else:
            st.success(f"{len(result)} row(s)" + (" - first 500 shown" if truncated else ""))
            if result:
                st.dataframe(pd.DataFrame(result), hide_index=True, width="stretch")
