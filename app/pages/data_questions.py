"""Phase 4 page: ask any question. See the route, the tables, the SQL, the
result, and the RAG answer when the question needs documents."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from core.sql.ask import ask_rag, pick_pack
from core.sql.generator import BOTH_NOTE, ask_sql
from core.sql.router import route

st.title("Ask the data")
st.caption(
    "Counts, lists and rankings go to SQL over every row. Questions about "
    "what a document says go to RAG. Some need both. Synthetic data only."
)

question = st.text_input(
    "Question",
    placeholder="How many admissions were followed by a readmission within 30 days?",
)
forced = st.radio("Route", ["automatic", "sql", "rag", "both"], horizontal=True)
pack = st.selectbox("Pack for document questions", ["automatic", "hospital", "pharma"])

if st.button("Ask", type="primary") and question.strip():
    where, reason = route(question)
    if forced != "automatic":
        where, reason = forced, "chosen by you"
    st.info(f"Route: **{where.upper()}** ({reason})")

    if where in ("sql", "both"):
        with st.spinner("Writing and running SQL..."):
            try:
                out = ask_sql(
                    question.strip(),
                    source="app",
                    note=BOTH_NOTE if where == "both" else "",
                )
            except Exception as exc:  # noqa: BLE001 - show any failure, do not crash
                st.error(f"SQL failed: {type(exc).__name__}: {exc}")
                st.stop()
        st.subheader("SQL answer")
        if out["cannot_answer"]:
            st.warning(f"The database cannot answer this: {out['answer']}")
        else:
            st.write(out["answer"])
            st.caption(f"Tables chosen: {', '.join(out['tables_selected']) or 'all'}")
            st.code(out["sql"] or "", language="sql")
            if out["errors"]:
                st.caption(
                    f"Fixed after {len(out['errors'])} failed attempt(s): "
                    f"{out['errors'][-1]}"
                )
            if out["result"]:
                res = out["result"]
                st.table(pd.DataFrame(res["rows"][:50], columns=res["columns"]))
                if res["truncated"] or len(res["rows"]) > 50:
                    st.caption("Only the first rows are shown.")
        st.caption("This query is recorded in data/warehouse/audit.jsonl.")

    if where in ("rag", "both"):
        chosen = pick_pack(question) if pack == "automatic" else pack
        with st.spinner(f"Asking the {chosen} pack..."):
            text = ask_rag(chosen, question.strip())
        st.subheader(f"Document answer ({chosen} pack)")
        st.text(text)
