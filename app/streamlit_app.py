"""Phase 3: Streamlit chat interface for the capstone packs."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.phase2.check_claims import check_claims, load_records
from src.phase3.pack_rag import answer, find_ids
from src.phase3.packs import collection_count, list_packs, load_pack

st.set_page_config(page_title="Capstone packs", layout="wide")
st.title("Capstone platform: ask a pack")


@st.cache_data
def patient_records():
    return load_records()


names = list_packs()
name = st.sidebar.selectbox("Pack", names)
pack = load_pack(name)
st.sidebar.markdown(f"**{pack.title}**")
st.sidebar.caption(f"Collection: {pack.collection}   Top results: {pack.top_k}")

count = collection_count(pack)
if count is None:
    st.warning(
        f"The collection '{pack.collection}' does not exist yet. "
        f"Run: python -m src.phase3.build_pack_index {pack.name}"
    )
    st.stop()
st.sidebar.caption(f"Indexed items: {count}")

question = st.text_area("Your question", height=90)
if st.button("Ask") and question.strip():
    with st.spinner("Searching and answering..."):
        try:
            result = answer(pack, question.strip())
        except Exception as exc:  # noqa: BLE001 - show any failure in the UI instead of crashing the app
            st.error(f"Could not answer: {type(exc).__name__}: {exc}")
            st.stop()

    st.subheader("Answer")
    st.markdown(result["answer"])

    cited = find_ids(pack, result["answer"])
    unverified = sorted(cited - set(result["sources"]))
    c1, c2 = st.columns(2)
    c1.metric("Citations verified", len(cited) - len(unverified))
    c2.metric("Citations not retrieved", len(unverified))
    if unverified:
        st.error("These cited IDs were not retrieved: " + ", ".join(unverified))

    if pack.raw.get("claim_check") == "patient_conditions":
        claims = check_claims(result["answer"], patient_records())
        if not claims:
            st.warning("Claim check: no bold condition names with IDs to check.")
        else:
            counts = {
                k: sum(c["status"] == k for c in claims)
                for k in ("supported", "wrong", "unclear")
            }
            st.write(
                f"Claim check: {counts['supported']} supported, "
                f"{counts['wrong']} wrong, {counts['unclear']} unclear"
            )
            problems = [c for c in claims if c["status"] != "supported"]
            if problems:
                st.dataframe(pd.DataFrame(problems), width="stretch")

    st.subheader("Retrieved sources")
    table = pd.DataFrame(
        [
            {
                "id": h["id"],
                "similarity": round(h["similarity"], 3),
                "cited": h["id"] in cited,
                "preview": h["text"][:140].replace("\n", " "),
            }
            for h in result["hits"]
        ]
    )
    st.dataframe(table, width="stretch")
    for h in result["hits"]:
        with st.expander(h["id"]):
            st.text(h["text"])
