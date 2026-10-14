"""Phase 2: turn Synthea tables into one text document per patient."""

import json

import pandas as pd

from src.common.config import PROJECT_ROOT, get_settings

OUTPUT = PROJECT_ROOT / "data" / "processed" / "patient_docs.jsonl"


def build_documents() -> list:
    s = get_settings()
    patients = pd.read_csv(s.synthea_dir / "patients.csv")
    conditions = pd.read_csv(s.synthea_dir / "conditions.csv")

    # Keep real diagnoses only, then remove duplicates per patient
    disorders = conditions[conditions["DESCRIPTION"].str.endswith("(disorder)")].copy()
    disorders["NAME"] = disorders["DESCRIPTION"].str.replace(
        " (disorder)", "", regex=False
    )
    disorders["ACTIVE"] = disorders["STOP"].isna()
    disorders = disorders.drop_duplicates(subset=["PATIENT", "NAME", "ACTIVE"])

    docs = []
    for p in patients.itertuples():
        mine = disorders[disorders["PATIENT"] == p.Id]
        active = sorted(mine.loc[mine["ACTIVE"], "NAME"])
        past = sorted(set(mine.loc[~mine["ACTIVE"], "NAME"]) - set(active))
        status = "deceased" if pd.notna(p.DEATHDATE) else "alive"

        text = (
            f"Patient {p.Id} | Gender: {p.GENDER} | Born: {p.BIRTHDATE[:4]} | Status: {status}\n"
            f"Active conditions: {'; '.join(active) or 'none recorded'}\n"
            f"Past conditions: {'; '.join(past) or 'none recorded'}"
        )
        docs.append({"id": p.Id, "text": text})
    return docs


def main():
    docs = build_documents()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")

    print(f"Documents written : {len(docs)}")
    print(f"Output file       : {OUTPUT}")
    print("\n--- Sample document ---")
    print(docs[0]["text"])


if __name__ == "__main__":
    main()
