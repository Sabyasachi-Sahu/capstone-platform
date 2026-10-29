from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("data/raw/synthea/output/csv")
OUT = Path("data/world")
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)

pat = pd.read_csv(SRC / "patients.csv")

# 1. PERSON - one master record per human
person = pd.DataFrame(
    {
        "person_id": [f"PER{i:06d}" for i in range(1, len(pat) + 1)],
        "synthea_id": pat["Id"],
        "first_name": pat["FIRST"],
        "last_name": pat["LAST"],
        "dob": pat["BIRTHDATE"],
        "gender": pat["GENDER"],
        "state": pat["STATE"],
    }
)

# 2. Role tables - same person_id, different roles
patient = person[["person_id", "synthea_id"]].assign(
    patient_id=lambda d: "PAT" + d.person_id.str[3:]
)

ph_mask = rng.random(len(person)) < 0.70  # ~70% insured
policyholder = person.loc[ph_mask, ["person_id"]].assign(
    policyholder_id=lambda d: "POL" + d.person_id.str[3:]
)

bc_mask = rng.random(len(person)) < 0.85  # ~85% bank customers
bank_customer = person.loc[bc_mask, ["person_id"]].assign(
    customer_id=lambda d: "CUS" + d.person_id.str[3:]
)

for name, df in {
    "person": person,
    "patient": patient,
    "policyholder": policyholder,
    "bank_customer": bank_customer,
}.items():
    df.to_parquet(OUT / f"{name}.parquet", index=False)
    print(f"{name:15s} rows = {len(df):,}")
