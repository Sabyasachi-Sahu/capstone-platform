"""Phase 4: copy the generated tables to the bronze layer and plant defects.

Every planted defect is written to the defect register, the ground truth for
the DataSteward pack. Defects go only into rows that no other row points to
(leaf rows), so removing them later cannot leave orphans behind, and each
row gets at most one defect. Fixed seed.
"""

import shutil
from datetime import timedelta

import numpy as np
import pandas as pd

from core.datagen.common import GEN, data_end, gen, write
from core.sql.catalog import foreign_keys, layer_tables, load
from core.sql.checks import LAYERS, REGISTER

SEED = 4404
BRONZE = LAYERS["bronze"]
# table, defect, column (None means the primary key), how many
PLAN = [
    ("bank_transaction", "duplicate", None, 5),
    ("bank_transaction", "null", "amount", 4),
    ("bank_transaction", "orphan", "account_id", 4),
    ("bank_transaction", "future_date", "txn_date", 2),
    ("bank_transaction", "after_death", "txn_date", 2),
    ("claim", "duplicate", None, 3),
    ("claim", "null", "amount_claimed", 3),
    ("claim", "orphan", "policy_id", 3),
    ("claim", "future_date", "submitted_date", 3),
    ("prescription", "duplicate", None, 3),
    ("prescription", "orphan", "drug_id", 3),
    ("prescription", "future_date", "start_date", 2),
    ("payout", "duplicate", None, 2),
    ("payout", "orphan", "account_id", 2),
    ("diagnosis", "duplicate", None, 2),
    ("diagnosis", "null", "description", 3),
    ("loan", "null", "principal", 2),
    ("loan", "after_death", "start_date", 2),
]
REGISTER_TYPE = {"future_date": "date_out_of_range", "after_death": "date_out_of_range"}


def referenced_keys() -> dict:
    """For each table, the primary keys that some other row points to."""
    used = {}
    for table in layer_tables():
        frame = gen(table)
        for column, parent, _ in foreign_keys(table):
            used.setdefault(parent, set()).update(frame[column].dropna())
    return used


def plant(rng) -> pd.DataFrame:
    end = data_end()
    death = gen("person").set_index("person_id")["death_date"]
    used = referenced_keys()
    tables = {t for t, *_ in PLAN}
    register = []
    for table in sorted(tables):
        frame = gen(table)
        pk = load()["tables"][table]["primary_key"]
        free = ~frame[pk].isin(used.get(table, set()))
        taken = set()
        extra_rows = []
        for t, defect, column, count in PLAN:
            if t != table:
                continue
            pool = free & ~frame.index.isin(taken)
            if column:
                pool &= frame[column].notna()
            if defect == "after_death":
                pool &= frame["person_id"].map(death).notna()
            choices = frame.index[pool].to_numpy()
            if len(choices) < count:
                raise SystemExit(f"{table}: only {len(choices)} rows for {defect}")
            for i in rng.choice(choices, size=count, replace=False):
                taken.add(i)
                key = frame.at[i, pk]
                if defect == "duplicate":
                    extra_rows.append(frame.loc[[i]])
                    detail, column_name = "row copied, same primary key", pk
                elif defect == "null":
                    frame.at[i, column] = None
                    detail, column_name = "value set to NULL", column
                elif defect == "orphan":
                    prefix = str(frame.at[i, column])[:3]
                    bad = f"{prefix}9{int(rng.integers(10000, 99999))}X"
                    frame.at[i, column] = bad
                    detail, column_name = (
                        f"points to {bad}, which does not exist",
                        column,
                    )
                elif defect == "future_date":
                    day = end + timedelta(days=int(rng.integers(30, 400)))
                    frame.at[i, column] = day
                    detail, column_name = f"set to {day}, after data end {end}", column
                else:
                    died = death[frame.at[i, "person_id"]]
                    day = died + timedelta(days=int(rng.integers(1, 120)))
                    frame.at[i, column] = day
                    detail, column_name = f"set to {day}, after death on {died}", column
                register.append(
                    {
                        "table_name": table,
                        "row_key": key,
                        "column_name": column_name,
                        "defect_type": REGISTER_TYPE.get(defect, defect),
                        "detail": detail,
                    }
                )
        frame = pd.concat([frame, *extra_rows], ignore_index=True)
        write(table, frame, BRONZE)
    out = pd.DataFrame(register).sort_values(["table_name", "defect_type", "row_key"])
    out.insert(0, "defect_id", [f"DEF{i:04d}" for i in range(1, len(out) + 1)])
    return out, tables


def main():
    rng = np.random.default_rng(SEED)
    if BRONZE.exists():
        shutil.rmtree(BRONZE)
    BRONZE.mkdir(parents=True)
    register, changed = plant(rng)
    for table in layer_tables():
        if table not in changed:
            shutil.copyfile(GEN / f"{table}.parquet", BRONZE / f"{table}.parquet")
    REGISTER.parent.mkdir(parents=True, exist_ok=True)
    register.to_csv(REGISTER, index=False)
    print(f"\nSeed {SEED}. Planted {len(register)} defects")
    print(register.groupby(["table_name", "defect_type"]).size().to_string())
    print(f"\nRegister: {REGISTER}")


if __name__ == "__main__":
    main()
