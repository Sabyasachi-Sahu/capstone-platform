"""Phase 4: data-quality checks generated from the schema catalogue.

Rules per table: unique, non-null primary key; required columns not null;
foreign keys that point to an existing row; dates between 1900-01-01 and the
end of the data; dates not after the person's death; plus the named
consistency rules in the catalogues. The same checks run on three layers:

  world   generated data. Must have no findings
  bronze  data with planted defects. Findings must match the defect register
  silver  cleaned data. Must have no findings
"""

import sys

import duckdb
import pandas as pd

from core.datagen.common import GEN, data_end
from core.sql.catalog import foreign_keys, layer_tables, load
from src.common.config import PROJECT_ROOT

LAKE = PROJECT_ROOT / "data" / "lake"
REGISTER = PROJECT_ROOT / "domain_packs" / "datasteward" / "defect_register.csv"
LAYERS = {"world": GEN, "bronze": LAKE / "bronze", "silver": LAKE / "silver"}
KEY = ["table_name", "row_key", "column_name", "defect_type"]


def open_layer(layer: str) -> duckdb.DuckDBPyConnection:
    """An in-memory connection with one view per table of the layer."""
    folder = LAYERS[layer]
    con = duckdb.connect()
    for table in layer_tables():
        path = folder / f"{table}.parquet"
        if not path.exists():
            raise SystemExit(f"{layer}: {path} not found")
        con.execute(
            f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{path.as_posix()}')"
        )
    return con


def check_sql(table: str, end) -> list:
    """(rule, defect_type, column, SQL returning row_key) for one table."""
    spec = load()["tables"][table]
    pk = spec["primary_key"]
    key = f"CAST({pk} AS VARCHAR)"
    out = [
        (
            "pk_unique",
            "duplicate",
            pk,
            (
                f"SELECT {key} AS row_key FROM {table} WHERE {pk} IS NOT NULL "
                f"GROUP BY {pk} HAVING COUNT(*) > 1"
            ),
        ),
    ]
    for column, col in spec["columns"].items():
        if col.get("required"):
            out.append(
                (
                    "required",
                    "null",
                    column,
                    f"SELECT {key} AS row_key FROM {table} WHERE {column} IS NULL",
                )
            )
    for column, parent, parent_col in foreign_keys(table):
        out.append(
            (
                "foreign_key",
                "orphan",
                column,
                (
                    f"SELECT {key} AS row_key FROM {table} t WHERE {column} IS NOT NULL "
                    f"AND NOT EXISTS (SELECT 1 FROM {parent} p "
                    f"WHERE p.{parent_col} = t.{column})"
                ),
            )
        )
    for column in spec.get("dates", []):
        out.append(
            (
                "date_range",
                "date_out_of_range",
                column,
                (
                    f"SELECT {key} AS row_key FROM {table} WHERE {column} "
                    f"NOT BETWEEN DATE '1900-01-01' AND DATE '{end}'"
                ),
            )
        )
        if spec.get("person_dated"):
            out.append(
                (
                    "after_death",
                    "date_out_of_range",
                    column,
                    (
                        f"SELECT {key} AS row_key FROM {table} t JOIN person p "
                        f"ON p.person_id = t.person_id "
                        f"WHERE t.{column} > p.death_date"
                    ),
                )
            )
    return out


def rule_sql() -> list:
    names = {t: t for t in layer_tables()}
    return [
        (r["name"], "consistency", r["column"], r["table"], r["sql"].format(**names))
        for r in load()["rules"]
    ]


def findings(con) -> pd.DataFrame:
    end = data_end()
    rows = []
    for table in layer_tables():
        for rule, kind, column, sql in check_sql(table, end):
            for (row_key,) in con.execute(sql).fetchall():
                rows.append((table, row_key, column, kind, rule))
    for rule, kind, column, table, sql in rule_sql():
        for (row_key,) in con.execute(sql).fetchall():
            rows.append((table, str(row_key), column, kind, rule))
    frame = pd.DataFrame(rows, columns=KEY + ["rule"])
    return frame.drop_duplicates(KEY).reset_index(drop=True)


def read_register() -> pd.DataFrame:
    if not REGISTER.exists():
        raise SystemExit(
            "Defect register not found. Run: python -m scripts.plant_defects"
        )
    # keep_default_na=False: the defect type "null" must stay text
    return pd.read_csv(REGISTER, dtype=str, keep_default_na=False)


def compare(found: pd.DataFrame, register: pd.DataFrame) -> pd.DataFrame:
    """Join findings to the register. defect_id is NULL for unexpected findings."""
    merged = found.merge(register[KEY + ["defect_id"]], on=KEY, how="left")
    return merged


def report(layer: str) -> int:
    con = open_layer(layer)
    found = findings(con)
    con.close()
    print(f"Layer {layer}: {len(found)} finding(s)")
    if layer != "bronze":
        if len(found):
            print(found.head(20).to_string(index=False))
        print("PASS" if found.empty else "FAIL: this layer must have no findings")
        return int(not found.empty)
    register = read_register()
    merged = compare(found, register)
    matched = int(merged["defect_id"].notna().sum())
    unexpected = merged[merged["defect_id"].isna()]
    missed = register[~register["defect_id"].isin(merged["defect_id"])]
    summary = merged.groupby(["table_name", "defect_type"]).size()
    print(summary.to_string())
    print(f"\nPlanted defects     : {len(register)}")
    print(f"Found and matched   : {matched}")
    print(f"Missed (not found)  : {len(missed)}")
    print(f"Unexpected findings : {len(unexpected)}")
    if len(missed):
        print(missed.to_string(index=False))
    if len(unexpected):
        print(unexpected.head(20).to_string(index=False))
    ok = matched == len(register) and unexpected.empty
    print("PASS" if ok else "FAIL")
    return int(not ok)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    layer = sys.argv[1] if len(sys.argv) > 1 else ""
    if layer not in LAYERS:
        raise SystemExit("Usage: python -m core.sql.checks world | bronze | silver")
    sys.exit(report(layer))


if __name__ == "__main__":
    main()
