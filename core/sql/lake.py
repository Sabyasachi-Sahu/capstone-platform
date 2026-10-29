"""Phase 4: build the silver and gold layers and the DuckDB warehouse.

bronze (planted defects) -> silver (cleaned, bad rows quarantined)
silver -> gold views -> data/lake/gold (Parquet, for Databricks)
silver + gold views + DataSteward tables -> data/warehouse/capstone.duckdb

The warehouse is built into a temporary file and swapped in only when every
check passes, so a failed build never replaces a working warehouse.
"""

import json
import shutil
import sys
import zipfile
from datetime import datetime

import duckdb
import pandas as pd

from core.datagen.common import data_end
from core.sql.catalog import foreign_keys, layer_tables, load
from core.sql.checks import LAKE, LAYERS, compare, findings, open_layer, read_register
from src.common.config import PROJECT_ROOT

WAREHOUSE = PROJECT_ROOT / "data" / "warehouse"
DB_FILE = WAREHOUSE / "capstone.duckdb"
HINTS_FILE = WAREHOUSE / "hints.json"
MANIFEST = LAKE / "manifest.json"
GENERATOR = {
    "person": "gen_hospital",
    "organisation": "gen_hospital",
    "encounter": "gen_hospital",
    "admission": "gen_hospital",
    "diagnosis": "gen_hospital",
    "hcp": "gen_pharma",
    "drug": "gen_pharma",
    "prescription": "gen_pharma",
    "bank_customer": "gen_banking",
    "account": "gen_banking",
    "loan": "gen_banking",
    "bank_transaction": "gen_banking",
    "policyholder": "gen_insurance",
    "policy": "gen_insurance",
    "claim": "gen_insurance",
    "payout": "gen_insurance",
}


def reason_sql(table: str, end) -> str:
    """A CASE expression that names why a bronze row cannot go to silver."""
    spec = load()["tables"][table]
    pk = spec["primary_key"]
    required = [c for c, col in spec["columns"].items() if col.get("required")]
    nulls = " OR ".join(f"b.{c} IS NULL" for c in required)
    orphans = (
        " OR ".join(
            f"(b.{c} IS NOT NULL AND NOT EXISTS "
            f"(SELECT 1 FROM {parent} p WHERE p.{pc} = b.{c}))"
            for c, parent, pc in foreign_keys(table)
        )
        or "FALSE"
    )
    dates = []
    for c in spec.get("dates", []):
        dates.append(f"b.{c} NOT BETWEEN DATE '1900-01-01' AND DATE '{end}'")
        if spec.get("person_dated"):
            dates.append(
                f"b.{c} > (SELECT death_date FROM person d "
                f"WHERE d.person_id = b.person_id)"
            )
    bad_dates = " OR ".join(dates) or "FALSE"
    return (
        f"CASE WHEN b._copy > 1 THEN 'duplicate' "
        f"WHEN {nulls or 'FALSE'} THEN 'null' "
        f"WHEN {orphans} THEN 'orphan' "
        f"WHEN {bad_dates} THEN 'date_out_of_range' END"
    ), pk


def build_silver(con) -> pd.DataFrame:
    end = data_end()
    bronze, silver = LAYERS["bronze"], LAYERS["silver"]
    quarantine = LAKE / "quarantine"
    for folder in (silver, quarantine):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
    summary = []
    for table in layer_tables():
        source = (bronze / f"{table}.parquet").as_posix()
        if not (bronze / f"{table}.parquet").exists():
            raise SystemExit(f"Bronze table missing: {source}. Run plant_defects.")
        reason, pk = reason_sql(table, end)
        con.execute(
            f"CREATE TEMP TABLE staged AS SELECT b.*, {reason} AS _reason FROM "
            f"(SELECT *, row_number() OVER (PARTITION BY {pk}) AS _copy "
            f"FROM read_parquet('{source}')) b"
        )
        con.execute(
            f"CREATE TABLE {table} AS SELECT * EXCLUDE (_copy, _reason) "
            f"FROM staged WHERE _reason IS NULL"
        )
        con.execute(
            f"COPY {table} TO '{(silver / f'{table}.parquet').as_posix()}' "
            f"(FORMAT parquet)"
        )
        removed = con.execute(
            "SELECT _reason, COUNT(*) FROM staged WHERE _reason IS NOT NULL GROUP BY 1"
        ).fetchall()
        if removed:
            con.execute(
                "COPY (SELECT * EXCLUDE (_copy) FROM staged "
                "WHERE _reason IS NOT NULL) "
                f"TO '{(quarantine / f'{table}.parquet').as_posix()}' (FORMAT parquet)"
            )
        for why, n in removed:
            summary.append({"table_name": table, "reason": why, "rows_removed": n})
        con.execute("DROP TABLE staged")
    return pd.DataFrame(summary, columns=["table_name", "reason", "rows_removed"])


def add_table(con, name: str, frame: pd.DataFrame):
    """Create a DataSteward table with the types in its catalogue."""
    spec = load()["tables"][name]
    cols = ", ".join(
        f'CAST("{c}" AS {col["type"]}) AS "{c}"' for c, col in spec["columns"].items()
    )
    con.register("frame", frame)
    con.execute(f"CREATE TABLE {name} AS SELECT {cols} FROM frame")
    con.unregister("frame")


def steward_tables(con, quarantine: pd.DataFrame):
    cat = load()
    register = read_register()
    bronze = open_layer("bronze")
    found = compare(findings(bronze), register)
    bronze.close()
    found.insert(0, "finding_id", [f"F{i:05d}" for i in range(1, len(found) + 1)])
    add_table(con, "defect_register", register)
    add_table(con, "dq_finding", found)
    add_table(con, "quarantine_summary", quarantine)

    lineage = []
    for table in layer_tables():
        script = f"scripts/{GENERATOR.get(table, 'unknown')}.py"
        lineage += [
            ("world", table, "script", script, "generated"),
            ("bronze", table, "world", table, "copy plus planted defects"),
            (
                "silver",
                table,
                "bronze",
                table,
                "remove duplicates; quarantine nulls, orphans and bad dates",
            ),
        ]
    for view, spec in cat["views"].items():
        lineage += [("gold", view, "silver", s, "view") for s in spec["sources"]]
    add_table(
        con,
        "lineage",
        pd.DataFrame(
            lineage,
            columns=[
                "target_layer",
                "target_table",
                "source_layer",
                "source_table",
                "transform",
            ],
        ),
    )

    tables, columns = [], []
    for name, spec in {**cat["tables"], **cat["views"]}.items():
        tables.append(
            {
                "table_name": name,
                "pack": spec["pack"],
                "kind": spec["kind"],
                "description": " ".join(spec.get("about", "").split()),
                "primary_key": spec.get("primary_key"),
                "row_count": None,
            }
        )
        for column, col in spec["columns"].items():
            ref = col.get("references")
            columns.append(
                {
                    "table_name": name,
                    "column_name": column,
                    "data_type": col["type"],
                    "description": col.get("about"),
                    "is_primary_key": column == spec.get("primary_key"),
                    "is_required": bool(col.get("required")),
                    "references_table": ref.split(".")[0] if ref else None,
                }
            )
    add_table(con, "catalog_column", pd.DataFrame(columns))
    add_table(con, "catalog_table", pd.DataFrame(tables))
    for row in tables:
        n = con.execute(f"SELECT COUNT(*) FROM {row['table_name']}").fetchone()[0]
        con.execute(
            "UPDATE catalog_table SET row_count = ? WHERE table_name = ?",
            [n, row["table_name"]],
        )


def hints(con) -> dict:
    out = {}
    cat = load()
    for name, spec in {**cat["tables"], **cat["views"]}.items():
        for column, col in spec["columns"].items():
            if col.get("hint"):
                rows = con.execute(
                    f"SELECT CAST({column} AS VARCHAR) FROM {name} "
                    f"WHERE {column} IS NOT NULL GROUP BY 1 "
                    f"ORDER BY COUNT(*) DESC, 1 LIMIT 25"
                ).fetchall()
                out[f"{name}.{column}"] = [r[0] for r in rows]
    return out


def build():
    WAREHOUSE.mkdir(parents=True, exist_ok=True)
    temp = WAREHOUSE / "capstone.building.duckdb"
    temp.unlink(missing_ok=True)
    con = duckdb.connect(str(temp))
    try:
        quarantine = build_silver(con)
        problems = findings(con)
        print(f"Silver checks       : {len(problems)} finding(s)")
        if len(problems):
            print(problems.head(20).to_string(index=False))
            raise SystemExit("Silver has findings. Warehouse not replaced.")

        register = read_register()
        expected = register.groupby(["table_name", "defect_type"]).size()
        removed = quarantine.set_index(["table_name", "reason"])["rows_removed"]
        removed.index.names = expected.index.names
        diff = pd.concat([expected, removed], axis=1, keys=["planted", "removed"])
        diff = diff.fillna(0).astype(int)
        print("\nRows removed between bronze and silver, against the register:")
        print(diff.to_string())
        if not (diff["planted"] == diff["removed"]).all():
            raise SystemExit("Quarantine does not match the register.")

        gold = LAYERS["silver"].parent / "gold"
        if gold.exists():
            shutil.rmtree(gold)
        gold.mkdir(parents=True)
        for view, spec in load()["views"].items():
            con.execute(f"CREATE VIEW {view} AS {spec['sql']}")
            con.execute(
                f"COPY (SELECT * FROM {view}) TO "
                f"'{(gold / f'{view}.parquet').as_posix()}' (FORMAT parquet)"
            )
        steward_tables(con, quarantine)
        hint_values = hints(con)

        manifest = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "data_end": str(data_end()),
            "layers": {},
        }
        for layer in ("bronze", "silver", "gold"):
            folder = LAKE / layer
            manifest["layers"][layer] = {
                p.stem: duckdb.execute(
                    f"SELECT COUNT(*) FROM '{p.as_posix()}'"
                ).fetchall()[0][0]
                for p in sorted(folder.glob("*.parquet"))
            }
    finally:
        con.close()

    temp.replace(DB_FILE)
    HINTS_FILE.write_text(json.dumps(hint_values, indent=1), encoding="utf-8")
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n{'table':20} {'bronze':>9} {'silver':>9}")
    for table in layer_tables():
        b = manifest["layers"]["bronze"][table]
        s = manifest["layers"]["silver"][table]
        print(f"{table:20} {b:>9,} {s:>9,}")
    for view, n in manifest["layers"]["gold"].items():
        print(f"{view:20} {'gold':>9} {n:>9,}")
    print(f"\nWarehouse : {DB_FILE}")
    print(f"Manifest  : {MANIFEST}")


def package():
    """One zip with bronze, silver, gold and the manifest, for Databricks."""
    if not MANIFEST.exists():
        raise SystemExit("Run: python -m core.sql.lake build")
    target = LAKE / "lake_upload.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(MANIFEST, "manifest.json")
        for layer in ("bronze", "silver", "gold"):
            for path in sorted((LAKE / layer).glob("*.parquet")):
                zf.write(path, f"{layer}/{path.name}")
    print(
        f"Wrote {target} ({target.stat().st_size / 1e6:.1f} MB, "
        f"{len(zipfile.ZipFile(target).namelist())} files)"
    )


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "build":
        build()
    elif command == "package":
        package()
    else:
        raise SystemExit("Usage: python -m core.sql.lake build | package")


if __name__ == "__main__":
    main()
