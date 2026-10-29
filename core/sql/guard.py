"""Phase 4: the only code path that runs model-written SQL.

Layers, each independent of the others:
  1. Parse check: exactly one statement, and it must be a SELECT.
  2. Allow-list: the parsed query may read only catalogue tables and views
     (or its own CTEs). No table functions, no files, no system catalogues.
  3. Read-only connection with file access and extension loading switched off,
     and settings locked so a query cannot switch them back on.
  4. A time limit (the query is interrupted) and a row cap.
"""

import json
import sys
import threading
import time

import duckdb
import yaml

from core.sql.catalog import allowed_names
from src.common.config import PROJECT_ROOT

DB_FILE = PROJECT_ROOT / "data" / "warehouse" / "capstone.duckdb"
TESTS = PROJECT_ROOT / "tests" / "sql" / "guard_p4.yaml"
MAX_ROWS = 200
TIMEOUT_SECONDS = 10
SAFE_CONFIG = {
    "enable_external_access": False,
    "autoload_known_extensions": False,
    "autoinstall_known_extensions": False,
    "lock_configuration": True,
}


class SqlRejectedError(ValueError):
    """The SQL broke a safety rule and was not run."""


_parser = duckdb.connect()


def _walk(node, tables: list, functions: list, ctes: set):
    if isinstance(node, dict):
        kind = node.get("type")
        if kind == "BASE_TABLE":
            tables.append(
                (
                    node.get("catalog_name") or "",
                    node.get("schema_name") or "",
                    node.get("table_name"),
                )
            )
        elif kind == "TABLE_FUNCTION":
            functions.append((node.get("function") or {}).get("function_name", "?"))
        for entry in (node.get("cte_map") or {}).get("map", []) or []:
            ctes.add(str(entry.get("key", "")).lower())
        for value in node.values():
            _walk(value, tables, functions, ctes)
    elif isinstance(node, list):
        for value in node:
            _walk(value, tables, functions, ctes)


def check(sql: str) -> tuple:
    """Return (clean SQL, tables read) or raise SqlRejectedError."""
    text = sql.strip().rstrip(";").strip()
    if not text:
        raise SqlRejectedError("empty query")
    try:
        statements = _parser.extract_statements(text)
    except duckdb.Error as exc:
        raise SqlRejectedError(f"parse error: {exc}") from exc
    if len(statements) != 1:
        raise SqlRejectedError("only one statement is allowed")
    if statements[0].type != duckdb.StatementType.SELECT:
        kind = str(statements[0].type).split(".")[-1]
        raise SqlRejectedError(f"only SELECT is allowed, got {kind}")
    tree = json.loads(
        _parser.execute("SELECT json_serialize_sql(?)", [text]).fetchall()[0][0]
    )
    if tree.get("error"):
        raise SqlRejectedError(f"parse error: {tree.get('error_message')}")
    tables, functions, ctes = [], [], set()
    _walk(tree, tables, functions, ctes)
    if functions:
        raise SqlRejectedError(
            f"table functions are not allowed: {sorted(set(functions))}"
        )
    allowed = {n.lower() for n in allowed_names()}
    read = set()
    for catalog, schema, name in tables:
        lowered = name.lower()
        if lowered in ctes and not schema:
            continue
        if catalog not in ("", "capstone") or schema not in ("", "main"):
            raise SqlRejectedError(
                "not an allowed table: "
                + ".".join(filter(None, (catalog, schema, name)))
            )
        if lowered not in allowed:
            raise SqlRejectedError(f"unknown table: {name}")
        read.add(lowered)
    return text, sorted(read)


def connect() -> duckdb.DuckDBPyConnection:
    if not DB_FILE.exists():
        raise SystemExit("Warehouse not found. Run: python -m core.sql.lake build")
    return duckdb.connect(str(DB_FILE), read_only=True, config=SAFE_CONFIG)


def run_sql(
    sql: str, max_rows: int = MAX_ROWS, timeout: float = TIMEOUT_SECONDS
) -> dict:
    """Check and run one query. Raises SqlRejectedError or duckdb.Error."""
    text, tables = check(sql)
    con = connect()
    timer = threading.Timer(timeout, con.interrupt)
    started = time.monotonic()
    timer.start()
    try:
        cursor = con.execute(text)
        rows = cursor.fetchmany(max_rows + 1)
        columns = [d[0] for d in cursor.description or []]
    except duckdb.InterruptException as exc:
        raise SqlRejectedError(f"stopped after the {timeout}s time limit") from exc
    finally:
        timer.cancel()
        con.close()
    return {
        "sql": text,
        "tables": tables,
        "columns": columns,
        "rows": [list(r) for r in rows[:max_rows]],
        "truncated": len(rows) > max_rows,
        "seconds": round(time.monotonic() - started, 3),
    }


def selftest() -> int:
    cases = yaml.safe_load(TESTS.read_text(encoding="utf-8"))["cases"]
    failures = 0
    for case in cases:
        try:
            run_sql(case["sql"], timeout=2)
            outcome, detail = "allowed", ""
        except SqlRejectedError as exc:
            outcome, detail = "blocked", f"guard: {exc}"
        except duckdb.Error as exc:
            outcome, detail = "blocked", f"duckdb: {type(exc).__name__}"
        ok = outcome == case["expect"]
        failures += int(not ok)
        shown = " ".join(case["sql"].split())[:52]
        print(f"{'PASS' if ok else 'FAIL'}  {outcome:8} {shown:52} {detail[:70]}")
    print(f"\n{len(cases) - failures} of {len(cases)} guard tests passed")
    return int(failures > 0)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    if args == ["selftest"]:
        sys.exit(selftest())
    if not args:
        raise SystemExit('Usage: python -m core.sql.guard selftest | "<SELECT ...>"')
    result = run_sql(" ".join(args))
    print(result["columns"])
    for row in result["rows"][:20]:
        print(row)
    print(
        f"({len(result['rows'])} row(s), tables {result['tables']}, "
        f"{result['seconds']}s)"
    )


if __name__ == "__main__":
    main()
