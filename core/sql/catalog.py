"""Phase 4: the schema catalogue. One file per pack plus the shared file.

Everything else (generators, checks, the lake build, table selection and the
prompt) reads the catalogue through this module, so a table is described once.
"""

import json
import re
import sys
from functools import lru_cache

import yaml

from src.common.config import PROJECT_ROOT

SHARED_FILE = PROJECT_ROOT / "core" / "sql" / "shared_schema.yaml"
PACKS_DIR = PROJECT_ROOT / "domain_packs"
PACKS = ["hospital", "pharma", "banking", "insurance", "datasteward"]
COLUMN_KEYS = {"type", "required", "about", "references", "hint"}
TYPES = {"VARCHAR", "DATE", "TIMESTAMP", "DOUBLE", "INTEGER", "BIGINT", "BOOLEAN"}
WORD = re.compile(r"[a-z0-9]+")


def catalog_files() -> list:
    return [SHARED_FILE] + [PACKS_DIR / p / "schema.yaml" for p in PACKS]


def _check(path, name: str, spec: dict):
    for column, col in spec["columns"].items():
        unknown = set(col) - COLUMN_KEYS
        if unknown:
            raise SystemExit(
                f"{path.name}: {name}.{column} has unknown keys {sorted(unknown)}. "
                "Put quotes around any 'about' text that contains a comma."
            )
        if col.get("type") not in TYPES:
            raise SystemExit(f"{path.name}: {name}.{column} has type {col.get('type')}")


@lru_cache(maxsize=1)
def load() -> dict:
    """Merge every catalogue file into one dictionary."""
    cat = {"tables": {}, "views": {}, "join_paths": [], "examples": [], "rules": []}
    for path in catalog_files():
        if not path.exists():
            raise SystemExit(f"Catalogue file missing: {path}")
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        pack = doc["pack"]
        for kind in ("tables", "views"):
            for name, spec in (doc.get(kind) or {}).items():
                if name in cat["tables"] or name in cat["views"]:
                    raise SystemExit(f"{path.name}: '{name}' is defined twice")
                _check(path, name, spec)
                spec = dict(spec, pack=pack, kind=kind[:-1])
                spec["warehouse_only"] = bool(doc.get("warehouse_only"))
                cat[kind][name] = spec
        cat["join_paths"] += [(pack, j) for j in doc.get("join_paths") or []]
        cat["examples"] += [dict(e, pack=pack) for e in doc.get("examples") or []]
        cat["rules"] += [dict(r, pack=pack) for r in doc.get("rules") or []]
    return cat


def layer_tables() -> list:
    """Tables that go through bronze and silver, parents before children."""
    tables = {n: s for n, s in load()["tables"].items() if not s["warehouse_only"]}
    order, seen = [], set()

    def visit(name, path=()):
        if name in seen:
            return
        if name in path:
            raise SystemExit(f"Circular references: {path + (name,)}")
        for col in tables[name]["columns"].values():
            parent = (col.get("references") or ".").split(".")[0]
            if parent in tables and parent != name:
                visit(parent, path + (name,))
        seen.add(name)
        order.append(name)

    for name in tables:
        visit(name)
    return order


def foreign_keys(table: str) -> list:
    """(column, parent_table, parent_column) for one table."""
    spec = load()["tables"][table]
    out = []
    for column, col in spec["columns"].items():
        if col.get("references"):
            parent, parent_col = col["references"].split(".")
            out.append((column, parent, parent_col))
    return out


def allowed_names() -> set:
    cat = load()
    return set(cat["tables"]) | set(cat["views"])


def tokens(text: str) -> set:
    words = set(WORD.findall(text.lower()))
    return words | {w[:-1] for w in words if w.endswith("s") and len(w) > 3}


STOP = {
    "how",
    "many",
    "what",
    "which",
    "the",
    "of",
    "a",
    "an",
    "is",
    "are",
    "in",
    "for",
    "by",
    "with",
    "and",
    "or",
    "to",
    "do",
    "does",
    "have",
    "has",
    "had",
    "there",
    "per",
    "each",
    "most",
    "total",
    "number",
    "count",
    "who",
    "were",
    "was",
    "on",
    "at",
    "their",
    "that",
    "than",
    "more",
    "least",
    "any",
    "all",
}


HINTS_FILE = PROJECT_ROOT / "data" / "warehouse" / "hints.json"


def _value_words() -> dict:
    """Words in the most frequent real values of each table, from the last build.

    Lets a question that names a value (for example "prediabetes") find the
    table that stores it. DataSteward tables are skipped: their values are
    table names and would match almost every question.
    """
    if not HINTS_FILE.exists():
        return {}
    tables = load()["tables"]
    out = {}
    hints = json.loads(HINTS_FILE.read_text(encoding="utf-8"))
    for key, values in hints.items():
        table = key.split(".")[0]
        if tables.get(table, {}).get("warehouse_only"):
            continue
        out.setdefault(table, set()).update(tokens(" ".join(values)) - STOP)
    return out


def select_tables(question: str, limit: int = 6) -> list:
    """Pick the tables and views most related to a question.

    Score = 3 per keyword or name match, 2 per match with a real value,
    1 per column-name match. Then add the tables needed to join them.
    """
    words = tokens(question) - STOP
    values = _value_words()
    scores = []
    for name, spec in {**load()["tables"], **load()["views"]}.items():
        keywords = map(str, spec.get("keywords", []))
        vocab = tokens(" ".join([name.replace("_", " "), *keywords]))
        cols = tokens(" ".join(spec["columns"]).replace("_", " "))
        score = (
            3 * len(words & vocab)
            + 2 * len(words & values.get(name, set()))
            + len(words & cols)
        )
        if score:
            scores.append((score, name))
    scores.sort(key=lambda s: (-s[0], s[1]))
    chosen = [name for _, name in scores[:limit]]
    return _bridge(chosen, limit + 2)


def _graph() -> dict:
    graph = {}
    for _, join in load()["join_paths"]:
        left, right = (side.strip().split(".")[0] for side in join.split("="))
        graph.setdefault(left, set()).add(right)
        graph.setdefault(right, set()).add(left)
    return graph


def _bridge(chosen: list, limit: int) -> list:
    """Add the tables needed to join the chosen ones (shortest join path)."""
    if len(chosen) < 2:
        return chosen
    graph, out = _graph(), list(chosen)
    for target in chosen[1:]:
        paths, seen = [[chosen[0]]], {chosen[0]}
        while paths:
            path = paths.pop(0)
            if path[-1] == target:
                out += [t for t in path if t not in out]
                break
            for nxt in sorted(graph.get(path[-1], ())):
                if nxt not in seen:
                    seen.add(nxt)
                    paths.append(path + [nxt])
    return out[:limit]


def card(names: list, hints: dict | None = None) -> str:
    """The schema text the model reads, for the chosen tables only."""
    cat = load()
    hints = hints or {}
    lines = ["# DuckDB schema", ""]
    for name in names:
        spec = cat["tables"].get(name) or cat["views"][name]
        label = "view" if spec["kind"] == "view" else "table"
        lines.append(f"## {name} ({label}, pack {spec['pack']})")
        lines.append(" ".join(spec.get("about", "").split()))
        for column, col in spec["columns"].items():
            text = f"- {column} {col['type']}"
            if column == spec.get("primary_key"):
                text += " PRIMARY KEY"
            if col.get("references"):
                text += f" -> {col['references']}"
            if col.get("about"):
                text += f": {col['about']}"
            lines.append(text)
            values = hints.get(f"{name}.{column}")
            if values:
                lines.append(f"    most frequent values: {'; '.join(values)}")
        lines.append("")
    joins = [
        j
        for _, j in cat["join_paths"]
        if j.split(".")[0] in names and j.split("= ")[1].split(".")[0] in names
    ]
    if joins:
        lines.append("## Join paths")
        lines += [f"- {j}" for j in joins]
    return "\n".join(lines)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cat = load()
    args = sys.argv[1:]
    if args[:1] == ["select"] and len(args) > 1:
        print(select_tables(" ".join(args[1:])))
        return
    by_pack = {}
    for name, spec in {**cat["tables"], **cat["views"]}.items():
        by_pack.setdefault(spec["pack"], []).append(name)
    for pack, names in by_pack.items():
        print(f"{pack:12} {len(names):2}  {', '.join(names)}")
    total = len(cat["tables"]) + len(cat["views"])
    print(f"\nTables and views : {total}")
    print(f"Layer order      : {' > '.join(layer_tables())}")
    print(f"Join paths       : {len(cat['join_paths'])}")
    print(f"Example questions: {len(cat['examples'])}")
    print(f"Consistency rules: {len(cat['rules'])}")


if __name__ == "__main__":
    main()
