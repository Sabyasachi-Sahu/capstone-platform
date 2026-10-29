"""Phase 4: text-to-SQL. Question -> tables -> SQL -> guard -> result -> answer.

1. Pick the tables most related to the question (catalogue words and join paths).
2. Ask the chat model for one DuckDB SELECT, using the schema card of those
   tables, the most frequent real values and worked examples from the same packs.
3. Run it through the guard. If it is rejected or fails, send the error back
   once, this time with the full catalogue (the first table choice may be wrong).
4. Explain the result in one or two sentences using only the returned rows.
5. Append every attempt to the audit log (data/warehouse/audit.jsonl).
"""

import json
import os
import re
import sys
import time
from datetime import datetime

import duckdb
import ollama

from core.sql.catalog import card, load, select_tables
from core.sql.guard import SqlRejectedError, run_sql
from src.common.config import PROJECT_ROOT, get_settings

WAREHOUSE = PROJECT_ROOT / "data" / "warehouse"
AUDIT = WAREHOUSE / "audit.jsonl"
HINTS = WAREHOUSE / "hints.json"
CANNOT = "CANNOT_ANSWER"
SQL_BLOCK = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

SYSTEM_PROMPT = """You write DuckDB SQL for a synthetic, cross-industry database
(hospital, pharma, banking, insurance and data governance), joined by person_id.
Rules:
- Use ONLY the tables and columns in the schema below. Never invent a name.
- Write exactly one SELECT (or WITH ... SELECT) query, in a ```sql code block.
- Count people with COUNT(DISTINCT person_id): a person can have many rows.
- Match text with ILIKE '%...%', using the wording in "most frequent values".
- Boolean columns are true/false: write WHERE is_active, or WHERE NOT is_active.
- Dates are DATE values. Use year(col), and date_diff('day', start, finish).
- Round money and averages to 2 decimals. Give computed columns short aliases.
- If the schema cannot answer the question, reply with one line:
  CANNOT_ANSWER: <the reason>
 
{schema}"""

SUMMARY_PROMPT = """Answer the question in one or two sentences, using ONLY the
query result below. Do not add numbers that are not in the result. If the result
was cut off, say that it shows only the first rows."""


def model_name() -> str:
    settings = get_settings()
    return (
        getattr(settings, "chat_model", None)
        or os.getenv("CHAT_MODEL")
        or "gpt-oss:20b-cloud"
    )


def chat(messages: list) -> str:
    response = ollama.chat(
        model=model_name(), messages=messages, options={"temperature": 0}
    )
    return response.message.content or ""


def hints() -> dict:
    if not HINTS.exists():
        raise SystemExit("hints.json not found. Run: python -m core.sql.lake build")
    return json.loads(HINTS.read_text(encoding="utf-8"))


def examples_for(tables: list, limit: int = 4) -> list:
    cat = load()
    packs = []
    for name in tables:
        spec = cat["tables"].get(name) or cat["views"].get(name)
        if spec["pack"] not in packs:
            packs.append(spec["pack"])
    chosen = [e for p in packs for e in cat["examples"] if e["pack"] == p]
    return chosen[:limit]


def build_messages(question: str, tables: list, note: str = "") -> list:
    schema = card(tables, hints())
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(schema=schema)}]
    for example in examples_for(tables):
        messages.append({"role": "user", "content": example["question"]})
        messages.append(
            {"role": "assistant", "content": f"```sql\n{example['sql'].strip()}\n```"}
        )
    messages.append({"role": "user", "content": question + note})
    return messages


def extract_sql(reply: str):
    """The SQL in a reply, None if the model refused, or "" if there is none."""
    if CANNOT in reply:
        return None
    block = SQL_BLOCK.search(reply)
    if block:
        return block.group(1).strip()
    text = reply.strip()
    return text if text.lower().startswith(("select", "with")) else ""


def result_as_text(result: dict, limit: int = 20) -> str:
    lines = [" | ".join(result["columns"])]
    lines += [" | ".join(str(v) for v in row) for row in result["rows"][:limit]]
    if result["truncated"] or len(result["rows"]) > limit:
        lines.append("(more rows not shown)")
    return "\n".join(lines)


def audit(entry: dict):
    WAREHOUSE.mkdir(parents=True, exist_ok=True)
    with AUDIT.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, default=str) + "\n")


BOTH_NOTE = (
    "\n\n(Answer only the parts of this question that need counting, totals or "
    "lists from the tables. Other parts are answered from documents separately. "
    "Return only numbers or names that are in the tables.)"
)


def ask_sql(
    question: str,
    summarise: bool = True,
    max_attempts: int = 2,
    source: str = "cli",
    note: str = "",
) -> dict:
    started = time.monotonic()
    selected = select_tables(question)
    everything = sorted(set(load()["tables"]) | set(load()["views"]))
    narrowed = bool(selected) and len(selected) < len(everything)
    out = {
        "question": question,
        "tables_selected": selected,
        "sql": None,
        "result": None,
        "answer": "",
        "attempts": 0,
        "errors": [],
        "cannot_answer": False,
    }
    messages = build_messages(question, selected if narrowed else everything, note)
    limit = max_attempts

    while out["attempts"] < limit:
        out["attempts"] += 1
        reply = chat(messages)
        sql = extract_sql(reply)
        if sql is None:
            # A refusal seen with only some tables may be wrong. Look once more
            # with the full catalogue. This extra look does not use up the retry
            # that fixes SQL errors.
            if narrowed:
                narrowed = False
                limit += 1
                out["errors"].append("refused with the chosen tables; retried with all")
                messages = build_messages(question, everything, note)
                continue
            out["cannot_answer"] = True
            out["answer"] = reply.split(CANNOT, 1)[1].lstrip(": ").strip()
            break
        out["sql"] = sql
        try:
            if not sql:
                raise SqlRejectedError("no SQL found in the reply")
            out["result"] = run_sql(sql)
            break
        except (SqlRejectedError, duckdb.Error) as exc:
            error = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
            out["errors"].append(error)
            narrowed = False
            messages = build_messages(question, everything, note)
            messages += [
                {"role": "assistant", "content": reply},
                {
                    "role": "user",
                    "content": f"That query failed with {error}. The full schema "
                    "is now in the system message. Reply with the corrected "
                    "query only.",
                },
            ]

    if out["result"] is None and not out["cannot_answer"]:
        out["answer"] = "No query could be run: " + out["errors"][-1]
    elif out["result"] is not None and summarise:
        out["answer"] = chat(
            [
                {"role": "system", "content": SUMMARY_PROMPT},
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nQuery result:\n"
                    f"{result_as_text(out['result'])}",
                },
            ]
        )

    status = (
        "cannot_answer"
        if out["cannot_answer"]
        else "ok"
        if out["result"] is not None
        else "failed"
    )
    audit(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "source": source,
            "model": model_name(),
            "question": question,
            "status": status,
            "tables_selected": selected,
            "tables_read": (out["result"] or {}).get("tables"),
            "sql": out["sql"],
            "attempts": out["attempts"],
            "errors": out["errors"],
            "rows": len((out["result"] or {}).get("rows", [])),
            "truncated": (out["result"] or {}).get("truncated"),
            "seconds": round(time.monotonic() - started, 2),
        }
    )
    return out


def print_result(out: dict):
    print(f"Question : {out['question']}")
    print(f"Tables   : {', '.join(out['tables_selected']) or '(full catalogue)'}")
    print(f"Attempts : {out['attempts']}")
    for error in out["errors"]:
        print(f"  error  : {error}")
    if out["cannot_answer"]:
        print(f"Cannot answer from the database: {out['answer']}")
        return
    print(f"\nSQL:\n{out['sql']}\n")
    if out["result"]:
        print(result_as_text(out["result"]))
        print(
            f"\n({len(out['result']['rows'])} row(s)"
            f"{', truncated' if out['result']['truncated'] else ''})"
        )
    print(f"\nAnswer: {out['answer']}")


def show_audit(last: int):
    if not AUDIT.exists():
        print("No audit entries yet.")
        return
    lines = AUDIT.read_text(encoding="utf-8").splitlines()[-last:]
    for line in lines:
        e = json.loads(line)
        print(
            f"{e['time']}  {e['status']:13} attempts={e['attempts']} "
            f"rows={e['rows']}  {e['question'][:60]}"
        )
        print(f"    {' '.join((e['sql'] or '').split())[:110]}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    if args[:1] == ["audit"]:
        show_audit(int(args[1]) if len(args) > 1 else 10)
    elif args:
        print_result(ask_sql(" ".join(args)))
    else:
        raise SystemExit('Usage: python -m core.sql.generator "<question>" | audit [N]')


if __name__ == "__main__":
    main()
