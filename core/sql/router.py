"""Phase 4: send a question to SQL, to RAG, or to both.

SQL: counts, totals, averages, rankings, breakdowns and complete lists,
     because they need every row.
RAG: what a document says (labels, warnings, records), with citations.
Both: a question that asks for a number and for document evidence.
The rules are plain patterns: free, instant, explainable and testable.
"""

import re
import sys

import yaml

from src.common.config import PROJECT_ROOT

SQL_RULES = [
    (r"\bhow many\b|\bnumber of\b|\bcount\b", "asks for a count"),
    (r"\b(average|mean|median|total|sum)\b", "asks for an aggregate"),
    (r"\b(percent|percentage|proportion|share|rate)\b", "asks for a ratio"),
    (
        r"\b(highest|lowest|largest|smallest|longest|shortest|maximum|minimum)\b",
        "asks for an extreme value",
    ),
    (
        r"\bmost common\b|\btop \d+\b|\bwhich year\b|\bin which year\b",
        "asks for a ranking",
    ),
    (
        r"\bwhich \w+(?: \w+)? (?:has|had|have|is|was) the most\b|\bthe most \w+\b",
        "asks for a ranking",
    ),
    (
        r"\b(list|show)\s+(all|every)\b|\b(every|all)\s+(patients?|people|customers?)\b",
        "asks for a complete list",
    ),
    (
        r"\b(tables?|columns?|primary key|lineage|layers?|defects?|findings?)\b",
        "asks about the data catalogue",
    ),
    (
        (
            r"\b(per|by|for each|each)\s+(year|month|gender|class|type|state|pack|product|"
            r"speciality|channel|reason|status|drug|label drug)\b"
        ),
        "asks for a breakdown",
    ),
]
RAG_RULES = [
    (r"\blabels?\b(?! drugs?)|\bleaflet\b", "asks about label text"),
    (r"\bwarn(s|ing|ings)?\b|\bside effects?\b|\bcontraindicat", "asks about warnings"),
    (r"\binteract(s|ion|ions)?\b|\bdos(e|age|ing)\b", "asks about label text"),
    (r"\bused to treat\b|\bindicat(ed|ion|ions)\b", "asks about label text"),
    (
        r"\b(say|says|mention|mentions|describe|describes)\b",
        "asks what a document says",
    ),
    (r"\bwhy\b|\bexplain\b|\bevidence\b|\bguidelines?\b", "asks for an explanation"),
    (
        r"^\s*(which|what) (patients|people)\b(?!.*\b(how many|number)\b)",
        "asks for examples from records",
    ),
]
TEST_FILE = PROJECT_ROOT / "tests" / "sql" / "router_p4.yaml"
SQL_SETS = ["hospital", "pharma", "banking", "insurance", "datasteward", "cross"]


def _first(rules, text):
    for pattern, reason in rules:
        if re.search(pattern, text):
            return reason
    return None


def route(question: str) -> tuple:
    """Return ('sql' | 'rag' | 'both', reason)."""
    text = question.lower()
    sql, rag = _first(SQL_RULES, text), _first(RAG_RULES, text)
    if sql and rag:
        return "both", f"{sql}, and {rag}"
    if sql:
        return "sql", sql
    return "rag", rag or "no count or list: answer from documents"


def evaluate() -> int:
    cases = yaml.safe_load(TEST_FILE.read_text(encoding="utf-8"))["questions"]
    for name in SQL_SETS:
        path = PROJECT_ROOT / "tests" / "sql" / f"{name}_p4.yaml"
        for q in yaml.safe_load(path.read_text(encoding="utf-8"))["questions"]:
            if q["type"] == "sql":
                cases.append({"id": q["id"], "route": "sql", "question": q["question"]})
    right, how_many, how_many_right = 0, 0, 0
    for case in cases:
        got, reason = route(case["question"])
        ok = got == case["route"]
        right += int(ok)
        if case["question"].lower().startswith("how many"):
            how_many += 1
            how_many_right += int(got == "sql" or got == "both")
        if not ok or case["id"].startswith("R"):
            mark = "OK  " if ok else "MISS"
            print(
                f"{mark} {case['id']:5} expected {case['route']:4} "
                f"got {got:4} ({reason})"
            )
    print(f"\nRouting accuracy        : {right} of {len(cases)}")
    print(f"How-many questions to SQL: {how_many_right} of {how_many}")
    return int(how_many_right != how_many)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1:] == ["eval"]:
        sys.exit(evaluate())
    if len(sys.argv) > 1:
        print(route(" ".join(sys.argv[1:])))
    else:
        raise SystemExit('Usage: python -m core.sql.router eval | "<question>"')


if __name__ == "__main__":
    main()
