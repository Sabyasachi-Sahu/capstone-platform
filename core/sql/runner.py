"""Phase 4: validate the SQL test sets and measure execution accuracy per pack.

validate [pack|all]                 run every gold query; check the test set
run --pack <pack|all|cross> [--runs N] [--only ID,ID] [--baseline]
baseline                            print the saved baseline
"""

import json
import statistics
import sys
from datetime import datetime

import yaml

from core.sql.catalog import PACKS, load, select_tables
from core.sql.generator import ask_sql
from core.sql.guard import run_sql
from core.sql.scoring import score
from src.common.config import PROJECT_ROOT

TESTS = PROJECT_ROOT / "tests" / "sql"
SCORES = PROJECT_ROOT / "tests" / "scores"
BASELINE = PROJECT_ROOT / "tests" / "baseline_sql.json"
SETS = PACKS + ["cross"]


def questions(pack: str) -> list:
    path = TESTS / f"{pack}_p4.yaml"
    if not path.exists():
        raise SystemExit(f"Test file not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))["questions"]


def validate(pack: str) -> int:
    examples = {e["question"].strip().lower() for e in load()["examples"]}
    bad, hits, total = 0, 0, 0
    print(f"--- {pack} ---")
    for q in questions(pack):
        if q["question"].strip().lower() in examples:
            print(f"{q['id']:5} FAIL  the question is also a prompt example")
            bad += 1
            continue
        if q["type"] != "sql":
            print(f"{q['id']:5} {q['type']:12} nothing to run")
            continue
        gold = run_sql(q["gold_sql"])
        rows = gold["rows"]
        status, note = "OK", ""
        flat = [v for row in rows for v in row]
        if gold["truncated"]:
            status, note = "FAIL", "  <- more rows than the cap; ask for less"
        elif q.get("expect_zero"):
            if flat not in ([0], [None]):
                status, note = "FAIL", "  <- expected 0: an integrity rule is broken"
        elif not rows or all(v in (None, 0) for v in flat):
            status, note = (
                "FAIL",
                "  <- empty or zero result: check the wording or data",
            )
        if q.get("tie_sql"):
            pair = [r[0] for r in run_sql(q["tie_sql"])["rows"]]
            if len(pair) == 2 and pair[0] == pair[1]:
                status, note = "FAIL", f"  <- tie at the cut-off {pair}"
        chosen = set(select_tables(q["question"]))
        found = set(gold["tables"]) <= chosen
        hits += int(found)
        total += 1
        sel = "sel ok  " if found else "sel MISS"
        bad += int(status != "OK")
        first = rows[0] if rows else []
        shown = str(first)[:40]
        print(f"{q['id']:5} {status:4} {sel} rows={len(rows):<3} first={shown}{note}")
    if total:
        print(f"Table selection recall: {hits}/{total}")
    return bad


def run(pack: str, runs: int, only: set, save: bool):
    qs = [q for q in questions(pack) if not only or q["id"] in only]
    golds = {q["id"]: run_sql(q["gold_sql"]) for q in qs if q["type"] == "sql"}
    records, run_scores = [], []
    for run_no in range(1, runs + 1):
        passed = 0
        for q in qs:
            try:
                out = ask_sql(q["question"], summarise=False, source=f"test:{pack}")
                scored = score(q, out, golds.get(q["id"]))
            except Exception as exc:  # noqa: BLE001 - one failure must not stop the run
                error = f"error: {type(exc).__name__}: {exc}"
                out = {"sql": None, "errors": [error], "tables_selected": []}
                scored = {"passed": False, "reasons": [error], "attempts": 0}
            passed += int(scored["passed"])
            mark = "PASS" if scored["passed"] else "FAIL"
            retry = " (retried)" if scored["attempts"] > 1 else ""
            why = (
                f"  <- {'; '.join(scored['reasons'])[:110]}"
                if scored["reasons"]
                else ""
            )
            print(f"run {run_no}  {q['id']:5} {q['type']:12} {mark}{retry}{why}")
            records.append(
                {
                    "run": run_no,
                    "id": q["id"],
                    "scenario": q.get("scenario"),
                    "sql": out.get("sql"),
                    "errors": out.get("errors"),
                    "tables_selected": out.get("tables_selected"),
                    **scored,
                }
            )
        run_scores.append(passed / len(qs))

    print("\nPer question (passes out of runs):")
    for q in qs:
        n = sum(r["passed"] for r in records if r["id"] == q["id"])
        print(f"  {q['id']:5} {q['type']:12} {n}/{runs}")
    mean = statistics.mean(run_scores)
    spread = statistics.pstdev(run_scores) if runs > 1 else 0.0
    retried = sum(r["attempts"] > 1 for r in records)
    print(
        f"\nExecution accuracy ({pack}): mean {mean:.2f}  spread {spread:.2f}  "
        f"min {min(run_scores):.2f}  max {max(run_scores):.2f}"
    )
    print(f"Answers that needed the retry: {retried} of {len(records)}")
    scenarios = {}
    for r in records:
        if r["scenario"]:
            scenarios.setdefault(r["scenario"], []).append(r["passed"])
    for name, results in sorted(scenarios.items()):
        rate = sum(results) / len(results)
        verdict = "PASS" if rate > 0.5 else "FAIL"
        print(
            f"Scenario {name}: {sum(results)}/{len(results)} answers correct  {verdict}"
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "runs": runs,
        "questions": len(qs),
        "run_scores": run_scores,
        "mean": round(mean, 4),
        "spread": round(spread, 4),
        "retried": retried,
        "created": stamp,
        "scenarios": {k: round(sum(v) / len(v), 4) for k, v in scenarios.items()},
    }
    SCORES.mkdir(parents=True, exist_ok=True)
    detail = SCORES / f"sql_{pack}_{stamp}.json"
    detail.write_text(
        json.dumps(
            {**summary, "pack": pack, "records": records}, indent=2, default=str
        ),
        encoding="utf-8",
    )
    print(f"Saved: {detail}")
    if save:
        if only:
            raise SystemExit("A baseline must use every question. Remove --only.")
        base = (
            json.loads(BASELINE.read_text(encoding="utf-8"))
            if BASELINE.exists()
            else {}
        )
        base[pack] = summary
        BASELINE.write_text(json.dumps(base, indent=2), encoding="utf-8")
        print(f"Baseline saved for '{pack}': {BASELINE}")


def show_baseline():
    if not BASELINE.exists():
        raise SystemExit("No baseline yet. Run with --baseline.")
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    print(f"{'set':12} {'mean':>5} {'spread':>6} {'runs':>4} {'qs':>3} {'retried':>7}")
    for pack in SETS:
        if pack in base:
            b = base[pack]
            print(
                f"{pack:12} {b['mean']:5.2f} {b['spread']:6.2f} {b['runs']:4} "
                f"{b['questions']:3} {b['retried']:7}"
            )
        else:
            print(f"{pack:12} (not run yet)")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    if args[:1] == ["validate"]:
        target = args[1] if len(args) > 1 else "all"
        packs = SETS if target == "all" else [target]
        bad = sum(validate(p) for p in packs)
        print(
            f"\n{bad} question(s) need attention"
            if bad
            else "\nAll gold queries run and give a usable answer"
        )
        sys.exit(int(bad > 0))
    elif args[:1] == ["run"] and "--pack" in args:
        target = args[args.index("--pack") + 1]
        runs = int(args[args.index("--runs") + 1]) if "--runs" in args else 3
        only = set()
        if "--only" in args:
            only = set(args[args.index("--only") + 1].split(","))
        for pack in SETS if target == "all" else [target]:
            run(pack, runs, only, "--baseline" in args)
    elif args[:1] == ["baseline"]:
        show_baseline()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
