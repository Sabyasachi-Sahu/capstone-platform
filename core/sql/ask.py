"""Phase 4: one entry point for any question. Route it, then answer with SQL,
with RAG (the Phase 3 pack answer), or with both.

RAG runs the Phase 3 command (python -m src.phase3.pack_rag <pack> "<question>")
in a child process, so this file does not depend on Phase 3 function names.
"""

import re
import subprocess
import sys

import yaml

from core.sql.generator import BOTH_NOTE, ask_sql, print_result
from core.sql.router import route
from src.common.config import PROJECT_ROOT

PHARMA_PACK = PROJECT_ROOT / "core" / "packs" / "pharma.yaml"


def pick_pack(question: str) -> str:
    drugs = yaml.safe_load(PHARMA_PACK.read_text(encoding="utf-8"))["source"]["drugs"]
    text = question.lower()
    if re.search(r"\blabels?\b", text) or any(d in text for d in drugs):
        return "pharma"
    return "hospital"


def ask_rag(pack: str, question: str) -> str:
    done = subprocess.run(
        [sys.executable, "-m", "src.phase3.pack_rag", pack, question],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=PROJECT_ROOT,
        timeout=300,
        check=False,
    )
    if done.returncode != 0:
        return f"RAG failed (exit {done.returncode}):\n{done.stderr[-1500:]}"
    return done.stdout


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    pack = None
    if "--pack" in args:
        i = args.index("--pack")
        pack = args[i + 1]
        args = args[:i] + args[i + 2 :]
    if not args:
        raise SystemExit('Usage: python -m core.sql.ask "<question>" [--pack <pack>]')
    question = " ".join(args)
    where, reason = route(question)
    print(f"Route: {where.upper()} ({reason})\n")
    if where in ("sql", "both"):
        note = BOTH_NOTE if where == "both" else ""
        print_result(ask_sql(question, source="ask", note=note))
    if where in ("rag", "both"):
        chosen = pack or pick_pack(question)
        print(f"\n--- RAG answer from the {chosen} pack ---\n")
        print(ask_rag(chosen, question))


if __name__ == "__main__":
    main()
