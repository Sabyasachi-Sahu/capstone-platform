"""Phase 2: check that each claim in an answer is true for the cited patient."""

import json
import re
import sys

from src.common.config import PROJECT_ROOT
from src.phase2.rag_answer import HYPHENS, ID_PATTERN, answer

DOCS_FILE = PROJECT_ROOT / "data" / "processed" / "patient_docs.jsonl"
BOLD_PATTERN = re.compile(r"\*\*(.+?)\*\*")
PAREN_PATTERN = re.compile(r"\(([^)]*)\)")


def load_records() -> dict:
    with DOCS_FILE.open(encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]
    return {d["id"]: d["text"] for d in docs}


def words(text: str) -> frozenset:
    found = re.findall(r"[a-z0-9]+", text.lower())
    return frozenset(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in found)


def record_conditions(text: str) -> set:
    found = set()
    for line in text.splitlines():
        if line.startswith(("Active conditions:", "Past conditions:")):
            for item in line.split(":", 1)[1].split(";"):
                found.add(words(item))
    return found


def name_variants(name: str) -> list:
    options = [PAREN_PATTERN.sub("", name)] + PAREN_PATTERN.findall(name)
    return [w for w in (words(o) for o in options) if w]


def matches(variants: list, conditions: set) -> bool:
    return any(v <= c for v in variants for c in conditions)


def check_claims(answer_text: str, records: dict) -> list:
    vocabulary = set()
    for text in records.values():
        vocabulary |= record_conditions(text)

    table = str.maketrans(HYPHENS, "-" * len(HYPHENS))
    results = []
    for line in answer_text.splitlines():
        ids = sorted(set(ID_PATTERN.findall(line.translate(table).lower())))
        for name in BOLD_PATTERN.findall(line):
            variants = name_variants(name)
            label = " ".join(name.split())
            for patient_id in ids:
                text = records.get(patient_id)
                if text is not None and matches(variants, record_conditions(text)):
                    status = "supported"
                elif matches(variants, vocabulary):
                    status = "wrong"
                else:
                    status = "unclear"
                results.append(
                    {"condition": label, "patient": patient_id, "status": status}
                )
    return results


def main():
    question = (
        " ".join(sys.argv[1:])
        or "What conditions commonly occur alongside type 2 diabetes?"
    )
    result = answer(question)
    claims = check_claims(result["answer"], load_records())
    print(f"Question: {question}\n")
    print(result["answer"])

    if not claims:
        print("\nWARNING: 0 claims found. Nothing was checked (answer format?).")
    count = {
        s: sum(c["status"] == s for c in claims)
        for s in ("supported", "wrong", "unclear")
    }
    print(
        f"\nClaim check: {count['supported']} supported, "
        f"{count['wrong']} wrong, {count['unclear']} unclear"
    )
    for c in claims:
        if c["status"] == "wrong":
            print(f"  WRONG: '{c['condition']}' is not in record {c['patient']}")
    for c in claims:
        if c["status"] == "unclear":
            print(f"  UNCLEAR: '{c['condition']}' (record {c['patient']})")


if __name__ == "__main__":
    main()
