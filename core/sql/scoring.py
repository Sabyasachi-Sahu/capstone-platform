"""Phase 4: execution accuracy. A generated query passes when its result
matches the result of the gold query.

Rules: same number of rows; every required gold column equals some column of
the generated result (extra columns, other names and another column order are
allowed); rows compared as a set; whole numbers must match exactly; decimals
may differ by rounding (0.05 or 0.1%, whichever is larger).
"""

from collections import Counter
from datetime import date, datetime

REFUSAL_MARKERS = (
    "cannot",
    "can't",
    "not available",
    "no information",
    "unable",
    "does not contain",
    "doesn't contain",
    "not in the",
    "no column",
    "no table",
    "not recorded",
    "not stored",
)


def norm(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()[:10]
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        return text.lower()


def close(gold, pred) -> bool:
    if gold == pred:
        return True
    if isinstance(gold, float) and isinstance(pred, float):
        if gold.is_integer():
            return abs(gold - pred) < 1e-9
        return abs(gold - pred) <= max(0.051, 0.001 * abs(gold))
    return False


def _order(value):
    return (value is None, value or 0.0)


def same_column(gold: list, pred: list) -> bool:
    numeric = all(isinstance(v, float) for v in gold + pred if v is not None)
    if numeric:
        return all(
            close(g, p)
            for g, p in zip(
                sorted(gold, key=_order), sorted(pred, key=_order), strict=True
            )
        )
    return Counter(gold) == Counter(pred)


def columns_of(result: dict) -> list:
    width = len(result["columns"])
    return [[norm(row[i]) for row in result["rows"]] for i in range(width)]


def compare(gold: dict, pred: dict, match_columns=None) -> list:
    """Reasons the prediction differs from the gold result (empty = match)."""
    if gold["truncated"] or pred["truncated"]:
        return ["result too large to compare (over the row cap)"]
    if len(pred["rows"]) != len(gold["rows"]):
        return [f"{len(pred['rows'])} row(s), expected {len(gold['rows'])}"]
    gold_cols, pred_cols = columns_of(gold), columns_of(pred)
    wanted = match_columns if match_columns is not None else range(len(gold_cols))
    unused = list(range(len(pred_cols)))
    reasons = []
    for i in wanted:
        hits = [j for j in unused if same_column(gold_cols[i], pred_cols[j])]
        if hits:
            unused.remove(hits[0])
        else:
            got = [c[:3] for c in pred_cols]
            reasons.append(f"gold column {i} {gold_cols[i][:3]} not found; got {got}")
    return reasons


def refused(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def score(question: dict, out: dict, gold: dict | None) -> dict:
    reasons = []
    if question["type"] == "unanswerable":
        if out["result"] is not None and not out["cannot_answer"]:
            reasons.append("ran a query for a question the data cannot answer")
        elif not (out["cannot_answer"] or refused(out["answer"])):
            reasons.append("did not say the data cannot answer")
    elif out["cannot_answer"]:
        reasons.append("said it cannot answer")
    elif out["result"] is None:
        reasons.append("no query ran: " + (out["errors"] or ["unknown"])[-1])
    else:
        reasons = compare(gold, out["result"], question.get("match_columns"))
    return {"passed": not reasons, "reasons": reasons, "attempts": out["attempts"]}
