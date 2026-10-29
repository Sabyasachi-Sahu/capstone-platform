"""Phase 4: banking tables (CreditGuard): bank_customer with KYC fields,
account, loan and bank_transaction. Fixed seed. Keyed on person_id.

Time rules: customers under 18 get a savings account only and no loans.
Nothing starts after the person's death or after the end of the data, and
transactions are generated for the last five years only, to keep the size small.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from core.datagen.common import add_years, data_end, gen, ids, world, write

SEED = 4401
HISTORY_YEARS = 5
TXN_PER_MONTH = {"current": 4.0, "savings": 0.8}
TYPES = {
    "current": (
        ["PAYMENT", "DEBIT", "TRANSFER", "CASH_OUT", "CASH_IN"],
        [0.45, 0.20, 0.15, 0.10, 0.10],
    ),
    "savings": (["TRANSFER", "CASH_IN"], [0.5, 0.5]),
}
CHANNEL = {
    "PAYMENT": ["card", "online"],
    "DEBIT": ["card"],
    "TRANSFER": ["online", "branch"],
    "CASH_OUT": ["atm", "branch"],
    "CASH_IN": ["branch", "atm"],
}
MEAN_AMOUNT = {
    "PAYMENT": 60,
    "DEBIT": 35,
    "TRANSFER": 400,
    "CASH_OUT": 120,
    "CASH_IN": 900,
}
LOAN = {  # type: (weight, principal low, principal high, terms in months)
    "personal": (0.40, 1_000, 25_000, [12, 24, 36, 60]),
    "auto": (0.25, 5_000, 45_000, [36, 48, 60, 72]),
    "mortgage": (0.20, 80_000, 600_000, [180, 240, 300, 360]),
    "medical": (0.15, 500, 20_000, [6, 12, 24, 36]),
}


def pick(rng, low, high):
    """A random date between two dates (inclusive)."""
    span = max((high - low).days, 0)
    return low + timedelta(days=int(rng.integers(0, span + 1)))


def build(rng):
    end = data_end()
    history_start = end - timedelta(days=365 * HISTORY_YEARS)
    person = gen("person").set_index("person_id")
    customers = world("bank_customer").sort_values("customer_id")
    customer_rows, account_rows, loan_rows, txn_parts = [], [], [], []

    for row in customers.itertuples():
        p = person.loc[row.person_id]
        birth = p["birth_date"]
        death = p["death_date"] if pd.notna(p["death_date"]) else None
        last = min(death, end) if death else end
        adult = add_years(birth, 18)
        is_adult = adult <= last
        earliest = max(adult, add_years(last, -15)) if is_adult else birth
        since = pick(rng, earliest, last)
        kyc = rng.choice(["verified", "pending", "expired"], p=[0.85, 0.08, 0.07])
        customer_rows.append(
            {
                "customer_id": row.customer_id,
                "person_id": row.person_id,
                "customer_since": since,
                "kyc_status": kyc,
                "kyc_verified_date": None
                if kyc == "pending"
                else pick(rng, since, last),
                "risk_rating": rng.choice(["low", "medium", "high"], p=[0.6, 0.3, 0.1]),
            }
        )

        kinds = ["current"] + (["savings"] if rng.random() < 0.5 else [])
        if not is_adult:
            kinds = ["savings"]
        for kind in kinds:
            opened = pick(rng, since, last)
            closed = None
            if death:
                closed = death
            elif rng.random() < 0.10:
                closed = pick(rng, opened, end)
            account_rows.append(
                {
                    "customer_id": row.customer_id,
                    "person_id": row.person_id,
                    "account_type": kind,
                    "open_date": opened,
                    "close_date": closed,
                    "status": "closed" if closed else "open",
                }
            )

        n_loans = rng.choice([0, 1, 2], p=[0.5, 0.4, 0.1]) if is_adult else 0
        for _ in range(n_loans):
            names = list(LOAN)
            kind = rng.choice(names, p=[LOAN[k][0] for k in names])
            _, low, high, terms = LOAN[kind]
            start = pick(rng, max(since, adult), last)
            term = int(rng.choice(terms))
            finished = start + timedelta(days=30 * term) < end
            if finished:
                status = rng.choice(["repaid", "defaulted"], p=[0.9, 0.1])
            else:
                status = rng.choice(
                    ["active", "in_arrears", "defaulted"], p=[0.85, 0.11, 0.04]
                )
            due = {
                "in_arrears": int(rng.integers(30, 121)),
                "defaulted": int(rng.integers(121, 366)),
            }.get(status, 0)
            loan_rows.append(
                {
                    "customer_id": row.customer_id,
                    "person_id": row.person_id,
                    "loan_type": kind,
                    "principal": round(float(rng.uniform(low, high)), 2),
                    "interest_rate": round(float(rng.uniform(3.0, 14.0)), 2),
                    "term_months": term,
                    "start_date": start,
                    "status": status,
                    "days_past_due": due,
                }
            )

    account = pd.DataFrame(account_rows).sort_values(["open_date", "customer_id"])
    account = account.reset_index(drop=True)
    account.insert(0, "account_id", ids("ACC", len(account), 6))
    loan = pd.DataFrame(loan_rows).sort_values(["start_date", "customer_id"])
    loan = loan.reset_index(drop=True)
    loan.insert(0, "loan_id", ids("LON", len(loan), 6))

    for acc in account.itertuples():
        death = person.loc[acc.person_id, "death_date"]
        first = max(acc.open_date, history_start)
        stops = [end, acc.close_date, death if pd.notna(death) else None]
        last = min(d for d in stops if d is not None)
        days = (last - first).days
        if days < 0:
            continue
        n = rng.poisson(TXN_PER_MONTH[acc.account_type] * (days + 1) / 30)
        if n == 0:
            continue
        names, weights = TYPES[acc.account_type]
        kinds = rng.choice(names, size=n, p=weights)
        amounts = np.round(
            rng.lognormal(0, 0.8, n) * np.array([MEAN_AMOUNT[k] for k in kinds]), 2
        )
        direction = np.where(
            kinds == "CASH_IN",
            "credit",
            np.where((kinds == "TRANSFER") & (rng.random(n) < 0.5), "credit", "debit"),
        )
        txn_parts.append(
            pd.DataFrame(
                {
                    "account_id": acc.account_id,
                    "person_id": acc.person_id,
                    "txn_date": [
                        first + timedelta(days=int(d))
                        for d in rng.integers(0, days + 1, n)
                    ],
                    "txn_type": kinds,
                    "direction": direction,
                    "amount": amounts,
                    "channel": [rng.choice(CHANNEL[k]) for k in kinds],
                }
            )
        )

    txn = pd.concat(txn_parts).sort_values(["txn_date", "account_id"])
    txn = txn.reset_index(drop=True)
    txn.insert(0, "txn_id", ids("TXN", len(txn), 7))
    return pd.DataFrame(customer_rows), account, loan, txn


def main():
    rng = np.random.default_rng(SEED)
    customer, account, loan, txn = build(rng)
    write("bank_customer", customer)
    write("account", account)
    write("loan", loan)
    write("bank_transaction", txn)
    print(f"\nSeed {SEED}. Loan status: {loan['status'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
