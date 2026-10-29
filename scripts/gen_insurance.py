"""Phase 4: insurance tables (ClaimSense): policyholder, policy, claim, payout.

Medical claims are seeded from Synthea claims (one per encounter), pharmacy
claims from prescriptions. A claim exists only while the person's policy was
in force. Time rules: claims follow encounters, payouts follow approved
claims, and nothing happens after the person's death. Fixed seed.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from core.datagen.common import add_years, data_end, gen, ids, synthea, world, write

SEED = 4402
POLICY_YEARS = 8
PRODUCTS = {
    "Basic": (0.4, 0.6, 1200),
    "Standard": (0.4, 0.8, 2400),
    "Premium": (0.2, 0.9, 4200),
}


def build_policy(rng, person: pd.DataFrame, holders: pd.DataFrame) -> pd.DataFrame:
    end = data_end()
    names = list(PRODUCTS)
    rows = []
    for row in holders.sort_values("policyholder_id").itertuples():
        p = person.loc[row.person_id]
        death = p["death_date"] if pd.notna(p["death_date"]) else None
        last = min(death, end) if death else end
        first = max(p["birth_date"], add_years(last, -POLICY_YEARS))
        start = first + timedelta(days=int(rng.integers(0, (last - first).days + 1)))
        product = rng.choice(names, p=[PRODUCTS[n][0] for n in names])
        _, rate, premium = PRODUCTS[product]
        if death:
            status, stop = "ended", death
        elif rng.random() < 0.10:
            status = "lapsed"
            stop = start + timedelta(days=int(rng.integers(0, (end - start).days + 1)))
        else:
            status, stop = "active", None
        rows.append(
            {
                "policyholder_id": row.policyholder_id,
                "person_id": row.person_id,
                "product": product,
                "coverage_rate": rate,
                "annual_premium": round(premium * float(rng.uniform(0.8, 1.3)), 2),
                "start_date": start,
                "end_date": stop,
                "status": status,
            }
        )
    policy = pd.DataFrame(rows).sort_values(["start_date", "policyholder_id"])
    policy = policy.reset_index(drop=True)
    policy.insert(0, "policy_id", ids("PLC", len(policy), 6))
    return policy


def in_force(events: pd.DataFrame, policy: pd.DataFrame) -> pd.DataFrame:
    """Keep events that happened while the person's policy was in force."""
    cover = policy[
        ["policy_id", "person_id", "start_date", "end_date", "coverage_rate"]
    ]
    merged = events.merge(cover, on="person_id", how="inner")
    stop = merged["end_date"].fillna(data_end())
    keep = (merged["service_date"] >= merged["start_date"]) & (
        merged["service_date"] <= stop
    )
    return merged[keep].drop(columns=["start_date", "end_date"])


def build_claim(rng, person: pd.DataFrame, policy: pd.DataFrame) -> pd.DataFrame:
    enc = gen("encounter")
    seed = synthea("claims.csv", ["Id", "PATIENTID", "APPOINTMENTID"])
    seed = seed.drop_duplicates("APPOINTMENTID")
    medical = seed.merge(
        enc, left_on="APPOINTMENTID", right_on="synthea_encounter_id", how="inner"
    )
    medical = pd.DataFrame(
        {
            "person_id": medical["person_id"],
            "claim_type": "medical",
            "encounter_id": medical["encounter_id"],
            "rx_id": None,
            "hcp_id": medical["hcp_id"],
            "synthea_claim_id": medical["Id"],
            "service_date": medical["encounter_date"],
            "amount_claimed": medical["total_cost"],
        }
    )
    rx = gen("prescription").dropna(subset=["encounter_id"])
    pharmacy = pd.DataFrame(
        {
            "person_id": rx["person_id"],
            "claim_type": "pharmacy",
            "encounter_id": rx["encounter_id"],
            "rx_id": rx["rx_id"],
            "hcp_id": rx["hcp_id"],
            "synthea_claim_id": None,
            "service_date": rx["start_date"],
            "amount_claimed": rx["total_cost"],
        }
    )
    claims = in_force(pd.concat([medical, pharmacy], ignore_index=True), policy)
    claims = claims.sort_values(
        ["service_date", "encounter_id", "claim_type", "rx_id"]
    ).reset_index(drop=True)

    death = claims["person_id"].map(person["death_date"])
    cap = death.where(death.notna(), data_end())
    lag = pd.to_timedelta(rng.integers(0, 31, len(claims)), unit="D")
    submitted = (pd.to_datetime(claims["service_date"]) + lag).dt.date
    claims["submitted_date"] = [min(s, c) for s, c in zip(submitted, cap, strict=True)]
    claims["status"] = rng.choice(
        ["approved", "denied", "pending"], len(claims), p=[0.85, 0.10, 0.05]
    )
    claims["amount_claimed"] = claims["amount_claimed"].fillna(0).round(2)
    approved = claims["status"] == "approved"
    claims["amount_approved"] = np.where(
        approved, (claims["amount_claimed"] * claims["coverage_rate"]).round(2), 0.0
    )
    claims.insert(0, "claim_id", ids("CLM", len(claims), 7))
    return claims.drop(columns=["coverage_rate"])


def build_payout(rng, person: pd.DataFrame, claim: pd.DataFrame) -> pd.DataFrame:
    paid = claim[claim["status"] == "approved"].copy()
    death = paid["person_id"].map(person["death_date"])
    cap = death.where(death.notna(), data_end())
    lag = pd.to_timedelta(rng.integers(3, 22, len(paid)), unit="D")
    due = (pd.to_datetime(paid["submitted_date"]) + lag).dt.date
    paid["payout_date"] = [min(d, c) for d, c in zip(due, cap, strict=True)]
    paid["by_transfer"] = rng.random(len(paid)) < 0.9

    acc = gen("account")
    acc["rank"] = acc["account_type"].map({"current": 0, "savings": 1})
    pairs = paid[["claim_id", "person_id", "payout_date"]].merge(
        acc[["account_id", "person_id", "open_date", "close_date", "rank"]],
        on="person_id",
    )
    open_then = (pairs["open_date"] <= pairs["payout_date"]) & (
        pairs["close_date"].isna() | (pairs["close_date"] >= pairs["payout_date"])
    )
    best = (
        pairs[open_then]
        .sort_values(["claim_id", "rank", "account_id"])
        .drop_duplicates("claim_id")
        .set_index("claim_id")["account_id"]
    )
    paid["account_id"] = paid["claim_id"].map(best)
    paid.loc[~paid["by_transfer"], "account_id"] = None
    paid["method"] = np.where(paid["account_id"].notna(), "bank_transfer", "cheque")
    paid["amount"] = paid["amount_approved"]
    paid = paid.sort_values(["payout_date", "claim_id"]).reset_index(drop=True)
    paid.insert(0, "payout_id", ids("PAY", len(paid), 7))
    return paid[
        [
            "payout_id",
            "claim_id",
            "person_id",
            "account_id",
            "payout_date",
            "amount",
            "method",
        ]
    ]


def main():
    rng = np.random.default_rng(SEED)
    person = gen("person").set_index("person_id")
    holders = world("policyholder")
    write("policyholder", holders[["policyholder_id", "person_id"]])
    policy = build_policy(rng, person, holders)
    write("policy", policy)
    claim = build_claim(rng, person, policy)
    write("claim", claim)
    payout = build_payout(rng, person, claim)
    write("payout", payout)
    print(
        f"\nSeed {SEED}. Claims by type: {claim['claim_type'].value_counts().to_dict()}"
    )
    print(f"Payouts by method: {payout['method'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
