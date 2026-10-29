"""Phase 4: pharma tables (PharmaSignal): hcp, drug and prescription.

HCPs come from Synthea providers. Drugs are the distinct medication products,
each linked to a pharma-pack label drug with the Phase 3 matching rule.
The prescriber of a prescription is the clinician of its encounter.
"""

import re

import pandas as pd
import yaml

from core.datagen.common import (
    drop_after_death,
    gen,
    ids,
    person_map,
    synthea,
    to_date,
    write,
)
from src.common.config import PROJECT_ROOT

PHARMA_PACK = PROJECT_ROOT / "core" / "packs" / "pharma.yaml"


def label_drugs() -> list:
    pack = yaml.safe_load(PHARMA_PACK.read_text(encoding="utf-8"))
    return pack["source"]["drugs"]


def build_hcp() -> pd.DataFrame:
    prov = synthea(
        "providers.csv", ["Id", "ORGANIZATION", "GENDER", "SPECIALITY", "STATE"]
    )
    org = gen("organisation")
    org_map = dict(zip(org["synthea_org_id"], org["organisation_id"], strict=True))
    return pd.DataFrame(
        {
            "hcp_id": ids("HCP", len(prov), 5),
            "synthea_provider_id": prov["Id"],
            "organisation_id": prov["ORGANIZATION"].map(org_map),
            "gender": prov["GENDER"],
            "speciality": prov["SPECIALITY"],
            "state": prov["STATE"],
        }
    )


def match_label(description: str, drugs: list):
    """Phase 3 rule: the drug name at the start of a word in the description."""
    lowered = description.lower()
    for drug in drugs:
        if re.search(rf"\b{re.escape(drug)}", lowered):
            return drug
    return None


def build_drug(meds: pd.DataFrame) -> pd.DataFrame:
    products = meds.drop_duplicates("CODE").sort_values("CODE").reset_index(drop=True)
    drugs = label_drugs()
    return pd.DataFrame(
        {
            "drug_id": ids("DRG", len(products), 4),
            "rxnorm_code": products["CODE"],
            "description": products["DESCRIPTION"],
            "label_drug": [match_label(d, drugs) for d in products["DESCRIPTION"]],
            "is_combination": products["DESCRIPTION"].str.contains(" / ", regex=False),
        }
    )


def build_prescription(meds: pd.DataFrame, drug: pd.DataFrame) -> pd.DataFrame:
    meds = meds.sort_values(["START", "PATIENT", "CODE"]).reset_index(drop=True)
    enc = gen("encounter")[["synthea_encounter_id", "encounter_id", "hcp_id"]]
    enc_id = dict(zip(enc["synthea_encounter_id"], enc["encounter_id"], strict=True))
    enc_hcp = dict(zip(enc["encounter_id"], enc["hcp_id"], strict=True))
    drug_id = dict(zip(drug["rxnorm_code"], drug["drug_id"], strict=True))
    encounter_id = meds["ENCOUNTER"].map(enc_id)
    stop = to_date(meds["STOP"])
    return pd.DataFrame(
        {
            "rx_id": ids("RX", len(meds), 7),
            "person_id": meds["PATIENT"].map(person_map()),
            "encounter_id": encounter_id,
            "hcp_id": encounter_id.map(enc_hcp),
            "drug_id": meds["CODE"].map(drug_id),
            "start_date": to_date(meds["START"]),
            "stop_date": stop,
            "is_active": pd.isna(stop),
            "dispenses": pd.to_numeric(meds["DISPENSES"], errors="coerce"),
            "total_cost": pd.to_numeric(meds["TOTALCOST"], errors="coerce"),
        }
    )


def main():
    write("hcp", build_hcp())
    cols = [
        "START",
        "STOP",
        "PATIENT",
        "ENCOUNTER",
        "CODE",
        "DESCRIPTION",
        "DISPENSES",
        "TOTALCOST",
    ]
    meds = synthea("medications.csv", cols)
    drug = build_drug(meds)
    write("drug", drug)
    rx = drop_after_death(build_prescription(meds, drug), "start_date", "prescription")
    write("prescription", rx)
    linked = drug.dropna(subset=["label_drug"])
    print(f"\nProducts linked to a label drug: {len(linked)} of {len(drug)}")
    counts = linked.groupby("label_drug").size()
    for name in label_drugs():
        print(f"  {name:20} {int(counts.get(name, 0))} product(s)")


if __name__ == "__main__":
    main()
