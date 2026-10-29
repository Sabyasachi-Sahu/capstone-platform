"""Phase 4: hospital tables (CareBridge) and the shared person table.

Reads Synthea patients, organisations, encounters and conditions and writes
person, organisation, encounter, admission and diagnosis to data/world/gen.
No random numbers are used, so the output is fully determined by Synthea.
"""

import numpy as np
import pandas as pd

from core.datagen.common import (
    drop_after_death,
    id_map,
    ids,
    person_map,
    save_meta,
    synthea,
    to_date,
    world,
    write,
)


def build_person() -> pd.DataFrame:
    person = world("person")
    patients = synthea("patients.csv", ["Id", "DEATHDATE"])
    death = dict(zip(patients["Id"], to_date(patients["DEATHDATE"]), strict=True))
    policy = set(world("policyholder")["person_id"])
    bank = set(world("bank_customer")["person_id"])
    return pd.DataFrame(
        {
            "person_id": person["person_id"],
            "birth_date": to_date(person["dob"].astype(str)),
            "death_date": person["synthea_id"].map(death),
            "gender": person["gender"],
            "state": person["state"],
            "is_policyholder": person["person_id"].isin(policy),
            "is_bank_customer": person["person_id"].isin(bank),
        }
    )


def build_organisation() -> pd.DataFrame:
    org = synthea("organizations.csv", ["Id", "NAME", "CITY", "STATE"])
    return pd.DataFrame(
        {
            "organisation_id": ids("ORG", len(org), 4),
            "synthea_org_id": org["Id"],
            "name": org["NAME"],
            "city": org["CITY"],
            "state": org["STATE"],
        }
    )


def build_encounter() -> pd.DataFrame:
    cols = [
        "Id",
        "START",
        "STOP",
        "PATIENT",
        "ORGANIZATION",
        "PROVIDER",
        "ENCOUNTERCLASS",
        "DESCRIPTION",
        "TOTAL_CLAIM_COST",
    ]
    enc = synthea("encounters.csv", cols).sort_values(["START", "Id"])
    enc = enc.reset_index(drop=True)
    start = pd.to_datetime(enc["START"].str[:19], errors="coerce")
    stop = pd.to_datetime(enc["STOP"].str[:19].replace("", None), errors="coerce")
    frame = pd.DataFrame(
        {
            "encounter_id": ids("ENC", len(enc), 7),
            "synthea_encounter_id": enc["Id"],
            "person_id": enc["PATIENT"].map(person_map()),
            "organisation_id": enc["ORGANIZATION"].map(
                id_map("organizations.csv", "ORG", 4)
            ),
            "hcp_id": enc["PROVIDER"].map(id_map("providers.csv", "HCP", 5)),
            "encounter_class": enc["ENCOUNTERCLASS"],
            "encounter_date": start.dt.date,
            "start_time": start,
            "stop_time": stop,
            "description": enc["DESCRIPTION"],
            "total_cost": pd.to_numeric(enc["TOTAL_CLAIM_COST"], errors="coerce"),
        }
    )
    return frame


def build_admission(encounter: pd.DataFrame) -> pd.DataFrame:
    stays = encounter[encounter["encounter_class"] == "inpatient"].copy()
    stays["admit_date"] = pd.to_datetime(stays["encounter_date"])
    discharge = pd.to_datetime(stays["stop_time"]).dt.normalize()
    stays["discharge_date"] = discharge.fillna(stays["admit_date"])
    stays = stays.sort_values(["person_id", "admit_date", "encounter_id"])
    next_admit = stays.groupby("person_id")["admit_date"].shift(-1)
    gap = (next_admit - stays["discharge_date"]).dt.days
    stays["days_to_next_admission"] = gap.astype("Int64")
    stays["readmitted_30d"] = gap.between(0, 30).fillna(False).astype(bool)
    stays["length_of_stay"] = (stays["discharge_date"] - stays["admit_date"]).dt.days
    stays = stays.sort_values(["admit_date", "encounter_id"]).reset_index(drop=True)
    stays["admission_id"] = ids("ADM", len(stays), 6)
    stays["admit_date"] = stays["admit_date"].dt.date
    stays["discharge_date"] = stays["discharge_date"].dt.date
    return stays[
        [
            "admission_id",
            "encounter_id",
            "person_id",
            "admit_date",
            "discharge_date",
            "length_of_stay",
            "days_to_next_admission",
            "readmitted_30d",
        ]
    ]


def build_diagnosis(encounter: pd.DataFrame) -> pd.DataFrame:
    cols = ["START", "STOP", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION"]
    con = synthea("conditions.csv", cols).sort_values(["START", "PATIENT", "CODE"])
    con = con.reset_index(drop=True)
    enc_map = dict(
        zip(encounter["synthea_encounter_id"], encounter["encounter_id"], strict=True)
    )
    stop = to_date(con["STOP"])
    frame = pd.DataFrame(
        {
            "diagnosis_id": ids("DX", len(con), 7),
            "person_id": con["PATIENT"].map(person_map()),
            "encounter_id": con["ENCOUNTER"].map(enc_map),
            "start_date": to_date(con["START"]),
            "stop_date": stop,
            "code": con["CODE"],
            "description": con["DESCRIPTION"],
            "is_active": pd.isna(stop),
        }
    )
    return frame


def main():
    person = build_person()
    write("person", person)
    write("organisation", build_organisation())
    encounter = build_encounter()
    unknown = int(encounter["person_id"].isna().sum())
    if unknown:
        raise SystemExit(f"{unknown} encounters belong to patients not in person")
    save_meta(data_end=max(encounter["encounter_date"]), source="Synthea")
    encounter = drop_after_death(encounter, "encounter_date", "encounter")
    write("encounter", encounter)
    write("admission", build_admission(encounter))
    diagnosis = drop_after_death(build_diagnosis(encounter), "start_date", "diagnosis")
    write("diagnosis", diagnosis)
    readmit = build_admission(encounter)["readmitted_30d"]
    print(f"\nData end date : {max(encounter['encounter_date'])}")
    print(f"Deceased      : {int(person['death_date'].notna().sum())}")
    print(f"Readmissions  : {int(np.sum(readmit))} admissions followed within 30 days")


if __name__ == "__main__":
    main()
