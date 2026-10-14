"""Profile Synthea patients.csv: Phase 1 debugging practice."""

import pandas as pd

from src.common.config import get_settings


def load_patients(synthea_dir):
    return pd.read_csv(synthea_dir / "patients.csv")


def profile(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "gender_counts": df["GENDER"].value_counts().to_dict(),
        "deceased": int(df["DEATHDATE"].notna().sum()),
    }


def main():
    s = get_settings()
    if s.synthea_dir is None:
        raise SystemExit("Set SYNTHEA_DIR in .env")
    df = load_patients(s.synthea_dir)
    summary = profile(df)
    for key, value in summary.items():
        print(f"{key:15}: {value}")


if __name__ == "__main__":
    main()
