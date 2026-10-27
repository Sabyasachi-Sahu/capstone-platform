"""Check that this machine can reach the openFDA drug-label API."""

import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
URL = "https://api.fda.gov/drug/label.json"


def main():
    params = {"search": "openfda.generic_name:metformin", "limit": 1}
    key = os.getenv("OPENFDA_API_KEY")
    if key:
        params["api_key"] = key
    try:
        r = requests.get(URL, params=params, timeout=30)
    except requests.exceptions.SSLError as exc:
        print("SSL error. A company proxy may be inspecting traffic.")
        print(f"  {type(exc).__name__}")
        sys.exit(1)
    except requests.exceptions.RequestException as exc:
        print("Network error. The site may be blocked, or you are offline.")
        print(f"  {type(exc).__name__}")
        sys.exit(1)

    print(f"HTTP status : {r.status_code}")
    if r.status_code != 200:
        print(r.text[:300])
        sys.exit(1)
    data = r.json()
    total = data["meta"]["results"]["total"]
    label = data["results"][0]
    openfda = label.get("openfda", {})
    sections = [k for k, v in label.items() if isinstance(v, list) and k != "openfda"]
    print(f"Matches     : {total}")
    print(f"Brand       : {openfda.get('brand_name')}")
    print(f"Generic     : {openfda.get('generic_name')}")
    print(f"Sections    : {len(sections)} (for example {sections[:5]})")
    print("openFDA reachable. Do not rely on this data for medical decisions.")


if __name__ == "__main__":
    main()
