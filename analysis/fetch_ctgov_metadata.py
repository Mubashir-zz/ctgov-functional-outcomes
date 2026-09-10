#!/usr/bin/env python3
"""Fetch trial-level metadata for the oncology corpus from ClinicalTrials.gov v2.

Companion to fetch_ctgov_outcomes.py, same query and same registration cutoff, so
the two tables join one-to-one on trial id.

Output columns (data/trial_metadata.csv.gz):
    nct_id, study_type, overall_status, phases, start_year, first_submit_date,
    enrollment, lead_sponsor_class, conditions, intervention_types,
    intervention_names

Everything downstream needs this table: cohort eligibility, the cisplatin
intervention flag, and the cancer-site / phase / era strata that the direct
standardisation adjusts for.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import time
import urllib.parse
import urllib.request

API = "https://clinicaltrials.gov/api/v2/studies"
CUTOFF = "2026-07-31"
FIELDS = ",".join(
    [
        "protocolSection.identificationModule.nctId",
        "protocolSection.statusModule.studyFirstSubmitDate",
        "protocolSection.statusModule.startDateStruct",
        "protocolSection.statusModule.overallStatus",
        "protocolSection.designModule.studyType",
        "protocolSection.designModule.phases",
        "protocolSection.designModule.designInfo",
        "protocolSection.designModule.enrollmentInfo",
        "protocolSection.sponsorCollaboratorsModule.leadSponsor",
        "protocolSection.conditionsModule.conditions",
        "protocolSection.armsInterventionsModule.interventions",
    ]
)
STATE = "/tmp/ctgov_meta_state.json"
PAGE_SIZE = 1000
RETRIES = 5
HEADER = [
    "nct_id",
    "study_type",
    "primary_purpose",
    "overall_status",
    "phases",
    "start_year",
    "first_submit_date",
    "enrollment",
    "lead_sponsor_class",
    "conditions",
    "intervention_types",
    "intervention_names",
]


def get_page(token: str | None) -> dict:
    params = {
        "query.cond": "cancer",
        "fields": FIELDS,
        "pageSize": str(PAGE_SIZE),
        "countTotal": "true",
    }
    if token:
        params["pageToken"] = token
    url = f"{API}?{urllib.parse.urlencode(params)}"
    last = None
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "projectb-fetch/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"page fetch failed after {RETRIES} attempts: {last}")


def clean(value: str) -> str:
    return (value or "").replace("\r", " ").replace("\n", " ").replace("|", "/")


def metadata_row(study: dict) -> list | None:
    proto = study.get("protocolSection", {})
    nct = proto.get("identificationModule", {}).get("nctId")
    status = proto.get("statusModule", {})
    submitted = status.get("studyFirstSubmitDate", "")
    if not nct or not submitted or submitted > CUTOFF:
        return None

    design = proto.get("designModule", {}) or {}
    sponsor = (proto.get("sponsorCollaboratorsModule", {}) or {}).get("leadSponsor", {}) or {}
    conditions = (proto.get("conditionsModule", {}) or {}).get("conditions") or []
    interventions = (proto.get("armsInterventionsModule", {}) or {}).get("interventions") or []

    start = (status.get("startDateStruct", {}) or {}).get("date", "")
    start_year = start[:4] if len(start) >= 4 and start[:4].isdigit() else ""

    return [
        nct,
        design.get("studyType", ""),
        (design.get("designInfo", {}) or {}).get("primaryPurpose", ""),
        status.get("overallStatus", ""),
        ";".join(design.get("phases") or []),
        start_year,
        submitted,
        (design.get("enrollmentInfo", {}) or {}).get("count", ""),
        sponsor.get("class", ""),
        "|".join(clean(c) for c in conditions),
        "|".join(clean(i.get("type", "")) for i in interventions),
        "|".join(
            clean(" ".join([i.get("name", "")] + (i.get("otherNames") or [])))
            for i in interventions
        ).lower(),
    ]


def load_state() -> dict:
    if os.path.exists(STATE):
        with open(STATE) as fh:
            return json.load(fh)
    return {"token": None, "pages": 0, "rows": 0, "done": False}


def save_state(state: dict) -> None:
    with open(STATE, "w") as fh:
        json.dump(state, fh)


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch CTGOV v2 oncology trial metadata")
    ap.add_argument("--out", required=True, help="output .csv.gz path")
    ap.add_argument("--summary-json")
    ap.add_argument("--restart", action="store_true")
    args = ap.parse_args()

    if args.restart and os.path.exists(STATE):
        os.remove(STATE)
    state = load_state()
    if state["done"]:
        print("already complete; use --restart to refetch")
        return 0

    plain = args.out[:-3] if args.out.endswith(".gz") else args.out + ".plain"
    mode = "at" if state["pages"] and os.path.exists(plain) else "wt"
    total = None

    with open(plain, mode, newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if mode == "wt":
            writer.writerow(HEADER)
        while True:
            payload = get_page(state["token"])
            if total is None:
                total = payload.get("totalCount")
            studies = payload.get("studies", [])
            if not studies:
                break
            for study in studies:
                row = metadata_row(study)
                if row:
                    writer.writerow(row)
                    state["rows"] += 1
            state["pages"] += 1
            state["token"] = payload.get("nextPageToken")
            save_state(state)
            print(f"page {state['pages']:>4d}  trials {state['rows']:>8,}", flush=True)
            if not state["token"]:
                break

    print("compressing...", flush=True)
    with open(plain, "rb") as src, gzip.open(args.out, "wb") as dst:
        while True:
            block = src.read(1 << 20)
            if not block:
                break
            dst.write(block)
    os.remove(plain)

    state["done"] = True
    save_state(state)
    summary = {
        "query": {"cond": "cancer", "registered_on_or_before": CUTOFF},
        "pages": state["pages"],
        "trials": state["rows"],
        "total_studies_matched": total,
        "output": args.out,
    }
    print(json.dumps(summary, indent=2))
    if args.summary_json:
        with open(args.summary_json, "w") as fh:
            fh.write(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
