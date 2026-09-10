#!/usr/bin/env python3
"""Fetch the oncology outcome corpus from ClinicalTrials.gov v2.

Writes the gzipped CSV the first-pass extractor expects:
    source_trial_id, measure, description

One row per registered outcome measure (primary, secondary, other).

Corpus definition, frozen with the study:
  - condition search: cancer
  - registrations on or before 2026-07-31 (studyFirstSubmitDate)
  - studies with at least one outcome measure

Resumable: the page token and row count are checkpointed outside the project
folder, so an interrupted run picks up where it stopped.
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
        "protocolSection.outcomesModule",
    ]
)
STATE = "/tmp/ctgov_fetch_state.json"
PAGE_SIZE = 1000
RETRIES = 5


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
        except Exception as exc:  # noqa: BLE001 - retry any transport error
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"page fetch failed after {RETRIES} attempts: {last}")


def outcome_rows(study: dict):
    proto = study.get("protocolSection", {})
    nct = proto.get("identificationModule", {}).get("nctId")
    submitted = proto.get("statusModule", {}).get("studyFirstSubmitDate", "")
    if not nct or not submitted or submitted > CUTOFF:
        return
    outcomes = proto.get("outcomesModule", {}) or {}
    for key in ("primaryOutcomes", "secondaryOutcomes", "otherOutcomes"):
        for item in outcomes.get(key) or []:
            measure = (item.get("measure") or "").replace("\r", " ").replace("\n", " ")
            description = (item.get("description") or "").replace("\r", " ").replace("\n", " ")
            if measure or description:
                yield nct, measure, description


def load_state() -> dict:
    if os.path.exists(STATE):
        with open(STATE) as fh:
            return json.load(fh)
    return {"token": None, "pages": 0, "rows": 0, "trials": 0, "done": False}


def save_state(state: dict) -> None:
    with open(STATE, "w") as fh:
        json.dump(state, fh)


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch CTGOV v2 oncology outcome corpus")
    ap.add_argument("--out", required=True, help="output .csv.gz path")
    ap.add_argument("--summary-json", help="write run summary here")
    ap.add_argument("--restart", action="store_true", help="ignore any checkpoint and start over")
    args = ap.parse_args()

    if args.restart and os.path.exists(STATE):
        os.remove(STATE)
    state = load_state()
    if state["done"]:
        print("already complete; use --restart to refetch")
        return 0

    # Fetch into a plain CSV: appending to a gzip stream is not safe to resume
    # if the process is interrupted mid-member. Compress once, at the end.
    plain = args.out[:-3] if args.out.endswith(".gz") else args.out + ".plain"
    mode = "at" if state["pages"] and os.path.exists(plain) else "wt"
    trials: set[str] = set()
    total = None

    with open(plain, mode, newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if mode == "wt":
            writer.writerow(["source_trial_id", "measure", "description"])
        while True:
            payload = get_page(state["token"])
            if total is None:
                total = payload.get("totalCount")
                print(f"total studies matching query: {total:,}" if total else "total unknown")
            studies = payload.get("studies", [])
            if not studies:
                break
            for study in studies:
                for nct, measure, description in outcome_rows(study):
                    writer.writerow([nct, measure, description])
                    state["rows"] += 1
                    trials.add(nct)
            state["pages"] += 1
            state["token"] = payload.get("nextPageToken")
            state["trials"] = len(trials)
            save_state(state)
            print(
                f"page {state['pages']:>4d}  rows {state['rows']:>9,}  trials {len(trials):>7,}",
                flush=True,
            )
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
        "outcome_rows": state["rows"],
        "trials_with_outcomes": state["trials"],
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
