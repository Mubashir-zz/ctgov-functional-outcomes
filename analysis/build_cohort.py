#!/usr/bin/env python3
"""Apply the study's cohort eligibility and check the rebuild against the manuscript.

Two nested populations, both as defined in the Methods.

Full corpus — supports the prevalence and misclassification-correction analysis:
    condition search "cancer", INTERVENTIONAL, registered on or before 2026-07-31,
    at least one registered outcome measure. Therapeutic and non-therapeutic
    designs are both retained, so the estimand is the proportion of outcome-bearing
    records the cancer search returns.

Restricted therapeutic subset — supports the treatment-domain comparisons:
    registered 1 January 2010 to 31 July 2026, primary purpose TREATMENT, and
    explicit malignancy evidence in the registered conditions.

Writes data/cohort.csv.gz (one row per corpus trial, covariates plus the 14 domain
flags and a therapeutic-subset flag) and prints the comparison against the
published figures.
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import re
import sys

DOMAINS = [
    "cognition", "neuropathy", "fatigue", "pain", "physical_function",
    "swallowing", "speech_voice", "hearing", "respiratory", "bowel",
    "urinary", "sexual_reproductive", "limb_lymphedema", "visual",
]

PUBLISHED_PREVALENCE = {
    "pain": 8.70, "fatigue": 4.46, "physical_function": 4.29, "bowel": 3.59,
    "cognition": 3.19, "urinary": 2.50, "respiratory": 1.88, "neuropathy": 1.81,
    "sexual_reproductive": 1.60, "swallowing": 1.21, "limb_lymphedema": 0.62,
    "speech_voice": 0.40, "visual": 0.38, "hearing": 0.21,
}
PUBLISHED = {
    "corpus": 96936,
    "any_domain_pct": 19.74,
    "therapeutic_subset": 55178,
    "date_and_purpose_eligible": 58221,
    "cisplatin_trials": 2653,
    "cisplatin_hearing": 28,
    "cisplatin_hearing_pct": 1.06,
}

THERAPEUTIC_FROM = "2010-01-01"

# Explicit malignancy evidence in the registered condition strings.
MALIGNANCY = re.compile(
    r"cancer|carcinom|neoplas|tumou?r|malignan|sarcom|lymphom|leuk|myelom|melanom"
    r"|gliom|blastom|mesotheliom|myelodysplas|myeloprolifer|nsclc|sclc|metasta"
    r"|polycythemia|amyloidos|myelofibros|oncolog"
)


def line(label: str, got: float, want: float, pct: bool = False) -> None:
    unit = "%" if pct else ""
    fmt = "{:>10.2f}" if pct else "{:>10,.0f}"
    delta = (got - want) / want * 100 if want else 0.0
    print(f"  {label:32s} {fmt.format(got)}{unit}   published {fmt.format(want)}{unit}   {delta:+6.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="data/trial_metadata.csv.gz")
    ap.add_argument("--labels", default="data/trial_domain_labels.csv.gz")
    ap.add_argument("--out", default="data/cohort.csv.gz")
    ap.add_argument("--summary-json")
    args = ap.parse_args()

    meta = {}
    with gzip.open(args.metadata, "rt") as fh:
        for row in csv.DictReader(fh):
            if row["study_type"] == "INTERVENTIONAL":
                meta[row["nct_id"]] = row

    labels: dict[str, set[str]] = collections.defaultdict(set)
    corpus: set[str] = set()
    with gzip.open(args.labels, "rt") as fh:
        for row in csv.DictReader(fh):
            tid = row["trial_id"]
            if tid not in meta:
                continue
            corpus.add(tid)
            if row["measured_firstpass"] == "1":
                labels[tid].add(row["domain"])

    header = [
        "nct_id", "phases", "start_year", "first_submit_date", "enrollment",
        "lead_sponsor_class", "conditions", "intervention_names", "therapeutic",
    ] + DOMAINS + ["any_domain"]

    counts = collections.Counter()
    n_any = 0
    n_date_purpose = 0
    therapeutic: list[dict] = []

    with gzip.open(args.out, "wt", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for tid in sorted(corpus):
            m = meta[tid]
            hits = labels.get(tid, set())
            any_flag = 1 if hits else 0
            n_any += any_flag
            for domain in hits:
                counts[domain] += 1

            date_purpose = (
                m["primary_purpose"] == "TREATMENT"
                and m["first_submit_date"] >= THERAPEUTIC_FROM
            )
            n_date_purpose += date_purpose
            is_ther = bool(date_purpose and MALIGNANCY.search(m["conditions"].lower()))
            if is_ther:
                therapeutic.append({"m": m, "hits": hits})

            writer.writerow(
                [
                    tid, m["phases"], m["start_year"], m["first_submit_date"],
                    m["enrollment"], m["lead_sponsor_class"], m["conditions"],
                    m["intervention_names"], int(is_ther),
                ]
                + [1 if d in hits else 0 for d in DOMAINS]
                + [any_flag]
            )

    n = len(corpus)
    any_pct = n_any / n * 100 if n else 0.0

    print("Corpus")
    line("outcome-bearing records", n, PUBLISHED["corpus"])
    line("any functional domain", any_pct, PUBLISHED["any_domain_pct"], pct=True)

    print("\nPer-domain registration prevalence")
    rows = []
    for domain in sorted(PUBLISHED_PREVALENCE, key=lambda d: -PUBLISHED_PREVALENCE[d]):
        rate = counts[domain] / n * 100 if n else 0.0
        published = PUBLISHED_PREVALENCE[domain]
        rows.append({"domain": domain, "rebuilt_pct": round(rate, 2),
                     "published_pct": published, "diff_pp": round(rate - published, 2)})
        print(f"  {domain:22s} {rate:>7.2f}%   published {published:>5.2f}%   {rate - published:+6.2f} pp")
    mean_abs = sum(abs(r["diff_pp"]) for r in rows) / len(rows)
    print(f"\n  mean absolute difference: {mean_abs:.3f} percentage points")

    print("\nRestricted therapeutic subset")
    line("date and purpose eligible", n_date_purpose, PUBLISHED["date_and_purpose_eligible"])
    line("after malignancy rule", len(therapeutic), PUBLISHED["therapeutic_subset"])

    cis = [t for t in therapeutic if "cisplatin" in t["m"]["intervention_names"]]
    cis_hearing = sum(1 for t in cis if "hearing" in t["hits"])
    print("\nAnchor case: cisplatin and hearing")
    line("cisplatin trials", len(cis), PUBLISHED["cisplatin_trials"])
    line("registering a hearing outcome", cis_hearing, PUBLISHED["cisplatin_hearing"])
    if cis:
        line("crude rate", cis_hearing / len(cis) * 100, PUBLISHED["cisplatin_hearing_pct"], pct=True)

    summary = {
        "eligibility": {
            "corpus": {
                "condition_search": "cancer",
                "study_type": "INTERVENTIONAL",
                "registered_on_or_before": "2026-07-31",
                "requires_outcome_measure": True,
            },
            "therapeutic_subset": {
                "registered_from": THERAPEUTIC_FROM,
                "primary_purpose": "TREATMENT",
                "malignancy_evidence": "regex over registered conditions",
            },
        },
        "rebuilt": {
            "corpus": n,
            "any_domain_pct": round(any_pct, 2),
            "date_and_purpose_eligible": n_date_purpose,
            "therapeutic_subset": len(therapeutic),
            "cisplatin_trials": len(cis),
            "cisplatin_hearing": cis_hearing,
        },
        "published": PUBLISHED,
        "domains": rows,
        "mean_absolute_difference_pp": round(mean_abs, 3),
        "output": args.out,
    }
    if args.summary_json:
        with open(args.summary_json, "w") as fh:
            fh.write(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
