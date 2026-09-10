#!/usr/bin/env python3
"""Assemble the frozen human reference standard and score the extractor against it.

Rebuilds the merge that was lost with the original scoring scripts, from the
clinician workbooks themselves. No model output enters the reference standard at
any point: where the two clinicians agree, their consensus is the truth; where
they disagree, the human adjudication is the truth.

Three sampling strata, as described in the Methods:
  calibration  569 items, two independent clinicians, 222 adjudicated disagreements
  negverify    450 extractor-negative items, single reviewer
  enrichment   350 items over seven sparse domains, single reviewer

Reads the workbooks from the labelling folder, applies the frozen v0.1 lexicon to
the same outcome text, and writes one row per (item, domain):

    data/reference_standard.csv
        item_id, stratum, domain, predicted, truth

Then prints the 2x2 table and raw sensitivity/specificity per domain.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import openpyxl

ANALYSIS = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS))
from run_firstpass_multidomain_extractor import DOMAINS, predicted_domains  # noqa: E402

LABELS = Path("/Users/mac/Documents/Research/Laber.py")

CALIBRATION_1 = "PROJECTB_NLP_VALIDATION_CLINICIAN_1_DRAFT.xlsx"
CALIBRATION_2 = "PROJECTB_NLP_VALIDATION_CLINICIAN_2_DRAFT.xlsx"
ADJUDICATION_FILES = [
    ("PROJECTB_NLP_VALIDATION_HUMAN_ADJUDICATION_QUEUE_v1.xlsx", "Adjudicate", 5),
    ("PROJECTB_NLP_VALIDATION_REMAINING_HUMAN_ADJUDICATION_v2_HUMANFILLED_2026-08-29.xlsx", "Adjudicate", 5),
]
NEGVERIFY = "PROJECTB_NLP_NEGVERIFY_LABELING_WORKBOOK_v1_FILLED.xlsx"
ENRICHMENT = "PROJECTB_NLP_SPARSE_NEG_ENRICH_LABELING_WORKBOOK_v1_2026-08-29_FILLED.xlsx"


def as_bit(value) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in {"0", "0.0"}:
        return 0
    if text in {"1", "1.0"}:
        return 1
    return None


def read_label_sheet(path: Path) -> dict[str, dict]:
    """A workbook whose 'Label' sheet is item_id, text, then the 14 domain columns."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb["Label"]
    rows = ws.iter_rows(values_only=True)
    header = [str(h).strip() if h is not None else "" for h in next(rows)]
    idx = {name: i for i, name in enumerate(header)}
    out: dict[str, dict] = {}
    for row in rows:
        if not row or not row[0]:
            continue
        item = str(row[0]).strip()
        out[item] = {
            "text": str(row[idx["registered_outcomes_text"]] or ""),
            "labels": {d: as_bit(row[idx[d]]) for d in DOMAINS if d in idx},
        }
    wb.close()
    return out


def read_adjudications() -> dict[tuple[str, str], int]:
    resolved: dict[tuple[str, str], int] = {}
    for filename, sheet, col in ADJUDICATION_FILES:
        path = LABELS / filename
        if not path.is_file():
            print(f"  ! missing adjudication file: {filename}", file=sys.stderr)
            continue
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb[sheet]
        rows = ws.iter_rows(values_only=True)
        next(rows)
        n = 0
        for row in rows:
            if not row or not row[0]:
                continue
            bit = as_bit(row[col])
            if bit is None:
                continue
            resolved[(str(row[0]).strip(), str(row[1]).strip())] = bit
            n += 1
        wb.close()
        print(f"  adjudications from {filename.split('_')[-1]:>28s}: {n}")
    return resolved


def read_enrichment() -> list[tuple[str, str, str, int]]:
    """Per-domain sheets: item_id, rule, text, LABEL."""
    path = LABELS / ENRICHMENT
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    out = []
    for ws in wb.worksheets:
        if ws.title not in DOMAINS:
            continue
        rows = ws.iter_rows(values_only=True)
        next(rows)
        for row in rows:
            if not row or not row[0]:
                continue
            bit = as_bit(row[3])
            if bit is None:
                continue
            out.append((str(row[0]).strip(), ws.title, str(row[2] or ""), bit))
    wb.close()
    return out


def load_full_text(item_map: str, outcomes: str) -> dict[str, str]:
    """item_id -> the trial's untruncated outcome text, where the mapping is known."""
    import csv as _csv
    import gzip as _gzip

    if not Path(item_map).is_file():
        print("  ! no item-trial map; falling back to the truncated workbook text")
        return {}
    mapping = {}
    with open(item_map) as fh:
        for row in _csv.DictReader(fh):
            mapping[row["item_id"]] = row["trial_id"]
    wanted = set(mapping.values())
    blocks: dict[str, list[str]] = collections.defaultdict(list)
    with _gzip.open(outcomes, "rt") as fh:
        for row in _csv.DictReader(fh):
            tid = row["source_trial_id"]
            if tid in wanted:
                blocks[tid].append(f"{row['measure']}. {row['description']}".strip())
    text = {t: " || ".join(v) for t, v in blocks.items()}
    out = {item: text[t] for item, t in mapping.items() if t in text}
    print(f"  full outcome text recovered for {len(out):,} of {len(mapping):,} mapped items")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/reference_standard.csv")
    ap.add_argument("--item-map", default="data/item_trial_map.csv")
    ap.add_argument("--outcomes", default="data/ctgov_outcomes.csv.gz")
    ap.add_argument("--summary-json")
    ap.add_argument("--text-mode", choices=["auto", "truncated", "matched-only"],
                    default="auto",
                    help="auto: full text where the trial is known, truncated otherwise. "
                         "truncated: workbook text for every record, so one rule applies "
                         "throughout. matched-only: keep just the records with full text.")
    args = ap.parse_args()

    print(f"Assembling the human reference standard (text-mode: {args.text_mode})")
    full_text = {} if args.text_mode == "truncated" else load_full_text(args.item_map, args.outcomes)

    def predict(item: str, fallback: str) -> tuple[set, str]:
        if item in full_text:
            return predicted_domains(full_text[item].lower(), rule_version="v0_1"), "full"
        return predicted_domains(fallback.lower(), rule_version="v0_1"), "truncated"

    def keep(source: str) -> bool:
        return not (args.text_mode == "matched-only" and source != "full")

    c1 = read_label_sheet(LABELS / CALIBRATION_1)
    c2 = read_label_sheet(LABELS / CALIBRATION_2)
    print(f"  calibration items: clinician 1 {len(c1)}, clinician 2 {len(c2)}")
    adjudicated = read_adjudications()

    records: list[dict] = []
    agree = disagree = unresolved = 0

    for item, rec in c1.items():
        other = c2.get(item)
        if other is None:
            continue
        predicted, source = predict(item, rec["text"])
        for domain in DOMAINS:
            a, b = rec["labels"].get(domain), other["labels"].get(domain)
            if a is None or b is None:
                continue
            if a == b:
                truth = a
                agree += 1
            else:
                disagree += 1
                truth = adjudicated.get((item, domain))
                if truth is None:
                    unresolved += 1
                    continue
            records.append(
                {"item_id": item, "stratum": "calibration", "domain": domain,
                 "predicted": int(domain in predicted), "truth": truth,
                 "text_source": source}
            ) if keep(source) else None

    print(f"  paired labels: {agree + disagree:,}  agree {agree:,}  disagree {disagree:,}"
          f"  ({agree / (agree + disagree) * 100:.2f}% raw agreement)")
    if unresolved:
        print(f"  ! {unresolved} disagreements had no human adjudication and were dropped")

    neg = read_label_sheet(LABELS / NEGVERIFY)
    print(f"  negative-verification items: {len(neg)}")
    for item, rec in neg.items():
        predicted, source = predict(item, rec["text"])
        for domain in DOMAINS:
            truth = rec["labels"].get(domain)
            if truth is None:
                continue
            records.append(
                {"item_id": item, "stratum": "negverify", "domain": domain,
                 "predicted": int(domain in predicted), "truth": truth,
                 "text_source": source}
            ) if keep(source) else None

    enrich = read_enrichment()
    print(f"  sparse-enrichment labels: {len(enrich)}")
    for item, domain, text, truth in enrich:
        predicted, source = predict(item, text)
        records.append(
            {"item_id": item, "stratum": "enrichment", "domain": domain,
             "predicted": int(domain in predicted), "truth": truth,
             "text_source": source}
        ) if keep(source) else None

    # Derived any-domain observation: the paper's headline estimand is "at least one of
    # the 14 domains", which is a different classifier from any single domain and needs its
    # own operating characteristics. Enrichment items are per-domain by construction and
    # cannot contribute to it.
    per_item: dict[str, dict] = collections.defaultdict(
        lambda: {"stratum": None, "pred": 0, "truth": 0, "n": 0, "source": "full"})
    for r in records:
        if r["stratum"] == "enrichment":
            continue
        a = per_item[r["item_id"]]
        a["stratum"] = r["stratum"]
        a["pred"] |= r["predicted"]
        a["truth"] |= r["truth"]
        a["n"] += 1
        if r["text_source"] != "full":
            a["source"] = r["text_source"]
    for item, a in per_item.items():
        if a["n"] < len(DOMAINS):
            continue  # incomplete label set cannot define "any"
        records.append({"item_id": item, "stratum": a["stratum"], "domain": "any_domain",
                        "predicted": a["pred"], "truth": a["truth"],
                        "text_source": a["source"]})
    print(f"  derived any-domain observations: "
          f"{sum(1 for r in records if r['domain'] == 'any_domain'):,}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["item_id", "stratum", "domain", "predicted", "truth", "text_source"])
        writer.writeheader()
        writer.writerows(records)
    n_full = sum(1 for r in records if r["text_source"] == "full")
    print(f"\nwrote {len(records):,} (item, domain) reference observations -> {args.out}")
    print(f"  predictions from full outcome text: {n_full:,} "
          f"({n_full / len(records) * 100:.1f}%); the rest from truncated workbook text")

    cells = collections.defaultdict(collections.Counter)
    for r in records:
        cells[r["domain"]][(r["predicted"], r["truth"])] += 1

    print(f"\n{'domain':22s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'TN':>6s}   {'Se':>6s} {'Sp':>6s} {'PPV':>6s}")
    table = []
    for domain in DOMAINS:
        c = cells[domain]
        tp, fp, fn, tn = c[(1, 1)], c[(1, 0)], c[(0, 1)], c[(0, 0)]
        se = tp / (tp + fn) if tp + fn else float("nan")
        sp = tn / (tn + fp) if tn + fp else float("nan")
        ppv = tp / (tp + fp) if tp + fp else float("nan")
        table.append({"domain": domain, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                      "sensitivity": None if tp + fn == 0 else round(se, 4),
                      "specificity": None if tn + fp == 0 else round(sp, 4),
                      "ppv": None if tp + fp == 0 else round(ppv, 4)})
        print(f"{domain:22s} {tp:>5d} {fp:>5d} {fn:>5d} {tn:>6d}   {se:>6.3f} {sp:>6.3f} {ppv:>6.3f}")

    if args.summary_json:
        with open(args.summary_json, "w") as fh:
            fh.write(json.dumps({
                "n_observations": len(records),
                "calibration_paired_labels": agree + disagree,
                "calibration_agreements": agree,
                "calibration_disagreements": disagree,
                "raw_agreement_pct": round(agree / (agree + disagree) * 100, 2),
                "unresolved_dropped": unresolved,
                "domains": table,
            }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
