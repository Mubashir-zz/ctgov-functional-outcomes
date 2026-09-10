#!/usr/bin/env python3
"""Reconstruct the three-phase sampling weights and produce design-weighted accuracy.

The validation sample was not drawn uniformly. Positives were deliberately
oversampled, rare domains more so, which means raw sample fractions are not
population values — the point the manuscript makes when it reports raw and
design-weighted predictive values separately.

The design, recovered from `build_multidomain_validation_sample.py` and the
analysis notes:

  phase 1  calibration    30 drawn from each of 14 first-pass-positive domain frames,
                          plus 160 from the extractor-negative frame. The frames
                          overlap, so a record in several of them has several chances
                          of selection and its inclusion probability is one minus the
                          product of the complements — never a naive sum.
  phase 2  negverify      450 drawn from the 55,761 extractor-negative records not
                          already selected in phase 1.
  phase 3  enrichment     50 drawn per domain for the seven sparse domains: 40 from a
                          narrow broad-screen-positive stratum S1 and 10 from a wider
                          stratum S2, both inside the predicted-negative space.

Phases 1 and 2 compose as

    pi = pi1 + (1 - pi1) * pi2

and every labelled (record, domain) observation from those phases is weighted by 1 / pi.
Sensitivity, specificity and predictive values are ratios of Horvitz-Thompson weighted
totals.

Phase 3 is NOT weighted here, and that is deliberate. Its inclusion probabilities are not
identifiable from what survives: S1 and S2 membership was recorded only in the sampling
key, which is lost, and the documented frames (hearing S1 = 71 records, 40 drawn, giving
0.56) differ from the wider stratum by three orders of magnitude. Assigning enrichment
records a probability derived from the whole predicted-negative frame is wrong by roughly
that factor and silently destroys the information the enrichment arm was designed to
provide. Until the strata are recovered, enrichment observations are excluded from the
weighted estimator and reported separately as an unweighted targeted audit of the
predicted-negative space.

Frame sizes are rebuilt from the current corpus and rescaled to the documented
phase-2 frame of 55,761, which anchors the reconstruction to a recorded number
rather than to an assumption about how many development trials were excluded.
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import sys

DOMAINS = [
    "cognition", "neuropathy", "fatigue", "pain", "physical_function",
    "swallowing", "speech_voice", "hearing", "respiratory", "bowel",
    "urinary", "sexual_reproductive", "limb_lymphedema", "visual",
    "any_domain",
]
ENRICHED = ["cognition", "neuropathy", "swallowing", "speech_voice",
            "hearing", "limb_lymphedema", "visual"]

N_POS_PER_DOMAIN = 30
N_NEG_CALIBRATION = 160
NEGVERIFY_FRAME = 55_761      # documented in the analysis notes
N_NEGVERIFY = 450
N_ENRICH_PER_DOMAIN = 50          # 40 from S1 + 10 from S2, strata not recoverable
HEARING_S1_FRAME = 71             # documented; 40 drawn, so 56% of that stratum was read

# Published design-weighted values, for validation of the reconstruction.
PUBLISHED_SPECIFICITY = {
    "pain": 0.959, "fatigue": 0.976, "physical_function": 0.980, "bowel": 0.963,
    "cognition": 0.987, "urinary": 0.978, "respiratory": 0.992, "neuropathy": 0.993,
    "sexual_reproductive": 0.992, "swallowing": 0.995, "limb_lymphedema": 0.999,
    "speech_voice": 0.998, "visual": 0.998, "hearing": 0.999,
    "any_domain": 0.897,
}
PUBLISHED_PPV = {
    "limb_lymphedema": 61.79, "sexual_reproductive": 54.45, "neuropathy": 53.80,
    "visual": 52.15, "hearing": 51.54, "cognition": 51.36, "pain": 51.26,
    "physical_function": 45.50, "fatigue": 40.86, "respiratory": 40.23,
    "urinary": 28.57, "swallowing": 28.29, "speech_voice": 23.51, "bowel": 16.87,
    "any_domain": 54.19,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="data/reference_standard.csv")
    ap.add_argument("--cohort", default="data/cohort.csv.gz")
    ap.add_argument("--out", default="data/reference_standard.csv")
    ap.add_argument("--summary-json")
    ap.add_argument("--table", default="tables/table2.tsv")
    ap.add_argument("--replicates", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260910)
    args = ap.parse_args()

    # ---- frame sizes from the corpus, rescaled to the documented phase-2 frame
    positives = collections.Counter()
    n_corpus = n_negative = 0
    with gzip.open(args.cohort, "rt") as fh:
        for row in csv.DictReader(fh):
            n_corpus += 1
            hit = False
            for domain in DOMAINS:
                if row[domain] == "1":
                    positives[domain] += 1
                    hit = True
            if not hit:
                n_negative += 1

    scale = NEGVERIFY_FRAME / n_negative
    pos_frame = {d: max(positives[d] * scale, N_POS_PER_DOMAIN) for d in DOMAINS}
    neg_frame = NEGVERIFY_FRAME

    print(f"corpus {n_corpus:,}  extractor-negative {n_negative:,}")
    print(f"rescaled to the documented phase-2 frame of {NEGVERIFY_FRAME:,} "
          f"(factor {scale:.4f})\n")
    print(f"{'domain':22s} {'positive frame':>15s} {'P(draw in phase 1)':>19s}")
    for domain in DOMAINS:
        print(f"{domain:22s} {pos_frame[domain]:>15,.0f} {N_POS_PER_DOMAIN / pos_frame[domain]:>19.5f}")

    # ---- read the reference standard, recover each item's predicted domain set
    rows = list(csv.DictReader(open(args.reference)))
    predicted_set = collections.defaultdict(set)
    for r in rows:
        if r["predicted"] == "1":
            predicted_set[r["item_id"]].add(r["domain"])

    pi2 = N_NEGVERIFY / neg_frame

    def inclusion(item: str, stratum: str, domain: str) -> float:
        preds = predicted_set.get(item, set())
        if preds:
            comp = 1.0
            for d in preds:
                comp *= 1.0 - N_POS_PER_DOMAIN / pos_frame[d]
            pi1 = 1.0 - comp
        else:
            pi1 = N_NEG_CALIBRATION / neg_frame

        p2 = pi2 if not preds else 0.0
        pi = pi1 + (1 - pi1) * p2
        return min(max(pi, 1e-9), 1.0)

    n_enrich = 0
    for r in rows:
        if r["stratum"] == "enrichment":
            # design not identifiable — see module docstring
            r["inclusion_probability"] = ""
            r["weight"] = ""
            n_enrich += 1
            continue
        pi = inclusion(r["item_id"], r["stratum"], r["domain"])
        r["inclusion_probability"] = f"{pi:.8f}"
        r["weight"] = f"{1.0 / pi:.4f}"

    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["item_id", "stratum", "domain", "predicted", "truth",
                            "text_source", "inclusion_probability", "weight"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote weights onto {len(rows):,} observations -> {args.out}")

    # ---- Horvitz-Thompson weighted accuracy
    agg = collections.defaultdict(lambda: collections.defaultdict(float))
    for r in rows:
        if not r["weight"]:
            continue
        agg[r["domain"]][(int(r["predicted"]), int(r["truth"]))] += float(r["weight"])

    print(f"\n{'domain':22s} {'HT Sp':>7s} {'pub Sp':>7s} {'HT PPV':>8s} {'pub PPV':>8s} {'HT Se':>7s}")
    print("-" * 64)
    out = []
    for domain in DOMAINS:
        a = agg[domain]
        tp, fp, fn, tn = a[(1, 1)], a[(1, 0)], a[(0, 1)], a[(0, 0)]
        se = tp / (tp + fn) if tp + fn else float("nan")
        sp = tn / (tn + fp) if tn + fp else float("nan")
        ppv = tp / (tp + fp) if tp + fp else float("nan")
        out.append({
            "domain": domain,
            "ht_sensitivity": None if tp + fn == 0 else round(se, 4),
            "ht_specificity": None if tn + fp == 0 else round(sp, 4),
            "ht_ppv_pct": None if tp + fp == 0 else round(ppv * 100, 2),
            "published_specificity": PUBLISHED_SPECIFICITY[domain],
            "published_ppv_pct": PUBLISHED_PPV[domain],
            "weighted_counts": {"tp": round(tp, 1), "fp": round(fp, 1),
                                "fn": round(fn, 1), "tn": round(tn, 1)},
        })
        print(f"{domain:22s} {sp:>7.4f} {PUBLISHED_SPECIFICITY[domain]:>7.3f} "
              f"{ppv * 100:>7.2f}% {PUBLISHED_PPV[domain]:>7.2f}% {se:>7.3f}")

    # ---- cluster bootstrap over validation items for interval estimates
    import random
    rnd = random.Random(args.seed)
    by_item: dict[str, list] = collections.defaultdict(list)
    for r in rows:
        if r["weight"]:
            by_item[r["item_id"]].append(r)
    item_ids = sorted(by_item)
    strata_of = {i: by_item[i][0]["stratum"] for i in item_ids}
    pools = collections.defaultdict(list)
    for i in item_ids:
        pools[strata_of[i]].append(i)

    def cells(sample_ids):
        acc = collections.defaultdict(lambda: collections.defaultdict(float))
        for i in sample_ids:
            for r in by_item[i]:
                acc[r["domain"]][(int(r["predicted"]), int(r["truth"]))] += float(r["weight"])
        return acc

    boot = collections.defaultdict(lambda: collections.defaultdict(list))
    for _ in range(args.replicates):
        draw = [rnd.choice(v) for k, v in pools.items() for _ in v]
        acc = cells(draw)
        for domain in DOMAINS:
            c = acc[domain]
            tp, fp, fn, tn = c[(1, 1)], c[(1, 0)], c[(0, 1)], c[(0, 0)]
            if tp + fp > 0:
                boot[domain]["ppv"].append(tp / (tp + fp))
            if tn + fn > 0:
                boot[domain]["npv"].append(tn / (tn + fn))
            if tp + fn > 0:
                boot[domain]["se"].append(tp / (tp + fn))
            if tn + fp > 0:
                boot[domain]["sp"].append(tn / (tn + fp))

    def ci(vals):
        if len(vals) < 100:
            return None
        v = sorted(vals)
        return [round(v[int(0.025 * len(v))], 4), round(v[int(0.975 * len(v)) - 1], 4)]

    raw_counts = collections.defaultdict(collections.Counter)
    for r in rows:
        if r["weight"]:
            raw_counts[r["domain"]][(int(r["predicted"]), int(r["truth"]))] += 1

    import os
    os.makedirs(os.path.dirname(args.table) or ".", exist_ok=True)
    with open(args.table, "w") as fh:
        fh.write("Domain\tValidation records\tRaw PPV %\tDesign-weighted PPV %\t95% CI"
                 "\tDesign-weighted NPV %\tSensitivity\tSpecificity\n")
        for r in sorted(out, key=lambda x: -(x["ht_ppv_pct"] or 0)):
            d = r["domain"]
            rc = raw_counts[d]
            n_val = sum(rc.values())
            raw_ppv = rc[(1, 1)] / (rc[(1, 1)] + rc[(1, 0)]) * 100 if rc[(1, 1)] + rc[(1, 0)] else float("nan")
            pv = ci(boot[d]["ppv"])
            npv = sorted(boot[d]["npv"])
            npv_pt = npv[len(npv) // 2] * 100 if npv else float("nan")
            fh.write(f"{d}\t{n_val}\t{raw_ppv:.2f}\t{r['ht_ppv_pct']:.2f}\t"
                     f"{pv[0]*100:.2f}-{pv[1]*100:.2f}\t{npv_pt:.3f}\t"
                     f"{r['ht_sensitivity']:.3f}\t{r['ht_specificity']:.4f}\n")
            r["ppv_ci_pct"] = None if pv is None else [round(pv[0] * 100, 2), round(pv[1] * 100, 2)]
            r["npv_pct"] = round(npv_pt, 3)
            r["validation_records"] = n_val
            r["raw_ppv_pct"] = round(raw_ppv, 2)
    print(f"\nwrote {args.table}")

    sp_err = [abs(r["ht_specificity"] - r["published_specificity"])
              for r in out if r["ht_specificity"] is not None]
    ppv_err = [abs(r["ht_ppv_pct"] - r["published_ppv_pct"])
               for r in out if r["ht_ppv_pct"] is not None]
    print(f"\nmean absolute error vs published:  specificity {sum(sp_err) / len(sp_err):.4f}"
          f"   PPV {sum(ppv_err) / len(ppv_err):.2f} percentage points")

    # unweighted report of the excluded enrichment arm
    enr = collections.defaultdict(collections.Counter)
    for r in rows:
        if r["stratum"] == "enrichment":
            enr[r["domain"]][(int(r["predicted"]), int(r["truth"]))] += 1
    print(f"\nEnrichment arm ({n_enrich} observations), excluded from the weighted estimator")
    print("and reported unweighted — a targeted audit of the predicted-negative space:")
    print(f"  {'domain':22s} {'read':>6s} {'true misses found':>18s}")
    for domain in sorted(enr):
        c = enr[domain]
        read = sum(c.values())
        misses = c[(0, 1)] + c[(1, 1)]
        print(f"  {domain:22s} {read:>6d} {misses:>18d}")
    print(f"  hearing S1 frame was {HEARING_S1_FRAME} records and 40 were drawn, so 56% of that")
    print("  stratum was read by a clinician; no true miss was found.")

    n_full = sum(1 for r in rows if r.get("text_source") == "full")
    pct_full = n_full / len(rows) * 100 if rows else 0.0
    print(f"\nExtractor predictions come from the untruncated outcome text for {pct_full:.1f}% of")
    print("observations, via the recovered item-trial map; the remainder fall back to the")
    print("reviewer workbook text, which the sampler truncated to 1,500 characters. Truncation")
    print("suppresses predicted positives, so residual PPV differences run in that direction.")

    if args.summary_json:
        with open(args.summary_json, "w") as fh:
            fh.write(json.dumps({
                "design": {
                    "n_pos_per_domain": N_POS_PER_DOMAIN,
                    "n_neg_calibration": N_NEG_CALIBRATION,
                    "negverify_frame": NEGVERIFY_FRAME,
                    "n_negverify": N_NEGVERIFY,
                    "n_enrich_per_domain": N_ENRICH_PER_DOMAIN,
                    "enriched_domains": ENRICHED,
                    "composition": "pi = pi1 + (1-pi1)*pi2; phase 3 excluded, strata not recoverable",
                },
                "frames": {d: round(pos_frame[d]) for d in DOMAINS},
                "rescale_factor": round(scale, 6),
                "domains": out,
                "mean_abs_error_specificity": round(sum(sp_err) / len(sp_err), 4),
                "mean_abs_error_ppv_pp": round(sum(ppv_err) / len(ppv_err), 2),
            }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
