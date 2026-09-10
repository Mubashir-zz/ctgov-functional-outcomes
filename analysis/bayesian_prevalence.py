#!/usr/bin/env python3
"""Misclassification-corrected prevalence with design-based, paired accuracy draws.

Replaces the Rogan-Gladen plug-in, which treats sensitivity and specificity as known
constants and can leave the parameter space, with a model that treats them as estimated,
keeps prevalence inside [0, 1] by construction, and preserves the dependence between the
two accuracy parameters.

Accuracy uncertainty comes from a cluster bootstrap of the validation design rather than
from Beta pseudo-counts. Each replicate resamples validation *items* with replacement
within sampling stratum, recomputes Horvitz-Thompson weighted sensitivity and specificity
from that resample, and yields one paired (Se, Sp) draw. This does three things a
Beta-per-cell model cannot:

  - Se and Sp come from the same resample, so their estimated dependence is carried
    through rather than assumed away.
  - Resampling whole items preserves the correlation between a record's 14 domain labels.
  - No effective-sample-size fudge is needed. An earlier version rescaled weighted counts
    by a single Kish factor across all four cells, which badly understated the information
    in the sensitivity denominator for sparse domains and overstated it for common ones.

Given a paired draw, the conditional posterior of prevalence is exact on a fine grid:

    pi ~ Beta(1, 1)        y ~ Binomial(N, pi*Se + (1 - pi)*(1 - Sp))

One pi is sampled per draw, so the collected pi values are draws from the joint posterior.

Enrichment-arm observations carry no weight (their sampling strata are not recoverable —
see sampling_weights.py) and are excluded here too.
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import sys

import numpy as np

DOMAINS = [
    "pain", "fatigue", "physical_function", "bowel", "cognition", "urinary",
    "respiratory", "neuropathy", "sexual_reproductive", "swallowing",
    "limb_lymphedema", "speech_voice", "visual", "hearing", "any_domain",
]

GRID_MAX = 0.30
GRID_N = 6001
DRAWS = 2000
SEED = 20260910


def rogan_gladen(p_obs: float, se: float, sp: float) -> float:
    denom = se + sp - 1.0
    return float("nan") if denom <= 0 else (p_obs + sp - 1.0) / denom


def load_reference(path: str):
    """items -> (stratum, weight, {domain: (pred, truth)}), weighted rows only."""
    items: dict[str, dict] = {}
    for row in csv.DictReader(open(path)):
        if not row.get("weight"):
            continue
        rec = items.setdefault(row["item_id"],
                               {"stratum": row["stratum"],
                                "w": float(row["weight"]), "d": {}})
        rec["d"][row["domain"]] = (int(row["predicted"]), int(row["truth"]))
    return items


def build_arrays(items: dict):
    ids = sorted(items)
    strata = sorted({items[i]["stratum"] for i in ids})
    idx_by_stratum = {s: np.array([k for k, i in enumerate(ids)
                                   if items[i]["stratum"] == s]) for s in strata}
    w = np.array([items[i]["w"] for i in ids])
    pred, truth = {}, {}
    for d in DOMAINS:
        pred[d] = np.array([items[i]["d"].get(d, (0, 0))[0] for i in ids], dtype=float)
        truth[d] = np.array([items[i]["d"].get(d, (0, 0))[1] for i in ids], dtype=float)
    return ids, idx_by_stratum, w, pred, truth


def accuracy(w, pred, truth, sel):
    """HT-weighted (Se, Sp) for one domain over the selected rows."""
    ww, p, t = w[sel], pred[sel], truth[sel]
    pos, neg = ww * t, ww * (1 - t)
    dp, dn = pos.sum(), neg.sum()
    se = float((pos * p).sum() / dp) if dp > 0 else float("nan")
    sp = float((neg * (1 - p)).sum() / dn) if dn > 0 else float("nan")
    return se, sp


def prevalence_draws(y: int, n: int, se: np.ndarray, sp: np.ndarray,
                     rng: np.random.Generator) -> np.ndarray:
    grid = np.linspace(0.0, GRID_MAX, GRID_N)
    out = np.empty(se.size)
    step = 250
    for a in range(0, se.size, step):
        s_e, s_p = se[a:a + step], sp[a:a + step]
        p = grid[:, None] * s_e[None, :] + (1.0 - grid[:, None]) * (1.0 - s_p[None, :])
        np.clip(p, 1e-12, 1 - 1e-12, out=p)
        ll = y * np.log(p) + (n - y) * np.log1p(-p)
        ll -= ll.max(axis=0, keepdims=True)
        wgt = np.exp(ll)
        wgt /= wgt.sum(axis=0, keepdims=True)
        cdf = np.cumsum(wgt, axis=0)
        u = rng.random(wgt.shape[1])
        k = (cdf < u[None, :]).sum(axis=0)
        np.clip(k, 0, GRID_N - 1, out=k)
        out[a:a + step] = grid[k]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="data/reference_standard.csv")
    ap.add_argument("--cohort", default="data/cohort.csv.gz")
    ap.add_argument("--replicates", type=int, default=DRAWS)
    ap.add_argument("--summary-json")
    ap.add_argument("--table", default="tables/table1.tsv")
    args = ap.parse_args()

    items = load_reference(args.reference)
    ids, idx_by_stratum, w, pred, truth = build_arrays(items)
    print(f"validation items carrying design weights: {len(ids):,} "
          f"across strata {sorted(idx_by_stratum)}")

    observed = collections.Counter()
    n_corpus = 0
    with gzip.open(args.cohort, "rt") as fh:
        for row in csv.DictReader(fh):
            n_corpus += 1
            for d in DOMAINS:
                observed[d] += int(row[d])
    print(f"corpus: {n_corpus:,} trials\n")

    rng = np.random.default_rng(SEED)
    full = np.arange(len(ids))
    point = {d: accuracy(w, pred[d], truth[d], full) for d in DOMAINS}

    boot = {d: {"se": [], "sp": []} for d in DOMAINS}
    print(f"cluster bootstrap of the validation design, {args.replicates:,} replicates...")
    for r in range(args.replicates):
        sel = np.concatenate([rng.choice(v, size=v.size, replace=True)
                              for v in idx_by_stratum.values()])
        for d in DOMAINS:
            se, sp = accuracy(w, pred[d], truth[d], sel)
            if np.isfinite(se) and np.isfinite(sp):
                boot[d]["se"].append(se)
                boot[d]["sp"].append(sp)
        if (r + 1) % 500 == 0:
            print(f"  {r + 1:,}", flush=True)

    hdr = (f"\n{'domain':22s} {'observed':>9s} {'Rogan-Gladen':>13s} "
           f"{'corrected':>10s} {'95% credible':>18s} {'draws':>7s}")
    print(hdr)
    print("-" * (len(hdr) - 1))
    results = []
    for d in DOMAINS:
        y, p_obs = observed[d], observed[d] / n_corpus
        se_hat, sp_hat = point[d]
        rg = rogan_gladen(p_obs, se_hat, sp_hat)
        se = np.asarray(boot[d]["se"])
        sp = np.asarray(boot[d]["sp"])
        if se.size < 100:
            print(f"{d:22s} {p_obs*100:>8.3f}%  not estimable")
            results.append({"domain": d, "observed_pct": round(p_obs * 100, 3),
                            "corrected_pct": None, "ci95_pct": None,
                            "note": "too few usable bootstrap replicates"})
            continue
        draws = prevalence_draws(y, n_corpus, se, sp, rng)
        med, lo, hi = np.percentile(draws, [50, 2.5, 97.5])
        corr = float(np.corrcoef(se, sp)[0, 1]) if se.std() > 0 and sp.std() > 0 else float("nan")
        print(f"{d:22s} {p_obs*100:>8.3f}% {max(rg,0)*100:>12.3f}% "
              f"{med*100:>9.3f}% {'[' + format(lo*100, '.3f') + ', ' + format(hi*100, '.3f') + ']':>18s}"
              f" {se.size:>7,}")
        results.append({
            "domain": d,
            "observed_pct": round(p_obs * 100, 3),
            "point_sensitivity": None if not np.isfinite(se_hat) else round(se_hat, 4),
            "point_specificity": None if not np.isfinite(sp_hat) else round(sp_hat, 4),
            "rogan_gladen_pct": None if not np.isfinite(rg) else round(rg * 100, 3),
            "corrected_pct": round(med * 100, 3),
            "ci95_pct": [round(lo * 100, 3), round(hi * 100, 3)],
            "bootstrap_draws": int(se.size),
            "se_sp_correlation": None if not np.isfinite(corr) else round(corr, 3),
        })

    import os
    os.makedirs(os.path.dirname(args.table) or ".", exist_ok=True)
    with open(args.table, "w") as fh:
        fh.write("Domain\tObserved %\tSensitivity\tSpecificity\tCorrected %\t95% credible\n")
        for r in results:
            if r.get("corrected_pct") is None:
                continue
            fh.write(f"{r['domain']}\t{r['observed_pct']:.2f}\t"
                     f"{r['point_sensitivity']:.3f}\t{r['point_specificity']:.3f}\t"
                     f"{r['corrected_pct']:.2f}\t"
                     f"{r['ci95_pct'][0]:.2f}-{r['ci95_pct'][1]:.2f}\n")
    print(f"\nwrote {args.table}")

    print("\nAccuracy draws are paired: sensitivity and specificity come from the same")
    print("resample of the validation design, so their dependence is propagated. Items are")
    print("resampled whole within stratum, preserving correlation across a record's domains.")
    print("Enrichment observations are excluded — their sampling strata are not recoverable.")

    if args.summary_json:
        with open(args.summary_json, "w") as fh:
            fh.write(json.dumps({
                "accuracy_uncertainty": "cluster bootstrap of the validation design, "
                                        "items resampled within stratum, paired (Se, Sp)",
                "prevalence_prior": "Beta(1,1), grid-exact conditional posterior",
                "replicates": args.replicates,
                "grid_points": GRID_N,
                "grid_max": GRID_MAX,
                "seed": SEED,
                "n_corpus": n_corpus,
                "n_validation_items": len(ids),
                "enrichment": "excluded, strata not recoverable",
                "domains": results,
            }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
