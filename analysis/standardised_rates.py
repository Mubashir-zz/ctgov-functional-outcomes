#!/usr/bin/env python3
"""Directly standardised agent-domain registration rates, with overlap-aware intervals.

The manuscript reports standardised rate ratios descriptively and withholds inferential
intervals, because the comparator is the whole cohort within the same strata and the
agent-exposed records are a subset of it. Numerator and denominator are therefore
correlated, and a delta-method interval that treats them as independent understates the
correlation and overstates precision.

A nonparametric bootstrap over trials solves this directly. Each replicate resamples
trials with replacement and recomputes *both* the agent rate and the comparator rate from
the same resample, so whatever correlation the overlap induces is carried through to the
distribution of their ratio. No independence assumption is made anywhere.

Standardisation follows the Methods: 11 cancer-site groups x 4 phase groups x 3
registration eras, restricted to strata in which the agent has at least one trial, with
the overall cohort's stratum distribution as the standard population.

    standardised agent rate      = sum_s w_s * p_agent(s)
    standardised comparator rate = sum_s w_s * p_cohort(s)
    rate ratio                   = the first over the second

w_s is renormalised over supported strata. Both rates and the ratio are recomputed inside
every replicate, including the stratum weights and the support set.

Runs on the restricted therapeutic subset, which is the population the manuscript uses for
these comparisons.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys

import numpy as np

# Agent-domain pairs examined in the manuscript, with the published values.
PAIRS = [
    ("cisplatin", "hearing", 1.72, 0.18, 9.39, "primary"),
    ("carboplatin", "hearing", 0.61, 0.17, 3.53, "secondary"),
    ("cisplatin", "neuropathy", None, None, 1.93, "secondary"),
    ("taxane", "neuropathy", None, None, 1.91, "secondary"),
    ("oxaliplatin", "neuropathy", None, None, 1.80, "secondary"),
    ("bortezomib", "neuropathy", 3.43, 2.43, None, "secondary"),
    ("imid", "neuropathy", None, None, 1.29, "secondary"),
    ("cisplatin", "limb_lymphedema", None, None, 0.70, "negative control"),
    ("cisplatin", "visual", None, None, 0.25, "negative control"),
    ("taxane", "visual", None, None, 1.00, "negative control"),
    ("bortezomib", "hearing", None, None, None, "negative control"),
    ("vincristine", "limb_lymphedema", None, None, None, "negative control"),
    # Excluded post hoc from the published control set after its crude ratio of 3.03 was
    # inspected. Retained here so the control set can be reported both ways: dropping a
    # control after looking at it is defensible only if the analysis without the drop is
    # also shown.
    ("cisplatin", "speech_voice", None, None, None, "negative control (post hoc excluded)"),
]

AGENT_PATTERNS = {
    "cisplatin": r"cisplatin|cddp|platinol",
    "carboplatin": r"carboplatin|paraplatin",
    "oxaliplatin": r"oxaliplatin|eloxatin",
    "bortezomib": r"bortezomib|velcade",
    "taxane": r"paclitaxel|docetaxel|taxane|taxol|taxotere|nab-paclitaxel|abraxane|cabazitaxel",
    "imid": r"thalidomide|lenalidomide|pomalidomide|revlimid|thalomid",
    "vincristine": r"vincristine|oncovin|vinblastine|vinorelbine",
}

# 11 cancer-site groups, matched in order; first match wins.
SITE_GROUPS = [
    ("haematologic", r"leuk|lymphom|myelom|myelodysplas|myeloprolifer|polycythemia|"
                     r"myelofibros|amyloidos|hodgkin"),
    ("cns", r"gliom|glioblastom|astrocytom|medulloblastom|meningiom|brain|"
            r"central nervous system|neuroblastom"),
    ("head_neck", r"head and neck|nasopharyn|oropharyn|laryn|oral cav|hypopharyn|"
                  r"salivary|tongue|thyroid"),
    ("thoracic", r"lung|nsclc|sclc|mesotheliom|thymom|pleural"),
    ("breast", r"breast"),
    ("gastrointestinal", r"colorect|colon|rectal|gastric|stomach|oesophag|esophag|"
                         r"pancrea|hepatocellul|liver|cholangio|biliary|anal|"
                         r"gastrointestinal|neuroendocrine"),
    ("genitourinary", r"prostat|bladder|urothelial|renal|kidney|testicul|germ cell|penile|ureter"),
    ("gynaecologic", r"ovarian|cervic|endometri|uterine|vulva|vaginal|gestational trophoblastic"),
    ("skin", r"melanom|basal cell|squamous cell carcinoma of the skin|cutaneous|merkel"),
    ("sarcoma", r"sarcom|osteosarcom|ewing|gist|gastrointestinal stromal"),
]


def site_group(conditions: str) -> str:
    text = conditions.lower()
    for name, pattern in SITE_GROUPS:
        if re.search(pattern, text):
            return name
    return "other"


def phase_group(phases: str) -> str:
    p = (phases or "").upper()
    if "PHASE3" in p:
        return "3"
    if "PHASE2" in p:
        return "2"
    if "PHASE1" in p:
        return "1"
    return "other"


def era(first_submit: str) -> str:
    year = first_submit[:4]
    if not year.isdigit():
        return "unknown"
    y = int(year)
    if y < 2015:
        return "2010-2014"
    if y < 2020:
        return "2015-2019"
    return "2020-2026"


def standardised(pos_a, n_a, pos_c, n_c):
    """Direct standardisation over strata the agent supports."""
    support = n_a > 0
    if not support.any():
        return float("nan"), float("nan")
    w = n_c[support].astype(float)
    total = w.sum()
    if total <= 0:
        return float("nan"), float("nan")
    w = w / total
    rate_a = float(np.dot(w, pos_a[support] / np.maximum(n_a[support], 1)))
    rate_c = float(np.dot(w, pos_c[support] / np.maximum(n_c[support], 1)))
    return rate_a, rate_c


def bca_interval(draws: np.ndarray, theta_hat: float, jack: np.ndarray,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Bias-corrected and accelerated percentile interval.

    A ratio of two standardised rates is skewed and biased enough that the plain
    percentile interval is noticeably off-centre; BCa corrects for both. Acceleration
    uses a delete-one-stratum jackknife, which is the natural blocking here because the
    strata are the standardisation cells.
    """
    from scipy.stats import norm

    draws = draws[np.isfinite(draws)]
    if draws.size < 100 or not np.isfinite(theta_hat):
        return float("nan"), float("nan")
    prop = float((draws < theta_hat).mean())
    if prop <= 0 or prop >= 1:
        return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    z0 = norm.ppf(prop)

    jack = jack[np.isfinite(jack)]
    if jack.size < 3:
        a = 0.0
    else:
        d = jack.mean() - jack
        denom = 6.0 * (np.sum(d ** 2) ** 1.5)
        a = float(np.sum(d ** 3) / denom) if denom > 0 else 0.0

    def endpoint(z_alpha):
        num = z0 + z_alpha
        return norm.cdf(z0 + num / (1 - a * num))

    lo_p = 100 * endpoint(norm.ppf(alpha / 2))
    hi_p = 100 * endpoint(norm.ppf(1 - alpha / 2))
    lo_p = min(max(lo_p, 0.1), 99.9)
    hi_p = min(max(hi_p, 0.1), 99.9)
    if hi_p <= lo_p:
        return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    return float(np.percentile(draws, lo_p)), float(np.percentile(draws, hi_p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="data/cohort.csv.gz")
    ap.add_argument("--replicates", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--summary-json")
    ap.add_argument("--table", default="tables/table3.tsv")
    args = ap.parse_args()

    strata_key: dict[tuple, int] = {}
    strat: list[int] = []
    interventions: list[str] = []
    domains_needed = sorted({d for _, d, *_ in PAIRS})
    flags: dict[str, list[int]] = {d: [] for d in domains_needed}

    with gzip.open(args.cohort, "rt") as fh:
        for row in csv.DictReader(fh):
            if row["therapeutic"] != "1":
                continue
            key = (site_group(row["conditions"]), phase_group(row["phases"]),
                   era(row["first_submit_date"]))
            strat.append(strata_key.setdefault(key, len(strata_key)))
            interventions.append(row["intervention_names"])
            for d in domains_needed:
                flags[d].append(int(row[d]))

    n = len(strat)
    n_strata = len(strata_key)
    strat_arr = np.asarray(strat, dtype=np.int64)
    dom = {d: np.asarray(v, dtype=np.float64) for d, v in flags.items()}
    agents = {
        name: np.fromiter((1 if re.search(pat, t) else 0 for t in interventions),
                          dtype=bool, count=n)
        for name, pat in AGENT_PATTERNS.items()
    }

    print(f"therapeutic subset: {n:,} trials across {n_strata} strata "
          f"({len(SITE_GROUPS) + 1} sites x 4 phases x 3 eras)")
    for name, mask in agents.items():
        print(f"  {name:14s} {int(mask.sum()):>6,} trials")

    def point(mask, d, idx=None):
        if idx is None:
            s, dv, m = strat_arr, dom[d], mask
        else:
            s, dv, m = strat_arr[idx], dom[d][idx], mask[idx]
        n_c = np.bincount(s, minlength=n_strata)
        pos_c = np.bincount(s, weights=dv, minlength=n_strata)
        n_a = np.bincount(s[m], minlength=n_strata)
        pos_a = np.bincount(s[m], weights=dv[m], minlength=n_strata)
        return standardised(pos_a, n_a, pos_c, n_c)

    rng = np.random.default_rng(args.seed)
    boots = {(a, d): [] for a, d, *_ in PAIRS}
    print(f"\nbootstrapping {args.replicates:,} replicates over trials...")
    for r in range(args.replicates):
        idx = rng.integers(0, n, n)
        # index once per replicate, then reuse across pairs
        s = strat_arr[idx]
        n_c = np.bincount(s, minlength=n_strata)
        dv = {d: dom[d][idx] for d in domains_needed}
        pos_c = {d: np.bincount(s, weights=dv[d], minlength=n_strata) for d in domains_needed}
        ms = {a: agents[a][idx] for a in agents}
        for agent, domain, *_ in PAIRS:
            m = ms[agent]
            sa = s[m]
            n_a = np.bincount(sa, minlength=n_strata)
            pos_a = np.bincount(sa, weights=dv[domain][m], minlength=n_strata)
            ra, rc = standardised(pos_a, n_a, pos_c[domain], n_c)
            boots[(agent, domain)].append(ra / rc if rc and rc > 0 else np.nan)
        if (r + 1) % 500 == 0:
            print(f"  {r + 1:,}", flush=True)

    # delete-one-stratum jackknife, for the BCa acceleration term
    jackknife = {(a, d): np.full(n_strata, np.nan) for a, d, *_ in PAIRS}
    for k in range(n_strata):
        keep = strat_arr != k
        if not keep.any():
            continue
        s_k = strat_arr[keep]
        n_c = np.bincount(s_k, minlength=n_strata)
        for agent, domain, *_ in PAIRS:
            dv = dom[domain][keep]
            m = agents[agent][keep]
            pos_c = np.bincount(s_k, weights=dv, minlength=n_strata)
            n_a = np.bincount(s_k[m], minlength=n_strata)
            pos_a = np.bincount(s_k[m], weights=dv[m], minlength=n_strata)
            ra_k, rc_k = standardised(pos_a, n_a, pos_c, n_c)
            if rc_k and rc_k > 0:
                jackknife[(agent, domain)][k] = ra_k / rc_k

    print(f"\n{'agent':13s} {'domain':18s} {'events':>7s} {'std %':>7s} {'comp %':>7s} "
          f"{'RR':>6s} {'BCa 95% CI':>18s} {'pub RR':>7s}")
    print("-" * 96)
    results = []
    for agent, domain, pub_std, pub_comp, pub_rr, role in PAIRS:
        mask = agents[agent]
        events = int(dom[domain][mask].sum())
        ra, rc = point(mask, domain)
        rr = ra / rc if rc and rc > 0 else float("nan")
        draws = np.asarray(boots[(agent, domain)], dtype=float)
        draws = draws[np.isfinite(draws)]

        if events == 0:
            # A nonparametric bootstrap cannot manufacture events the sample does not
            # contain. Reporting 0 with an interval of [0, 0] is false precision.
            print(f"{agent:13s} {domain:18s} {events:>7d} {ra * 100:>7.2f} {rc * 100:>7.2f} "
                  f"{'-':>6s} {'not estimable':>18s} {'-':>7s}")
            results.append({
                "agent": agent, "domain": domain, "role": role,
                "n_agent_trials": int(mask.sum()), "agent_events": 0,
                "standardised_pct": 0.0,
                "comparator_pct": None if not np.isfinite(rc) else round(rc * 100, 3),
                "rate_ratio": None, "ci95_bca": None, "ci95_percentile": None,
                "status": "not estimable: zero agent events",
                "published_rate_ratio": pub_rr,
            })
            continue

        jack = np.array([jackknife[(agent, domain)][k] for k in range(n_strata)])
        lo_b, hi_b = bca_interval(draws, rr, jack)
        lo_p, hi_p = (np.percentile(draws, [2.5, 97.5]) if draws.size >= 100
                      else (float("nan"), float("nan")))
        ci = f"[{lo_b:.2f}, {hi_b:.2f}]" if np.isfinite(lo_b) else "not estimable"
        pub = f"{pub_rr:.2f}" if pub_rr is not None else "-"
        print(f"{agent:13s} {domain:18s} {events:>7d} {ra * 100:>7.2f} {rc * 100:>7.2f} "
              f"{rr:>6.2f} {ci:>18s} {pub:>7s}")
        results.append({
            "agent": agent, "domain": domain, "role": role,
            "n_agent_trials": int(mask.sum()), "agent_events": events,
            "standardised_pct": None if not np.isfinite(ra) else round(ra * 100, 3),
            "comparator_pct": None if not np.isfinite(rc) else round(rc * 100, 3),
            "rate_ratio": None if not np.isfinite(rr) else round(rr, 3),
            "ci95_bca": None if not np.isfinite(lo_b) else [round(lo_b, 3), round(hi_b, 3)],
            "ci95_percentile": None if not np.isfinite(lo_p) else [round(lo_p, 3), round(hi_p, 3)],
            "bootstrap_replicates_used": int(draws.size),
            "published_standardised_pct": pub_std,
            "published_comparator_pct": pub_comp,
            "published_rate_ratio": pub_rr,
        })

    import os
    os.makedirs(os.path.dirname(args.table) or ".", exist_ok=True)
    with open(args.table, "w") as fh:
        fh.write("Pair\tAgent\tDomain\tTrials\tMeasuring\tStandardised rate %\t"
                 "Standardised rate ratio\t95% CI (BCa)\n")
        for r in results:
            rr = "not estimable" if r["rate_ratio"] is None else f"{r['rate_ratio']:.2f}"
            civ = r.get("ci95_bca")
            cis = "not estimable" if not civ else f"{civ[0]:.2f}-{civ[1]:.2f}"
            fh.write(f"{r['role']}\t{r['agent']}\t{r['domain']}\t{r['n_agent_trials']}\t"
                     f"{r['agent_events']}\t{r['standardised_pct']:.2f}\t{rr}\t{cis}\n")
    print(f"\nwrote {args.table}")

    print("\nIntervals are BCa bootstrap over trials, with percentile intervals kept in the")
    print("run summary. Because each replicate")
    print("recomputes the agent rate, the comparator rate, the stratum weights and the")
    print("support set together, the correlation induced by the agent records being a")
    print("subset of the comparator is carried through rather than assumed away.")

    if args.summary_json:
        with open(args.summary_json, "w") as fh:
            fh.write(json.dumps({
                "population": "restricted therapeutic subset",
                "n_trials": n,
                "n_strata": n_strata,
                "standardisation": "11 site groups x 4 phase groups x 3 eras; "
                                   "cohort stratum distribution as standard, "
                                   "renormalised over agent-supported strata",
                "interval_method": "nonparametric bootstrap over trials, numerator and "
                                   "denominator recomputed jointly; BCa primary, "
                                   "percentile reported alongside",
                "replicates": args.replicates,
                "seed": args.seed,
                "pairs": results,
            }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
