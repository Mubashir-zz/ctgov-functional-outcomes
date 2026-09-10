#!/usr/bin/env python3
"""Sponsor-class association with functional-outcome registration, crude and adjusted.

Included because it is the clearest demonstration in this corpus of the composition
hazard the paper argues for: the crude contrast and the adjusted contrast point in
opposite directions, and only the adjusted one is interpretable.

Model: any functional domain registered ~ industry sponsor + trial phase + log enrolment
+ registration era, logistic, on the restricted therapeutic subset.
"""
from __future__ import annotations

import argparse, csv, gzip, json, math, sys
import numpy as np


def phase_group(p):
    p = (p or "").upper()
    return "3" if "PHASE3" in p else "2" if "PHASE2" in p else "1" if "PHASE1" in p else "other"


def era(d):
    y = d[:4]
    if not y.isdigit():
        return "unknown"
    y = int(y)
    return "2010-2014" if y < 2015 else "2015-2019" if y < 2020 else "2020-2026"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="data/cohort.csv.gz")
    ap.add_argument("--summary-json")
    args = ap.parse_args()

    rows = []
    with gzip.open(args.cohort, "rt") as fh:
        for r in csv.DictReader(fh):
            if r["therapeutic"] != "1":
                continue
            rows.append(r)
    print(f"therapeutic subset: {len(rows):,}")

    y = np.array([int(r["any_domain"]) for r in rows], dtype=float)
    industry = np.array([1.0 if r["lead_sponsor_class"] == "INDUSTRY" else 0.0 for r in rows])

    n_ind, n_non = int(industry.sum()), int((1 - industry).sum())
    crude_ind = float(y[industry == 1].mean() * 100)
    crude_non = float(y[industry == 0].mean() * 100)
    print(f"crude: industry {crude_ind:.2f}% (n={n_ind:,})   "
          f"non-industry {crude_non:.2f}% (n={n_non:,})")

    phases = sorted({phase_group(r["phases"]) for r in rows})
    eras = sorted({era(r["first_submit_date"]) for r in rows})
    cols = [industry]
    names = ["industry"]
    for p in phases[1:]:
        cols.append(np.array([1.0 if phase_group(r["phases"]) == p else 0.0 for r in rows]))
        names.append(f"phase_{p}")
    for e in eras[1:]:
        cols.append(np.array([1.0 if era(r["first_submit_date"]) == e else 0.0 for r in rows]))
        names.append(f"era_{e}")
    enrol = np.array([math.log1p(float(r["enrollment"])) if r["enrollment"] not in ("", None)
                      else 0.0 for r in rows])
    cols.append(enrol)
    names.append("log_enrolment")
    X = np.column_stack([np.ones(len(rows))] + cols)

    # Newton-Raphson logistic
    beta = np.zeros(X.shape[1])
    for _ in range(60):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-9, None)
        z = eta + (y - mu) / w
        XtW = X.T * w
        beta_new = np.linalg.solve(XtW @ X, XtW @ z)
        if np.max(np.abs(beta_new - beta)) < 1e-10:
            beta = beta_new
            break
        beta = beta_new
    eta = X @ beta
    mu = 1.0 / (1.0 + np.exp(-eta))
    w = np.clip(mu * (1 - mu), 1e-9, None)
    cov = np.linalg.inv((X.T * w) @ X)
    se = np.sqrt(np.diag(cov))

    i = names.index("industry") + 1
    or_ = math.exp(beta[i])
    lo, hi = math.exp(beta[i] - 1.96 * se[i]), math.exp(beta[i] + 1.96 * se[i])
    zstat = beta[i] / se[i]
    print(f"\nadjusted odds ratio, industry sponsorship: {or_:.3f} "
          f"(95% CI {lo:.3f}-{hi:.3f}), z = {zstat:.2f}")
    print("adjusted for trial phase, log enrolment and registration era.")
    direction = ("reverses" if (crude_ind < crude_non) == (or_ > 1) else "does not reverse")
    print(f"The crude and adjusted contrasts: adjustment {direction} the association.")

    if args.summary_json:
        with open(args.summary_json, "w") as fh:
            fh.write(json.dumps({
                "n": len(rows), "n_industry": n_ind, "n_non_industry": n_non,
                "crude_industry_pct": round(crude_ind, 2),
                "crude_non_industry_pct": round(crude_non, 2),
                "adjusted_or": round(or_, 3), "ci95": [round(lo, 3), round(hi, 3)],
                "z": round(zstat, 2),
                "covariates": ["phase", "log enrolment", "registration era"],
                "reverses_after_adjustment": direction == "reverses",
            }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
