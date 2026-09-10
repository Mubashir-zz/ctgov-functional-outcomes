"""Invariants the analysis outputs must satisfy.

These are not unit tests of the statistics — they are the checks that catch the failure
modes this project actually hit: tables drifting from the code that generates them,
weights leaking onto records whose sampling design is unknown, and interval columns that
do not contain their own point estimate.
"""
from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))

DOMAINS = 14


def read_tsv(name):
    with open(ROOT / "tables" / name) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def summary(name):
    with open(ROOT / "provenance" / name) as fh:
        return json.load(fh)


def test_extractor_selftest_passes():
    from run_firstpass_multidomain_extractor import _selftest
    _selftest()


def test_domain_labels_are_complete():
    n = 0
    with gzip.open(ROOT / "data" / "trial_domain_labels.csv.gz", "rt") as fh:
        for _ in csv.DictReader(fh):
            n += 1
    assert n % DOMAINS == 0, "every trial must carry a row for all 14 domains"


def test_enrichment_rows_carry_no_weight():
    """Enrichment strata are not recoverable; weighting them would be an invention."""
    seen = weighted = 0
    with open(ROOT / "data" / "reference_standard.csv") as fh:
        for row in csv.DictReader(fh):
            if row["stratum"] == "enrichment":
                seen += 1
                if row.get("weight"):
                    weighted += 1
    assert seen > 0
    assert weighted == 0


def test_inclusion_probabilities_are_probabilities():
    with open(ROOT / "data" / "reference_standard.csv") as fh:
        for row in csv.DictReader(fh):
            pi = row.get("inclusion_probability")
            if pi:
                assert 0 < float(pi) <= 1


def test_reference_standard_matches_the_published_validation():
    s = summary("reference_standard_summary.json")
    assert s["calibration_paired_labels"] == 7966
    assert s["calibration_disagreements"] == 222
    assert s["unresolved_dropped"] == 0
    assert round(s["raw_agreement_pct"], 2) == 97.21


@pytest.mark.parametrize("name", ["table1.tsv", "table2.tsv", "table3.tsv"])
def test_tables_exist_and_are_populated(name):
    rows = read_tsv(name)
    assert len(rows) >= 13


def test_corrected_prevalence_lies_inside_its_interval():
    for r in read_tsv("table1.tsv"):
        lo, hi = (float(x) for x in r["95% credible"].split("-"))
        assert lo <= float(r["Corrected %"]) <= hi, r["Domain"]
        assert 0 <= lo <= hi <= 100


def test_rate_ratios_lie_inside_their_intervals():
    for r in read_tsv("table3.tsv"):
        if r["Standardised rate ratio"] == "not estimable":
            assert r["95% CI (BCa)"] == "not estimable"
            assert int(r["Measuring"]) == 0
            continue
        lo, hi = (float(x) for x in r["95% CI (BCa)"].split("-"))
        assert lo <= float(r["Standardised rate ratio"]) <= hi, r["Agent"]


def test_zero_event_pairs_are_not_given_a_ratio():
    for r in read_tsv("table3.tsv"):
        if int(r["Measuring"]) == 0:
            assert r["Standardised rate ratio"] == "not estimable"


def test_predictive_values_are_percentages():
    for r in read_tsv("table2.tsv"):
        assert 0 <= float(r["Design-weighted PPV %"]) <= 100
        assert 0 <= float(r["Specificity"]) <= 1


def test_bayesian_run_used_paired_design_draws():
    s = summary("bayesian_summary.json")
    assert "bootstrap" in s["accuracy_uncertainty"].lower()
    assert s["enrichment"].startswith("excluded")
    assert s["replicates"] >= 1000
