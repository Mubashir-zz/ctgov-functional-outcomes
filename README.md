# Functional-outcome registration in oncology trials — cisplatin/hearing anchor analysis

Cross-sectional analysis of 93,371 outcome-bearing ClinicalTrials.gov cancer records, classified
across 14 functional domains by a deterministic lexicon extractor, corrected for classification
error using a human reference standard and Horvitz–Thompson design weights. The primary clinical
test is cisplatin against hearing-outcome registration; everything else supports that correction
or guards it against composition confounding.

Headline: of 2,189 modern therapeutic cisplatin trials, 27 register a hearing outcome — 1.23%
crude, 1.72% standardised against a 0.15% matched comparator, rate ratio 11.88 (95% CI
6.34–21.92). Corrected for classification error, hearing registration across the whole corpus is
0.12% (0.06–0.17) against an observed 0.21%.

Every number in this repository and in the manuscript comes from one analytic release. Nothing is
carried over from the earlier release that produced the first draft.

## Layout

```
MANUSCRIPT.docx      the manuscript
MANUSCRIPT.md        same text, editable
README.md            this file
figures/             figure1.png, figure2.png — regenerated from tables/
tables/              table1.tsv, table2.tsv, table3.tsv — written by the scripts, not by hand
references/          REFERENCES.ris
analysis/            the pipeline
tests/               invariants the outputs must satisfy
data/                corpus, metadata, labels, cohort, reference standard
provenance/          one run summary per stage, plus the original analysis notes
```

One manuscript, one README, one set of tables. A regenerated figure or table replaces the old one.

## Pipeline

| script | what it does |
|---|---|
| `fetch_ctgov_outcomes.py` | registered outcome text → `data/ctgov_outcomes.csv.gz` |
| `fetch_ctgov_metadata.py` | trial metadata → `data/trial_metadata.csv.gz` |
| `run_firstpass_multidomain_extractor.py` | 14-domain lexicon, rule version 0.1, applied unchanged |
| `build_cohort.py` | cohort eligibility → `data/cohort.csv.gz` |
| `recover_item_trial_map.py` | matches workbook items back to trials → `data/item_trial_map.csv` |
| `build_reference_standard.py` | human reference standard → `data/reference_standard.csv` |
| `sampling_weights.py` | inclusion probabilities and design weights → `tables/table2.tsv` |
| `bayesian_prevalence.py` | misclassification correction → `tables/table1.tsv` |
| `standardised_rates.py` | direct standardisation, BCa intervals → `tables/table3.tsv` |
| `sponsor_association.py` | crude and adjusted sponsor-class contrast |
| `build_figure.py` | regenerates both figures and refreshes the manuscript's tables |

```bash
export SSL_CERT_FILE=$(python3 -c "import certifi; print(certifi.where())")   # python.org builds
python3 analysis/fetch_ctgov_outcomes.py  --out data/ctgov_outcomes.csv.gz
python3 analysis/fetch_ctgov_metadata.py  --out data/trial_metadata.csv.gz
python3 analysis/run_firstpass_multidomain_extractor.py \
    --src data/ctgov_outcomes.csv.gz --out data/trial_domain_labels.csv.gz \
    --rule-version v0_1 --summary-json provenance/extractor_run_summary.json
python3 analysis/build_cohort.py             --summary-json provenance/cohort_summary.json
python3 analysis/recover_item_trial_map.py
python3 analysis/build_reference_standard.py --summary-json provenance/reference_standard_summary.json
python3 analysis/sampling_weights.py         --summary-json provenance/sampling_weights_summary.json
python3 analysis/bayesian_prevalence.py      --summary-json provenance/bayesian_summary.json
python3 analysis/standardised_rates.py       --summary-json provenance/standardised_rates_summary.json
python3 analysis/sponsor_association.py      --summary-json provenance/sponsor_summary.json
python3 analysis/build_figure.py
pytest tests/ -q
```

Both fetches are resumable and take roughly seven minutes each, longer when the registry
throttles. Requires `pandas`, `numpy`, `scipy`, `matplotlib`, `openpyxl`, `python-docx`.

## Cohort

Condition search "cancer", interventional, registered on or before 2026-07-31, at least one
registered outcome measure: 93,371 records of 96,578 candidates. Therapeutic subset — registered
from 2010-01-01, primary purpose treatment, explicit malignancy evidence: 51,227 of 55,655
date-and-purpose-eligible records.

## Reference standard

Assembled directly from the clinician workbooks: consensus where the two reviewers agreed, the
documented human adjudication where they did not, no model output anywhere. It reproduces the
published validation exactly — 7,966 paired calibration labels, 97.21% raw agreement, 222
adjudicated disagreements.

**The enrichment arm is excluded from the weighted estimator.** It sampled 40 records from a
narrow broad-screen stratum and 10 from a wider one, per domain; for hearing the narrow stratum
held only 71 records, so its inclusion probability is about 0.56, three orders of magnitude away
from the wider frame. Membership was recorded only in the sampling key, which is lost, so those
probabilities are not identifiable and assigning them anyway would be an invention. The arm is
reported instead as an unweighted audit of the predicted-negative space: 350 records read, seven
true misses found (neuropathy 3, limb function 3, speech/voice 1), none for hearing.

## Accuracy uncertainty

A cluster bootstrap of the validation design, not per-cell approximations. Each replicate
resamples validation items whole within stratum and recomputes weighted sensitivity and
specificity together, so their dependence is propagated and the correlation among a record's 14
domain labels survives. Prevalence is then drawn from its exact conditional posterior given each
paired accuracy draw, bounded in the unit interval by construction — no boundary estimates, and no
assumed-sensitivity tier for the domains with no observed false negatives.

## Text-source sensitivity

Reviewer workbooks truncated outcome text at 1,500 characters; the corpus was classified on full
text. Matching restored full text for 88.2% of items, and matching is not independent of content —
matched items have a median length of 1,304 characters against 746. Three analyses:

| text source | specificity mean absolute error vs the earlier release |
|---|---|
| full where known, workbook text otherwise (primary) | 0.0009 |
| full text only, matched items | 0.0018 |
| truncated workbook text throughout | 0.0085 |

Truncation suppresses predicted positives and inflates apparent predictive value. The primary
analysis sits between the two extremes.

## Results

`tables/table1.tsv` observed and corrected prevalence with credible intervals;
`tables/table2.tsv` design-weighted predictive values, sensitivity and specificity;
`tables/table3.tsv` standardised agent–domain rates with BCa intervals.

Standardised rate ratios use bias-corrected and accelerated bootstrap intervals with a
delete-one-stratum jackknife; percentile intervals are kept in the run summary. Pairs contributing
no events are reported as not estimable rather than as zero.

## Agreement with the earlier release

The release that produced the first draft is no longer retrievable. Against its published figures
the current one agrees to a mean absolute difference of 0.021 percentage points on per-domain
prevalence (largest single difference 0.07, pain), 0.0020 on design-weighted specificity and 1.96
percentage points on design-weighted predictive value. The current corpus is 3.7% smaller and its
therapeutic subset 7.2% smaller.

## What is and is not in this repository

Tracked: the pipeline, the tests, the tables and figures it produces, the human reference
standard (`data/reference_standard.csv`), the recovered item-to-trial map, the classifier scores,
and one run summary per stage.

Not tracked: the four large derived files under `data/` — the outcome corpus, the trial metadata,
the domain labels and the cohort. They are rebuilt by the two fetch steps and the extractor, and
the registry they come from is public. The manuscript is not published here either; it is an
unsubmitted draft.

Rebuilding from an empty `data/` directory takes about twenty minutes, most of it waiting on the
ClinicalTrials.gov API.

## Status

Draft for scientific review. Not submitted. The corresponding-author affiliation and contact
details are placeholders in the manuscript.
