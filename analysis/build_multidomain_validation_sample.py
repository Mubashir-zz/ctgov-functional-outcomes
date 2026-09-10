# -*- coding: utf-8 -*-
"""
Build the NLP-001 14-domain external-validation sample — ready for two blinded human reviewers.
Design: DISJOINT from the four-cancer development corpus; ENRICHED per domain (so rare domains
like hearing/visual get enough positives to estimate sensitivity); retains inverse-probability
sampling weights for design-weighted Se/Sp; BLINDED (no extractor prediction, no sponsor in the
reviewer workbook; neutral item_ids). Emits: blinded workbook, sealed private key, manifest, README.

Importing this module does not rebuild the sample. Existing 2026-08-27 artifacts are frozen
unless --force is passed after explicit review.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
FIRSTPASS = ROOT / "PROJECTB_EXTRACTOR_FIRSTPASS_TRIAL_DOMAIN_v0_1_2026-08-27.csv.gz"
REPFRAME = ROOT / "ctgov_oncology_candidate_replication_frame_v2.csv.gz"
OUTCOMES = ROOT / "ctgov_oncology_candidate_normalized_v2_outcomes.csv.gz"
DOMAINS = [
    "cognition", "neuropathy", "fatigue", "pain", "physical_function", "swallowing",
    "speech_voice", "hearing", "respiratory", "bowel", "urinary", "sexual_reproductive",
    "limb_lymphedema", "visual",
]
SEED = 20260827
N_POS_PER_DOMAIN = 30
N_NEG = 160
WB = ROOT / "PROJECTB_NLP_VALIDATION_BLINDED_v1_2026-08-27.csv"
KEY = ROOT / "PROJECTB_NLP_VALIDATION_SEALED_KEY_PRIVATE_v1_2026-08-27.csv"
MAN = ROOT / "PROJECTB_NLP_VALIDATION_MANIFEST_v1_2026-08-27.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild and overwrite the frozen 2026-08-27 validation sample artifacts",
    )
    args = parser.parse_args(argv)
    existing = [path for path in (WB, KEY, MAN) if path.exists()]
    if existing and not args.force:
        print(
            "Frozen NLP-001 sample already exists "
            f"({', '.join(path.name for path in existing)}). "
            "Refusing to overwrite. Pass --force only after explicit review."
        )
        return 0

    random.seed(SEED)
    pred = defaultdict(set)
    for chunk in pd.read_csv(FIRSTPASS, compression="gzip", chunksize=300_000):
        for trial_id, domain, measured in zip(chunk.trial_id, chunk.domain, chunk.measured_firstpass):
            if measured == 1:
                pred[trial_id].add(domain)

    replication = pd.read_csv(REPFRAME, compression="gzip", usecols=["source_trial_id", "seen_in_four_cancer_development"])
    dev = set(
        replication.loc[
            replication["seen_in_four_cancer_development"].astype(str).isin(["True", "true", "1"]),
            "source_trial_id",
        ]
    )
    universe = {trial_id for trial_id in pred if trial_id not in dev}
    universe.update(trial_id for trial_id in set(replication.source_trial_id) if trial_id not in dev)
    pos_pool = {domain: [trial_id for trial_id in universe if domain in pred.get(trial_id, set())] for domain in DOMAINS}
    neg_pool = [trial_id for trial_id in universe if not pred.get(trial_id, set())]
    picked = {}

    def draw(pool, k, stratum):
        take = random.sample(pool, min(k, len(pool)))
        for trial_id in take:
            if trial_id not in picked or len(pool) < picked[trial_id][1]:
                picked[trial_id] = (stratum, len(pool), min(k, len(pool)))

    for domain in DOMAINS:
        draw(pos_pool[domain], N_POS_PER_DOMAIN, f"pos:{domain}")
    draw(neg_pool, N_NEG, "neg:all")

    items = list(picked.keys())
    weights = {trial_id: (picked[trial_id][1] / picked[trial_id][2]) for trial_id in items}
    want = set(items)
    blocks = defaultdict(list)
    for chunk in pd.read_csv(
        OUTCOMES,
        compression="gzip",
        usecols=["source_trial_id", "hierarchy", "measure", "description"],
        chunksize=200_000,
        dtype=str,
    ):
        chunk = chunk[chunk.source_trial_id.isin(want)].fillna("")
        for trial_id, hierarchy, measure, description in zip(
            chunk.source_trial_id, chunk.hierarchy, chunk.measure, chunk.description
        ):
            blocks[trial_id].append(f"[{hierarchy}] {measure}. {description}".strip())

    def block_text(trial_id):
        text = " || ".join(blocks.get(trial_id, []))
        return (text[:1500] + " …") if len(text) > 1500 else text

    random.shuffle(items)
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with WB.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", "registered_outcomes_text"] + DOMAINS + ["notes"])
        for index, trial_id in enumerate(items, 1):
            writer.writerow([f"ITEM{index:04d}", block_text(trial_id)] + [""] * len(DOMAINS) + [""])
    with KEY.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "item_id", "trial_id", "sampling_stratum", "stratum_size", "n_drawn",
            "ip_weight", "firstpass_predicted_domains",
        ])
        for index, trial_id in enumerate(items, 1):
            stratum, size, drawn = picked[trial_id]
            writer.writerow([
                f"ITEM{index:04d}", trial_id, stratum, size, drawn, round(weights[trial_id], 4),
                "|".join(sorted(pred.get(trial_id, set()))),
            ])
    manifest = dict(
        created_utc=timestamp,
        seed=SEED,
        n_items=len(items),
        domains=DOMAINS,
        disjoint_from="four_cancer_development_corpus",
        n_excluded_dev=len(dev),
        strata={stratum: sum(1 for trial_id in items if picked[trial_id][0] == stratum) for stratum in sorted({picked[trial_id][0] for trial_id in items})},
        n_pos_per_domain_target=N_POS_PER_DOMAIN,
        n_neg_target=N_NEG,
        weighting="inverse_probability = stratum_size / n_drawn (design-weighted Se/Sp)",
        blinding="workbook excludes extractor prediction and sponsor; neutral item_ids; key sealed/private",
    )
    MAN.write_text(json.dumps(manifest, indent=2))
    print(f"n_items={len(items)}  (excluded {len(dev)} development trials)")
    print("strata:", manifest["strata"])
    print(f"wrote:\n  {WB.name}\n  {KEY.name}\n  {MAN.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
