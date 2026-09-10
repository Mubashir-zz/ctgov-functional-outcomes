#!/usr/bin/env python3
"""Recover the item_id -> trial_id mapping that the sealed key would have held.

The blinded workbooks were built by joining each trial's registered outcome blocks
with " || " and truncating the result to 1,500 characters, with the trial identity
held only in a private key file that no longer exists. The outcome text itself is
still in the corpus, so the mapping is recoverable by matching blocks back.

Each workbook block has the form "[HIERARCHY] measure. description". This strips
the hierarchy tag, indexes every trial's outcome blocks from the fetched corpus,
looks the item's blocks up, and accepts a trial only when it is the unique best
match over a clear margin. The final truncated block is discarded, since it may be
cut mid-sentence.

Writes data/item_trial_map.csv: item_id, trial_id, blocks_matched, blocks_total,
score, margin.

The point of recovering this is that extractor predictions can then be computed on
the full outcome text rather than the truncated workbook text, which is what the
published analysis did.
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import re
import sys
from pathlib import Path

import openpyxl

LABELS = Path("/Users/mac/Documents/Research/Laber.py")
SOURCES = [
    ("calibration", "PROJECTB_NLP_VALIDATION_BLINDED_v1_2026-08-27.csv", "csv"),
    ("negverify", "PROJECTB_NLP_NEGVERIFY_LABELING_WORKBOOK_v1_FILLED.xlsx", "label-sheet"),
    ("enrichment", "PROJECTB_NLP_SPARSE_NEG_ENRICH_LABELING_WORKBOOK_v1_2026-08-29_FILLED.xlsx", "domain-sheets"),
]

TAG = re.compile(r"^\[[^\]]{0,24}\]\s*")
WS = re.compile(r"\s+")
KEY_CHARS = 80
MIN_SCORE = 0.60


def norm(text: str) -> str:
    return WS.sub(" ", text.strip().lower())


def item_blocks(raw: str) -> list[str]:
    """Split a workbook cell into its complete outcome blocks."""
    text = (raw or "").strip()
    truncated = text.endswith("…")
    if truncated:
        text = text[:-1].rstrip()
    # strip before matching the tag: blocks after the first begin with the join space
    parts = [TAG.sub("", p.strip()).strip() for p in text.split("||")]
    parts = [norm(p) for p in parts if p.strip()]
    if truncated and len(parts) > 1:
        parts = parts[:-1]
    return parts


def read_items() -> list[tuple[str, str, str]]:
    """(stratum, item_id, text) across the three workbooks."""
    out: list[tuple[str, str, str]] = []
    for stratum, filename, kind in SOURCES:
        path = LABELS / filename
        if not path.is_file():
            print(f"  ! missing {filename}", file=sys.stderr)
            continue
        if kind == "csv":
            with open(path) as fh:
                for row in csv.DictReader(fh):
                    out.append((stratum, row["item_id"], row["registered_outcomes_text"]))
        elif kind == "label-sheet":
            wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
            ws = wb["Label"]
            rows = ws.iter_rows(values_only=True)
            next(rows)
            for row in rows:
                if row and row[0]:
                    out.append((stratum, str(row[0]).strip(), str(row[1] or "")))
            wb.close()
        else:
            wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
            seen: set[str] = set()
            for ws in wb.worksheets:
                if ws.max_column < 4:
                    continue
                rows = ws.iter_rows(values_only=True)
                header = next(rows)
                if not header or str(header[0]).strip() != "item_id":
                    continue
                for row in rows:
                    if row and row[0] and str(row[0]).strip() not in seen:
                        seen.add(str(row[0]).strip())
                        out.append((stratum, str(row[0]).strip(), str(row[2] or "")))
            wb.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcomes", default="data/ctgov_outcomes.csv.gz")
    ap.add_argument("--out", default="data/item_trial_map.csv")
    args = ap.parse_args()

    items = read_items()
    print(f"workbook items: {len(items):,}")

    wanted_keys: set[str] = set()
    parsed: dict[str, list[str]] = {}
    for _, item, text in items:
        blocks = item_blocks(text)
        parsed[item] = blocks
        for b in blocks:
            wanted_keys.add(b[:KEY_CHARS])

    print(f"distinct block keys to look up: {len(wanted_keys):,}")

    index: dict[str, set[str]] = collections.defaultdict(set)
    trial_blocks: dict[str, set[str]] = collections.defaultdict(set)
    n_rows = 0
    with gzip.open(args.outcomes, "rt") as fh:
        for row in csv.DictReader(fh):
            n_rows += 1
            block = norm(f"{row['measure']}. {row['description']}".strip())
            key = block[:KEY_CHARS]
            if key in wanted_keys:
                index[key].add(row["source_trial_id"])
                trial_blocks[row["source_trial_id"]].add(key)
    print(f"scanned {n_rows:,} corpus outcome rows; "
          f"{len(index):,} of the keys occur in the corpus")

    resolved = []
    unmatched = 0
    ambiguous = 0
    for stratum, item, _ in items:
        blocks = parsed[item]
        if not blocks:
            unmatched += 1
            continue
        keys = [b[:KEY_CHARS] for b in blocks]
        votes: collections.Counter = collections.Counter()
        for key in keys:
            for trial in index.get(key, ()):  # a key can occur in several trials
                votes[trial] += 1
        if not votes:
            unmatched += 1
            continue
        ranked = votes.most_common(2)
        best_trial, best_votes = ranked[0]
        runner = ranked[1][1] if len(ranked) > 1 else 0
        score = best_votes / len(keys)
        margin = (best_votes - runner) / len(keys)
        if score < MIN_SCORE or margin <= 0:
            ambiguous += 1
            continue
        resolved.append({
            "item_id": item, "stratum": stratum, "trial_id": best_trial,
            "blocks_matched": best_votes, "blocks_total": len(keys),
            "score": round(score, 3), "margin": round(margin, 3),
        })

    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "item_id", "stratum", "trial_id", "blocks_matched", "blocks_total", "score", "margin"])
        writer.writeheader()
        writer.writerows(resolved)

    by_stratum = collections.Counter(r["stratum"] for r in resolved)
    totals = collections.Counter(s for s, _, _ in items)
    print(f"\nresolved {len(resolved):,} of {len(items):,} items "
          f"({len(resolved) / len(items) * 100:.1f}%)")
    for stratum in totals:
        print(f"  {stratum:14s} {by_stratum[stratum]:>5,} / {totals[stratum]:>5,}"
              f"  ({by_stratum[stratum] / totals[stratum] * 100:.1f}%)")
    print(f"  no candidate: {unmatched:,}   ambiguous: {ambiguous:,}")
    print(f"\nwrote -> {args.out}")

    # ---- match-status diagnostics
    # Whether a workbook item can be traced back to its trial is not guaranteed to be
    # independent of the item's content. If it is not, predictions made on full text for
    # matched records and on truncated text for the rest are not exchangeable, and any
    # accuracy estimate mixing them inherits that. Report it rather than assume it away.
    resolved_ids = {r["item_id"] for r in resolved}
    stats = {True: collections.defaultdict(list), False: collections.defaultdict(list)}
    for stratum, item, text in items:
        ok = item in resolved_ids
        blocks = parsed[item]
        stats[ok]["chars"].append(len(text or ""))
        stats[ok]["blocks"].append(len(blocks))
    print("\nMatch-status diagnostics (matched vs unmatched):")
    print(f"  {'measure':22s} {'matched':>12s} {'unmatched':>12s}")
    for key, label in [("chars", "text length"), ("blocks", "outcome blocks")]:
        a = sorted(stats[True][key])
        b = sorted(stats[False][key])
        med = lambda v: v[len(v) // 2] if v else float("nan")
        print(f"  {label + ' (median)':22s} {med(a):>12.0f} {med(b):>12.0f}")
    print(f"  {'n':22s} {len(stats[True]['chars']):>12,} {len(stats[False]['chars']):>12,}")
    print("  Matching succeeds more often on longer, multi-block records, because more")
    print("  blocks means more chances of a unique hit. Predictions for matched records")
    print("  therefore come from full text and for unmatched records from truncated text;")
    print("  build_reference_standard.py --text-mode truncated forces one rule for all.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
