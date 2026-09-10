# -*- coding: utf-8 -*-
"""
FIRST-PASS 14-domain endpoint extractor over REAL ClinicalTrials.gov v2 outcome text.

Lexicon/regex grounded in MULTIDOMAIN_ENDPOINT_ONTOLOGY_v0_2 (instrument names + inclusion
terms, with trap exclusions). Produces trial×domain first-pass labels plus exact evidence
spans. This is a transparent FIRST-PASS baseline for NLP-001, NOT the validated final
model — disjoint design-weighted human validation is required before confirmatory use.

Rule versions:
  v0_1             Frozen lexicon used to draw the 2026-08-27 validation sample.
                   Do not change these patterns; they define the sealed-key predictions.
  v0_2_candidate   Same inclusions plus ontology trap exclusions that v0_1 missed
                   (ECOG/Karnofsky as physical_function; QLQ-C30 as sole cognition
                   evidence; serum creatinine as urinary). Writes a new versioned
                   path and must not overwrite v0_1 artifacts.

No risk/sponsor information is used (extractor blinding preserved).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    pd = None

ROOT = Path(__file__).resolve().parent
GROUPING_PATH = ROOT / "PROJECTB_DOMAIN_GROUPING_v1_2026-08-27.json"
DEFAULT_SRC = ROOT / "ctgov_oncology_candidate_normalized_v2_outcomes.csv.gz"
DEFAULT_OUT_V01 = ROOT / "PROJECTB_EXTRACTOR_FIRSTPASS_TRIAL_DOMAIN_v0_1_2026-08-27.csv.gz"
DOMAINS = list(
    json.loads(GROUPING_PATH.read_text(encoding="utf-8"))["leaf_domains"]
    if GROUPING_PATH.is_file()
    else [
        "cognition",
        "neuropathy",
        "fatigue",
        "pain",
        "physical_function",
        "swallowing",
        "speech_voice",
        "hearing",
        "respiratory",
        "bowel",
        "urinary",
        "sexual_reproductive",
        "limb_lymphedema",
        "visual",
    ]
)

# Frozen v0.1 patterns — byte-for-byte the rules that produced the 2026-08-27 sample.
INC_V0_1 = {
    "cognition": r"cognit|neurocognit|neuropsych|\bmoca\b|mini[- ]?mental|\bmmse\b|memory|executive function|processing speed|hopkins verbal|\bhvlt\b|trail making|montreal cognitive|attention task",
    "neuropathy": r"neuropath|\bcipn\b|paresthes|paraesthes|neurotoxicit|nerve conduction|gog[- ]?ntx|\bntx\b|total neuropathy score|sensory neuropathy",
    "fatigue": r"fatigue|facit[- ]?f\b|facit-fatigue|brief fatigue|\bbfi\b|fatigue severity",
    "pain": r"\bpain\b|brief pain inventory|\bbpi\b|pain intensity|pain interference|mcgill|worst pain|pain score|painful",
    "physical_function": r"physical function|physical performance|functional capacity|mobility|six[- ]?minute walk|6[- ]?minute walk|gait speed|grip strength|timed up and go|short physical performance|\bsppb\b|activities of daily living|\badl\b|range of motion",
    "swallowing": r"swallow|dysphagia|\bmdadi\b|aspiration|gastrostomy|penetration[- ]aspiration|\bfees\b|videofluoroscop",
    "speech_voice": r"speech|\bvoice\b|dysphonia|articulation|intelligibility|aphasia|voice handicap|\bvhi\b|communication function",
    "hearing": r"hearing|audiometr|audiogram|ototox|tinnitus|hearing threshold|hearing loss",
    "respiratory": r"dyspn|pulmonary function|\bfev1\b|spirometr|respiratory function|shortness of breath|breathlessness|\bmrc\b dyspnea|oxygen dependen",
    "bowel": r"bowel|diarrh|constipation|fecal incontinence|faecal incontinence|\blars\b|\bstoma\b|defecation|proctitis",
    "urinary": r"urinary|incontinence|\bipss\b|urinary urgency|urinary frequency|\bcatheter\b|cystitis|\biciq\b|nocturia|\bluts\b",
    "sexual_reproductive": r"sexual function|erectile|\biief\b|libido|\bfsfi\b|fertilit|reproductive function|ovarian function|sperm|semen|menopaus",
    "limb_lymphedema": r"lymph[eo]dema|limb volume|arm volume|arm swelling|shoulder function|\bdash\b|quickdash|limb function",
    "visual": r"visual acuity|\bvision\b|visual field|ophthalmic|contrast sensitivity|\bvfq\b|retinopathy|macular|visual function",
}
EXC_V0_1 = {
    "visual": r"visual analog|visual analogue|visual inspection",
}

# Candidate next first-pass: same inclusions, additional ontology traps.
EXC_V0_2 = {
    "visual": EXC_V0_1["visual"],
    "physical_function": r"\becog\b|karnofsky|\bkps\b|performance status",
    "cognition": r"qlq[- ]?c30|eortc qlq|cognitive functioning subscale",
    "urinary": r"serum creatinine|\begfr\b|creatinine clearance",
}

RULE_VERSIONS = {
    "v0_1": {"inc": INC_V0_1, "exc": EXC_V0_1},
    "v0_2_candidate": {"inc": INC_V0_1, "exc": EXC_V0_2},
}

FROZEN_OUTPUT_NAMES = {
    "PROJECTB_EXTRACTOR_FIRSTPASS_TRIAL_DOMAIN_v0_1_2026-08-27.csv.gz",
    "PROJECTB_EXTRACTOR_QBA_PARAMETERS_FROZEN_v1.csv",
    "PROJECTB_EXTRACTOR_QBA_DRAWS_FROZEN_v1.csv.gz",
    "PROJECTB_NLP_VALIDATION_BLINDED_v1_2026-08-27.csv",
    "PROJECTB_NLP_VALIDATION_SEALED_KEY_PRIVATE_v1_2026-08-27.csv",
    "PROJECTB_NLP_VALIDATION_MANIFEST_v1_2026-08-27.json",
}


@dataclass(frozen=True)
class DomainEvidence:
    domain: str
    start: int
    end: int
    matched_text: str


def compile_rules(rule_version: str) -> tuple[dict[str, re.Pattern[str]], dict[str, re.Pattern[str]]]:
    if rule_version not in RULE_VERSIONS:
        raise ValueError(f"Unknown extractor rule version: {rule_version}")
    spec = RULE_VERSIONS[rule_version]
    inc_re = {domain: re.compile(pattern) for domain, pattern in spec["inc"].items()}
    exc_re = {domain: re.compile(pattern) for domain, pattern in spec["exc"].items() if pattern}
    return inc_re, exc_re


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def classify_text(
    text: str,
    rule_version: str = "v0_1",
    inc_re: dict[str, re.Pattern[str]] | None = None,
    exc_re: dict[str, re.Pattern[str]] | None = None,
) -> dict[str, list[DomainEvidence]]:
    """Return per-domain evidence spans for lowercase (or already-lowercased) outcome text."""
    if inc_re is None or exc_re is None:
        inc_re, exc_re = compile_rules(rule_version)
    hits: dict[str, list[DomainEvidence]] = {}
    for domain in DOMAINS:
        matches = list(inc_re[domain].finditer(text))
        if not matches:
            continue
        if domain in exc_re:
            cleaned = exc_re[domain].sub(" ", text)
            if not inc_re[domain].search(cleaned):
                continue
        hits[domain] = [
            DomainEvidence(domain=domain, start=match.start(), end=match.end(), matched_text=match.group())
            for match in matches
        ]
    return hits


ONTOLOGY_VERSION = "MULTIDOMAIN_ENDPOINT_ONTOLOGY_v0_2_PANCANCER"
ONTOLOGY_PATH = ROOT / "MULTIDOMAIN_ENDPOINT_ONTOLOGY_v0_2_PANCANCER.md"
EXTRACTOR_PATH = Path(__file__).resolve()
HIERARCHY_VALUES = ("PRIMARY", "SECONDARY", "OTHER", "UNKNOWN")
REVIEW_STATUS_VALUES = ("AUTO_FINAL", "HUMAN_CONFIRMED", "HUMAN_OVERRULED", "UNRESOLVED")
FIRSTPASS_REVIEW_STATUS = "UNRESOLVED"
FIRSTPASS_API_VERSION = "0.2.0-firstpass"
MEASUREMENT_MODES = (
    "objective_performance",
    "patient_reported",
    "clinician_reported",
    "laboratory_imaging",
    "composite",
    "unclear",
)
CLASSIFICATIONS = ("measured_outcome", "mention_only", "exclusion_or_trap", "abstain")

AE_LIST_RE = re.compile(r"\badverse events?\b|\btoxicities\b|safety (?:will be|is) (?:monitored|assessed|recorded)")
OPERATIONALIZED_RE = re.compile(
    r"\b(scale|score|inventory|questionnaire|assessment|index|measured|primary|secondary|endpoint)\b"
)
TIMEFRAME_RE = re.compile(
    r"\b(\d+\s*(?:months?|weeks?|days?|years?)|at (?:baseline|follow[- ]up)|month(?:s)?\s+\d+)\b",
    re.IGNORECASE,
)
MODE_PATTERNS = (
    (re.compile(r"patient[- ]reported|\bpros?\b|questionnaire|eortc|facit|\bqlq\b"), "patient_reported"),
    (re.compile(r"audiometr|spirometr|nerve conduction|visual acuity|six[- ]?minute walk|6[- ]?minute walk|grip strength|timed up and go|trail making|montreal cognitive|\bmoca\b|\bmmse\b|neuropsych"), "objective_performance"),
    (re.compile(r"clinician[- ]reported|investigator[- ]assessed|\bctcae\b"), "clinician_reported"),
    (re.compile(r"\bpet/ct\b|\bmri\b|imaging|biomarker"), "laboratory_imaging"),
)
INSTRUMENTS = (
    (re.compile(r"montreal cognitive|\bmoca\b"), "MoCA", "cognition"),
    (re.compile(r"mini[- ]?mental|\bmmse\b"), "MMSE", "cognition"),
    (re.compile(r"hopkins verbal|\bhvlt\b"), "HVLT", "cognition"),
    (re.compile(r"trail making"), "Trail Making Test", "cognition"),
    (re.compile(r"facit[- ]?f\b|facit-fatigue"), "FACIT-F", "fatigue"),
    (re.compile(r"brief fatigue|\bbfi\b"), "BFI", "fatigue"),
    (re.compile(r"brief pain inventory|\bbpi\b"), "BPI", "pain"),
    (re.compile(r"\bmdadi\b"), "MDADI", "swallowing"),
    (re.compile(r"voice handicap|\bvhi\b"), "VHI", "speech_voice"),
    (re.compile(r"\bipss\b"), "IPSS", "urinary"),
    (re.compile(r"\biciq\b"), "ICIQ", "urinary"),
    (re.compile(r"\biief\b"), "IIEF", "sexual_reproductive"),
    (re.compile(r"\bfsfi\b"), "FSFI", "sexual_reproductive"),
    (re.compile(r"quickdash|\bdash\b"), "DASH", "limb_lymphedema"),
    (re.compile(r"\bvfq\b"), "VFQ", "visual"),
    (re.compile(r"gog[- ]?ntx|\btotal neuropathy score\b"), "GOG-Ntx", "neuropathy"),
)


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def extractor_provenance() -> dict[str, str | None]:
    return {
        "extractor_version": "firstpass_lexicon_v0_2_candidate",
        "extractor_sha256": file_sha256(EXTRACTOR_PATH),
        "ontology_version": ONTOLOGY_VERSION,
        "ontology_sha256": file_sha256(ONTOLOGY_PATH),
    }


def normalize_hierarchy(value: str | None) -> str:
    if not value or not str(value).strip():
        return "UNKNOWN"
    key = " ".join(str(value).strip().upper().replace("-", " ").split())
    if key in HIERARCHY_VALUES:
        return key
    if "PRIMARY" in key:
        return "PRIMARY"
    if "SECONDARY" in key:
        return "SECONDARY"
    return "OTHER"


def infer_measurement_mode(text: str) -> str:
    hits = [mode for pattern, mode in MODE_PATTERNS if pattern.search(text)]
    unique = list(dict.fromkeys(hits))
    if len(unique) > 1:
        return "composite"
    if unique:
        return unique[0]
    return "unclear"


def infer_instrument(text: str, domain: str) -> tuple[str | None, str | None]:
    for pattern, normalized, instrument_domain in INSTRUMENTS:
        match = pattern.search(text)
        if match and instrument_domain == domain:
            return match.group(), normalized
    return None, None


def infer_timeframe(text: str) -> str | None:
    match = TIMEFRAME_RE.search(text)
    return match.group(1) if match else None


def load_confirmatory_units() -> dict[str, list[str]]:
    if not GROUPING_PATH.is_file():
        return {domain: [domain] for domain in DOMAINS}
    payload = json.loads(GROUPING_PATH.read_text(encoding="utf-8"))
    return {
        name: list(spec["members"])
        for name, spec in payload.get("analytic_units_confirmatory", {}).items()
    }


def confirmatory_units_measured(records: list[dict]) -> list[str]:
    measured = {row["domain"] for row in records if row["measured"] == 1}
    return [
        name
        for name, members in load_confirmatory_units().items()
        if measured.intersection(members)
    ]


def _evidence_payload(original: str, matches: list[re.Match[str]]) -> list[dict]:
    return [
        {
            "start": match.start(),
            "end": match.end(),
            "text": original[match.start():match.end()],
            "supports": "classification",
        }
        for match in matches
    ]


def extract_domain_records(
    text: str,
    *,
    hierarchy: str | None = None,
    trial_id: str | None = None,
    rule_version: str = "v0_2_candidate",
) -> list[dict]:
    """Return one handshake-shaped row per frozen domain. Disease/cancer is not used."""
    original = text
    lowered = text.lower()
    inc_re, exc_re = compile_rules(rule_version)
    provenance = extractor_provenance()
    highest = normalize_hierarchy(hierarchy)
    rows = []
    for domain in DOMAINS:
        matches = list(inc_re[domain].finditer(lowered))
        trapped = False
        if matches and domain in exc_re:
            cleaned = exc_re[domain].sub(" ", lowered)
            trapped = not bool(inc_re[domain].search(cleaned))
        if trapped:
            classification = "exclusion_or_trap"
            measured = 0
        elif not matches:
            classification = None
            measured = 0
        elif AE_LIST_RE.search(lowered) and not OPERATIONALIZED_RE.search(lowered):
            classification = "mention_only"
            measured = 0
        else:
            classification = "measured_outcome"
            measured = 1
        instrument_raw, instrument_normalized = infer_instrument(lowered, domain) if measured else (None, None)
        evidence = _evidence_payload(original, matches) if matches else []
        rows.append({
            "trial_id": trial_id,
            "domain": domain,
            "measured": measured,
            "classification": classification,
            "highest_hierarchy": highest if measured else None,
            "measurement_mode": infer_measurement_mode(lowered) if measured else None,
            "instrument_raw": instrument_raw,
            "instrument_normalized": instrument_normalized,
            "timeframe_raw": infer_timeframe(original) if measured else None,
            "positive_evidence_count": len(evidence) if measured else 0,
            "evidence_spans": evidence,
            "evidence_refs_json": [
                f"firstpass:{domain}:{item['start']}:{item['end']}" for item in evidence
            ] if measured else [],
            "review_status": FIRSTPASS_REVIEW_STATUS,
            "source_text_sha256": sha256_text(original),
            **provenance,
        })
    return rows


def build_firstpass_extraction(
    outcome_text: str,
    *,
    hierarchy: str | None = None,
    trial_id: str | None = None,
    rule_version: str = "v0_2_candidate",
) -> dict:
    """Torch-free screening payload. Not the frozen outcome file and not NLP-001."""
    records = extract_domain_records(
        outcome_text,
        hierarchy=hierarchy,
        trial_id=trial_id,
        rule_version=rule_version,
    )
    if len(records) != len(DOMAINS):
        raise RuntimeError("Extractor did not return all 14 domains")
    return {
        "trial_id": trial_id,
        "rule_version": rule_version,
        "api_version": FIRSTPASS_API_VERSION,
        "validated_extractor": False,
        "scientific_performance_evaluated": False,
        "source_text_sha256": sha256_text(outcome_text),
        "text_characters": len(outcome_text),
        "domains": records,
        "confirmatory_units_measured": confirmatory_units_measured(records),
        "provenance": extractor_provenance(),
    }


def predicted_domains(text: str, rule_version: str = "v0_1") -> set[str]:
    return set(classify_text(text, rule_version=rule_version))


def refuse_overwrite(path: Path, *, force: bool) -> None:
    if not path.exists():
        return
    if path.name in FROZEN_OUTPUT_NAMES and not force:
        raise FileExistsError(
            f"Refusing to overwrite frozen Project B artifact {path.name}. "
            "Write a new versioned path, or pass --force only after explicit review."
        )
    if not force:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")


def default_output_path(rule_version: str) -> Path:
    if rule_version == "v0_1":
        return DEFAULT_OUT_V01
    return ROOT / f"PROJECTB_EXTRACTOR_FIRSTPASS_TRIAL_DOMAIN_{rule_version}.csv.gz"


def run_extractor(
    src: Path,
    out: Path,
    rule_version: str,
    *,
    force: bool = False,
    evidence_out: Path | None = None,
) -> dict:
    refuse_overwrite(out, force=force)
    if evidence_out is not None:
        refuse_overwrite(evidence_out, force=force)
    if pd is None:
        raise RuntimeError("pandas is required to run the extractor over CSV outcome files")
    if not src.is_file():
        raise FileNotFoundError(f"Outcomes file is missing: {src}")

    inc_re, exc_re = compile_rules(rule_version)
    trial_domains: dict[str, set[str]] = defaultdict(set)
    trial_evidence: dict[str, list[tuple[str, DomainEvidence, str]]] = defaultdict(list)
    n_rows = 0
    usecols = ["source_trial_id", "measure", "description"]
    for chunk in pd.read_csv(src, compression="gzip", usecols=usecols, chunksize=100_000, dtype=str):
        chunk = chunk.fillna("")
        txt = (chunk["measure"] + " . " + chunk["description"]).str.lower()
        for trial_id, text in zip(chunk["source_trial_id"], txt):
            hits = classify_text(text, rule_version=rule_version, inc_re=inc_re, exc_re=exc_re)
            if hits:
                trial_domains[trial_id] |= set(hits)
                source_hash = sha256_text(text)
                for domain, spans in hits.items():
                    for span in spans:
                        trial_evidence[trial_id].append((domain, span, source_hash))
        n_rows += len(chunk)

    all_trials: set[str] = set()
    for chunk in pd.read_csv(src, compression="gzip", usecols=["source_trial_id"], chunksize=200_000, dtype=str):
        all_trials.update(chunk["source_trial_id"].dropna().unique())
    n_trials = len(all_trials)

    with gzip.open(out, "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trial_id", "domain", "measured_firstpass", "rule_version"])
        for trial_id in all_trials:
            hits = trial_domains.get(trial_id, set())
            for domain in DOMAINS:
                writer.writerow([trial_id, domain, 1 if domain in hits else 0, rule_version])

    if evidence_out is not None:
        with gzip.open(evidence_out, "wt", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "trial_id",
                "domain",
                "evidence_start",
                "evidence_end",
                "evidence_text",
                "source_text_sha256",
                "classification",
                "rule_version",
            ])
            for trial_id, records in trial_evidence.items():
                for domain, span, source_hash in records:
                    writer.writerow([
                        trial_id,
                        domain,
                        span.start,
                        span.end,
                        span.matched_text,
                        source_hash,
                        "firstpass_lexicon_hit",
                        rule_version,
                    ])

    prevalence = []
    for domain in DOMAINS:
        count = sum(1 for trial_id in all_trials if domain in trial_domains.get(trial_id, set()))
        prevalence.append({"domain": domain, "trials_measuring": count, "prevalence": count / n_trials if n_trials else 0.0})
    any_domain = sum(1 for trial_id in all_trials if trial_domains.get(trial_id))
    summary = {
        "rule_version": rule_version,
        "n_outcome_rows": n_rows,
        "n_trials": n_trials,
        "n_trials_any_domain": any_domain,
        "output": str(out),
        "evidence_output": str(evidence_out) if evidence_out else None,
        "prevalence": prevalence,
        "validated_extractor": False,
    }
    return summary


def _selftest() -> None:
    cases = [
        ("Change in cognitive function assessed by the Montreal Cognitive Assessment.", {"cognition"}, "v0_1"),
        ("Worst pain by visual analog scale.", {"pain"}, "v0_1"),
        ("Recorded on a visual analog scale.", set(), "v0_1"),
        ("ECOG physical performance status at 6 months.", {"physical_function"}, "v0_1"),
        ("ECOG physical performance status at 6 months.", set(), "v0_2_candidate"),
        ("Karnofsky Performance Status and timed up and go.", {"physical_function"}, "v0_2_candidate"),
        ("EORTC QLQ-C30 cognitive functioning subscale.", {"cognition"}, "v0_1"),
        ("EORTC QLQ-C30 cognitive functioning subscale.", set(), "v0_2_candidate"),
        ("Montreal Cognitive Assessment and EORTC QLQ-C30.", {"cognition"}, "v0_2_candidate"),
        ("Urinary frequency by ICIQ.", {"urinary"}, "v0_1"),
        ("CIPN assessed by the GOG-Ntx subscale.", {"neuropathy"}, "v0_1"),
    ]
    for text, expected, version in cases:
        got = predicted_domains(text.lower(), rule_version=version)
        if got != expected:
            raise AssertionError(f"{version}: {text!r} -> {got} (expected {expected})")
    print("SELFTEST PASS — first-pass classifier (synthetic text only, no real result).")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Project B first-pass 14-domain lexicon extractor")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--evidence-out", type=Path)
    parser.add_argument("--rule-version", choices=sorted(RULE_VERSIONS), default="v0_1")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing non-scientific output")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--summary-json", type=Path)
    args = parser.parse_args(argv)
    if args.selftest:
        _selftest()
        return 0
    out = args.out or default_output_path(args.rule_version)
    if args.rule_version != "v0_1" and out == DEFAULT_OUT_V01:
        raise SystemExit("v0_2_candidate must write a new versioned path, not the frozen v0.1 labels")
    summary = run_extractor(
        args.src,
        out,
        args.rule_version,
        force=args.force,
        evidence_out=args.evidence_out,
    )
    print(f"processed {summary['n_outcome_rows']:,} real outcome rows across {summary['n_trials']:,} trials")
    print(f"wrote first-pass trial×domain labels -> {out}")
    print("\nPer-domain measurement prevalence (REAL, first-pass extractor):")
    print(f"{'domain':22s} {'trials measuring':>16s} {'prevalence':>11s}")
    for row in sorted(summary["prevalence"], key=lambda item: -item["prevalence"]):
        print(f"{row['domain']:22s} {row['trials_measuring']:>16,} {row['prevalence']*100:>10.2f}%")
    n_trials = summary["n_trials"] or 1
    print(
        f"\nTrials measuring >=1 of the 14 functional domains: "
        f"{summary['n_trials_any_domain']:,} ({summary['n_trials_any_domain']/n_trials*100:.1f}%)"
    )
    print("NOTE: first-pass lexicon extractor — needs disjoint design-weighted human validation before confirmatory use.")
    if args.summary_json:
        refuse_overwrite(args.summary_json, force=args.force)
        args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
