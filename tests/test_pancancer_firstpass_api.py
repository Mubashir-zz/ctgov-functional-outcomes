#!/usr/bin/env python3
"""Pan-cancer first-pass ontology records and lexicon API tests.

Synthetic registry-style text only. Not NLP-001 sensitivity/specificity evidence.
"""

from __future__ import annotations

import inspect
import json
import unittest
from pathlib import Path

from run_firstpass_multidomain_extractor import (
    DOMAINS,
    REVIEW_STATUS_VALUES,
    build_firstpass_extraction,
    extract_domain_records,
    load_confirmatory_units,
    predicted_domains,
)


HANDSHAKE_SCHEMA = (
    Path(__file__).resolve().parent.parent
    / "PROJECTB_CODEX_CLAUDE_ESTIMATOR_HANDSHAKE_SCHEMA_v1.json"
)
GROUPING = (
    Path(__file__).resolve().parent.parent
    / "analysis"
    / "PROJECTB_DOMAIN_GROUPING_v1_2026-08-27.json"
)
_MISSING = "frozen interface artefact not part of the current analytic release"

class PanCancerOntologyRecordTests(unittest.TestCase):
    def test_returns_all_fourteen_domains(self) -> None:
        rows = extract_domain_records("Overall survival.")
        self.assertEqual([row["domain"] for row in rows], list(DOMAINS))
        self.assertTrue(all(row["measured"] == 0 for row in rows))

    def test_does_not_infer_cognition_from_a_cancer_name(self) -> None:
        rows = extract_domain_records("Interventional trial in metastatic pancreatic adenocarcinoma.")
        cognition = next(row for row in rows if row["domain"] == "cognition")
        self.assertEqual(cognition["measured"], 0)
        self.assertIsNone(cognition["classification"])

    def test_moca_is_measured_cognition_with_instrument_and_span(self) -> None:
        text = "Change in cognitive function assessed by the Montreal Cognitive Assessment at 6 months."
        rows = extract_domain_records(text, hierarchy="PRIMARY")
        cognition = next(row for row in rows if row["domain"] == "cognition")
        self.assertEqual(cognition["measured"], 1)
        self.assertEqual(cognition["classification"], "measured_outcome")
        self.assertEqual(cognition["instrument_normalized"], "MoCA")
        self.assertEqual(cognition["highest_hierarchy"], "PRIMARY")
        self.assertEqual(cognition["measurement_mode"], "objective_performance")
        self.assertTrue(cognition["timeframe_raw"])
        span = cognition["evidence_spans"][0]
        self.assertEqual(text[span["start"]:span["end"]], span["text"])
        self.assertEqual(cognition["review_status"], "UNRESOLVED")
        self.assertIn(cognition["review_status"], REVIEW_STATUS_VALUES)
        self.assertTrue(cognition["evidence_refs_json"])

    def test_v0_2_qlq_c30_is_exclusion_or_trap_not_measured(self) -> None:
        text = "EORTC QLQ-C30 cognitive functioning subscale."
        self.assertEqual(predicted_domains(text.lower(), "v0_2_candidate"), set())
        cognition = next(
            row for row in extract_domain_records(text) if row["domain"] == "cognition"
        )
        self.assertEqual(cognition["measured"], 0)
        self.assertEqual(cognition["classification"], "exclusion_or_trap")
        self.assertGreater(len(cognition["evidence_spans"]), 0)

    def test_ae_list_fatigue_is_mention_only(self) -> None:
        text = "Adverse events including fatigue will be recorded."
        fatigue = next(row for row in extract_domain_records(text) if row["domain"] == "fatigue")
        self.assertEqual(fatigue["classification"], "mention_only")
        self.assertEqual(fatigue["measured"], 0)

    @unittest.skipUnless(GROUPING.exists(), _MISSING)
    def test_confirmatory_units_follow_domain_grouping(self) -> None:
        units = load_confirmatory_units()
        self.assertIn("physical_musculoskeletal_function", units)
        self.assertEqual(
            set(units["physical_musculoskeletal_function"]),
            {"physical_function", "limb_lymphedema"},
        )
        self.assertEqual(set(units), {
            "cognition",
            "neuropathy",
            "fatigue",
            "pain",
            "respiratory_function",
            "bowel",
            "urinary",
            "sexual_reproductive",
            "physical_musculoskeletal_function",
            "sensory_communication_function",
        })

    def test_payload_builder_does_not_accept_cancer_type(self) -> None:
        self.assertNotIn("cancer_type", inspect.signature(build_firstpass_extraction).parameters)
        payload = build_firstpass_extraction(
            "CIPN assessed by the GOG-Ntx subscale.",
            hierarchy="secondary",
            trial_id="SYNTH-PAN-001",
        )
        self.assertEqual(len(payload["domains"]), 14)
        self.assertFalse(payload["validated_extractor"])
        self.assertNotIn("cancer_type", payload)
        neuropathy = next(item for item in payload["domains"] if item["domain"] == "neuropathy")
        self.assertEqual(neuropathy["measured"], 1)
        self.assertEqual(neuropathy["instrument_normalized"], "GOG-Ntx")
        self.assertEqual(neuropathy["highest_hierarchy"], "SECONDARY")
        self.assertEqual(neuropathy["review_status"], "UNRESOLVED")
        self.assertIn("neuropathy", payload["confirmatory_units_measured"])

    @unittest.skipUnless(HANDSHAKE_SCHEMA.exists(), _MISSING)
    def test_handshake_outcome_columns_are_present(self) -> None:
        schema = json.loads(
            Path(__file__).resolve().parents[1]
            .joinpath("PROJECTB_CODEX_CLAUDE_ESTIMATOR_HANDSHAKE_SCHEMA_v1.json")
            .read_text(encoding="utf-8")
        )
        row = extract_domain_records("CIPN assessed by the GOG-Ntx subscale.", trial_id="SYNTH-PAN-001")[0]
        required = set(schema["files"]["outcome"]["required_columns"])
        self.assertTrue(required.issubset(row.keys()))
        self.assertEqual(row["review_status"], "UNRESOLVED")


class PanCancerFirstpassApiTests(unittest.TestCase):
    def test_extract_endpoint_does_not_use_cancer_type(self) -> None:
        try:
            from fastapi.testclient import TestClient
            from projectb_pancancer_firstpass_api import app
        except ImportError:
            self.skipTest("fastapi is not installed")
        client = TestClient(app)
        about = client.get("/about").json()
        self.assertTrue(about["not_the_cognitive_bert_classifier"])
        self.assertTrue(about["cancer_type_not_used"])
        self.assertIs(about["validated_extractor"], False)
        response = client.post("/extract", json={
            "outcome_text": "CIPN assessed by the GOG-Ntx subscale.",
            "hierarchy": "secondary",
            "trial_id": "SYNTH-PAN-001",
        })
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(len(body["domains"]), 14)
        self.assertFalse(body["validated_extractor"])
        neuropathy = next(item for item in body["domains"] if item["domain"] == "neuropathy")
        self.assertEqual(neuropathy["measured"], 1)
        self.assertEqual(neuropathy["instrument_normalized"], "GOG-Ntx")
        self.assertNotIn("cancer_type", body)

    def test_rejects_empty_text_without_echoing_payload_keys_as_success(self) -> None:
        try:
            from fastapi.testclient import TestClient
            from projectb_pancancer_firstpass_api import app
        except ImportError:
            self.skipTest("fastapi is not installed")
        client = TestClient(app)
        response = client.post("/extract", json={"outcome_text": "   "})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main(verbosity=2)
