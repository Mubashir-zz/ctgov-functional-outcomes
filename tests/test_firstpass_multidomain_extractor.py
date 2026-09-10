#!/usr/bin/env python3
"""Unit tests for the first-pass 14-domain lexicon extractor.

Synthetic registry-style text only. These tests are not sensitivity/specificity
evidence and must not be cited as NLP-001 validation.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from run_firstpass_multidomain_extractor import (
    DEFAULT_OUT_V01,
    FROZEN_OUTPUT_NAMES,
    classify_text,
    default_output_path,
    predicted_domains,
    refuse_overwrite,
    sha256_text,
)


class FirstPassExtractorTests(unittest.TestCase):
    def test_importing_the_module_does_not_require_the_outcomes_file(self) -> None:
        self.assertTrue(DEFAULT_OUT_V01.name in FROZEN_OUTPUT_NAMES)

    def test_v0_1_recovers_named_cognitive_instrument(self) -> None:
        text = "change in cognitive function assessed by the montreal cognitive assessment."
        hits = classify_text(text, rule_version="v0_1")
        self.assertIn("cognition", hits)
        span = hits["cognition"][0]
        self.assertEqual(text[span.start:span.end], span.matched_text)
        self.assertTrue(span.matched_text)

    def test_v0_1_visual_analog_is_not_visual_function(self) -> None:
        self.assertEqual(predicted_domains("recorded on a visual analog scale.", "v0_1"), set())
        self.assertEqual(predicted_domains("worst pain by visual analog scale.", "v0_1"), {"pain"})

    def test_v0_1_does_not_treat_ecog_alone_as_physical_function(self) -> None:
        self.assertEqual(predicted_domains("ecog performance status at 6 months.", "v0_1"), set())

    def test_v0_1_fires_physical_performance_status_phrasing(self) -> None:
        self.assertEqual(
            predicted_domains("ecog physical performance status at 6 months.", "v0_1"),
            {"physical_function"},
        )

    def test_v0_2_drops_ecog_physical_performance_status(self) -> None:
        self.assertEqual(
            predicted_domains("ecog physical performance status at 6 months.", "v0_2_candidate"),
            set(),
        )

    def test_v0_2_keeps_independent_physical_function_next_to_karnofsky(self) -> None:
        self.assertEqual(
            predicted_domains("karnofsky performance status and timed up and go.", "v0_2_candidate"),
            {"physical_function"},
        )

    def test_v0_2_qlq_c30_subscale_is_not_cognition_unless_independently_named(self) -> None:
        qlq = "eortc qlq-c30 cognitive functioning subscale."
        self.assertEqual(predicted_domains(qlq, "v0_1"), {"cognition"})
        self.assertEqual(predicted_domains(qlq, "v0_2_candidate"), set())
        both = "montreal cognitive assessment and eortc qlq-c30."
        self.assertEqual(predicted_domains(both, "v0_2_candidate"), {"cognition"})

    def test_neuropathy_and_hearing_inclusions(self) -> None:
        self.assertEqual(
            predicted_domains("cipn assessed by the gog-ntx subscale.", "v0_1"),
            {"neuropathy"},
        )
        self.assertEqual(
            predicted_domains("hearing threshold by audiometry.", "v0_1"),
            {"hearing"},
        )

    def test_source_text_hash_is_stable(self) -> None:
        self.assertEqual(len(sha256_text("montreal cognitive assessment")), 64)

    def test_v0_2_default_output_is_not_the_frozen_v0_1_file(self) -> None:
        self.assertNotEqual(default_output_path("v0_2_candidate"), DEFAULT_OUT_V01)
        self.assertTrue(default_output_path("v0_2_candidate").name.startswith("PROJECTB_EXTRACTOR_FIRSTPASS"))

    def test_refuse_overwrite_protects_frozen_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            frozen = Path(temporary) / "PROJECTB_EXTRACTOR_FIRSTPASS_TRIAL_DOMAIN_v0_1_2026-08-27.csv.gz"
            frozen.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                refuse_overwrite(frozen, force=False)
            refuse_overwrite(frozen, force=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
