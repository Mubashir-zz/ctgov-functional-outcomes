#!/usr/bin/env python3
"""Pan-cancer first-pass 14-domain lexicon API.

This is not the cognitive BERT v7/v8 classifier. It does not take cancer type
as input: disease and intervention cannot be used to infer an outcome label.
It is a research-screening first-pass over registry outcome text and is not
externally validated. Production cognitive v7 is unchanged.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from run_firstpass_multidomain_extractor import (
    CLASSIFICATIONS,
    DOMAINS,
    FIRSTPASS_API_VERSION,
    HIERARCHY_VALUES,
    MEASUREMENT_MODES,
    REVIEW_STATUS_VALUES,
    build_firstpass_extraction,
    extractor_provenance,
)

API_VERSION = FIRSTPASS_API_VERSION
MAX_TEXT_CHARACTERS = 100_000
RULE_VERSION = "v0_2_candidate"


class ExtractionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome_text: str = Field(min_length=1, max_length=MAX_TEXT_CHARACTERS)
    hierarchy: str | None = Field(default=None, max_length=64)
    trial_id: str | None = Field(default=None, max_length=128)

    @field_validator("outcome_text")
    @classmethod
    def reject_whitespace_only(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("outcome_text cannot be empty")
        return value


class DomainRecord(BaseModel):
    trial_id: str | None
    domain: str
    measured: int
    classification: str | None
    highest_hierarchy: str | None
    measurement_mode: str | None
    instrument_raw: str | None
    instrument_normalized: str | None
    timeframe_raw: str | None
    positive_evidence_count: int
    evidence_spans: list[dict[str, Any]]
    evidence_refs_json: list[str]
    review_status: Literal["AUTO_FINAL", "HUMAN_CONFIRMED", "HUMAN_OVERRULED", "UNRESOLVED"]
    source_text_sha256: str
    extractor_version: str
    extractor_sha256: str | None
    ontology_version: str
    ontology_sha256: str | None


class ExtractionResponse(BaseModel):
    trial_id: str | None
    rule_version: Literal["v0_2_candidate"]
    api_version: str
    validated_extractor: bool
    scientific_performance_evaluated: bool
    source_text_sha256: str
    text_characters: int
    domains: list[DomainRecord]
    confirmatory_units_measured: list[str]
    provenance: dict[str, str | None]


app = FastAPI(
    title="Project B pan-cancer first-pass extractor",
    description=(
        "Lexicon first-pass over 14 patient-centered domains. "
        "Not clinical decision support. Not the cognitive BERT classifier."
    ),
    version=API_VERSION,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "product": "pan-cancer-firstpass", "api_version": API_VERSION}


@app.get("/about")
def about() -> dict:
    return {
        "purpose": "Research screening of registry outcome text for 14 pan-cancer functional domains.",
        "deployment_status": "first-pass lexicon candidate; not externally validated",
        "not_the_cognitive_bert_classifier": True,
        "cancer_type_not_used": True,
        "validated_extractor": False,
        "domains": list(DOMAINS),
        "measurement_modes": list(MEASUREMENT_MODES),
        "classifications": list(CLASSIFICATIONS),
        "hierarchies": list(HIERARCHY_VALUES),
        "review_status_values": list(REVIEW_STATUS_VALUES),
        "rule_version": RULE_VERSION,
        "api_version": API_VERSION,
        **extractor_provenance(),
    }


@app.post("/extract", response_model=ExtractionResponse)
def extract(req: ExtractionRequest) -> ExtractionResponse:
    payload = build_firstpass_extraction(
        req.outcome_text,
        hierarchy=req.hierarchy,
        trial_id=req.trial_id,
        rule_version=RULE_VERSION,
    )
    if len(payload["domains"]) != 14:
        raise HTTPException(status_code=500, detail="Extractor did not return all 14 domains")
    return ExtractionResponse.model_validate(payload)
