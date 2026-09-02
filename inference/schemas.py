from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RasterSummary(BaseModel):
    name: str
    format: str
    bytes: int
    width: int | None = None
    height: int | None = None
    bands: int | None = None
    dtype: str | None = None
    crs: str | None = None
    modality: str = "AMBIGUOUS"
    modality_confidence: float = 0.0
    modality_note: str | None = None
    status: str = "PASSED"
    note: str | None = None


class Evidence(BaseModel):
    source: str = "uploaded_pixels"
    changed_area_percent: float | None = None
    bbox_pixels: list[int] | None = None
    mask_png_base64: str | None = None
    agreement: float | None = None


class AnalysisResponse(BaseModel):
    runtime: str = "real"
    model_version: str
    task: str
    claim: str
    where: str
    magnitude: str
    sensorCase: str
    limit: str
    confidence: int = Field(ge=0, le=100)
    decision: str
    ai_summary: str
    inputSummary: list[RasterSummary]
    evidence: Evidence | None = None
    trace: list[list[str]]
    diagnostics: dict[str, Any] = Field(default_factory=dict)
