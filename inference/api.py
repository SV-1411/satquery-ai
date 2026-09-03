from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .models.registry import ModelRegistry
from .orchestration.executor import Executor
from .orchestration.router import route_query
from .preprocessing.geospatial_checks import pair_check
from .preprocessing.raster_loader import load_raster
from .schemas import AnalysisResponse, RasterSummary

app = FastAPI(title="SatQuery AI Inference", version="0.1.0")
registry = ModelRegistry()
executor = Executor(registry)


def summary(raster, payload_size: int) -> RasterSummary:
    return RasterSummary(name=raster.name, format="GEOTIFF" if raster.is_geospatial else raster.name.rsplit(".", 1)[-1].upper(), bytes=payload_size, width=raster.width, height=raster.height, bands=raster.bands, dtype=raster.dtype, crs=raster.crs, modality=raster.modality, modality_confidence=raster.modality_confidence, modality_note=raster.modality_note)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "runtime": "real" if registry.ready else "blocked", "device": str(registry.device), "gpu": registry.device.type == "cuda", "model_ready": registry.ready, "model_version": registry.version, "checkpoint": str(registry.checkpoint), "ollama_model": executor.vlm.model, "ollama_ready": executor.vlm.available(), "load_error": registry.load_error}


@app.post("/analyse", response_model=AnalysisResponse)
async def analyse(query: str = Form(...), files: list[UploadFile] = File(...), modalityA: str = Form("AUTO"), modalityB: str = Form("AUTO"), modalityC: str = Form("AUTO"), modalityD: str = Form("AUTO")) -> AnalysisResponse:
    if not query.strip() or not 1 <= len(files) <= 4:
        raise HTTPException(status_code=422, detail="Send a query and between one and four raster files.")
    payloads = [await item.read() for item in files]
    try:
        hints = [modalityA, modalityB, modalityC, modalityD]
        rasters = [load_raster(item.filename or f"observation-{i}", payload, hints[i]) for i, (item, payload) in enumerate(zip(files, payloads))]
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    task = route_query(query, len(rasters))
    modalities = {raster.modality for raster in rasters}
    if task == "OPTICAL_SAR_FUSION" and not {"OPTICAL", "SAR"}.issubset(modalities):
        raise HTTPException(status_code=422, detail="Optical/SAR fusion requires at least one OPTICAL and one SAR upload. Use the modality selectors or rename the files.")
    if task == "BI_TEMPORAL_CHANGE_VQA" and len(rasters) >= 2:
        first, last = rasters[0], rasters[-1]
        compatible, message = pair_check(first, last)
        if not compatible:
            raise HTTPException(status_code=422, detail=message)
    elif task == "OPTICAL_SAR_FUSION":
        optical = next(raster for raster in rasters if raster.modality == "OPTICAL")
        sar = next(raster for raster in rasters if raster.modality == "SAR")
        compatible, message = pair_check(optical, sar)
        if not compatible:
            raise HTTPException(status_code=422, detail=message)
    trace = [["01", "RASTER_VALIDATOR", "PASSED"], ["02", "QUERY_ROUTER", task]]
    if task == "BI_TEMPORAL_CHANGE_VQA" and len(rasters) >= 2:
        result = executor.change(rasters[0], rasters[-1], query)
        tool = "PIXEL_CHANGE_ANALYST"
    elif task == "OPTICAL_SAR_FUSION" and len(rasters) >= 2:
        optical = next(raster for raster in rasters if raster.modality == "OPTICAL")
        sar = next(raster for raster in rasters if raster.modality == "SAR")
        result = executor.fusion(optical, sar, query)
        tool = "OPTICAL_SAR_FUSION_BASELINE"
    elif task == "MULTI_OBSERVATION_SYNTHESIS":
        result = executor.multi_summary(rasters, query)
        tool = "MULTI_OBSERVATION_CATALOG"
    else:
        result = executor.single(rasters[0], task, query)
        tool = "SATQUERY_PIXEL_MODEL" if registry.ready else "SPECTRAL_BASELINE"
    summary_model = result.get("diagnostics", {}).get("summary_model", "grounded-template-fallback")
    trace.extend([["03", "MODALITY_CLASSIFIER", "/".join(raster.modality for raster in rasters)], ["04", tool, "COMPLETE"], ["05", "EVIDENCE_BUILDER", "COMPLETE"], ["06", "OLLAMA_QWEN2.5-VL", summary_model]])
    return AnalysisResponse(runtime="real", model_version=registry.version, task=task, inputSummary=[summary(raster, len(payload)) for raster, payload in zip(rasters, payloads)], trace=trace, **result)
