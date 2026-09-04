from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
import torch
from PIL import Image

from ..models.registry import ModelRegistry
from .ollama_vlm import OllamaVisionSummarizer
from ..preprocessing.raster_loader import RasterData, model_channels


def _mask_png(mask: np.ndarray, target_shape: tuple[int, int] | None = None) -> str:
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    if target_shape and image.size != (target_shape[1], target_shape[0]):
        image = image.resize((target_shape[1], target_shape[0]), Image.Resampling.NEAREST)
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _rgb_png(rgb: np.ndarray) -> str:
    image = Image.fromarray((np.clip(rgb, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8), mode="RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _display_source(raster: RasterData) -> np.ndarray:
    """Select display bands without changing the uploaded image aspect ratio."""
    if raster.modality == "SAR":
        return np.repeat(raster.data[:1], 3, axis=0)
    elif raster.data.shape[0] >= 4:
        # BigEarthNet Sentinel-2 uses B01, B02, B03, B04 order at its front.
        return raster.data[[3, 2, 1]]
    source = raster.data[:3]
    if source.shape[0] == 1:
        return np.repeat(source, 3, axis=0)
    if source.shape[0] == 2:
        return np.concatenate([source, source[:1]], axis=0)
    return source


def _raster_preview(raster: RasterData, size: int = 256) -> np.ndarray:
    """Create the fixed-size grid used by the pixel model."""
    return model_channels(_display_source(raster), size=size)[:3]


def _display_rgb(raster: RasterData, max_dimension: int = 1024) -> np.ndarray:
    """Create a clear, aspect-preserving display image for the user interface."""
    source = _display_source(raster)
    height, width = source.shape[1:]
    scale = min(1.0, max_dimension / max(height, width))
    if scale >= 1.0:
        return source
    new_height, new_width = max(1, round(height * scale)), max(1, round(width * scale))
    tensor = torch.from_numpy(source).unsqueeze(0)
    return torch.nn.functional.interpolate(tensor, size=(new_height, new_width), mode="bilinear", align_corners=False).squeeze(0).numpy()


def _preview_png(raster: RasterData) -> str:
    return _rgb_png(_display_rgb(raster))


def _preview_grid(rasters: list[RasterData]) -> str:
    previews = [_raster_preview(raster, size=192) for raster in rasters[:4]]
    if not previews:
        return ""
    columns = 2 if len(previews) > 1 else 1
    rows = int(np.ceil(len(previews) / columns))
    grid = np.zeros((3, rows * 192, columns * 192), dtype=np.float32)
    for index, preview in enumerate(previews):
        y, x = divmod(index, columns)
        grid[:, y * 192:(y + 1) * 192, x * 192:(x + 1) * 192] = preview
    return _rgb_png(grid)


def _semantic_overlay(masks: list[tuple[np.ndarray, tuple[int, int, int]]], target_shape: tuple[int, int] | None = None) -> str:
    """Paint explainable evidence classes over the same 256px map grid."""
    if not masks:
        return ""
    height, width = target_shape or masks[0][0].shape
    canvas = np.zeros((3, height, width), dtype=np.float32)
    for mask, color in masks:
        if mask.shape != (height, width):
            mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255, mode="L").resize((width, height), Image.Resampling.NEAREST)) > 0
        for channel, value in enumerate(color):
            canvas[channel][mask] = value / 255.0
    return _rgb_png(canvas)


def _score_layer(score: np.ndarray, color: tuple[int, int, int]) -> str:
    low, high = np.percentile(score, [2, 98])
    normalized = np.clip((score - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    canvas = np.stack([normalized * (value / 255.0) for value in color], axis=0)
    return _rgb_png(canvas)


def _optical_signal_masks(channels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return explainable optical signals, including dark coastal water."""
    red, green, blue = channels
    brightness = channels.mean(axis=0)
    blue_advantage = blue - np.maximum(red, green)
    # Open water in the included coastal sample is dark, so requiring a fixed
    # positive blue score incorrectly rejected most of the ocean. Use relative
    # colour dominance plus a brightness gate; this remains a signal, not a
    # supervised land-cover segmentation.
    water = (blue_advantage >= np.percentile(blue_advantage, 70)) & (brightness <= np.percentile(brightness, 65))
    water_score = blue_advantage
    vegetation_score = green - red
    vegetation = (vegetation_score >= max(float(np.percentile(vegetation_score, 70)), 0.015)) & ~water
    built_score = brightness - np.abs(red - green) - 0.5 * np.abs(red - blue)
    # >= matters for normalized rasters whose upper percentile can equal 1.0.
    built = (built_score >= np.percentile(built_score, 72)) & ~water & ~vegetation
    return brightness, water_score, water, vegetation_score, vegetation, built_score, built


def _optical_scene_description(water_fraction: float, vegetation_fraction: float, built_fraction: float) -> str:
    if water_fraction >= 0.18 and built_fraction >= 0.18:
        return "mixed coastal water and urban land"
    if water_fraction >= 0.25:
        return "water-dominant or coastal land"
    if built_fraction >= 0.28:
        return "built-up or urban land"
    if vegetation_fraction >= 0.18:
        return "vegetated or agricultural land"
    return "mixed land cover"


def _single_semantic_evidence(raster: RasterData) -> tuple[str, list[str], list[dict[str, str]]]:
    """Return visual layers that are valid for the uploaded sensor capabilities."""
    channels = _display_rgb(raster, max_dimension=1024)
    brightness = channels.mean(axis=0)
    layers: list[dict[str, str]] = [{
        "id": "natural-colour", "title": "UPLOADED IMAGE", "status": "AVAILABLE",
        "meaning": "Display-normalized view of the uploaded pixels.", "png_base64": _rgb_png(channels),
    }]
    if raster.modality == "SAR":
        low_backscatter = brightness < np.percentile(brightness, 20)
        layers.extend([
            {"id": "backscatter", "title": "SAR BACKSCATTER", "status": "AVAILABLE", "meaning": "Relative radar backscatter, not a land-cover label.", "png_base64": _score_layer(brightness, (0, 220, 255))},
            {"id": "water-signal", "title": "LOW-BACKSCATTER SIGNAL", "status": "AVAILABLE", "meaning": "Low radar return can be water-like but needs analyst review.", "png_base64": _semantic_overlay([(low_backscatter, (0, 220, 255))])},
        ])
        overlay, legend = _semantic_overlay([(low_backscatter, (0, 220, 255))]), ["CYAN = LOW-BACKSCATTER SIGNAL"]
    else:
        _, water_score, water, vegetation_score, vegetation, built_score, built = _optical_signal_masks(channels)
        layers.extend([
            {"id": "water-signal", "title": "WATER-CONSISTENT SIGNAL", "status": "AVAILABLE", "meaning": "Blue-dominant spectral signal; it is not a verified water boundary.", "png_base64": _score_layer(water_score, (30, 130, 255))},
            {"id": "vegetation", "title": "VEGETATION SIGNAL", "status": "AVAILABLE", "meaning": "Green-over-red signal, shown as a relative index.", "png_base64": _score_layer(vegetation_score, (50, 220, 90))},
            {"id": "built-signal", "title": "BUILT-UP-LIKE SIGNAL", "status": "AVAILABLE", "meaning": "Bright, neutral-colour texture signal; it needs a land-cover model for confirmation.", "png_base64": _score_layer(built_score, (255, 170, 35))},
            {"id": "surface-brightness", "title": "SURFACE BRIGHTNESS", "status": "AVAILABLE", "meaning": "Image brightness only. This is not a temperature measurement.", "png_base64": _score_layer(brightness, (255, 255, 255))},
        ])
        overlay = _semantic_overlay([(vegetation, (50, 220, 90)), (built, (255, 170, 35)), (water, (30, 130, 255))])
        legend = ["BLUE = WATER-CONSISTENT", "GREEN = VEGETATION SIGNAL", "ORANGE = BUILT-UP-LIKE SIGNAL"]
    layers.extend([
        {"id": "temperature", "title": "SURFACE TEMPERATURE", "status": "NOT AVAILABLE", "meaning": "Requires calibrated thermal-infrared bands and acquisition metadata; RGB/SAR pixels cannot supply true temperature."},
        {"id": "atmosphere", "title": "AIR / ATMOSPHERE", "status": "NOT AVAILABLE", "meaning": "Requires atmospheric products or dedicated retrieval bands; it is not inferred from one uploaded image."},
    ])
    return overlay, legend, layers


def _bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _heuristic_water(channels: np.ndarray) -> np.ndarray:
    return _optical_signal_masks(channels)[2]


class Executor:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.vlm = OllamaVisionSummarizer()

    def _with_vlm_summary(self, rasters: list[RasterData], query: str, result: dict) -> dict:
        generated, summary_model = self.vlm.summarize(rasters, query, {
            "claim": result["claim"],
            "where": result["where"],
            "magnitude": result["magnitude"],
            "sensorCase": result["sensorCase"],
            "confidence": result["confidence"],
            "decision": result["decision"],
            "limit": result["limit"],
            "diagnostics": result.get("diagnostics", {}),
        })
        if generated:
            # Keep the on-screen conclusion numerically and semantically tied to
            # measured fields. Qwen is retained as an auditable local-language
            # draft, but cannot replace the validated summary with a hallucinated
            # interpretation of the rendered image.
            result.setdefault("diagnostics", {})["qwen_summary_draft"] = generated
        result.setdefault("diagnostics", {})["summary_model"] = summary_model
        return result

    @torch.inference_mode()
    def _model_mask(self, raster: RasterData) -> tuple[np.ndarray | None, float, int | None, float]:
        if not self.registry.ready:
            return None, 0.0, None, 0.0
        array = model_channels(raster.data)
        tensor = torch.from_numpy(array).unsqueeze(0).to(self.registry.device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.registry.device.type == "cuda"):
            logits, scene_logits = self.registry.model(tensor)
        probabilities = logits.softmax(dim=1)[0]
        confidence, labels = probabilities.max(dim=0)
        mask = labels.cpu().numpy() == 1
        scene_probabilities = scene_logits.softmax(dim=1)[0]
        scene_confidence, scene_label = scene_probabilities.max(dim=0)
        return mask, float(confidence.mean().item()), int(scene_label.item()), float(scene_confidence.item())

    def single(self, raster: RasterData, task: str, query: str) -> dict:
        channels = _raster_preview(raster)
        model_mask, model_confidence, scene_label, scene_probability = self._model_mask(raster)
        spectral_mask = _heuristic_water(channels) if raster.modality == "OPTICAL" else channels.mean(axis=0) < np.percentile(channels.mean(axis=0), 20)
        # The trained model supplies scene context; the evidence mask must still
        # be supported by the sensor-specific pixel signal.
        mask = spectral_mask
        if model_mask is not None and scene_label == 1:
            candidate = model_mask & spectral_mask
            if candidate.mean() >= 0.005:
                mask = candidate
        water_percent = float(mask.mean() * 100)
        bbox = _bbox(mask)
        scene_names = {0: "vegetated or agricultural land", 1: "water", 2: "built-up land"}
        model_scene = scene_names.get(scene_label, "unknown") if scene_label is not None else "unavailable"
        if raster.modality == "OPTICAL":
            _, _, water_signal, _, vegetation_signal, _, built_signal = _optical_signal_masks(channels)
            water_fraction = float(water_signal.mean())
            vegetation_fraction = float(vegetation_signal.mean())
            built_fraction = float(built_signal.mean())
            scene = _optical_scene_description(water_fraction, vegetation_fraction, built_fraction)
            model_conflict = scene_label == 0 and scene not in {"vegetated or agricultural land", "mixed land cover"}
        else:
            water_fraction = water_percent / 100.0
            vegetation_fraction = 0.0
            built_fraction = 0.0
            scene = model_scene if model_scene != "unknown" else ("water-consistent SAR signal" if water_percent > 2 else "mixed SAR scene")
            model_conflict = False
        if task == "TEXT_GUIDED_GROUNDING":
            claim = "WATER-CONSISTENT PIXELS WERE LOCALIZED FROM THE UPLOADED OBSERVATION."
            where = f"PIXEL BOUNDS {bbox}" if bbox else "NO STABLE WATER REGION RELEASED"
        elif task == "SINGLE_IMAGE_CAPTIONING":
            claim = f"THE SCENE CONTAINS {scene.upper()} WITH {water_percent:.1f}% WATER-CONSISTENT AREA."
            where = f"WATER-CONSISTENT REGION / PIXEL BOUNDS {bbox}" if bbox else "NO WATER REGION RELEASED"
        else:
            claim = f"THE UPLOADED OBSERVATION IS CONSISTENT WITH {scene.upper()}."
            where = f"WATER-CONSISTENT REGION / PIXEL BOUNDS {bbox}" if bbox else "NO STABLE REGION RELEASED"
        evidence_support = max(water_fraction, vegetation_fraction, built_fraction)
        confidence = int(np.clip(48 + evidence_support * 24 + (0 if model_conflict else scene_probability * 8), 0, 78)) if self.registry.ready else 48
        if raster.modality == "OPTICAL":
            conflict_note = f" The learned scene head separately signaled {model_scene}, but it disagreed with the mixed-scene evidence, so that head was not used as the headline classification." if model_conflict else ""
            ai_summary = f"AI SUMMARY: Visible pixel signals indicate {scene}: approximately {water_fraction * 100:.1f}% water-consistent, {built_fraction * 100:.1f}% built-up-like, and {vegetation_fraction * 100:.1f}% vegetation signal. These are relative evidence masks, not verified land-cover boundaries or a field survey.{conflict_note}"
        else:
            ai_summary = f"AI SUMMARY: The uploaded SAR observation is most consistent with {scene}. Its low-backscatter signal covers {water_percent:.1f}% of the uploaded pixels. This is a relative radar signal, not a verified land-cover boundary or field survey."
        display_shape = _display_rgb(raster).shape[1:]
        semantic_png, legend, layers = _single_semantic_evidence(raster)
        return self._with_vlm_summary([raster], query, {
            "claim": claim,
            "where": where,
            "magnitude": f"{water_percent:.1f}% WATER / {built_fraction * 100:.1f}% BUILT-LIKE / {vegetation_fraction * 100:.1f}% VEGETATION SIGNAL" if raster.modality == "OPTICAL" else f"{water_percent:.1f}% LOW-BACKSCATTER PIXELS",
            "sensorCase": "UPLOADED PIXELS: MODEL + SPECTRAL BASELINE AGREEMENT" if self.registry.ready else "UPLOADED PIXELS: SPECTRAL BASELINE ONLY",
            "limit": "Pixel evidence is not a substitute for analyst review; verify georegistration and class labels before operational use.",
            "confidence": confidence,
            "decision": "TRIAGE-READY / HUMAN CONFIRMATION ADVISED" if confidence >= 60 else "LOW CONFIDENCE / HUMAN REVIEW REQUIRED",
            "ai_summary": ai_summary,
            "evidence": {"source": "uploaded_pixels", "bbox_pixels": bbox, "mask_png_base64": _mask_png(mask, display_shape), "visual_png_base64": _preview_png(raster), "semantic_png_base64": semantic_png, "legend": legend, "analysis_path": f"{task} -> SATQUERY PIXEL MODEL + SPECTRAL SIGNAL -> LOCALIZED EVIDENCE", "layers": layers, "map_label": "UPLOADED IMAGE + QUERY EVIDENCE"},
            "diagnostics": {"model_confidence": round(model_confidence, 4), "model_scene": model_scene, "model_scene_probability": round(scene_probability, 4), "model_scene_conflict": model_conflict, "water_percent": round(water_percent, 3), "vegetation_percent": round(vegetation_fraction * 100, 3), "built_percent": round(built_fraction * 100, 3), "confidence_basis": "relative sensor-signal support; not validated accuracy"},
        })

    def change(self, before: RasterData, after: RasterData, query: str) -> dict:
        first = model_channels(before.data)
        second = model_channels(after.data)
        difference = np.mean(np.abs(second - first), axis=0)
        threshold = float(np.percentile(difference, 90))
        mask = difference >= max(threshold, 0.08)
        changed_percent = float(mask.mean() * 100)
        confidence = int(np.clip(52 + min(changed_percent, 30), 0, 85))
        ai_summary = f"AI SUMMARY: Comparing the first and last observations on the common grid, {changed_percent:.1f}% of pixels changed beyond the {threshold:.3f} baseline. This detects visual/spectral difference; it does not by itself prove flooding, construction, or another cause."
        display_shape = _display_rgb(after).shape[1:]
        return self._with_vlm_summary([before, after], query, {
            "claim": "PIXEL-LEVEL CHANGE WAS DETECTED BETWEEN THE TWO UPLOADED OBSERVATIONS.",
            "where": f"CHANGE BOUNDS {(_bbox(mask) or 'NOT STABLE')} IN THE COMMON PIXEL GRID",
            "magnitude": f"{changed_percent:.1f}% OF THE COMMON GRID FLAGGED / THRESHOLD {threshold:.3f}",
            "sensorCase": "BEFORE/AFTER SPECTRAL DIFFERENCE SUPPORTS THE CHANGE SIGNAL",
            "limit": "This baseline detects appearance change, not causal land-use change; validate acquisition dates, clouds, and registration.",
            "confidence": confidence,
            "decision": "TRIAGE-READY / HUMAN CONFIRMATION ADVISED" if confidence >= 60 else "LOW CONFIDENCE / HUMAN REVIEW REQUIRED",
            "ai_summary": ai_summary,
            "evidence": {"source": "uploaded_pixels", "changed_area_percent": changed_percent, "bbox_pixels": _bbox(mask), "mask_png_base64": _mask_png(mask, display_shape), "visual_png_base64": _preview_png(after), "before_png_base64": _preview_png(before), "after_png_base64": _preview_png(after), "semantic_png_base64": _semantic_overlay([(mask, (255, 50, 35))], display_shape), "legend": ["RED = PIXELS THAT CHANGED BETWEEN UPLOADS"], "analysis_path": "BI-TEMPORAL CHANGE -> COMMON-GRID PIXEL DIFFERENCE -> CHANGE EVIDENCE", "layers": [{"id": "before", "title": "BEFORE IMAGE", "status": "AVAILABLE", "meaning": "First uploaded observation.", "png_base64": _preview_png(before)}, {"id": "after", "title": "LATEST IMAGE", "status": "AVAILABLE", "meaning": "Latest uploaded observation used as the map base.", "png_base64": _preview_png(after)}, {"id": "change", "title": "CHANGE EVIDENCE", "status": "AVAILABLE", "meaning": "Red pixels exceeded the measured difference threshold.", "png_base64": _semantic_overlay([(mask, (255, 50, 35))], display_shape)}], "map_label": "LATEST IMAGE + RED CHANGE EVIDENCE"},
            "diagnostics": {"threshold": round(threshold, 5), "changed_pixels": int(mask.sum())},
        })

    def fusion(self, optical: RasterData, sar: RasterData, query: str) -> dict:
        optical_channels = model_channels(optical.data)
        sar_channels = model_channels(sar.data)
        optical_mask = _heuristic_water(optical_channels)
        sar_signal = sar_channels.mean(axis=0)
        sar_mask = sar_signal < np.percentile(sar_signal, 20)
        agreement = float((optical_mask == sar_mask).mean())
        confidence = int(np.clip(40 + agreement * 45, 0, 85))
        ai_summary = f"AI SUMMARY: Optical and SAR were analyzed separately and compared on the shared grid. Their simple evidence masks agree on {agreement * 100:.1f}% of pixels. Higher agreement supports a stronger joint signal; disagreement should trigger review rather than be hidden."
        display_shape = _display_rgb(optical).shape[1:]
        return self._with_vlm_summary([optical, sar], query, {
            "claim": "OPTICAL AND SAR SIGNALS WERE COMPARED AS COMPLEMENTARY EVIDENCE.",
            "where": f"COMMON EVIDENCE BOUNDS {(_bbox(optical_mask | sar_mask) or 'NOT STABLE')}",
            "magnitude": f"{float((optical_mask | sar_mask).mean() * 100):.1f}% COMBINED SIGNAL AREA",
            "sensorCase": f"OPTICAL/SAR PIXEL AGREEMENT: {agreement * 100:.1f}%\nOPTICAL AND SAR ARE EXPLICITLY REPORTED SEPARATELY",
            "limit": "SAR intensity is not a land-cover label; confirm calibration, incidence angle, and co-registration for operational decisions.",
            "confidence": confidence,
            "decision": "TRIAGE-READY / HUMAN CONFIRMATION ADVISED" if confidence >= 60 else "SENSOR DISAGREEMENT / HUMAN REVIEW REQUIRED",
            "ai_summary": ai_summary,
            "evidence": {"source": "uploaded_pixels", "agreement": agreement, "bbox_pixels": _bbox(optical_mask | sar_mask), "mask_png_base64": _mask_png(optical_mask | sar_mask, display_shape), "visual_png_base64": _preview_png(optical), "after_png_base64": _preview_png(sar), "semantic_png_base64": _semantic_overlay([(optical_mask & sar_mask, (0, 220, 255)), (optical_mask & ~sar_mask, (255, 210, 0)), (sar_mask & ~optical_mask, (255, 0, 200))], display_shape), "legend": ["CYAN = BOTH SENSORS", "YELLOW = OPTICAL ONLY", "MAGENTA = SAR ONLY"], "analysis_path": "OPTICAL + SAR FUSION -> SENSOR-SPECIFIC SIGNALS -> AGREEMENT EVIDENCE", "layers": [{"id": "optical", "title": "OPTICAL IMAGE", "status": "AVAILABLE", "meaning": "Uploaded optical observation.", "png_base64": _preview_png(optical)}, {"id": "sar", "title": "SAR BACKSCATTER", "status": "AVAILABLE", "meaning": "Uploaded radar observation, display-normalized.", "png_base64": _preview_png(sar)}, {"id": "agreement", "title": "SENSOR AGREEMENT", "status": "AVAILABLE", "meaning": "Cyan means both sensors; yellow or magenta means one sensor only.", "png_base64": _semantic_overlay([(optical_mask & sar_mask, (0, 220, 255)), (optical_mask & ~sar_mask, (255, 210, 0)), (sar_mask & ~optical_mask, (255, 0, 200))], display_shape)}], "map_label": "OPTICAL IMAGE + FUSED SENSOR EVIDENCE"},
            "diagnostics": {"agreement": round(agreement, 4)},
        })

    def multi_summary(self, rasters: list[RasterData], query: str) -> dict:
        modality_text = ", ".join(f"{index + 1}: {raster.modality} ({raster.width}x{raster.height})" for index, raster in enumerate(rasters))
        means = [float(raster.data.mean()) for raster in rasters]
        spread = float(np.std(means)) if len(means) > 1 else 0.0
        confidence = int(np.clip(55 + (15 if spread < 0.12 else 0), 0, 70))
        asks_for_change = any(token in query.lower() for token in ("change", "changed", "between", "before", "after"))
        return self._with_vlm_summary(rasters, query, {
            "claim": "MULTIPLE UPLOADED OBSERVATIONS WERE CATALOGUED AND COMPARED.",
            "where": "NO SINGLE CHANGE REGION RELEASED: A SPATIAL CHANGE MAP REQUIRES EXACTLY TWO MATCHED OBSERVATIONS.",
            "magnitude": f"{len(rasters)} OBSERVATIONS / MEAN-SIGNAL SPREAD {spread:.3f}",
            "sensorCase": modality_text,
            "limit": "This upload contains more than one comparison candidate. Use exactly two co-registered images for a change map, or one optical plus one SAR image for fusion.",
            "confidence": confidence,
            "decision": "CONTEXT READY / SPECIFY ANALYTIC QUESTION",
            "ai_summary": f"AI SUMMARY: I received {len(rasters)} observations: {modality_text}. Their normalized mean-signal spread is {spread:.3f}. {'No change map was released because more than two images were uploaded; select one matched before/after pair for that test.' if asks_for_change else 'No change or land-cover claim is released until a specific matched-pair or fusion question is selected.'}",
            "evidence": {"source": "uploaded_pixels", "visual_png_base64": _preview_grid(rasters), "analysis_path": "MULTI-OBSERVATION SYNTHESIS -> INPUT CATALOGUE -> SUMMARY ONLY", "layers": [{"id": f"upload-{index}", "title": f"UPLOAD {index + 1}", "status": "AVAILABLE", "meaning": f"{raster.modality} observation included in the summary.", "png_base64": _preview_png(raster)} for index, raster in enumerate(rasters)], "map_label": "UPLOADED OBSERVATION GRID"},
            "diagnostics": {"mean_signals": [round(value, 4) for value in means], "query": query},
        })
