from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
import torch
from PIL import Image

from ..models.registry import ModelRegistry
from .ollama_vlm import OllamaVisionSummarizer
from ..preprocessing.raster_loader import RasterData, model_channels


def _mask_png(mask: np.ndarray) -> str:
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _heuristic_water(channels: np.ndarray) -> np.ndarray:
    red, green, blue = channels[0], channels[1], channels[2]
    score = blue - 0.55 * red - 0.25 * green
    threshold = float(np.percentile(score, 90))
    return score > max(threshold, 0.08)


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
            result["ai_summary"] = generated
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
        channels = model_channels(raster.data)
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
        scene = scene_names.get(scene_label, "mixed land cover") if scene_label is not None else ("water-consistent and built-up land" if water_percent > 2 else "mixed land cover")
        if task == "TEXT_GUIDED_GROUNDING":
            claim = "WATER-CONSISTENT PIXELS WERE LOCALIZED FROM THE UPLOADED OBSERVATION."
            where = f"PIXEL BOUNDS {bbox}" if bbox else "NO STABLE WATER REGION RELEASED"
        elif task == "SINGLE_IMAGE_CAPTIONING":
            claim = f"THE SCENE CONTAINS {scene.upper()} WITH {water_percent:.1f}% WATER-CONSISTENT AREA."
            where = f"WATER-CONSISTENT REGION / PIXEL BOUNDS {bbox}" if bbox else "NO WATER REGION RELEASED"
        else:
            claim = f"THE UPLOADED OBSERVATION IS CONSISTENT WITH {scene.upper()}."
            where = f"WATER-CONSISTENT REGION / PIXEL BOUNDS {bbox}" if bbox else "NO STABLE REGION RELEASED"
        confidence = int(np.clip(45 + scene_probability * 35 + model_confidence * 20, 0, 92)) if self.registry.ready else 48
        ai_summary = f"AI SUMMARY: This {raster.modality.lower()} observation is most consistent with {scene}. The evidence mask highlights {water_percent:.1f}% water-consistent pixels in the uploaded raster. This is an observation of spectral/backscatter signal, not a guaranteed land-cover label."
        return self._with_vlm_summary([raster], query, {
            "claim": claim,
            "where": where,
            "magnitude": f"{water_percent:.1f}% WATER-CONSISTENT PIXELS",
            "sensorCase": "UPLOADED PIXELS: MODEL + SPECTRAL BASELINE AGREEMENT" if self.registry.ready else "UPLOADED PIXELS: SPECTRAL BASELINE ONLY",
            "limit": "Pixel evidence is not a substitute for analyst review; verify georegistration and class labels before operational use.",
            "confidence": confidence,
            "decision": "TRIAGE-READY / HUMAN CONFIRMATION ADVISED" if confidence >= 60 else "LOW CONFIDENCE / HUMAN REVIEW REQUIRED",
            "ai_summary": ai_summary,
            "evidence": {"source": "uploaded_pixels", "bbox_pixels": bbox, "mask_png_base64": _mask_png(mask)},
            "diagnostics": {"model_confidence": round(model_confidence, 4), "water_percent": round(water_percent, 3)},
        })

    def change(self, before: RasterData, after: RasterData) -> dict:
        first = model_channels(before.data)
        second = model_channels(after.data)
        difference = np.mean(np.abs(second - first), axis=0)
        threshold = float(np.percentile(difference, 90))
        mask = difference >= max(threshold, 0.08)
        changed_percent = float(mask.mean() * 100)
        confidence = int(np.clip(52 + min(changed_percent, 30), 0, 85))
        ai_summary = f"AI SUMMARY: Comparing the first and last observations on the common grid, {changed_percent:.1f}% of pixels changed beyond the {threshold:.3f} baseline. This detects visual/spectral difference; it does not by itself prove flooding, construction, or another cause."
        return self._with_vlm_summary([before, after], "Compare the first and last uploaded observations.", {
            "claim": "PIXEL-LEVEL CHANGE WAS DETECTED BETWEEN THE TWO UPLOADED OBSERVATIONS.",
            "where": f"CHANGE BOUNDS {(_bbox(mask) or 'NOT STABLE')} IN THE COMMON PIXEL GRID",
            "magnitude": f"{changed_percent:.1f}% OF THE COMMON GRID FLAGGED / THRESHOLD {threshold:.3f}",
            "sensorCase": "BEFORE/AFTER SPECTRAL DIFFERENCE SUPPORTS THE CHANGE SIGNAL",
            "limit": "This baseline detects appearance change, not causal land-use change; validate acquisition dates, clouds, and registration.",
            "confidence": confidence,
            "decision": "TRIAGE-READY / HUMAN CONFIRMATION ADVISED" if confidence >= 60 else "LOW CONFIDENCE / HUMAN REVIEW REQUIRED",
            "ai_summary": ai_summary,
            "evidence": {"source": "uploaded_pixels", "changed_area_percent": changed_percent, "bbox_pixels": _bbox(mask), "mask_png_base64": _mask_png(mask)},
            "diagnostics": {"threshold": round(threshold, 5), "changed_pixels": int(mask.sum())},
        })

    def fusion(self, optical: RasterData, sar: RasterData) -> dict:
        optical_channels = model_channels(optical.data)
        sar_channels = model_channels(sar.data)
        optical_mask = _heuristic_water(optical_channels)
        sar_signal = sar_channels.mean(axis=0)
        sar_mask = sar_signal < np.percentile(sar_signal, 20)
        agreement = float((optical_mask == sar_mask).mean())
        confidence = int(np.clip(40 + agreement * 45, 0, 85))
        ai_summary = f"AI SUMMARY: Optical and SAR were analyzed separately and compared on the shared grid. Their simple evidence masks agree on {agreement * 100:.1f}% of pixels. Higher agreement supports a stronger joint signal; disagreement should trigger review rather than be hidden."
        return self._with_vlm_summary([optical, sar], "Compare optical and SAR evidence.", {
            "claim": "OPTICAL AND SAR SIGNALS WERE COMPARED AS COMPLEMENTARY EVIDENCE.",
            "where": f"COMMON EVIDENCE BOUNDS {(_bbox(optical_mask | sar_mask) or 'NOT STABLE')}",
            "magnitude": f"{float((optical_mask | sar_mask).mean() * 100):.1f}% COMBINED SIGNAL AREA",
            "sensorCase": f"OPTICAL/SAR PIXEL AGREEMENT: {agreement * 100:.1f}%\nOPTICAL AND SAR ARE EXPLICITLY REPORTED SEPARATELY",
            "limit": "SAR intensity is not a land-cover label; confirm calibration, incidence angle, and co-registration for operational decisions.",
            "confidence": confidence,
            "decision": "TRIAGE-READY / HUMAN CONFIRMATION ADVISED" if confidence >= 60 else "SENSOR DISAGREEMENT / HUMAN REVIEW REQUIRED",
            "ai_summary": ai_summary,
            "evidence": {"source": "uploaded_pixels", "agreement": agreement, "bbox_pixels": _bbox(optical_mask | sar_mask), "mask_png_base64": _mask_png(optical_mask | sar_mask)},
            "diagnostics": {"agreement": round(agreement, 4)},
        })

    def multi_summary(self, rasters: list[RasterData], query: str) -> dict:
        modality_text = ", ".join(f"{index + 1}: {raster.modality} ({raster.width}x{raster.height})" for index, raster in enumerate(rasters))
        means = [float(raster.data.mean()) for raster in rasters]
        spread = float(np.std(means)) if len(means) > 1 else 0.0
        confidence = int(np.clip(55 + (15 if spread < 0.12 else 0), 0, 70))
        return self._with_vlm_summary(rasters, query, {
            "claim": "MULTIPLE UPLOADED OBSERVATIONS WERE CATALOGUED AND COMPARED.",
            "where": "COMMON SPATIAL REGION NOT RELEASED WITHOUT A SPECIFIC CHANGE OR FUSION REQUEST",
            "magnitude": f"{len(rasters)} OBSERVATIONS / MEAN-SIGNAL SPREAD {spread:.3f}",
            "sensorCase": modality_text,
            "limit": "Ask a specific change, fusion, grounding, or captioning question for a spatial claim.",
            "confidence": confidence,
            "decision": "CONTEXT READY / SPECIFY ANALYTIC QUESTION",
            "ai_summary": f"AI SUMMARY: I received {len(rasters)} observations: {modality_text}. Their normalized mean-signal spread is {spread:.3f}. No change or land-cover claim is released because the query does not specify which relationship to test.",
            "evidence": {"source": "uploaded_pixels"},
            "diagnostics": {"mean_signals": [round(value, 4) for value in means], "query": query},
        })
