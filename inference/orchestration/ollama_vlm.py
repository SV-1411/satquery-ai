from __future__ import annotations

import base64
import json
import os
from io import BytesIO
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from ..preprocessing.raster_loader import RasterData


class OllamaVisionSummarizer:
    """Optional Qwen2.5-VL report writer constrained by measured evidence."""

    def __init__(self) -> None:
        self.url = os.getenv("SATQUERY_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
        self.model = os.getenv("SATQUERY_OLLAMA_MODEL", "qwen2.5vl:3b")

    def available(self) -> bool:
        try:
            request = Request(f"{self.url}/api/tags", method="GET")
            with urlopen(request, timeout=3) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return any(item.get("name") == self.model for item in payload.get("models", []))
        except (OSError, URLError, ValueError):
            return False

    @staticmethod
    def _preview(raster: RasterData) -> str:
        """Keep local-VLM image payloads small enough for reliable GPU responses."""
        channels = raster.data
        if raster.modality == "SAR":
            image = np.repeat(channels[0:1], 3, axis=0)
        elif channels.shape[0] >= 4:
            # BigEarthNet order starts B01, B02, B03, B04; use B04/B03/B02.
            image = channels[[3, 2, 1]]
        else:
            image = channels[:3]
            if image.shape[0] == 1:
                image = np.repeat(image, 3, axis=0)
        rgb = np.moveaxis(np.clip(image, 0, 1), 0, -1)
        image = Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")
        image.thumbnail((512, 512), Image.Resampling.BILINEAR)
        buffer = BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def summarize(self, rasters: list[RasterData], query: str, evidence: dict) -> tuple[str | None, str]:
        if not self.available():
            return None, "grounded-template-fallback"
        prompt = (
            "You are the plain-language report writer for a remote-sensing analysis tool. "
            "Explain the measured result for a non-specialist in 2 to 3 short, simple sentences. "
            "First say what was uploaded, then the main measured finding, then the important caution or next step. "
            "Use ONLY the supplied evidence JSON and visible image previews. Do not invent a date, place, sensor, object, cause, or measurement. "
            "Do not replace or contradict the claim, confidence, percentage, bounds, or limitation. "
            "Explicitly say when a result is spectral/backscatter evidence rather than a guaranteed land-cover label. "
            "Return plain text beginning with 'AI SUMMARY:'.\n\n"
            f"USER QUESTION: {query}\n"
            f"MEASURED EVIDENCE JSON: {json.dumps(evidence, separators=(',', ':'))}"
        )
        message = {"role": "user", "content": prompt, "images": [self._preview(raster) for raster in rasters[:4]]}
        body = json.dumps({"model": self.model, "stream": False, "options": {"temperature": 0.1}, "messages": [message]}).encode("utf-8")
        try:
            request = Request(f"{self.url}/api/chat", data=body, headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(request, timeout=75) as response:
                payload = json.loads(response.read().decode("utf-8"))
            text = str(payload.get("message", {}).get("content", "")).strip()
            if text:
                return text if text.upper().startswith("AI SUMMARY:") else f"AI SUMMARY: {text}", self.model
        except (OSError, URLError, ValueError, TimeoutError):
            pass
        return None, "grounded-template-fallback"
