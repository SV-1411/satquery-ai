from pathlib import Path
import unittest

from fastapi.testclient import TestClient

from inference.api import app
from inference.orchestration.router import route_query


class InferenceTests(unittest.TestCase):
    def test_router_selects_required_workflows(self):
        self.assertEqual(route_query("What changed between these dates?", 2), "BI_TEMPORAL_CHANGE_VQA")
        self.assertEqual(route_query("Use optical and SAR together.", 2), "OPTICAL_SAR_FUSION")
        self.assertEqual(route_query("Highlight the water body.", 1), "TEXT_GUIDED_GROUNDING")

    def test_real_api_returns_uploaded_pixel_evidence(self):
        sample = Path("public/evidence-map.png")
        with sample.open("rb") as handle:
            response = TestClient(app).post("/analyse", data={"query": "Describe the land-cover visible in this image."}, files={"files": (sample.name, handle, "image/png")})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["runtime"], "real")
        self.assertTrue(body["model_version"].startswith("satquery-eurosat"))
        self.assertEqual(body["evidence"]["source"], "uploaded_pixels")
        self.assertTrue(body["evidence"]["mask_png_base64"])


if __name__ == "__main__":
    unittest.main()

