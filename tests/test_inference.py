from pathlib import Path
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from inference.api import app, executor
from inference.orchestration.router import route_query


class InferenceTests(unittest.TestCase):
    def test_router_selects_required_workflows(self):
        self.assertEqual(route_query("What changed between these dates?", 2), "BI_TEMPORAL_CHANGE_VQA")
        self.assertEqual(route_query("Use optical and SAR together.", 2), "OPTICAL_SAR_FUSION")
        self.assertEqual(route_query("Highlight the water body.", 1), "TEXT_GUIDED_GROUNDING")

    def test_real_api_returns_uploaded_pixel_evidence(self):
        sample = Path("public/evidence-map.png")
        with patch.object(executor.vlm, "summarize", return_value=(None, "test-template")):
            with sample.open("rb") as handle:
                response = TestClient(app).post("/analyse", data={"query": "Describe the land-cover visible in this image."}, files={"files": (sample.name, handle, "image/png")})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["runtime"], "real")
        self.assertTrue(body["model_version"].startswith("satquery-eurosat"))
        self.assertEqual(body["evidence"]["source"], "uploaded_pixels")
        self.assertTrue(body["evidence"]["mask_png_base64"])
        self.assertTrue(body["evidence"]["visual_png_base64"])
        self.assertTrue(body["evidence"]["semantic_png_base64"])
        self.assertTrue(body["evidence"]["legend"])
        self.assertIn("SATQUERY PIXEL MODEL", body["evidence"]["analysis_path"])
        self.assertEqual(body["evidence"]["layers"][-1]["status"], "NOT AVAILABLE")

    def test_change_returns_previews_for_the_new_uploads(self):
        before = Image.new("RGB", (64, 64), color=(20, 80, 40))
        after = Image.new("RGB", (64, 64), color=(190, 40, 30))
        from io import BytesIO
        before_data, after_data = BytesIO(), BytesIO()
        before.save(before_data, format="PNG")
        after.save(after_data, format="PNG")
        with patch.object(executor.vlm, "summarize", return_value=(None, "test-template")):
            response = TestClient(app).post(
                "/analyse",
                data={"query": "What changed between these dates?"},
                files=[
                    ("files", ("before.png", before_data.getvalue(), "image/png")),
                    ("files", ("after.png", after_data.getvalue(), "image/png")),
                ],
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        evidence = body["evidence"]
        self.assertEqual(body["task"], "BI_TEMPORAL_CHANGE_VQA")
        self.assertTrue(evidence["mask_png_base64"])
        self.assertTrue(evidence["before_png_base64"])
        self.assertTrue(evidence["after_png_base64"])
        self.assertTrue(evidence["semantic_png_base64"])
        self.assertEqual(evidence["legend"], ["RED = PIXELS THAT CHANGED BETWEEN UPLOADS"])
        self.assertEqual(evidence["layers"][-1]["title"], "CHANGE EVIDENCE")
        self.assertEqual(evidence["visual_png_base64"], evidence["after_png_base64"])
        self.assertNotEqual(evidence["before_png_base64"], evidence["after_png_base64"])

    def test_more_than_two_change_inputs_are_not_silently_dropped(self):
        image = Image.new("RGB", (32, 32), color=(30, 70, 40))
        from io import BytesIO
        payload = BytesIO()
        image.save(payload, format="PNG")
        files = [("files", (f"observation-{index}.png", payload.getvalue(), "image/png")) for index in range(3)]
        with patch.object(executor.vlm, "summarize", return_value=(None, "test-template")):
            response = TestClient(app).post("/analyse", data={"query": "What changed between these dates?"}, files=files)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["task"], "MULTI_OBSERVATION_SYNTHESIS")
        self.assertEqual(len(body["inputSummary"]), 3)
        self.assertIn("exactly two", body["limit"].lower())


if __name__ == "__main__":
    unittest.main()

