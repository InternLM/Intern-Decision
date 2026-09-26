"""Image upload integration checks: evidence, cleanup, limits and thinking."""

import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from src.inference.config import InferenceConfig
from src.inference.pipeline import DecisionPipeline
from src.inference.thinking import ThinkingModel
from src.service.app import create_app
from tests.check_inference import FakeEngine, FakeThinker, row


def uploaded(color="red", size=(12, 8), format="PNG", **kwargs):
    with Image.new("RGB", size, color) as image:
        data = io.BytesIO()
        image.save(data, format=format, **kwargs)
    mime = "image/jpeg" if format == "JPEG" else "image/png"
    return {
        "name": "../../not-a-server-path.png",
        "type": mime,
        "data": f"data:{mime};base64," + base64.b64encode(data.getvalue()).decode(),
    }


class ImageEngine(FakeEngine):
    def __init__(self):
        self.paths = []
        self.seen = []
        self.fail = False

    def predict(self, request):
        self.paths.extend(request.get("images", []))
        for path in request.get("images", []):
            with Image.open(path) as image:
                self.seen.append((image.size, image.getpixel((0, 0))))
        if self.fail:
            raise ValueError("Test model failure")
        return super().predict(request)


class ImageUploadTests(unittest.TestCase):
    def setUp(self):
        self.engine = ImageEngine()
        self.pipeline = DecisionPipeline(self.engine, InferenceConfig(checkpoint="/unused"), FakeThinker())
        self.client = TestClient(create_app(self.pipeline))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def test_multimodal_order_cleanup_and_no_paths_in_response(self):
        result = self.client.post("/v1/decisions", json=row() | {"images": [uploaded(), uploaded("blue")]})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(self.engine.seen, [((12, 8), (255, 0, 0)), ((12, 8), (0, 0, 255))])
        for path in self.engine.paths:
            self.assertFalse(Path(path).parent.exists())
            self.assertNotIn(path, result.text)
        self.assertNotIn("data:image", result.text)

    def test_failed_model_also_removes_files(self):
        self.engine.fail = True
        result = self.client.post("/v1/decisions", json=row() | {"images": [uploaded()]})
        self.assertEqual(result.status_code, 422)
        self.assertTrue(self.engine.paths)
        self.assertFalse(Path(self.engine.paths[0]).parent.exists())

    def test_thinking_keeps_same_pixels_after_files_removed(self):
        request = row() | {"images": [uploaded("blue")], "thinking": {"enabled": True}}
        local = self.client.post("/v1/decisions", json=request).json()
        session = self.pipeline._sessions[local["request_id"]]
        self.assertNotIn("images", session.row)
        data_url = session.row["image_contents"][0]
        thinker = ThinkingModel.__new__(ThinkingModel)
        messages = thinker.messages(session.row, "payment")
        self.assertEqual(messages[1]["content"][0]["image_url"]["url"], data_url)
        with Image.open(io.BytesIO(base64.b64decode(data_url.partition(",")[2]))) as image:
            self.assertEqual(image.getpixel((0, 0)), (0, 0, 255))
        self.assertNotIn("initial", json.dumps(messages))
        task = local["thinking"]["tasks"][0]
        stream = self.client.post(task["stream_url"], json={"field": task["field"], "token": task["token"]})
        self.assertIn("event: final", stream.text)
        self.assertFalse(Path(self.engine.paths[0]).parent.exists())

    def test_malformed_and_paths_are_rejected_without_leaving_files(self):
        invalid = [
            ["/etc/passwd"],
            [{"data": "https://example.invalid/a.png"}],
            "not-a-list",
            [uploaded()] * 9,
            [{"type": "image/png", "data": "not base64"}],
            [{"type": "image/png", "data": base64.b64encode(b"not image data").decode()}],
            [uploaded() | {"data": uploaded()["data"].replace("image/png", "image/jpeg")}],
            [uploaded(), {"data": "invalid"}],
        ]
        with tempfile.TemporaryDirectory() as directory:
            native_mkdtemp = tempfile.mkdtemp

            def private_dir(**kwargs):
                return native_mkdtemp(dir=directory, **kwargs)

            with patch("src.service.uploads.tempfile.mkdtemp", side_effect=private_dir):
                for images in invalid:
                    result = self.client.post("/v1/decisions", json=row() | {"images": images})
                    self.assertEqual(result.status_code, 422, result.text)
                    self.assertEqual(list(Path(directory).iterdir()), [])
        self.assertEqual(self.engine.paths, [])

    def test_limits_and_exif_orientation(self):
        with patch("src.service.uploads.MAX_IMAGE_PIXELS", 10):
            result = self.client.post("/v1/decisions", json=row() | {"images": [uploaded()]})
            self.assertEqual(result.status_code, 422)
        with patch("src.service.uploads.MAX_UPLOAD_TOTAL", 1):
            result = self.client.post("/v1/decisions", json=row() | {"images": [uploaded()]})
            self.assertEqual(result.status_code, 422)
        exif = Image.Exif()
        exif[274] = 6
        result = self.client.post("/v1/decisions", json=row() | {"images": [uploaded(format="JPEG", exif=exif)]})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(self.engine.seen[-1][0], (8, 12))

    def test_animated_gif_rejected(self):
        data = io.BytesIO()
        with Image.new("RGB", (8, 8), "red") as a, Image.new("RGB", (8, 8), "blue") as b:
            a.save(data, "GIF", save_all=True, append_images=[b], duration=100)
        item = {"type": "image/gif", "data": base64.b64encode(data.getvalue()).decode()}
        result = self.client.post("/v1/decisions", json=row() | {"images": [item]})
        self.assertEqual(result.status_code, 422)
        self.assertIn("Animated", result.text)


if __name__ == "__main__":
    unittest.main()
