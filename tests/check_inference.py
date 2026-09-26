"""CPU regression checks for shared inference and concurrent streaming handoffs."""

import asyncio
import importlib.abc
import json
import os
import sys
import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from src.inference.config import InferenceConfig, ThinkingConfig
from src.inference.pipeline import DecisionPipeline, thinking_options, validate_request
from src.inference.thinking import ThinkingError, ThinkingModel
from src.service.app import create_app
from src.service.examples import examples


class NoXTuner(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "xtuner" or fullname.startswith("xtuner."):
            raise AssertionError("Pure HF interface imported XTuner")
        return None


sys.meta_path.insert(0, NoXTuner())


def row():
    return {
        "state": "The receipt says paid. Rating: 1.",
        "questions": {
            "payment": {"type": "choice", "criteria": {"paid": "Paid", "due": "Due"}},
            "rating": {"type": "score", "criteria": ["bad", "good"]},
            "balance": {"type": "noul", "instructions": "Is there a balance?"},
        },
    }


class FakeEngine:
    def predict(self, request):
        from src.inputs.schema import _options

        answers = {}
        for key, question in request["questions"].items():
            labels = list(dict(_options(question)))
            probabilities = dict(zip(labels, [0.6, 0.4]))
            answer = {"type": question["type"], "confidence": 0.6, "probabilities": probabilities}
            if question["type"] == "score":
                answer.update(score=0.4, legend=dict(_options(question)))
            elif question["type"] == "noul":
                answer["noul"] = 0.4
            else:
                answer["choice"] = labels[0]
            answers[key] = answer
        return {
            "answers": answers,
            "usage": {"input_tokens": 10, "output_tokens": len(answers)},
            "timing": {"inference_ms": 0.1},
        }


class FakeThinker:
    available = True

    def __init__(self):
        self.calls = self.active = self.peak = 0
        self.fail = False

    async def stream(self, request, name):
        self.calls += 1
        self.active += 1
        self.peak = max(self.active, self.peak)
        try:
            await asyncio.sleep(0.02)
            if self.fail:
                raise ThinkingError("Provider unavailable.")
            yield {"event": "delta", "field": name, "text": "answer"}
            yield {
                "event": "choice",
                "field": name,
                "choice": {"payment": "due", "rating": "1", "balance": "yes"}[name],
            }
        finally:
            self.active -= 1


def pipeline():
    return DecisionPipeline(FakeEngine(), InferenceConfig(checkpoint="/unused"), FakeThinker())


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_first_parallel_and_no_fake_probabilities(self):
        p = pipeline()
        request = row() | {"thinking": {"enabled": True}}
        local = p.predict(request)
        self.assertEqual(p.thinker.calls, 0)
        self.assertEqual(len(local["thinking"]["tasks"]), 3)
        final = await p.resolve(request)
        self.assertEqual(p.thinker.peak, 3)
        for answer in final["answers"].values():
            self.assertEqual(answer["source"], "thinking")
            self.assertIsNone(answer["confidence"])
            self.assertIsNone(answer["probabilities"])
            self.assertEqual(answer["initial"]["confidence"], 0.6)
        self.assertEqual(final["answers"]["rating"]["score"], 1)
        self.assertEqual(final["answers"]["balance"]["noul"], 1)
        self.assertEqual(final["answers"]["payment"]["choice"], "due")

    async def test_threshold_boundary_and_defaults(self):
        p = pipeline()
        self.assertEqual(p.predict(row())["thinking"]["tasks"], [])
        result = p.predict(row() | {"thinking": {"enabled": True, "confidence_threshold": 0.6}})
        self.assertEqual(result["thinking"]["tasks"], [])
        for value in (True, -1, 1.1, float("nan"), "0.7"):
            with self.assertRaises(ValueError):
                thinking_options({"thinking": {"confidence_threshold": value}}, p.config.thinking)

    async def test_error_replay_duplicate_and_cancellation(self):
        p = pipeline()
        p.thinker.fail = True
        local = p.predict(row() | {"thinking": {"enabled": True}})
        task = local["thinking"]["tasks"][0]
        session, cached = p.claim(local["request_id"], task["field"], task["token"])
        with self.assertRaises(RuntimeError):
            p.claim(local["request_id"], task["field"], task["token"])
        with self.assertRaises(KeyError):
            p.claim(local["request_id"], task["field"], "wrong")
        events = [e async for e in p.stream_claimed(session, task["field"], cached)]
        self.assertEqual(events[-1]["event"], "error")
        self.assertEqual(session.result["answers"]["payment"]["source"], "local")
        session, cached = p.claim(local["request_id"], task["field"], task["token"])
        self.assertEqual([e async for e in p.stream_claimed(session, task["field"], cached)], [cached])
        self.assertEqual(p.thinker.calls, 1)
        task = local["thinking"]["tasks"][1]
        session, cached = p.claim(local["request_id"], task["field"], task["token"])
        stream = p.stream_claimed(session, task["field"])
        await anext(stream)
        await stream.aclose()
        self.assertEqual(session.states[task["field"]], "cancelled")

    async def test_expiry_capacity_and_sanitization(self):
        p = pipeline()
        dirty = row()
        dirty["targets"] = {"payment": {"label": "SECRET_GOLD"}}
        dirty["provenance"] = "SECRET_GOLD"
        dirty["questions"]["payment"]["answer"] = {"choice": "SECRET_GOLD"}
        self.assertNotIn("SECRET_GOLD", json.dumps(validate_request(dirty)))
        p.config.max_sessions = 1
        result = p.predict(row() | {"thinking": {"enabled": True}})
        with self.assertRaises(ValueError):
            p.predict(row() | {"thinking": {"enabled": True}})
        session = p._sessions[result["request_id"]]
        session.created -= 10000
        with self.assertRaises(KeyError):
            task = result["thinking"]["tasks"][0]
            p.claim(result["request_id"], task["field"], task["token"])

    async def test_pure_hf_import(self):
        from src.inference.hf_backend import HFBackend

        self.assertIsNotNone(HFBackend)
        self.assertFalse(any(k.startswith("xtuner") for k in sys.modules))


class ChunkedStream(httpx.AsyncByteStream):
    def __init__(self, text):
        self.data = text.encode()

    async def __aiter__(self):
        for index in range(0, len(self.data), 3):
            yield self.data[index : index + 3]


def sse(choice="due", finish="stop", done=True):
    parts = [
        {"choices": [{"delta": {"reasoning_content": "PRIVATE_REASONING"}, "index": 0}]},
        {"choices": [{"delta": {"content": json.dumps({"choice": choice})}, "index": 0}]},
        {"choices": [{"delta": {}, "finish_reason": finish, "index": 0}]},
    ]
    return "".join("data: " + json.dumps(p) + "\r\n\r\n" for p in parts) + ("data: [DONE]\r\n\r\n" if done else "")


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    async def collect(self, body, status=200):
        def handle(request):
            content = json.loads(request.content)
            self.assertEqual(content["model"], "your-thinking-model")
            self.assertNotIn("SECRET_GOLD", json.dumps(content))
            return httpx.Response(status, stream=ChunkedStream(body))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            with patch.dict(os.environ, {"LLM_BASE_URL": "https://example.invalid/v1", "LLM_API_KEY": "TEST_SECRET"}):
                thinker = ThinkingModel(ThinkingConfig(), client)
            dirty = row()
            dirty["targets"] = "SECRET_GOLD"
            dirty["questions"]["payment"]["answer"] = "SECRET_GOLD"
            return [e async for e in thinker.stream(dirty, "payment")]

    async def test_fragmented_stream_ignores_reasoning(self):
        events = await self.collect(sse())
        self.assertEqual(events[-1]["choice"], "due")
        self.assertNotIn("PRIVATE_REASONING", json.dumps(events))

    async def test_invalid_label_truncated_and_provider_failure(self):
        for body, status in (
            (sse("wrong"), 200),
            (sse(finish="length"), 200),
            (sse(done=False), 200),
            ("TEST_SECRET", 429),
            ("data: invalid\n\n", 200),
        ):
            with self.assertRaises(ThinkingError) as raised:
                await self.collect(body, status)
            self.assertNotIn("TEST_SECRET", str(raised.exception))


class ServiceTests(unittest.TestCase):
    def test_http_and_ui_contract(self):
        p = pipeline()
        with TestClient(create_app(p)) as client:
            request = row() | {"thinking": {"enabled": True}}
            response = client.post("/v1/decisions", json=request)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(p.thinker.calls, 0)
            result = response.json()
            task = result["thinking"]["tasks"][0]
            stream = client.post(task["stream_url"], json={"field": task["field"], "token": task["token"]})
            self.assertEqual(stream.status_code, 200)
            self.assertIn("event: final", stream.text)
            self.assertIn('"source": "thinking"', stream.text)
            self.assertEqual(p.thinker.calls, 1)
            self.assertEqual(
                client.post(task["stream_url"], json={"field": task["field"], "token": task["token"]}).text,
                stream.text[stream.text.index("event: final") :],
            )
            self.assertEqual(client.post("/v1/jev", json=row()).status_code, 200)
            for bad in ({}, row() | {"images": ["/private"]}, row() | {"thinking": {"model": "injection"}}):
                self.assertEqual(client.post("/v1/decisions", json=bad).status_code, 422)
            content = client.get("/").text + client.get("/demo.js").text + client.get("/demo-meta").text
            for forbidden in ("Qwen3.5-4B", "Jev API", "LLM_API_KEY", "TEST_SECRET"):
                self.assertNotIn(forbidden, content)
            image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
            before_upload = p.thinker.calls
            upload = client.post(
                "/v1/decisions",
                json=row()
                | {"images": [{"name": "pixel.png", "type": "image/png", "data": "data:image/png;base64," + image}]},
            )
            self.assertEqual(upload.status_code, 200, upload.text)
            self.assertEqual(p.thinker.calls, before_upload)
            for example in examples():
                validate_request(example)


if __name__ == "__main__":
    unittest.main()
