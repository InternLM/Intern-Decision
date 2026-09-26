"""Shared request validation, local inference, and parallel thinking sessions."""

import asyncio
import base64
import copy
import math
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.inference.temperature import argmax
from src.inference.thinking import ThinkingError
from src.inputs.schema import ANSWER_SYMBOLS, compile_row


def validate_request(request, allow_images=False):
    if not isinstance(request, dict) or "state" not in request:
        raise ValueError("Supply a JSON object containing state and questions.")
    images = request.get("images") or []
    if images and not allow_images:
        raise ValueError("Pass image bytes through the HTTP upload interface.")
    if images:
        if not isinstance(images, list) or not 1 <= len(images) <= 8:
            raise ValueError("Upload 1–8 images.")
        if not all(isinstance(image, str) and image.startswith("/") for image in images):
            raise ValueError("Uploaded images must be server-managed files.")
    questions = request.get("questions")
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 16:
        raise ValueError("Supply 1–16 questions.")
    clean = {}
    for name, question in questions.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Question names must be nonempty strings.")
        if not isinstance(question, dict) or question.get("type") not in {"choice", "score", "noul"}:
            raise ValueError("Question type must be choice, score, or noul.")
        kind = question["type"]
        criteria = question.get("criteria")
        if kind == "choice" and not isinstance(criteria, dict):
            raise ValueError("Choice criteria must be an object.")
        if kind == "score":
            if not isinstance(criteria, list | dict):
                raise ValueError("Score criteria must be a list or object.")
            if isinstance(criteria, dict):
                try:
                    if not all(math.isfinite(float(key)) for key in criteria):
                        raise ValueError
                except (ValueError, TypeError) as exc:
                    raise ValueError("Score option keys must be finite numbers.") from exc
        if kind != "noul" and not 1 <= len(criteria) <= len(ANSWER_SYMBOLS):
            raise ValueError(f"Supply 1–{len(ANSWER_SYMBOLS)} options.")
        clean[name] = {
            key: copy.deepcopy(question[key]) for key in ("type", "instructions", "criteria") if key in question
        }
    row = {"state": copy.deepcopy(request["state"]), "questions": clean}
    if images:
        row["images"] = list(images)
    compile_row(row, include_targets=False)
    return row


def thinking_options(request, config):
    options = request.get("thinking", {})
    if not isinstance(options, dict) or set(options) - {"enabled", "confidence_threshold"}:
        raise ValueError("thinking accepts only enabled and confidence_threshold.")
    enabled = options.get("enabled", config.enabled)
    threshold = options.get("confidence_threshold", config.confidence_threshold)
    if not isinstance(enabled, bool):
        raise ValueError("thinking.enabled must be boolean.")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, int | float)
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("confidence_threshold must be a finite number in [0, 1].")
    return enabled, float(threshold)


def final_answer(initial, choice):
    answer = {
        "type": initial["type"],
        "decision": choice,
        "source": "thinking",
        "confidence": None,
        "probabilities": None,
        "initial": copy.deepcopy(initial),
    }
    if initial["type"] == "choice":
        answer["choice"] = choice
    elif initial["type"] == "score":
        answer["score"] = float(choice)
        answer["legend"] = initial["legend"]
    else:
        # A hard choice, not a calibrated probability estimate from the thinker.
        answer["noul"] = float(choice == "yes")
    return answer


@dataclass
class Session:
    row: dict
    result: dict
    tokens: dict
    created: float = field(default_factory=time.monotonic)
    states: dict = field(default_factory=dict)
    finals: dict = field(default_factory=dict)


class DecisionPipeline:
    def __init__(self, engine, config, thinker):
        self.engine, self.config, self.thinker = engine, config, thinker
        self._model_lock = threading.Lock()
        self._sessions_lock = threading.Lock()
        self._sessions = {}

    def _cleanup(self):
        now = time.monotonic()
        for key, session in list(self._sessions.items()):
            if now - session.created > self.config.session_ttl_seconds and "running" not in session.states.values():
                del self._sessions[key]

    def predict(self, request, allow_images=False):
        started = time.perf_counter()
        row = validate_request(request, allow_images=allow_images)
        enabled, threshold = thinking_options(request, self.config.thinking)
        if enabled and not self.thinker.available:
            raise ValueError("Thinking model credentials are not configured.")
        with self._model_lock:
            queued_ms = (time.perf_counter() - started) * 1000
            result = self.engine.predict(row)
        for answer in result["answers"].values():
            answer["source"] = "local"
            answer["decision"] = argmax(answer["probabilities"])
        request_id = secrets.token_hex(16)
        result.update(model="intern-decision", request_id=request_id)
        result["usage"]["decision_count"] = len(result["answers"])
        result.setdefault("timing", {}).update(
            queue_ms=round(queued_ms, 2),
            server_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        uncertain = [
            name
            for name, answer in result["answers"].items()
            if enabled and max(answer["probabilities"].values()) < threshold
        ]
        tokens = {name: secrets.token_urlsafe(24) for name in uncertain}
        result["thinking"] = {
            "enabled": enabled,
            "confidence_threshold": threshold,
            "tasks": [
                {"field": name, "token": tokens[name], "stream_url": f"/v1/decisions/{request_id}/thinking"}
                for name in uncertain
            ],
        }
        if uncertain:
            # Upload files are removed after the local forward. Preserve only the
            # validated image bytes for later thinking streams, never file paths.
            session_row = {key: value for key, value in row.items() if key != "images"}
            if row.get("images"):
                session_row["image_contents"] = [
                    "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")
                    for path in row["images"]
                ]
            with self._sessions_lock:
                self._cleanup()
                if len(self._sessions) >= self.config.max_sessions:
                    raise ValueError("Thinking session capacity reached; retry later.")
                self._sessions[request_id] = Session(
                    session_row,
                    copy.deepcopy(result),
                    tokens,
                    states=dict.fromkeys(uncertain, "pending"),
                )
        # No network call has started. The caller can display this JSON immediately.
        return result

    def claim(self, request_id, name, token):
        with self._sessions_lock:
            self._cleanup()
            session = self._sessions.get(request_id)
            if session is None:
                raise KeyError("Thinking session expired or not found.")
            if not isinstance(name, str) or name not in session.tokens or not isinstance(token, str):
                raise KeyError("Thinking task not found.")
            if not secrets.compare_digest(session.tokens[name], token):
                raise KeyError("Thinking task not found.")
            state = session.states[name]
            if state == "running":
                raise RuntimeError("This thinking task is already running.")
            if name in session.finals:
                return session, session.finals[name]
            if state == "cancelled":
                raise RuntimeError("This thinking task was cancelled; submit a new request.")
            session.states[name] = "running"
            return session, None

    async def stream_claimed(self, session, name, cached=None):
        if cached:
            yield copy.deepcopy(cached)
            return
        started = time.perf_counter()
        try:
            yield {"event": "start", "field": name}
            chosen = None
            async for event in self.thinker.stream(session.row, name):
                if event["event"] == "choice":
                    chosen = event["choice"]
                else:
                    yield event
            if chosen is None:
                raise ThinkingError("Thinking stream ended without a decision.")
            initial = session.result["answers"][name]
            # Validate even injected adapters: unknown options can never become decisions.
            if chosen not in initial["probabilities"]:
                raise ThinkingError("Thinking model chose an invalid option.")
            answer = final_answer(initial, chosen)
            event = {
                "event": "final",
                "field": name,
                "answer": answer,
                "thinking_ms": round((time.perf_counter() - started) * 1000, 2),
            }
            with self._sessions_lock:
                session.result["answers"][name] = answer
                session.states[name] = "complete"
                session.finals[name] = event
            yield copy.deepcopy(event)
        except ThinkingError as exc:
            event = {"event": "error", "field": name, "message": str(exc), "retained_initial": True}
            with self._sessions_lock:
                session.states[name] = "error"
                session.finals[name] = event
            yield event
        except Exception:
            # Do not leak provider bodies, evidence, or credentials in HTTP errors.
            event = {
                "event": "error",
                "field": name,
                "message": "Thinking request failed; initial decision retained.",
                "retained_initial": True,
            }
            with self._sessions_lock:
                session.states[name] = "error"
                session.finals[name] = event
            yield event
        finally:
            with self._sessions_lock:
                if session.states[name] == "running":
                    session.states[name] = "cancelled"

    async def resolve(self, request):
        """Non-HTTP interface using the same pipeline, with concurrent handoffs."""
        result = await asyncio.to_thread(self.predict, request)

        async def run(task):
            session, cached = self.claim(result["request_id"], task["field"], task["token"])
            async for event in self.stream_claimed(session, task["field"], cached):
                if event["event"] == "final":
                    result["answers"][task["field"]] = event["answer"]
                elif event["event"] == "error":
                    result.setdefault("thinking_errors", {})[task["field"]] = event["message"]

        await asyncio.gather(*(run(task) for task in result["thinking"]["tasks"]))
        return result
