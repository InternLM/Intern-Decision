"""Validated inference configuration; credentials are resolved only on the server."""

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[2]


class ThinkingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    model: str = "your-thinking-model"
    confidence_threshold: float = Field(default=0.7, ge=0, le=1)
    base_url_env: str = "LLM_BASE_URL"
    api_key_env: str = "LLM_API_KEY"
    timeout_seconds: float = Field(default=300, gt=0, le=600)
    connect_timeout_seconds: float = Field(default=20, gt=0, le=120)
    trust_env: bool = True
    max_parallel: int = Field(default=16, ge=1, le=32)
    max_tokens: int = Field(default=2048, ge=16, le=16384)


class InferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: Literal["hf", "xtuner"] = "hf"
    checkpoint: str
    processor_path: str | None = None
    media_root: str = ""
    calibration_path: str | None = None
    max_length: int = Field(default=8192, ge=1)
    device: str = "cuda"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    attn_implementation: Literal["sdpa", "eager", "flash_attention_2"] = "sdpa"
    thinking: ThinkingConfig = Field(default_factory=ThinkingConfig)
    session_ttl_seconds: int = Field(default=600, ge=30, le=3600)
    max_sessions: int = Field(default=128, ge=1, le=4096)

    @classmethod
    def load(cls, path=None):
        from dotenv import load_dotenv

        # No credential values are returned to the client or written to logs.
        load_dotenv(ROOT / ".env", override=False)
        data = json.loads(Path(path).read_text()) if path else {}
        for env, key in (
            ("MODEL_CHECKPOINT", "checkpoint"),
            ("MODEL_PATH", "processor_path"),
            ("MEDIA_ROOT", "media_root"),
            ("CALIBRATION_PATH", "calibration_path"),
            ("INFERENCE_BACKEND", "backend"),
        ):
            if os.environ.get(env):
                data[key] = os.environ[env]
        if os.environ.get("MAX_INPUT_LENGTH"):
            data["max_length"] = int(os.environ["MAX_INPUT_LENGTH"])
        config = cls.model_validate(data)
        for key in ("checkpoint", "processor_path", "media_root", "calibration_path"):
            value = getattr(config, key)
            if value and not Path(value).is_absolute():
                setattr(config, key, str((ROOT / value).resolve()))
        return config
