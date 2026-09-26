"""Build the existing masked Qwen execution with dimensions read from the pinned HF model."""

import json
from pathlib import Path

from xtuner.v1.model.compose.qwen3_5.qwen3_5_config import Qwen3_5_ProjectorConfig, Qwen3_5_VisionConfig
from xtuner.v1.model.dense.qwen3_5_text import Qwen3_5_VLTextDense4BConfig
from xtuner.v1.module.attention import GatedDeltaNetConfig, MHAConfig
from xtuner.v1.module.rope import RopeParametersConfig

from src.model.qwen import DecisionQwenConfig


def decision_config_from_hf(path, **kwargs):
    hf = json.loads((Path(path) / "config.json").read_text())
    assert hf["model_type"] == "qwen3_5"
    t, v = hf["text_config"], hf["vision_config"]
    assert t.get("full_attention_interval", 4) == 4
    expected = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(t["num_hidden_layers"])]
    assert t["layer_types"] == expected
    assert not t["attention_bias"] and t["attention_dropout"] == 0 and t["attn_output_gate"]
    assert not t.get("mlp_only_layers") and not v.get("deepstack_visual_indexes")
    r = t["rope_parameters"]
    assert r["rope_type"] == "default" and r["mrope_interleaved"]
    vision = Qwen3_5_VisionConfig(
        **{
            k: v[k]
            for k in [
                "depth",
                "hidden_size",
                "intermediate_size",
                "hidden_act",
                "in_channels",
                "patch_size",
                "spatial_merge_size",
                "temporal_patch_size",
                "num_position_embeddings",
                "initializer_range",
            ]
        },
        num_attention_heads=v["num_heads"],
    )
    projector = Qwen3_5_ProjectorConfig(
        vision_hidden_size=v["hidden_size"],
        text_hidden_size=t["hidden_size"],
        spatial_merge_size=v["spatial_merge_size"],
    )
    text = Qwen3_5_VLTextDense4BConfig(
        **{
            k: t[k]
            for k in [
                "vocab_size",
                "max_position_embeddings",
                "eos_token_id",
                "num_hidden_layers",
                "hidden_size",
                "intermediate_size",
                "rms_norm_eps",
                "hidden_act",
            ]
        },
        tie_word_embeddings=hf["tie_word_embeddings"],
        attention=MHAConfig(
            with_gate=t["attn_output_gate"],
            num_attention_heads=t["num_attention_heads"],
            num_key_value_heads=t["num_key_value_heads"],
            head_dim=t["head_dim"],
            qk_norm=True,
            rms_norm_eps=t["rms_norm_eps"],
            rms_norm_type="zero_centered",
        ),
        linear_attention=GatedDeltaNetConfig(
            num_value_heads=t["linear_num_value_heads"],
            num_key_heads=t["linear_num_key_heads"],
            key_head_dim=t["linear_key_head_dim"],
            value_head_dim=t["linear_value_head_dim"],
            conv_kernel_dim=t["linear_conv_kernel_dim"],
            hidden_act=t["hidden_act"],
            rms_norm_eps=t["rms_norm_eps"],
        ),
        rope_parameters_cfg=RopeParametersConfig(
            rope_theta=r["rope_theta"],
            rope_type="qwen3_vl",
            mrope_section=r["mrope_section"],
            partial_rotary_factor=r["partial_rotary_factor"],
        ),
    )
    assert v["out_hidden_size"] == t["hidden_size"]
    return DecisionQwenConfig(
        vision_config=vision,
        projector_config=projector,
        text_config=text,
        **{k: hf[k] for k in ["image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"]},
        **kwargs,
    )
