"""Masked autoregressive Qwen3.5 training configuration.

The universal JSONL compiler in src.inputs.masked puts a single decision marker
in each assistant field. XTuner supplies the distributed/FSDP engine while the
masked labels keep the loss on those marker positions.
"""

from __future__ import annotations

import os
from pathlib import Path

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.datasets import DataloaderConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.model.base import HFSaveCfg
from xtuner.v1.train import TrainerConfig

from src.inputs.dataset import UniversalDatasetConfig
from src.inputs.masked import MaskedOpenaiTokenizeFunctionConfig, MaskedQwen3VLTokenizeFnConfig
from src.model.size_config import decision_config_from_hf


def _env_flag(name: str, default: bool = False) -> bool:
    return os.environ.get(name, "1" if default else "0").lower() in {"1", "true", "yes", "on"}


MODEL_PATH = os.environ["MODEL_PATH"]
DATA_PATH = os.environ["DATA_PATH"]
MEDIA_ROOT = os.environ.get("MEDIA_ROOT", "")
WORK_DIR = os.environ.get("WORK_DIR", "outputs/train")
SAMPLE_MAX_LENGTH = int(os.environ.get("SAMPLE_MAX_LENGTH", "8192"))
PACK_MAX_LENGTH = int(os.environ.get("PACK_MAX_LENGTH", str(SAMPLE_MAX_LENGTH)))
IS_MULTIMODAL = bool(MEDIA_ROOT)

# The projector is part of the vision path. Keeping it frozen is required when
# only the language backbone is fine-tuned.
model_cfg = decision_config_from_hf(
    MODEL_PATH,
    # Torch 2.6 removed composable FSDP's ignored_params argument. Keep
    # fp32-only norm parameters in the normal sharding path for compatibility.
    hf_save_cfg=HFSaveCfg(fp32_keys_pattern=None),
    only_llm_forward=not IS_MULTIMODAL,
    freeze_vision=True,
    freeze_projector=True,
    freeze_language=False,
    compile_cfg=_env_flag("MODEL_COMPILE", False),
)

if IS_MULTIMODAL:
    tokenize_fn = MaskedQwen3VLTokenizeFnConfig(
        chat_template="qwen3.5-vl",
        processor_path=MODEL_PATH,
        media_root=MEDIA_ROOT,
        max_length=SAMPLE_MAX_LENGTH,
        llm_pack_weight=float(os.environ.get("LLM_PACK_WEIGHT", "1.0")),
        visual_pack_weight=float(os.environ.get("VISUAL_PACK_WEIGHT", "0.0")),
        max_pixels=int(os.environ.get("MAX_PIXELS", str(16384 * 32 * 32))),
        rand_video_max_frames=int(os.environ.get("RAND_VIDEO_MAX_FRAMES", "24")),
    )
    dataset_class = "VLMJsonlDataset"
    dataset_name = "vlm"
    collator_name = "src.inputs.masked.masked_qwen3_vl_collator"
else:
    tokenize_fn = MaskedOpenaiTokenizeFunctionConfig(
        chat_template="qwen3.5-vl",
        max_length=SAMPLE_MAX_LENGTH,
    )
    dataset_class = "JsonlDataset"
    dataset_name = "text"
    collator_name = "src.inputs.masked.masked_sft_llm_collator"

dataloader_config = DataloaderConfig(
    dataset_config_list=[
        {
            "dataset": UniversalDatasetConfig(
                name=dataset_name,
                anno_path=DATA_PATH,
                class_name=dataset_class,
                media_root=MEDIA_ROOT,
                sample_ratio=float(os.environ.get("DATASET_SAMPLE_RATIO", "1.0")),
                cache_dir=os.path.join(WORK_DIR, "jsonl_cache"),
                disable_filter=True,
            ),
            "tokenize_fn": tokenize_fn,
        }
    ],
    pack_level=os.environ.get("PACK_LEVEL", "soft"),
    pack_max_length=PACK_MAX_LENGTH,
    pack_to_max_length=True,
    pack_chunk_size=int(os.environ.get("PACK_CHUNK_SIZE", "10000")),
    pack_workers=int(os.environ.get("PACK_WORKERS", "4")),
    global_pack=_env_flag("GLOBAL_PACK", True),
    group_by_length=_env_flag("GROUP_BY_LENGTH", True),
    collator=collator_name,
    pack_extra_buffer_size=int(os.environ.get("PACK_EXTRA_BUFFER_SIZE", "20")),
    num_workers=int(os.environ.get("DATALOADER_NUM_WORKERS", "4")),
)

trainer = TrainerConfig(
    model_cfg=model_cfg,
    load_from=MODEL_PATH,
    tokenizer_path=MODEL_PATH,
    strict_load=True,
    resume_cfg=None,
    fsdp_cfg=FSDPConfig(
        recompute_ratio=float(os.environ.get("RECOMPUTE_RATIO", "1.0")),
        vision_recompute_ratio=float(os.environ.get("VISION_RECOMPUTE_RATIO", "0.0")),
        torch_compile=_env_flag("MODEL_COMPILE", False),
        checkpoint_preserve_rng_state=False,
    ),
    optim_cfg=AdamWConfig(
        lr=float(os.environ.get("LR", "2e-6")),
        weight_decay=float(os.environ.get("WEIGHT_DECAY", "0.01")),
        foreach=False,
    ),
    dataloader_cfg=dataloader_config,
    lr_cfg=LRConfig(
        lr_type="cosine",
        warmup_ratio=float(os.environ.get("WARMUP_RATIO", "0.03")),
        lr_min=float(os.environ.get("LR_MIN", "2e-7")),
    ),
    loss_cfg=CELossConfig(
        mode=os.environ.get("LOSS_MODE", "chunk"),
        chunk_size=int(os.environ.get("LOSS_CHUNK_SIZE", "1024")),
        loss_reduction=os.environ.get("LOSS_REDUCTION", "token"),
    ),
    global_batch_size=int(os.environ.get("GLOBAL_BATCH_SIZE", "16")),
    total_epoch=int(os.environ.get("TOTAL_EPOCH", "2")),
    sp_size=int(os.environ.get("SP_SIZE", "1")),
    checkpoint_interval=int(os.environ.get("CHECKPOINT_INTERVAL", "500")),
    hf_interval=int(os.environ.get("HF_INTERVAL", "500")),
    work_dir=Path(WORK_DIR),
)

# Final HF checkpoint only; no automatic resume or benchmark-based selection.
trainer.seed = 42
trainer.checkpoint_interval = None
trainer.snapshot_interval = None
trainer.hf_interval = None
trainer.auto_resume = False
