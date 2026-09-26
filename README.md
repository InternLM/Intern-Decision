<h1 align="center">Intern-Decision</h1>

<p align="center">
  <a href="https://huggingface.co/spaces/internlm/intern-decision"><strong>🎮 Demo</strong></a>
  &nbsp; · &nbsp;
  <a href="https://huggingface.co/collections/internlm/intern-decision"><strong>🤗 Model Weights</strong></a>
</p>

<p align="center">
  <a href="README_zh-CN.md">简体中文</a> ·
  <a href="docs/DATA.md">Data interface</a> ·
  <a href="docs/EVALUATION.md">Reproduce evaluations</a>
</p>

Intern-Decision turns a state, optional images, and multiple typed questions into
decisions with probabilities. It fine-tunes the language backbone of Qwen3.5
while freezing the vision tower and projector. Supported question types are
`choice`, `score`, and `noul` (yes/no).

This release contains training, two inference backends, benchmark scoring,
temperature calibration, and a browser demo. It also includes the **96-case
[distribution calibration benchmark](docs/CALIBRATION_BENCHMARK.md)**,
its exact references, deterministic generator, and offline scorer, plus the
seven accuracy test suites and checkpoint-specific temperature presets. Training
data, private calibration/validation records, images, preparation pipelines, and
model weights are not included. The runtime schema, tokenization, and collation
support your own records. Demo examples are synthetic interface illustrations.

<h2 align="center">🎬 Demonstrations</h2>

<p align="center">Click any preview to watch the full video.</p>

<table align="center">
  <tr>
    <td align="center" width="50%">
      <p align="center"><strong>🍄 Mario</strong></p>
      <a href="assets/demos/Mario.mp4">
        <img src="assets/demos/Mario.gif" width="420" alt="🍄 Mario" />
      </a>
    </td>
    <td align="center" width="50%">
      <p align="center"><strong>🖱️ Mouse Moving</strong></p>
      <a href="assets/demos/Mouse-Moving.mp4">
        <img src="assets/demos/Mouse-Moving.gif" width="420" alt="🖱️ Mouse Moving" />
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <p align="center"><strong>🛡️ Doom · Defend</strong></p>
      <a href="assets/demos/Doom-Defend.mp4">
        <img src="assets/demos/Doom-Defend.gif" width="420" alt="🛡️ Doom · Defend" />
      </a>
    </td>
    <td align="center" width="50%">
      <p align="center"><strong>💚 Doom · Health Gathering</strong></p>
      <a href="assets/demos/Doom-Health-Gathering.mp4">
        <img src="assets/demos/Doom-Health-Gathering.gif" width="420" alt="💚 Doom · Health Gathering" />
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" colspan="2">
      <p align="center"><strong>🌐 Browser Use</strong></p>
      <a href="assets/demos/Browser-Use.mp4">
        <img src="assets/demos/Browser-Use.gif" width="420" alt="🌐 Browser Use" />
      </a>
    </td>
  </tr>
</table>

## Method

The objective is `autoregressive masked language modeling`. The assistant input contains a complete
JSON skeleton with one `<decision>` token per field. Ground-truth answer symbols
appear only in labels, never in the input. Training uses ordinary causal
next-token alignment: the logit immediately **before** a marker predicts that
field's answer. All fields share one causal forward pass.

Training cross-entropy uses the full vocabulary. Inference restricts logits to
the legal single-token answer symbols, applies softmax, and maps them back to
the original labels. Question and option ordering must be preserved.

## Results

Intern-Decision-4B reaches **90.02% average accuracy** across seven suites (Jev: 88.74%),
**44.16 ms mean local request latency** on an RTX 4090, and **0.550 Brier / 0.089 ECE**
on the 96-case calibration pilot (Jev: 0.595 / 0.130). The protocols and scope differ:
see [accuracy](#benchmark-accuracy), [latency](#rtx-4090-inference-latency), and [calibration](#known-distribution-calibration-benchmark).

### Benchmark accuracy

Accuracy columns are percentages; Average is the unweighted mean of the seven
displayed suites. The seven suites contain 10,751 rows and 12,351 decisions;
per-suite denominators are listed in [Evaluation](#evaluation-seven-suites).
Brier and ECE are **Jevbench-Hard** metrics on 111 items. This table uses
**max(P)** confidence and 10 equal-width bins for ECE, displayed as a fraction;
Brier is the raw multiclass sum. Intern-Decision probabilities use the fitted
temperatures below, with unchanged argmax predictions. Baselines use their
reported/default probabilities; no additional temperature was fitted here.

| Model | Easy | Original | Hard | Typed Decision | ToolACE | AG News | WildJailBreak | Average | Brier ↓ | ECE ↓ | T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [Jev](https://docs.typesafe.ai/api) | 100.00 | 98.61 | 72.07 | 73.35 | 91.29 | 89.57 | 96.29 | 88.74 | 0.358420 | 0.094685 | — |
| [Laya](https://github.com/NandhaKishorM/laya) | 95.83 | 72.22 | 28.83 | 35.95 | 63.87 | 92.84 | 14.84 | 57.77 | 0.804244 | 0.246455 | — |
| [SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev) | 100.00 | 98.61 | 61.26 | 62.80 | 85.16 | 89.22 | 92.53 | 84.23 | 0.498012 | 0.112213 | 1.0 |
| [Kev](https://github.com/jaredpalmer/kev) | 100.00 | 93.06 | 45.05 | 65.60 | 87.42 | 89.82 | 75.97 | 79.56 | 0.738253 | 0.262200 | 2.962195 |
| [JevK5](https://github.com/allebee/jevk5) | 100.00 | 97.22 | 73.87 | 64.50 | 80.97 | 89.13 | 90.45 | 85.16 | 0.366162 | 0.046672 | 1.532 |
| Intern-Decision-0.8B | 97.92 | 80.56 | 52.25 | 77.35 | 94.52 | 88.61 | 64.48 | 79.38 | 0.529519 | 0.065710 | 2.747761 |
| Intern-Decision-2B | 100.00 | 84.72 | 63.96 | 79.35 | 96.45 | 89.96 | 78.33 | 84.68 | 0.437256 | 0.100208 | 2.100509 |
| Intern-Decision-4B | 100.00 | 98.61 | 73.87 | 80.55 | 96.45 | 90.82 | 89.86 | 90.02 | 0.346762 | 0.065267 | 1.992418 |

See [the reproduction guide](docs/EVALUATION.md).

## RTX 4090 inference latency

Local measurements use one NVIDIA RTX 4090, BF16, SDPA, native Hugging Face
inference, Transformers 5.14.1 and flash-linear-attention 0.4.2. Each request
contains 289 input tokens and three fields (choice, yes/no and score), answered
in one forward pass with the checkpoint-specific temperature applied.

| **Model** | **Mean** | **Median / P50** | **P95** |
| -------------------- | --------- | ---------------- | --------- |
| Jev | 109.70 ms | 106.30 ms | 146.70 ms |
| Intern-Decision-0.8B | 33.98 ms | 33.44 ms | 37.50 ms |
| Intern-Decision-2B | 33.28 ms | 33.15 ms | 33.55 ms |
| Intern-Decision-4B | 44.16 ms | 44.03 ms | 44.60 ms |

## Known-distribution calibration benchmark

The bundled [96-case benchmark](benchmarks/known-distribution-pilot-v1/README.md)
is a small calibration pilot with six categories, 24 families and 48 paired
settings. Questions ask for random
outcomes; exact reference distributions are stored separately from model inputs.
It is evaluation-only: do not fit temperatures or train on its references.

### Evaluation results

All three settings have **96/96 valid predictions**, with 16 cases per category.
Intern-Decision-4B is evaluated with its final checkpoint and fixed XTuner backend. Temperature **T=1.992418**
was fitted on a separate calibration split; this benchmark was not used for
fitting or selection. Calibration preserved every predicted argmax label.
Jev results use `jev-1.13.0`.

Each cell is **expected multiclass Brier / expected ECE**; lower is better.
Brier includes irreducible outcome randomness. ECE uses 10 equal-width bins
and valid returned confidence, falling back to max(P), unlike the max(P)-only
hard-benchmark table above. These are scores
against exact reference distributions, not sampled-label accuracy.

| Category | Intern-Decision-4B, uncalibrated | Intern-Decision-4B, calibrated | Jev |
|---|---:|---:|---:|
| Direct randomness and support | 0.483 / 0.181 | 0.421 / 0.129 | 0.490 / 0.216 |
| Composed events and mixtures | 0.677 / 0.254 | 0.577 / 0.150 | 0.682 / 0.274 |
| History, conditioning, and hidden state | 0.711 / 0.219 | 0.613 / 0.108 | 0.657 / 0.113 |
| Daily evidence and observation bias | 0.701 / 0.328 | 0.575 / 0.210 | 0.603 / 0.114 |
| Selective disclosure and probability puzzles | 0.540 / 0.119 | 0.510 / 0.049 | 0.483 / 0.138 |
| Sequential and combinatorial processes | 0.656 / 0.180 | 0.605 / 0.058 | 0.657 / 0.116 |
| **Overall (pooled)** | 0.628 / 0.213 | 0.550 / 0.089 | 0.595 / 0.130 |

Calibration reduces overall Brier from **0.628 to 0.550** and expected ECE
from **0.213 to 0.089**, below Jev's **0.595 / 0.130** on this pilot.
Overall Brier averages the 96 cases; overall ECE is recomputed across all
cases and is **not** the arithmetic mean of category ECE. The paired option
variants make this a small diagnostic study, not 96 independent trials.

### Run the evaluation

Score your saved model/API predictions using Python 3.10+ (standard library only):

```bash
python -m src.eval.score_known_distribution \
  --dataset benchmarks/known-distribution-pilot-v1 \
  --predictions /path/to/predictions.jsonl \
  --output outputs/calibration-benchmark
```

The fresh output directory contains `summary.md` (category and overall Brier/ECE),
`metrics.json` (all metrics and bins), and `scored.jsonl` (per-case scores).
Overall ECE pools all cases; it is not the mean of category ECE.
See [the scoring guide](docs/CALIBRATION_BENCHMARK.md) for response formats,
inference examples, formulas, validation and limitations.

## Installation

Use Linux, Python 3.12, and CUDA for the trained backends. Run commands from
this repository root. Keep HF and XTuner in separate virtual environments:
their tested Transformers versions differ. Checkpoints are local HF directories
including weights, processor, tokenizer, chat template, and `<decision>` token.

### Native Hugging Face inference

```bash
python3.12 -m venv .venv-hf
source .venv-hf/bin/activate
python -m pip install -r requirements-inference.txt -r requirements-eval.txt
```

This backend loads `Qwen3_5ForConditionalGeneration` directly and requires no
XTuner import or compatibility patch. Do not put `runtime/xtuner` on its
`PYTHONPATH`. A compatible PyTorch CUDA wheel must be available for your system.

### XTuner training and inference

Model execution uses XTuner with the supplied runtime adaptations. The
upstream source revision is pinned; no third-party source or datasets are
vendored into this publication folder.

```bash
python3.12 -m venv .venv-xtuner
source .venv-xtuner/bin/activate
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-xtuner.txt -r requirements-eval.txt
mkdir -p third_party
git clone https://github.com/InternLM/xtuner.git third_party/xtuner
git -C third_party/xtuner checkout fb51baebdf91b03bd39255c4427edac9aca82865
python -m pip install --no-deps -e third_party/xtuner
# Install a flash-attn 2.8.3 wheel/build matching your Torch/CUDA toolchain.
python -m pip install flash-attn==2.8.3 --no-build-isolation
export XTUNER_PATH="$PWD/third_party/xtuner"
```

The explicit dependency override preserves Transformers 4.57.0 for this
XTuner path. `runtime/xtuner/sitecustomize.py` supplies the tested Torch 2.6
compatibility hooks; launchers enable it only for XTuner. CUDA development
tools may be needed by optional extensions. The HF environment remains
independent. Different kernels and BF16 rounding can produce probability
differences between backends; do not claim bitwise backend equivalence.

## Inference

Create a request JSON using [docs/DATA.md](docs/DATA.md), then:

```bash
export MODEL_CHECKPOINT=/path/to/checkpoint
python -m src.inference --backend hf --input /path/to/request.json
```

For XTuner, activate its environment and explicitly enable its runtime:

```bash
export PYTHONPATH="$PWD/runtime/xtuner:$PWD:$XTUNER_PATH"
export XTUNER_HF_IMPL=1 XTUNER_USE_FA3=0
python -m src.inference --backend xtuner --input /path/to/request.json
```

The XTuner backend uses CUDA/BF16. HF additionally supports `device`, `dtype`
and `attn_implementation` in `configs/inference/default.json`. Set `MODEL_PATH`
only if the matching processor is stored separately. `MEDIA_ROOT` resolves
local image paths for CLI use. `CALIBRATION_PATH` enables a checkpoint-specific
calibration artifact; otherwise temperature is 1.0. Do not reuse a temperature
from another checkpoint or treat a sampling temperature as calibration.

## Training on your own data

Supply an already prepared JSONL file matching the documented contract.
The loader does not perform dataset construction, relabeling, split generation,
or format conversion. Review your records and splits before training.

```bash
export MODEL_PATH=/path/to/base-qwen35-instruction-checkpoint
export DATA_PATH=/path/to/your/train.jsonl
export MEDIA_ROOT=/path/to/your/media  # omit for text-only data
export WORK_DIR="$PWD/outputs/train-001"  # must not exist
export NPROC_PER_NODE=4
bash scripts/train.sh
```

`configs/training/qwen35.py` reads architecture dimensions from the base
checkpoint and supports the 0.8B, 2B, 4B and 9B dense Qwen3.5 variants.
The launcher records the user-selected configuration and hashes, performs
training on user-supplied records, and saves only the final HF checkpoint. It does not automatically
resume a prior run. Runtime tokenization rejects invalid/overlength rows.
`training-config.json` is separate from the HF checkpoint's `config.json`.

Verify the saved weights and initial per-rank load audits:

```bash
python -m scripts.verify_checkpoint --base "$MODEL_PATH" \
  --checkpoint /path/to/final/hf-checkpoint \
  --audit-dir "$WORK_DIR/weight-audit" --ranks 4 \
  --output "$WORK_DIR/checkpoint-verification.json"
```

## Evaluation: seven suites

Install the pinned Jevbench scoring package via `requirements-eval.txt`.
The exact evaluation records are bundled under `benchmarks/accuracy-v1/`.
Follow [docs/EVALUATION.md](docs/EVALUATION.md) to verify them and reproduce both
tables. No dataset conversion scripts are needed.

| Suite | Rows | Decisions |
|---|---:|---:|
| Jevbench Easy / Original / Hard | 48 / 72 / 111 | 48 / 72 / 111 |
| AG News | 7,600 | 7,600 |
| ToolACE | 310 | 310 |
| Typed Decision | 400 | 2,000 |
| WildJailBreak | 2,210 | 2,210 |

```bash
python -m src.eval.verify_bundle
python -m src.eval.jev --backend hf --checkpoint "$MODEL_CHECKPOINT" \
  --suite --output outputs/eval-t1
```

Use `--backend xtuner` in the XTuner environment for the reported scoring
backend. `--public-only` runs the three public Jevbench suites without requiring
the other datasets. The default paths use the bundled records; `--test-root`,
`EVAL_DATA_ROOT` and `JEVBENCH_DATA_ROOT` can override them for separate experiments.

Outputs include raw candidate probabilities, correct/total counts, accuracy,
ECE, Brier, hard TVD, input hashes, and checkpoint hashes. Hard TVD is evaluated
only on the 10 public items with `gold_probs`. Gold labels and distributions
never enter the model prompt. For multiple processes, use `--worker R --workers N`
on distinct GPUs, then the same command with `--merge --workers N` after all
workers succeed. Merge validates prediction identity, coverage, hashes,
temperature, and backend.

## Selecting temperature

For the released models, first use the hash-verified presets described in
[docs/EVALUATION.md](docs/EVALUATION.md). The following is for your own models.

Fit one positive scalar per checkpoint by **NLL on calibration data only**.
Supply your own disjoint records with `validation_role` set to `calibration`
or `selection`; private calibration and validation split details are not disclosed.
Keep calibration and validation disjoint from training and benchmark tests.

```bash
python -m src.eval.collect --backend hf --checkpoint "$MODEL_CHECKPOINT" \
  --data /path/to/your/validation.jsonl --output outputs/validation-predictions.jsonl
python -m src.eval.fit --checkpoint "$MODEL_CHECKPOINT" \
  --predictions outputs/validation-predictions.jsonl --output outputs/calibration
python -m src.eval.replay --checkpoint "$MODEL_CHECKPOINT" \
  --baseline outputs/eval-t1 --calibration outputs/calibration/calibration.json \
  --output outputs/eval-calibrated
export CALIBRATION_PATH="$PWD/outputs/calibration/calibration.json"
```

Again, use XTuner collection for an XTuner evaluation. Optional flags
`--expected-fit-rows` and `--expected-validation-rows` validate counts you
specify for your own data. The optimizer bisects the convex NLL derivative in inverse
temperature over T ∈ [0.01, 100] and saves the search curve and separate
validation scores. The artifact is fixed before benchmark replay.

Replay applies `softmax(log(p) / T)` to saved candidate probabilities and
checks **zero changed decisions**. It writes before/after metrics and
`verification.json`. It does not optimize benchmark ECE or select checkpoints.
NLL-optimal temperature need not minimize ECE on another dataset. Calibration
artifacts are bound to checkpoint metadata hashes; regenerate one after
changing weights. If relocating a checkpoint, update its artifact's
`checkpoint` path only after checking all stored file hashes.

## Browser demo and HTTP API

```bash
export MODEL_CHECKPOINT=/path/to/checkpoint
export INFERENCE_BACKEND=hf  # or xtuner in its separate environment
bash scripts/demo.sh
```

Open <http://127.0.0.1:7860>. The demo supports typed questions, multiple
image uploads, probabilities, and calibrated confidence. API endpoints:
`POST /v1/decisions` (alias `/v1/jev`), `GET /health`, and interactive `/docs`.
HTTP accepts uploaded image bytes, not server file paths or remote URLs.
Uploads are temporary and removed after inference. CLI local paths are for
trusted callers. Default serving is loopback, with no request-content logging.
Add authentication and request-size limits before exposing it publicly.

An optional external thinking handoff is disabled by default. To enable it,
configure your own model in the inference config and provide `LLM_BASE_URL`
and `LLM_API_KEY` through the environment. Only explicitly enabled requests
can send evidence to that provider; handoff outputs are not calibrated local
probabilities and are not used in the reported evaluations.

## Checks and layout

```bash
python -m unittest tests.check_inference tests.check_image_uploads tests.test_release
python -m tests.check_temperature
```

These CPU checks use synthetic fixtures and mocked models; they do not launch
training, claim checkpoint quality, or replace GPU validation of a new runtime.
In the XTuner environment, with `MODEL_PATH` pointing to a local base processor
and the XTuner `PYTHONPATH` above, run `python -m tests.check_training_runtime`
for CPU checks of causal label alignment, packing, image tokens and convolution
output/gradient parity. It creates synthetic media temporarily and loads no model weights.

```text
configs/         Portable inference and training configuration
src/inputs/      Runtime schema, tokenization and collation only
src/model/       Qwen architecture and frozen-vision training adaptations
src/inference/   Native HF and XTuner engines, temperature scaling
src/eval/        Scoring, calibration collection/fitting and replay
src/service/     HTTP endpoint and browser demo
runtime/xtuner/  Explicit XTuner compatibility environment
scripts/         Training launcher and checkpoint verification
tests/           Synthetic CPU regression checks
docs/DATA.md     Expected user-supplied input format
```

Third-party dependencies retain their own licenses: Qwen model terms apply to
weights; XTuner is Apache-2.0; Jevbench is MIT. Neither weights nor third-party
repositories are redistributed in this folder.
