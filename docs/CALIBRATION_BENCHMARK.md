# Known-distribution calibration benchmark

This synthetic, evaluation-only pilot contains **96 cases across six categories**:
24 families, two parameter settings per family, and two option-order variants per
setting. Each category has 16 cases. Variants share a group; there are 48 distinct
parameter settings, not 96 independent trials.

| Category | 中文 | Example families |
|---|---|---|
| Direct randomness and support | 直接随机性与可能性支持集 | Fair dice, candy jars, certain outcomes, rare wins |
| Composed events and mixtures | 复合事件与混合分布 | Dice sums, independent hits, backup reliability, route mixtures |
| History, conditioning, and hidden state | 历史信息与条件概率 | Coin streaks, socks without replacement, restricted observations, hidden states |
| Daily evidence and observation bias | 日常证据与观测偏差 | Screening, repeated/correlated tests, bus waiting |
| Selective disclosure and probability puzzles | 选择性披露与概率谜题 | Informed/uninformed Monty Hall, selective offers, child-selection protocols |
| Sequential and combinatorial processes | 序列过程与组合过程 | Sticker collection, birthday collisions, gambler's ruin, guaranteed rewards |

The questions ask for **outcomes**, such as “What color is the selected sock?”,
not “Which action maximizes success?” or “Compute the probability.” The model's
native choice interface supplies probabilities over the options. The reference
is the exact distribution of outcomes under the stated mechanism. Selecting its
most likely outcome is not proof that a future random outcome was predicted
correctly. No sampled labels or hard-label accuracy are used here.

## Included files

Under `benchmarks/known-distribution-pilot-v1/`:

- `inputs.jsonl`: only `id`, `state`, and `questions`; 96 model-facing records.
- `references.jsonl`: exact rational probabilities, numeric probabilities,
  category/family, paired group IDs, parameters, and derivations.
- `manifest.json`: revision, counts, SHA-256 hashes, and `training_ready: false`.
- `README.md`: every base case and its derivation; evaluator-facing, not evidence.

Only the input file should be used for inference. Preserve its original option
ordering, full state, question instructions, and IDs. Never send the references,
derivations, dataset README, or expected answers to the model. Reject overlength
inputs instead of truncating them. Fit any temperature on a separate dataset;
do not select checkpoints or temperatures using this evaluation set.

## Obtain predictions

For the complete checkpoint-to-table workflow, including fixed temperature
presets and the raw/calibrated runner, see [EVALUATION.md](EVALUATION.md).

Use the repository's configured inference environment for local models. The
following example uses the existing HF backend; the XTuner backend works in its
separate environment. This example does not load references or labels:

```python
import json
from pathlib import Path

from src.inference.engine import DecisionEngine

engine = DecisionEngine(
    checkpoint="/path/to/checkpoint",
    processor_path="/path/to/matching/processor",
    backend="hf",
    # calibration_path="/path/to/prefitted/calibration.json",
)
dataset = Path("benchmarks/known-distribution-pilot-v1/inputs.jsonl")
with dataset.open(encoding="utf-8") as source, Path("predictions.jsonl").open(
    "x", encoding="utf-8"
) as output:
    for line in source:
        row = json.loads(line)
        response = engine.predict({"state": row["state"], "questions": row["questions"]})
        output.write(json.dumps({**response, "id": row["id"]}, ensure_ascii=False) + "\n")
```

For hosted Jev, send only `state`, `questions`, and the requested `model` in the
API request. Retain the ID locally and attach it to the saved response. Use your
own API credentials through the environment, never commit them. Other models
can be evaluated by mapping their native probabilities to the same label IDs;
do not invent probabilities for a label-only model. Record checkpoint/model
version, backend, prompt revision, any temperature, and hashes of raw predictions.

Accepted prediction formats (one JSON object per physical line):

```json
{"id":"known-distribution-pilot-v1/monty_informed_host/01/canonical","model":"your-model","answers":{"decision":{"type":"choice","choice":"other_unopened_door","probabilities":{"initial_door":0.4,"other_unopened_door":0.6},"confidence":0.6}}}
```

These numbers illustrate the response format, not an actual model result.
Alternatively, wrap the response as
`{"id": "...", "status": 200, "response": {"model": "...", "answers": {...}}}`.
Prediction lines may be reordered: the scorer joins by exact ID. Probability
keys must cover exactly the options in that question.

## Score offline

Run from the release repository root with **Python 3.10+**. Scoring needs only
the standard library: no pip installation, model, CUDA, network, or API key.

```bash
python -m src.eval.score_known_distribution \
  --dataset benchmarks/known-distribution-pilot-v1 \
  --predictions predictions.jsonl \
  --output outputs/known-distribution/my-model
```

Use a fresh output directory for each model/temperature. Outputs:

- `summary.md`: category rows and a pooled overall row, with expected Brier,
  expected ECE, excess expected Brier, TVD, and valid/total counts.
- `metrics.json`: full scores, confidence bins, category/family/difficulty/variant
  breakdowns, impossible-outcome mass, option-order sensitivity, and hashes.
- `scored.jsonl`: normalized probabilities and per-case scores/errors.

Unknown or duplicate IDs and corrupted dataset hashes are rejected. Missing or
invalid answers reduce coverage, are excluded from numeric metrics, and cause a
nonzero exit status. They are never replaced by guessed probabilities. Every
comparison should report coverage; the complete benchmark requires 96/96 valid
outputs. Invalid JSON is a fatal input error.

## Exact scoring definitions

For case i, let p be the normalized model probability vector and q the exact
reference distribution (converted from rational fractions for scoring).

- **Expected multiclass Brier** = sum_k (p_k - q_k)^2 + 1 - sum_k q_k^2.
  Equivalently, average the ordinary multiclass Brier score over all possible
  outcomes y weighted by q_y. This uses a sum over classes, not division by the
  number of classes, and includes irreducible randomness.
- **Excess expected Brier** = sum_k (p_k - q_k)^2. Zero is optimal. A perfect
  uniform six-sided die prediction has excess Brier 0 and expected Brier 5/6.
- **TVD** = 0.5 * sum_k abs(p_k - q_k).
- **Expected ECE**: select k* = argmax(p), breaking ties lexicographically. The
  expected correctness is q_k*, not a hard 0/1 label. Use a finite numeric
  `confidence` in [0,1] when supplied (including zero); otherwise use max(p).
  Group cases into 10 equal-width bins [0,.1), ..., [.9,1]. ECE is the sum over
  nonempty bins of (bin count / valid count) * abs(mean confidence - mean
  expected correctness). The supplied `choice` is retained for inspection;
  scoring uses argmax(p). Confidence should describe that selected outcome.
- **Impossible-outcome mass** sums p_k for q_k = 0; average it only over cases
  containing impossible options.
- **Option-order sensitivity** is mean TVD between paired original/reversed
  predictions, aligned by outcome label.

Probabilities must be finite numbers in [0,1] with sums within 0.02 of one.
All accepted maps are normalized, including small float roundoff. Sums farther
than 0.001 from one are additionally flagged as rounded responses.

**Overall Brier is the mean across valid cases. Overall ECE pools all cases and
recomputes bins; it is not the mean of category ECE.** Because each category has
16 cases, the mean of category Brier equals overall Brier when coverage is full.
The per-case score of a model matching q can still have nonzero confidence ECE
if its separate confidence differs from max(p). Thus full-distribution quality
and scalar-confidence calibration are distinct measurements.

This is a small diagnostic pilot, not a full leaderboard. Use paired parameter
groups when estimating uncertainty, and do not interpret 96 rows as 96
independent experiments. Synthetic medical/financial wording makes no claim
about real-world event rates.

## Reproduce and validate

The deterministic generator uses exact fractions, enumeration, dynamic
programming, and analytic formulas. Regeneration requires a new directory:

```bash
python -m src.eval.known_distribution --output outputs/regenerated-pilot
python -m pip install pytest
python -m pytest tests/test_known_distribution.py tests/test_benchmark_bundle.py -q
```

Tests independently enumerate Monty Hall, child selection, dice/binomial,
sticker and sensor cases, solve gambler's ruin with rational linear algebra,
and check scoring, confidence fallback, prompt isolation, hashes, and coverage.
Oracle predictions are test fixtures only, never reported model results.
