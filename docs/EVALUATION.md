# Reproducing evaluation tables / 评测复现指南

This guide reproduces the Intern-Decision accuracy rows and the known-distribution
calibration pilot from local checkpoints. The test records, scoring code, and
fixed temperature presets are included. Weights are obtained separately. Training
data and private temperature-fitting/validation records are not included.

本指南提供准确率表和校准 pilot 表的完整运行顺序。测试集、评分代码和固定温度预设
已包含在仓库中；权重需单独准备。不需要内部校准数据即可应用预设并复现评测流程。
所有命令从仓库根目录执行，输出目录必须是新目录。

## 1. Install the reported inference backend

Use Linux, Python 3.12 and a CUDA GPU with sufficient memory. Follow the
[XTuner installation commands](../README.md#xtuner-training-and-inference),
including the pinned XTuner commit and evaluation requirements. Then:

```bash
source .venv-xtuner/bin/activate
export XTUNER_PATH="$PWD/third_party/xtuner"
export PYTHONPATH="$PWD/runtime/xtuner:$PWD:$XTUNER_PATH"
export XTUNER_HF_IMPL=1 XTUNER_USE_FA3=0
```

Both tables' Intern-Decision rows used XTuner. The [HF backend](../README.md#native-hugging-face-inference)
also supports accuracy inference with `--backend hf` in its separate environment.
Kernel and BF16 differences can change probabilities and occasionally labels;
report HF results as a different execution setting. Released presets are bound
to XTuner. For HF calibration, fit a new artifact using step 7 in the HF environment.

复现已报告数字请选择 XTuner。两个后端分别安装，不要在 HF 环境中加载 XTuner 的
兼容适配。硬件、算子或依赖版本变化可能引入数值差异，不保证跨环境逐位相同。

## 2. Verify the included test sets

```bash
python -m src.eval.verify_bundle
```

This CPU-only command checks SHA-256 hashes and counts for both bundles.
Both lines must report `"passed": true`. The accuracy bundle is:

| Suite | File under `benchmarks/accuracy-v1/` | Rows | Decisions |
|---|---|---:|---:|
| Jevbench-Easy | `jevbench/easy.jsonl` | 48 | 48 |
| Jevbench-Original | `jevbench/original.jsonl` | 72 | 72 |
| Jevbench-Hard | `jevbench/hard.jsonl` | 111 | 111 |
| Typed Decision | `typed_decisions/test.jsonl` | 400 | 2,000 |
| ToolACE | `toolace/test.jsonl` | 310 | 310 |
| AG News | `agnews/test.jsonl` | 7,600 | 7,600 |
| WildJailBreak | `wildjailbreak/test.jsonl` | 2,210 | 2,210 |
| Total | | 10,751 | 12,351 |

The [accuracy manifest](../benchmarks/accuracy-v1/manifest.json) records file
hashes and upstream attribution. Model evidence, field/option order and scoring
labels match the evaluated records; internal provenance metadata is excluded.
The upstream datasets retain their own terms; repository code licensing does
not grant new dataset rights. The bundle includes the upstream Jevbench MIT
notice and the Apache-2.0 license text for sources declaring that license. AG News's source metadata reports an unknown
license; consult its upstream terms before further redistribution or use.

The [pilot bundle](../benchmarks/known-distribution-pilot-v1/README.md) contains
96 inputs and separate exact-distribution references. Its 48 paired settings
are not 96 independent trials. Neither bundle may be used to fit temperature
or choose a checkpoint in this evaluation protocol.

## 3. Choose a checkpoint and bind its temperature

Obtain the corresponding released Intern-Decision checkpoint and put the entire
HF directory locally, including weight shards, index, tokenizer, processor,
configuration and chat template. Use the matching Intern-Decision model size;
do not use an upstream Qwen model in place of the fine-tuned checkpoint.

For the 4B model:

```bash
export MODEL_CHECKPOINT=/path/to/Intern-Decision-4B
export MODEL_KEY=intern-decision-4b
export OUT=outputs/intern-decision-4b
python -m src.eval.use_temperature --model "$MODEL_KEY" \
  --checkpoint "$MODEL_CHECKPOINT" --output "$OUT/calibration.json"
```

For other sizes, change all three variables and repeat steps 3–4:

| Table model | Preset key | Fixed T (display rounded) |
|---|---|---:|
| Intern-Decision-0.8B | `intern-decision-0.8b` | 2.747760550702957 |
| Intern-Decision-2B | `intern-decision-2b` | 2.100509348277736 |
| Intern-Decision-4B | `intern-decision-4b` | 1.9924182353655278 |

The command checks inference metadata **and weight-shard hashes**, then writes
a calibration artifact bound to your local path. Hash mismatches are errors;
do not edit the preset to force a different checkpoint to pass. Verification
reads the weight files on CPU and can take time. No temperature search occurs
on the test data. The full-precision values in
[temperature-presets.json](../benchmarks/temperature-presets.json) are used.

温度由独立校准记录上的 NLL 搜索得到。预设只保存应用温度所需的信息，不包含内部
拟合数据。权重校验失败时，应检查是否下载了正确版本，不能忽略哈希检查。

## 4. Evaluate at T=1, then replay the fixed temperature

Remove any custom dataset-root overrides before using the bundled defaults:

```bash
unset JEVBENCH_DATA_ROOT EVAL_DATA_ROOT
python -m src.eval.jev --backend xtuner --checkpoint "$MODEL_CHECKPOINT" \
  --suite --output "$OUT/accuracy-t1"
python -m src.eval.replay --checkpoint "$MODEL_CHECKPOINT" \
  --baseline "$OUT/accuracy-t1" --calibration "$OUT/calibration.json" \
  --output "$OUT/accuracy-calibrated"
```

The first command runs inference for all seven suites. Its `complete.json`
contains counts, metrics, data/source/checkpoint hashes; each suite also has
raw `*.predictions.jsonl` and metrics. The second command uses saved probabilities
on CPU, applies `softmax(log(p)/T)`, and verifies **zero changed decisions**.
Require `accuracy-calibrated/verification.json` to report `passed: true` and
`changed_predictions: 0`. Its `comparison.json` holds before/after metrics.

One process is sufficient. For parallel inference, run the same first command
once per GPU with distinct `--worker R --workers N` arguments and a shared fresh
output root, setting `CUDA_VISIBLE_DEVICES` individually. After every worker
succeeds, run that command with `--merge --workers N` instead of worker arguments.
Merge checks identities, coverage, input hashes, backend and temperature before
creating `complete.json`. Do not run a serial evaluation into the worker output.

## 5. Generate the score-only accuracy table

After completing the three sizes:

```bash
python -m src.eval.tables --kind accuracy \
  --report 'Intern-Decision-0.8B=outputs/intern-decision-0.8b/accuracy-calibrated/comparison.json' \
  --report 'Intern-Decision-2B=outputs/intern-decision-2b/accuracy-calibrated/comparison.json' \
  --report 'Intern-Decision-4B=outputs/intern-decision-4b/accuracy-calibrated/comparison.json' \
  --output outputs/accuracy-table.md
```

The renderer uses actual correct/total counts, with accuracy displayed as numbers
such as `98.61`, without counts or percent signs. Average is the unweighted
mean of the **seven displayed suites**. For reference, the published seven-suite averages
are 79.38, 84.68 and 90.02. Use unrounded values when computing the average.

Brier and ECE are calibrated **Jevbench-Hard** metrics, not averages over suites.
Brier is the multiclass sum `sum((p[k] - 1[k == y])**2)`; ECE uses maximum
candidate probability and 10 equal-width bins, displayed as a fraction.
Hard TVD is available in the JSON report for the 10 public `gold_probs` cases.

## 6. Run and render the calibration pilot

Restore the 4B variables from step 3. The runner makes one unscaled model forward
per input, writes raw and calibrated responses, checks unchanged argmax labels,
and scores both with the same exact references:

```bash
python -m src.eval.run_pilot --backend xtuner --checkpoint "$MODEL_CHECKPOINT" \
  --calibration "$OUT/calibration.json" --output "$OUT/pilot"
python -m src.eval.tables --kind pilot \
  --report "Intern-Decision-4B, uncalibrated=$OUT/pilot/raw/metrics.json" \
  --report "Intern-Decision-4B, calibrated=$OUT/pilot/calibrated/metrics.json" \
  --output outputs/pilot-table.md
```

Require `pilot/verification.json` to pass and both metrics files to show 96 valid
rows with `complete: true`. Each category cell is **expected Brier / expected
ECE**. Expected Brier includes irreducible outcome randomness. Pilot ECE uses
valid returned confidence (otherwise max(P)), and reference probability of the
predicted outcome as expected correctness. Overall ECE pools all cases; it is
not the mean of category ECE. See [metric definitions](CALIBRATION_BENCHMARK.md).
The published 4B pooled values round to `0.628 / 0.213` before calibration and
`0.550 / 0.089` after it.

For an external model/API, send **only `state` and `questions` from `inputs.jsonl`**.
Keep IDs outside the prompt for response matching. Never send references,
derivations, labels or expected distributions. Save one native response per row:

```json
{"id":"<input ID>","answers":{"decision":{"type":"choice","probabilities":{"<label>":0.5,"<other label>":0.5},"confidence":0.5}}}
```

Use each input's exact label set and actual returned probabilities. Then:

```bash
python -m src.eval.score_known_distribution \
  --dataset benchmarks/known-distribution-pilot-v1 \
  --predictions /path/to/external-predictions.jsonl --output outputs/external-pilot
```

Add `--report 'Jev=outputs/external-pilot/metrics.json'` to the pilot table command
to include that result. The reported Jev row used `jev-1.13.0`; a hosted service
may change. External accuracy baselines likewise require their specific models,
adapters and outputs. This release does not include those baseline runners or
historical responses, so the commands here regenerate Intern-Decision rows directly,
not the complete external-baseline comparison from nothing.

## 7. Select a temperature for your own model (optional)

Supply your own canonical records with `validation_role` equal to `calibration`
or `selection` as described in [DATA.md](DATA.md#calibration-split). They must be
disjoint from training, the two bundled test sets, and each other. Never feed
either bundled test set to the fitting command. Use the same backend for
collection and benchmark inference:

```bash
python -m src.eval.collect --backend xtuner --checkpoint "$MODEL_CHECKPOINT" \
  --data /path/to/your/validation.jsonl --output outputs/custom-validation.jsonl
python -m src.eval.fit --checkpoint "$MODEL_CHECKPOINT" \
  --predictions outputs/custom-validation.jsonl --output outputs/custom-calibration
```

`fit` minimizes NLL on calibration rows using the convex inverse-temperature
derivative over T in [0.01, 100]. Selection rows report separate validation
metrics and do not choose T. Inspect `search-curve.json`, boundary behavior and `validation.json`. Freeze `calibration.json` before steps 4 and 6. The NLL
optimum need not minimize ECE on another distribution. The private search data
behind released presets are not distributed, so their fitting experiment cannot
be independently reconstructed from this release.

## Interpretation

Public Jevbench is a development diagnostic, not independent held-out evidence
or the complete leaderboard. Pilot variants are paired and form a small diagnostic
study. Report checkpoint, backend, dependencies, temperature, test hashes and raw
outputs together. A matching number of rows alone does not prove matching data.

中文摘要：先校验测试文件，再绑定对应权重的温度，运行 T=1 评测并离线重放校准，
最后由 JSON 结果生成表格。准确率按决策统计，Average 为七项准确率的算术平均；准确率表
中的 Brier/ECE 来自 Hard，pilot 则使用精确参考分布的期望指标。测试集不能用于训练、
温度搜索或 checkpoint 选择。
