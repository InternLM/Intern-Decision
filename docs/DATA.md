# Data interface / 数据格式

This document describes the runtime data interface. The release includes the
accuracy test suites and known-distribution pilot described in
[EVALUATION.md](EVALUATION.md). Training data, private calibration/validation
records, media, and their preparation pipelines are excluded. The example below
is synthetic and was written for this documentation.

本文说明运行时数据接口；已发布的测试数据与复现步骤见 [EVALUATION.md](EVALUATION.md)。
不发布训练数据、内部校准/验证记录、图片及其处理流程。下面是独立编写的合成示例。
本文不描述模型训练数据的来源、组成、数量或配比。
This interface specification does not disclose the models' training data sources,
composition, quantities, or mixing proportions.

## Canonical record

Training and non-Jevbench evaluation read UTF-8 JSONL: one complete JSON object
per physical line. The expanded object below is for readability. Inference
accepts a single JSON object without `targets`. Do not supply exported chat
messages as training records.

```json
{
  "id": "synthetic-format-example",
  "state": {"left_box": "red", "right_box": "blue"},
  "images": [],
  "questions": {
    "box": {
      "type": "choice",
      "instructions": "Which box is red?",
      "criteria": {"left": "The left box", "right": "The right box"}
    },
    "red_exists": {
      "type": "noul",
      "instructions": "Is at least one box red?"
    },
    "red_count": {
      "type": "score",
      "instructions": "How many boxes are red?",
      "criteria": ["Zero", "One", "Two"]
    }
  },
  "targets": {
    "box": {"label": "left"},
    "red_exists": {"label": "yes"},
    "red_count": {"label": "1"}
  }
}
```

| Field | Contract |
|---|---|
| `id` | Stable row identity; required by evaluation and calibration |
| `state` | JSON value containing only visible evidence |
| `questions` | Nonempty object; preserve field order |
| `type` | `choice`, `score`, or `noul` |
| `instructions` | The question or decision rule |
| `criteria` | `choice`: ordered label→description object; `score`: ordered list or numeric-key object; optional for `noul` |
| `targets[field].label` | Required hard label for training/evaluation; not a prompt input |
| `images` | Optional ordered list of local image paths for training/CLI; empty for text-only rows |
| `validation_role` | Calibration collection only: `calibration` or `selection` |

`choice` labels follow the insertion order of `criteria`; the compiler maps
them to `A`, `B`, … without changing their meaning. `score` list labels are
strings `"0"`, `"1"`, … . `noul` always orders labels as `no`, `yes`.
The runtime has 62 single-token candidate symbols. The HTTP interface accepts
1–16 fields per request; do not mistake that service limit for a dataset split.

问题和选项顺序必须保持不变。`targets` 只提供监督标签；ID、来源、标准答案、解释和
`gold_probs` 不会作为模型证据。训练使用硬标签，不使用软概率目标。
历史记录中的 `questions[field].answer` 也可被读取，但建议统一使用 `targets`，
不要在两个位置填写互相矛盾的答案。

## Images

For trusted training/CLI records, use e.g. `"images": ["view-1.png", "view-2.png"]`.
Relative paths resolve against `MEDIA_ROOT`; absolute paths are also supported.
Multiple images retain their input order. All referenced files must exist.
No images or example media files are included in this release. The tokenizer
rejects failed or overlength records rather than replacing them or dropping them.

HTTP requests use image bytes instead of local paths:

```text
"images": [{"type": "image/png", "data": "<base64-encoded image bytes>"}]
```

That placeholder is a format description, not an executable request. Data URLs
are also accepted. The demo constructs uploads automatically. Limits: 8 images,
12 MB per upload, 32 MB combined, 16 million pixels per image, static images
only. Client filenames never become server paths.

HTTP 上传只接受图片内容，不接受服务器文件路径或远程 URL；服务会校验并清理临时文件。

## Evaluation contracts

Public Jevbench uses its upstream single-question format (`question`,
`expected`, and optional `gold_probs`); keep those files unchanged. Its adapter
only creates the model request at evaluation time. Scoring uses the pinned
upstream package. The other four suites include canonical records in this directory layout:

```text
benchmarks/accuracy-v1/
  agnews/test.jsonl
  toolace/test.jsonl
  typed_decisions/test.jsonl
  wildjailbreak/test.jsonl
```

Required reference counts are documented in the README. Typed Decision has multiple fields per row: report **decision-level** correct/total,
not the fraction of rows with every field correct. Public hard TVD compares
probabilities with `gold_probs` on its 10 annotated items; never put that field
in the prompt. ECE uses maximum candidate probability and argmax correctness.
The multiclass Brier sum is `sum((p[k] - 1[k == target])**2 for k in labels)`.

评测不提供数据转换功能。若使用自行准备的其他版本或划分，必须明确说明与原实验不同；
相同的样本数量本身不能证明数据一致。

## Calibration split

Provide independent canonical validation records with one of these fields:

```text
"validation_role": "calibration"
"validation_role": "selection"
```

Only `calibration` rows fit T; `selection` rows report validation metrics and
never affect fitting. Use distinct IDs and ensure there is no content/group
overlap across training, calibration, validation, and test splits. ID checking
cannot detect semantic duplicates. Private split details are not disclosed.
The calibration workflow consumes user-supplied split assignments.

校准集用于最小化 NLL，独立验证集仅报告效果。不要在测试集上搜索 T，也不要选择
测试分数最高的 checkpoint。温度缩放改变概率，不应改变候选答案的 argmax。

## Output

Each answer contains its type, candidate `probabilities`, and maximum
probability `confidence`. `choice` returns the selected original label;
`noul` returns the probability of `yes`; `score` returns the expected numeric
score (which can be noninteger). Evaluation still uses the argmax label for
accuracy. A calibrated result additionally records the temperature used.
