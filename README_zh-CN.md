<h1 align="center">Intern-Decision</h1>

<p align="center">
  <a href="https://huggingface.co/spaces/internlm/intern-decision"><strong>🎮 在线演示</strong></a>
  &nbsp; · &nbsp;
  <a href="https://huggingface.co/collections/internlm/intern-decision"><strong>🤗 模型权重</strong></a>
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <a href="docs/DATA.md">数据格式</a> ·
  <a href="docs/EVALUATION.md">评测复现指南</a>
</p>

Intern-Decision 将状态、可选图片和多个结构化问题转换为决策与概率。
模型基于 Qwen3.5 微调语言主干，同时冻结视觉编码器和投影层。
支持 `choice`（选择）、`score`（评分）和 `noul`（是非）三类问题。

本目录提供训练、双后端推理、基准评分、温度校准和浏览器演示代码，并包含
**[96 条概率分布校准基准](docs/CALIBRATION_BENCHMARK.md)**、精确参考分布、
确定性生成器和离线评分脚本，以及七项准确率测试集和绑定 checkpoint 的温度预设。
训练数据、内部校准/验证记录、图片、数据准备流程和模型权重不包含在本目录中。
运行时 schema、tokenization 和 collator 支持用户自行提供的记录。
演示中的例子是独立编写的合成接口示例。

<h2 align="center">🎬 演示视频</h2>

<p align="center">点击预览，观看完整视频。</p>

<table align="center">
  <tr>
    <td align="center" width="50%">
      <p align="center"><strong>🍄 马里奥</strong></p>
      <a href="assets/demos/Mario.mp4">
        <img src="assets/demos/Mario.gif" width="420" alt="🍄 马里奥" />
      </a>
    </td>
    <td align="center" width="50%">
      <p align="center"><strong>🖱️ 鼠标移动</strong></p>
      <a href="assets/demos/Mouse-Moving.mp4">
        <img src="assets/demos/Mouse-Moving.gif" width="420" alt="🖱️ 鼠标移动" />
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <p align="center"><strong>🛡️ Doom · 防守</strong></p>
      <a href="assets/demos/Doom-Defend.mp4">
        <img src="assets/demos/Doom-Defend.gif" width="420" alt="🛡️ Doom · 防守" />
      </a>
    </td>
    <td align="center" width="50%">
      <p align="center"><strong>💚 Doom · 收集生命值</strong></p>
      <a href="assets/demos/Doom-Health-Gathering.mp4">
        <img src="assets/demos/Doom-Health-Gathering.gif" width="420" alt="💚 Doom · 收集生命值" />
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" colspan="2">
      <p align="center"><strong>🌐 浏览器操作</strong></p>
      <a href="assets/demos/Browser-Use.mp4">
        <img src="assets/demos/Browser-Use.gif" width="420" alt="🌐 浏览器操作" />
      </a>
    </td>
  </tr>
</table>

## 方法

训练目标为**自回归掩码语言建模（autoregressive masked language modeling）**。
assistant 输入包含完整的 JSON 骨架，每个字段对应一个 `<decision>` 标记。
真实答案符号只出现在 labels 中，不进入输入。训练采用普通因果 next-token 对齐：
标记**前一位置**的 logit 预测该字段的答案。所有字段共享一次因果前向计算。

训练交叉熵覆盖完整词表；推理将 logits 限制在合法的单 token 答案符号上，
执行 softmax 后映射回原始标签。问题和选项的顺序必须保留。

## 结果

Intern-Decision-4B 在七项基准上的**平均准确率为 90.02%**（Jev：88.74%），
在 RTX 4090 上的**本地请求平均延迟为 44.16 ms**，在 96 条校准 pilot 上达到
**0.550 Brier / 0.089 ECE**（Jev：0.595 / 0.130）。这些结果的评测协议和范围不同，
分别见[基准准确率](#基准准确率)、[推理延迟](#rtx-4090-推理延迟)和[校准基准](#已知分布校准基准)。

### 基准准确率

准确率列以百分数表示；Average 为表中七项准确率的算术平均。
七项共 10,751 行、12,351 个决策，各项分母见[七项基准评测](#七项基准评测)。
Brier 和 ECE 是 **111 条 Jevbench-Hard 的指标**。该表使用 **max(P)** 作为置信度，
ECE 采用 10 个等宽区间，以小数显示；Brier 使用原始多类别求和定义。
Intern-Decision 的概率应用下表中的拟合温度，argmax 预测保持不变。
基线使用其已报告或默认的概率，本次未为它们额外拟合温度。

| 模型 | Easy | Original | Hard | Typed Decision | ToolACE | AG News | WildJailBreak | Average | Brier ↓ | ECE ↓ | T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [Jev](https://docs.typesafe.ai/api) | 100.00 | 98.61 | 72.07 | 73.35 | 91.29 | 89.57 | 96.29 | 88.74 | 0.358420 | 0.094685 | — |
| [Laya](https://github.com/NandhaKishorM/laya) | 95.83 | 72.22 | 28.83 | 35.95 | 63.87 | 92.84 | 14.84 | 57.77 | 0.804244 | 0.246455 | — |
| [SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev) | 100.00 | 98.61 | 61.26 | 62.80 | 85.16 | 89.22 | 92.53 | 84.23 | 0.498012 | 0.112213 | 1.0 |
| [Kev](https://github.com/jaredpalmer/kev) | 100.00 | 93.06 | 45.05 | 65.60 | 87.42 | 89.82 | 75.97 | 79.56 | 0.738253 | 0.262200 | 2.962195 |
| [JevK5](https://github.com/allebee/jevk5) | 100.00 | 97.22 | 73.87 | 64.50 | 80.97 | 89.13 | 90.45 | 85.16 | 0.366162 | 0.046672 | 1.532 |
| Intern-Decision-0.8B | 97.92 | 80.56 | 52.25 | 77.35 | 94.52 | 88.61 | 64.48 | 79.38 | 0.529519 | 0.065710 | 2.747761 |
| Intern-Decision-2B | 100.00 | 84.72 | 63.96 | 79.35 | 96.45 | 89.96 | 78.33 | 84.68 | 0.437256 | 0.100208 | 2.100509 |
| Intern-Decision-4B | 100.00 | 98.61 | 73.87 | 80.55 | 96.45 | 90.82 | 89.86 | 90.02 | 0.346762 | 0.065267 | 1.992418 |

复现步骤见[评测指南](docs/EVALUATION.md)。

## RTX 4090 推理延迟

本地测量使用单张 NVIDIA RTX 4090、BF16、SDPA、Hugging Face 原生推理、
Transformers 5.14.1 和 flash-linear-attention 0.4.2。每个请求包含 289 个输入 token
和三个字段（选择、是非、评分），在一次前向计算中完成，并应用对应 checkpoint 的温度。

| **模型** | **平均值** | **中位数 / P50** | **P95** |
| -------------------- | --------- | ---------------- | --------- |
| Jev | 109.70 ms | 106.30 ms | 146.70 ms |
| Intern-Decision-0.8B | 33.98 ms | 33.44 ms | 37.50 ms |
| Intern-Decision-2B | 33.28 ms | 33.15 ms | 33.55 ms |
| Intern-Decision-4B | 44.16 ms | 44.03 ms | 44.60 ms |

## 已知分布校准基准

随仓库提供的 [96 条基准](benchmarks/known-distribution-pilot-v1/README.md)
是一个小规模校准 pilot，包含六个类别、24 个问题族和 48 组成对设置。
问题询问随机结果；精确参考分布与模型输入分开保存。
该基准仅用于评测，不得使用其参考答案训练模型或拟合温度。

### 评测结果

三种设置均获得 **96/96 条有效预测**，每个类别包含 16 条。
Intern-Decision-4B 使用最终 checkpoint 和固定的 XTuner 后端评测。
温度 **T=1.992418** 在独立校准划分上拟合，本基准未参与拟合或选择。
温度校准保留了每条预测的 argmax 标签。Jev 使用 `jev-1.13.0`。

每个单元格为**期望多类别 Brier / 期望 ECE**，越低越好。
Brier 包含结果本身不可消除的随机性。ECE 使用 10 个等宽区间，优先采用返回的有效
置信度，否则使用 max(P)；这与上方 Hard 表仅使用 max(P) 的口径不同。
这些指标根据精确参考分布计算，不是针对采样标签计算的准确率。

| 类别 | Intern-Decision-4B, 未校准 | Intern-Decision-4B, 已校准 | Jev |
|---|---:|---:|---:|
| 直接随机性与可能性支持集 | 0.483 / 0.181 | 0.421 / 0.129 | 0.490 / 0.216 |
| 复合事件与混合分布 | 0.677 / 0.254 | 0.577 / 0.150 | 0.682 / 0.274 |
| 历史信息、条件概率与隐藏状态 | 0.711 / 0.219 | 0.613 / 0.108 | 0.657 / 0.113 |
| 日常证据与观测偏差 | 0.701 / 0.328 | 0.575 / 0.210 | 0.603 / 0.114 |
| 选择性披露与概率谜题 | 0.540 / 0.119 | 0.510 / 0.049 | 0.483 / 0.138 |
| 序列与组合过程 | 0.656 / 0.180 | 0.605 / 0.058 | 0.657 / 0.116 |
| **总体（合并计算）** | 0.628 / 0.213 | 0.550 / 0.089 | 0.595 / 0.130 |

校准使总体 Brier 从 **0.628 降至 0.550**，期望 ECE 从 **0.213 降至 0.089**，
低于 Jev 在本 pilot 上的 **0.595 / 0.130**。
总体 Brier 对 96 条样本取平均；总体 ECE 合并所有样本重新计算，
**不是**各类别 ECE 的算术平均。选项顺序变体成对出现，
因此这是一个小规模诊断研究，不能视为 96 次独立试验。

### 运行评测

使用 Python 3.10+ 对保存的模型/API 预测评分，仅需标准库：

```bash
python -m src.eval.score_known_distribution \
  --dataset benchmarks/known-distribution-pilot-v1 \
  --predictions /path/to/predictions.jsonl \
  --output outputs/calibration-benchmark
```

新建输出目录包含 `summary.md`（分类及总体 Brier/ECE）、`metrics.json`
（完整指标和分桶）以及 `scored.jsonl`（逐条评分）。
总体 ECE 合并全部样本计算，不是类别 ECE 的均值。
响应格式、推理示例、公式、验证方法和局限见[评分指南](docs/CALIBRATION_BENCHMARK.md)。

## 安装

模型后端使用 Linux、Python 3.12 和 CUDA。请从仓库根目录执行命令。
HF 和 XTuner 使用不同的已验证 Transformers 版本，应分别创建虚拟环境。
checkpoint 为本地 HF 目录，须包含权重、processor、tokenizer、chat template
和 `<decision>` token。

### Hugging Face 原生推理

```bash
python3.12 -m venv .venv-hf
source .venv-hf/bin/activate
python -m pip install -r requirements-inference.txt -r requirements-eval.txt
```

该后端直接加载 `Qwen3_5ForConditionalGeneration`，无需导入 XTuner 或加载兼容补丁。
不要将 `runtime/xtuner` 加入其 `PYTHONPATH`。
系统需要可用且兼容的 PyTorch CUDA wheel。

### XTuner 训练与推理

模型通过 XTuner 和随附的运行时适配执行。上游源码版本已固定；
本发布目录不内置 XTuner 的第三方源码或数据集。

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
# 安装与 Torch/CUDA 工具链匹配的 flash-attn 2.8.3 wheel，或从源码编译。
python -m pip install flash-attn==2.8.3 --no-build-isolation
export XTUNER_PATH="$PWD/third_party/xtuner"
```

显式依赖覆盖为 XTuner 路径保留 Transformers 4.57.0。
`runtime/xtuner/sitecustomize.py` 提供已验证的 Torch 2.6 兼容适配，
启动脚本仅在 XTuner 环境中启用。可选扩展可能需要 CUDA 开发工具。
HF 环境保持独立。不同算子和 BF16 舍入可能造成后端间的概率差异，
不能假定两个后端逐位一致。

## 推理

按照 [docs/DATA.md](docs/DATA.md) 准备请求 JSON，然后运行：

```bash
export MODEL_CHECKPOINT=/path/to/checkpoint
python -m src.inference --backend hf --input /path/to/request.json
```

使用 XTuner 时，激活对应环境并显式启用其运行时：

```bash
export PYTHONPATH="$PWD/runtime/xtuner:$PWD:$XTUNER_PATH"
export XTUNER_HF_IMPL=1 XTUNER_USE_FA3=0
python -m src.inference --backend xtuner --input /path/to/request.json
```

XTuner 后端使用 CUDA/BF16。HF 还支持通过 `configs/inference/default.json`
配置 `device`、`dtype` 和 `attn_implementation`。
仅在匹配的 processor 单独存储时设置 `MODEL_PATH`。
`MEDIA_ROOT` 用于解析 CLI 本地图片路径。`CALIBRATION_PATH` 启用对应 checkpoint
的校准文件；不设置则使用温度 1.0。不要复用其他 checkpoint 的温度，
也不要将采样温度当作置信度校准。

## 使用自己的数据训练

请提供符合数据接口约定、已准备好的 JSONL 文件。
加载器不负责数据集构建、重新标注、划分生成或格式转换。
训练前请检查记录和划分。

```bash
export MODEL_PATH=/path/to/base-qwen35-instruction-checkpoint
export DATA_PATH=/path/to/your/train.jsonl
export MEDIA_ROOT=/path/to/your/media  # 纯文本数据可省略
export WORK_DIR="$PWD/outputs/train-001"  # 必须为新目录
export NPROC_PER_NODE=4
bash scripts/train.sh
```

`configs/training/qwen35.py` 从基础 checkpoint 读取架构维度，
支持 Qwen3.5 的 0.8B、2B、4B 和 9B dense 版本。
启动脚本记录用户选择的配置和哈希，在用户提供的记录上训练，并仅保存最终 HF checkpoint。
脚本不会自动恢复此前运行；运行时 tokenization 会拒绝无效或超长记录。
`training-config.json` 与 HF checkpoint 的 `config.json` 分开保存。

验证保存的权重和每个 rank 的初始加载审计：

```bash
python -m scripts.verify_checkpoint --base "$MODEL_PATH" \
  --checkpoint /path/to/final/hf-checkpoint \
  --audit-dir "$WORK_DIR/weight-audit" --ranks 4 \
  --output "$WORK_DIR/checkpoint-verification.json"
```

## 七项基准评测

通过 `requirements-eval.txt` 安装固定版本的 Jevbench 评分包。
实际评测记录位于 `benchmarks/accuracy-v1/`。
按 [docs/EVALUATION.md](docs/EVALUATION.md) 校验文件并复现两张结果表，
无需运行数据转换脚本。

| 基准 | 行数 | 决策数 |
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

复现已报告的评分后端时，请在 XTuner 环境中使用 `--backend xtuner`。
`--public-only` 仅运行三项公开 Jevbench 基准，无需其他数据集。
默认路径指向随附记录；可通过 `--test-root`、`EVAL_DATA_ROOT` 和
`JEVBENCH_DATA_ROOT` 覆盖路径，用于独立实验。

输出包括原始候选概率、正确数/总数、准确率、ECE、Brier、Hard TVD、
输入哈希和 checkpoint 哈希。Hard TVD 仅在含 `gold_probs` 的 10 条公开记录上计算。
真实标签和参考分布不会进入模型提示词。多进程运行时，在不同 GPU 上分别指定
`--worker R --workers N`；全部 worker 成功后，再用同一命令加上
`--merge --workers N` 合并。合并过程检查预测身份、覆盖率、哈希、温度和后端。

## 温度选择

已发布模型请优先使用 [docs/EVALUATION.md](docs/EVALUATION.md) 中
经过哈希验证的温度预设。以下流程用于自己的模型。

每个 checkpoint 拟合一个正标量温度，目标为**仅在校准数据上最小化 NLL**。
请自行提供互不重叠的记录，将 `validation_role` 设置为 `calibration` 或 `selection`；
内部校准和验证划分的具体信息不披露。
校准集、验证集应与训练集和基准测试集保持不重叠。

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

XTuner 评测应同样使用 XTuner 收集预测。
可选参数 `--expected-fit-rows` 和 `--expected-validation-rows` 用于检查
用户自行指定的数据行数。优化器在 T ∈ [0.01, 100] 内，对逆温度下凸 NLL
的导数进行二分搜索，并保存搜索曲线和独立验证分数。
校准文件在 benchmark 重放前固定。

Replay 对保存的候选概率应用 `softmax(log(p) / T)`，检查**零决策变化**，
并输出校准前后指标和 `verification.json`。
它不会优化 benchmark ECE 或选择 checkpoint。NLL 最优温度未必能最小化
另一数据集上的 ECE。校准文件绑定 checkpoint 元数据哈希；修改权重后需重新生成。
移动 checkpoint 时，须先检查所有已记录的文件哈希，再更新校准文件中的 `checkpoint` 路径。

## 浏览器演示与 HTTP API

```bash
export MODEL_CHECKPOINT=/path/to/checkpoint
export INFERENCE_BACKEND=hf  # 或在独立环境中使用 xtuner
bash scripts/demo.sh
```

打开 <http://127.0.0.1:7860>。演示支持结构化问题、多图片上传、概率和校准置信度。
API 端点包括 `POST /v1/decisions`（别名 `/v1/jev`）、`GET /health` 和交互文档 `/docs`。
HTTP 接收上传的图片字节，不接受服务器文件路径或远程 URL。
上传内容为临时文件，推理后删除；CLI 本地路径仅供可信调用方使用。
服务默认只监听本机，不记录请求内容。公开部署前应添加身份认证和请求大小限制。

可选的外部思考转交默认关闭。启用时，在推理配置中指定自己的模型，
并通过环境变量提供 `LLM_BASE_URL` 和 `LLM_API_KEY`。
只有显式启用的请求才会向该服务发送证据；转交输出不是本地校准概率，
也未用于已报告的评测。

## 检查与目录

```bash
python -m unittest tests.check_inference tests.check_image_uploads tests.test_release
python -m tests.check_temperature
```

这些 CPU 检查使用合成样例和 mock 模型，不启动训练，也不能证明 checkpoint 的质量
或替代新运行时的 GPU 验证。在 XTuner 环境中，将 `MODEL_PATH` 指向本地基础模型
processor，并设置上述 XTuner `PYTHONPATH`，然后运行
`python -m tests.check_training_runtime`，可在 CPU 上检查因果标签对齐、packing、
图像 token 以及卷积输出/梯度的一致性。测试仅临时创建合成媒体，不加载模型权重。

```text
configs/         可移植的推理与训练配置
src/inputs/      运行时 schema、tokenization 和 collation
src/model/       Qwen 架构与冻结视觉模块的训练适配
src/inference/   HF 原生及 XTuner 引擎、温度缩放
src/eval/        评分、校准预测收集、拟合与重放
src/service/     HTTP 接口和浏览器演示
runtime/xtuner/  显式启用的 XTuner 兼容环境
scripts/         训练启动器与 checkpoint 验证
tests/           使用合成样例的 CPU 回归检查
docs/DATA.md     用户输入数据的格式约定
```

第三方依赖遵循各自许可证：权重适用 Qwen 模型条款，XTuner 为 Apache-2.0，
Jevbench 为 MIT。本目录不再分发模型权重或第三方代码仓库。
