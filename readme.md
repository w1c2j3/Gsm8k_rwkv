# RWKV-7 GSM8K Evaluation System

## 1. 项目概述 (Project Overview)

本项目是一个专为 **RWKV-7** 架构设计的 **GSM8K (Grade School Math)** 数据集自动化评测系统。

针对 RWKV 模型在数学推理任务中的特性，本系统设计了 **Two-Stage Logic (双阶段逻辑)** 采样策略，并深度利用了 RWKV 的 **RNN State Caching (状态缓存)** 技术。

### 核心价值

* **双阶段采样 (Two-Stage Sampling)：** 将“逻辑推理 (CoT)”与“数值收敛 (Answer)”解耦，分别配置最佳采样参数，最大化模型推理潜力。
* **极致效率 (Efficiency)：** 利用 RWKV 的 State 机制，Prefill 阶段仅计算一次。在 `passes > 1` 的场景下，通过 `state_snapshot` 实现零开销的分支探索，大幅降低显存占用与计算时间。
* **自动化流水线 (Pipeline)：** 内置数据集自动下载、脏数据清洗、正则数值提取与 Pass@k 统计，开箱即用。

---

## 2. 核心机制 (Core Mechanism)

本系统在推理过程中严格区分两个阶段，逻辑如下：

| 阶段 | 任务目标 | 关键行为 | 采样策略 (配置) |
| :--- | :--- | :--- | :--- |
| **Stage 0: Prefill** | 题目编码 | 读取题目，生成初始 Hidden State ($\mathbf{S}_0$) 并缓存快照。 | N/A |
| **Stage 1: Reasoning** | 逻辑推导 | 强制触发 `<think>` 标签，进行长思维链推理。 | `temp=0.3`, `top_p=0.3`<br>(低熵，保证逻辑连贯) |
| **Stage 2: Answer** | 数值收敛 | 强制输出 `Therefore, the answer is \(\boxed{`，引导模型输出最终数值。 | `top_k=1`<br>(Greedy, 强制收敛，消除格式噪声) |

### 状态管理与数学原理 (State Management)

RWKV 作为 RNN 模型，其核心状态更新遵循以下形式：

$$
\mathbf{S}_t = \mathbf{S}_{t-1} \cdot (\text{diag}(\mathbf{w}_t) + \mathbf{a}_t^\top \mathbf{b}_t) + \mathbf{v}_t^\top \mathbf{k}_t
$$

利用这一特性，我们在 Stage 0 结束时保存 `state_snapshot`。当执行 `passes > 1` (即对同一题尝试多次) 时，系统无需重复计算题目的 Prompt，而是直接从 Snapshot 克隆状态开始 Stage 1。这使得多次推理的上下文切换成本降低至 $O(1)$。

---

## 3. 环境准备 (Prerequisites)

### 硬件要求

* **GPU:** 推荐 NVIDIA RTX 3090/4090 或更高级别 (本项目基于 **RTX 5070Ti** 验证)。
* **VRAM:** 视模型参数量而定，RWKV-7 0.4B/1.5B 可在 8GB+ 显存流畅运行。

### 安装依赖

推荐使用 `uv` 进行依赖管理（以支持最新的 PyTorch Nightly 版本）：

## 4. 快速开始 (Quick Start)

### 基础运行

最简启动命令，使用默认参数运行评估：

```bash
python gsm8k_rollout.py \
  --model "models/rwkv7-g1a-0.4b" \
  --output "results/gsm8k_eval.jsonl"
```

### Pass@k 测试
### 启用 8 次尝试（passes=8）并增加 CoT 长度限制：
```bash
python gsm8k_rollout.py \
  --model "models/rwkv7-g1a-1.5b" \
  --batch 8 \
  --passes 8 \
  --cot_max_len 1024 \
  --final_max_len 128 \
  --output "results/gsm8k_pass8.jsonl"
```

## 5. 参数配置说明 (Configuration)
脚本支持通过 argparse 灵活调整评测行为：
| 参数项 | 类型 | 默认值 | 说明 | 业务影响 |
|------|------|--------|------|----------|
| `--model` | `str (Path)` | `-` | 模型权重路径（`.pth`） | **必填** |
| `--input` | `str` | `eval/...` | 测试集路径 | 缺失自动下载 |
| `--batch` | `int` | `16` | 并行推理批次大小 | 显存允许下越大越快 |
| `--passes` | `int` | `1` | 每题尝试次数（Pass@k） | `N>1` 测稳定性 |
| `--cot_max_len` | `int` | `512` | Stage 1 最大 Token 数 | 复杂题建议 ≥1024 |
| `--final_max_len` | `int` | `64` | Stage 2 Token 上限 | 仅数值输出 |
| `--limit` | `int` | `0` | 测试样本数限制 | `0` = 全量（1319） |

## 6. 输出结果与指标 (Output & Metrics)
### 控制台输出示例
```bash
Q: James decides to run 3 sprints...
Gold: 540 | Pred: 540 | exact=True
Generated: ...<think>First, calculate the distance...
Therefore, the answer is \( \boxed{540} \)
----------------------------------------
GSM8K rollout done. samples=1319 acc=0.452
```
### 结果文件（jsonl）
由 --output 指定的文件将包含完整评测细节：
```bash
{
  "question": "A robe takes 2 bolts of...",
  "gold": "3",
  "pred": "3",
  "correct": true,
  "gen": "User: ...\n\nAssistant: <think>...Therefore, the answer is \\(\\boxed{3}"
}
```
字段说明：
question：原始问题
gold：标准答案（已清洗）
pred：模型预测数值
correct：是否命中
gen：完整生成文本（含 CoT）

## 7. 代码结构 (Code Structure)
load_questions：数据加载与预处理（含特例修复）
_normalize_gold_answer / extract_number：答案清洗与数值提取
sample_top_k_top_p_with_penalty：自定义采样内核（Penalty + 动态温度）
main：主评测循环（State Snapshot 复用 + Two-Stage 切换）
