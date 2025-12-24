#  RWKV-7 GSM8K Rollout 评测方案报告

**框架基础**：BlinkDL/Albatross 驱动的自定义评测脚本


## 1. 系统定义 (System Definition)

本系统采用 **Two-Stage Logic (两阶段逻辑)** 采样流程，通过 `argparse` 灵活配置不同阶段的最大长度：

| 阶段 | 任务目标 | 核心策略                                          | 默认长度限制 |
| :--- | :--- |:----------------------------------------------| :--- |
| **Stage 1: Reasoning** | 逻辑推导 | 强制触发 `<think>` 标签，低温度探索路径，确保选择概率最高的Token      | `--cot_max_len 512` |
| **Stage 2: Answer** | 数值收敛 | 引导 `Therefore, the answer is \(\boxed{`     | `--final_max_len 64` |


## 2. 输入与状态管理 (Input & State Management)

### 2.1 数据构造与预处理
* **数据集**: 自动下载或读取 OpenAI 的 **GSM8K** 测试集。（1319条）
* **BOS 注入**: 严格遵循 RWKV 标准，在 `prompt_tokens` 起始端注入 `[0]`。
* **并行预填充**: 利用 `model.forward_batch` 一次性处理所有题目 Token，将上下文压缩进隐藏状态。

> ### 📄 Prefill 与状态快照 (Snapshot)
> ---
> ```python
> # 1. 初始状态卫生：生成零状态并执行 Batch Forward
> state_prefill = model.generate_zero_state(bsz)
> out_prefill = model.forward_batch(prompt_tokens, state_prefill)
> 
> # 2. 状态持久化：克隆预填充后的状态作为后续所有 Pass 的起点
> state_snapshot = [x.clone() for x in state_prefill]
> ```

### 2.2 Pass@K 性能优化逻辑
在多轮采样（Pass@k）中，系统仅在 `attempt 0` 执行一次题目 Prefill，后续轮次通过 `.clone()` 实现秒级重置。
> ### 📄 核心循环与早停掩码
> ---
> ```python
> for attempt in range(max(1, args_cli.passes)):
>     # 从快照恢复状态，避免重复 Prefill 带来的算力浪费
>     state_cot = [x.clone() for x in state_snapshot]
>     
>     # 检查掩码：若该题已在先前轮次做对，则不再进行采样
>     for i in range(bsz):
>         if cot_finished[i] or got_correct[i]:
>             cot_finished[i] = True
>             continue
> ```

## 3. 答案提取与匹配逻辑 (Matching Logic)

### 3.1 格式化引导
系统在 Stage 2 拼接固定后缀：`Therefore, the answer is \(\boxed{{`。

### 3.2 严谨数值判定
代码内置 `_is_numeric_equal` 函数，通过浮点数转换与绝对误差阈值判别正误。

> ### 📄 数值比对实现
> ---
> ```python
> def _is_numeric_equal(a, b, tol=1e-6):
>     try:
>         # 计算预测值与标答的绝对误差
>         return abs(float(a) - float(b)) <= tol
>     except Exception:
>         return False # 无法解析为数值时判定为 False
> ```

## 4. 关键采样参数 (Critical Parameters)



针对 RWKV-7 模型特性，双阶段采用差异化配置：

| 参数项 | Stage 1 (CoT) | Stage 2 (Final) | 针对 RWKV 的设计意图                                   |
| :--- | :--- | :--- |:------------------------------------------------|
| **Temperature** | `0.3` | `0.8` | **CoT**: 低温保证选择概率最高；**Final**: 属于“标准问答”建议值。 |
| **Top_P** | `0.3` | `0.3` | 截断概率分布尾部，剔除噪声 Token。                            |
| **Top_K** | `50` | `1 (Greedy)` | **Final 阶段强制强制收敛**，确保数值格式绝对确定。                  |
| **Alpha Presence** | `0.5` | `0.0` | **CoT**: 强惩罚防止循环复读；**Final**: 允许数字自然出现。         |
| **Alpha Decay** | `0.99` | `0.99` | **核心参数**：控制 RNN 状态随 Token 推进的衰减，维持长期记忆。         |

## 5. 快速运行指令 (CLI)

### 环境安装（nightly cu128）
我的常用显卡是5070ti，是120结构，所以使用12.8版本的torch，虚拟环境的是uv
```bash
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
uv pip install -r requirements.txt
```

### 使用说明
- 将 `--model` 替换为本地 RWKV7 模型前缀路径（不含 `.pth`），可以直接使用绝对路径。
- 不传 `--input` 时默认自动下载官方 GSM8K test split；`--question` 可直接评单题。
- `--passes` 控制 Pass@k；`--cot_max_len` / `--final_max_len` 为两阶段长度。
- check.py是用于检查答案的，避免在运行的时候没有完整记录下来

```bash
 python gsm8k_rollout.py `                                                                                                                                                                                     
>>   --model "path" `
>>   --batch 8 `
>>   --cot_max_len 512 `
>>   --final_max_len 64 `
>>   --passes 1 `
>>   --output "out\gsm8k_0p1b.jsonl"
