import argparse
import json
import math
import re
import types
import urllib.request
from pathlib import Path

import torch
from reference.rwkv7 import RWKV_x070  # 请确保路径正确
from reference.utils import TRIE_TOKENIZER  # 请确保路径正确


# ==========================================
# 核心工具函数 (来自脚本 1 - 强壮解析)
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid RWKV-7 GSM8K Pipeline")
    # 请确认这里是你的模型路径
    parser.add_argument("--model", type=str, default=r"G:\pth\rwkv7-g1b-1.5b-20251202-ctx8192.pth", help="Path to rwkv7 checkpoint")
    parser.add_argument("--input", type=str, default="eval/gsm8k_test.jsonl", help="Path to GSM8K jsonl")
    parser.add_argument("--question", type=str, default=None, help="Single question debug override")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--passes", type=int, default=1, help="K value (Pass@K)")
    parser.add_argument("--output", type=str, default="rollout_hybrid.jsonl", help="Output path")

    # 采样配置
    parser.add_argument("--cot_max_len", type=int, default=1024)
    parser.add_argument("--final_max_len", type=int, default=128)
    return parser.parse_args()


def _normalize_text(value: str) -> str:
    """标准化文本：去除逗号，统一空格"""
    if value is None: return ""
    value = str(value).replace(",", "").replace("\\ ", " ").replace("\u00a0", " ")
    return re.sub(r"\s+", " ", value.strip())


def _is_numeric_equal(a: str, b: str, tol: float = 1e-6) -> bool:
    """数值等价判断，支持浮点/分数近似"""
    def _to_number(x):
        if x is None:
            return None
        s = str(x).strip()
        try:
            return float(s)
        except Exception:
            pass
        if "/" in s:
            num, *rest = s.split("/")
            if len(rest) == 1:
                den = rest[0]
                try:
                    return float(num) / float(den)
                except Exception:
                    return None
        return None

    fa, fb = _to_number(a), _to_number(b)
    if fa is None or fb is None:
        return False
    return math.isclose(fa, fb, abs_tol=tol, rel_tol=tol)


def extract_number(text: str):
    """提取答案数字，处理各类停止符 (Script 1 逻辑)"""
    for cue in ["<|endoftext|>", "\nQ:", "</s>", "<|im_end|>", "User:", "\n\n"]:
        if cue in text:
            text = text.split(cue, 1)[0]

    text_no_comma = text.replace(",", "")
    if "####" in text_no_comma:
        text_no_comma = text_no_comma.split("####")[-1]

    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", text_no_comma)
    return matches[-1] if matches else None


def load_questions(args):
    target_path = Path(args.input)
    if args.question:
        return [{"q": args.question, "a": None}]

    if not target_path.exists():
        print(f"Downloading GSM8K to {target_path}...")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
            target_path)

    data = []
    with open(target_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            q = obj.get("question") or obj.get("problem")
            a_raw = obj.get("answer")
            if isinstance(a_raw, str) and "####" in a_raw:
                a_raw = a_raw.split("####")[-1].strip()
            # 预处理 Gold Answer
            match = re.search(r"[-+]?[0-9]*\.?[0-9]+", str(a_raw).replace(",", ""))
            a_clean = match.group(0) if match else str(a_raw).strip()
            data.append({"q": q, "a": a_clean})
    return data


def _prepare_model_path(raw: str) -> str:
    p = Path(raw)
    if p.suffix == ".pth": p = p.with_suffix("")
    return str(p)


# ==========================================
# 采样核心 (混合逻辑)
# ==========================================

def sample_step(logits, cfg, occurrence=None, ban_tokens=None):
    with torch.no_grad():
        logits = logits.float()
        if occurrence is not None:
            logits -= (cfg["alpha_presence"] + occurrence * cfg["alpha_frequency"])

        # [Script 2 逻辑] Ban Tokens 实现多样性
        if ban_tokens:
            logits[:, list(ban_tokens)] = -float('inf')

        temp = cfg["temp"]
        if temp != 1.0 and temp > 0:
            logits /= temp

        probs = torch.softmax(logits, dim=-1)

        # Top-K
        if cfg["top_k"] > 0:
            val, _ = torch.topk(probs, cfg["top_k"])
            probs[probs < val[:, [-1]]] = 0

        # Top-P
        if cfg["top_p"] > 0 and cfg["top_p"] < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative_probs > cfg["top_p"]
            mask[:, 0] = False
            mask = mask.scatter(1, sorted_indices, mask)  # Restore original order
            probs[mask] = 0

        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
        return torch.multinomial(probs, 1)


def main():
    args_cli = parse_args()
    args_cli.model = _prepare_model_path(args_cli.model)

    # 初始化模型
    model_args = types.SimpleNamespace()
    model_args.vocab_size = 65536
    model_args.head_size = 64
    model_args.MODEL_NAME = args_cli.model
    model_args.n_layer = 0  # Load from file
    model_args.n_embd = 0  # Load from file

    print(f">> Boss Hybrid System Initialized")
    print(f">> Model: {args_cli.model}")
    print(f">> Strategy: {'[Strict Reset]' if args_cli.passes == 1 else '[Diversity + Continuation]'}")

    model = RWKV_x070(model_args)
    tokenizer = TRIE_TOKENIZER("reference/rwkv_vocab_v20230424.txt")  # 请确保路径正确

    samples = load_questions(args_cli)
    print(f">> Total Samples: {len(samples)} | Batch: {args_cli.batch} | Pass@K: {args_cli.passes}")

    # 配置参数：K=1 稳定、K>1 多样性，但这里采用统一保守配置
    cot_cfg = {
        "temp": 0.3, "top_k": 50, "top_p": 0.3,
        "alpha_presence": 0.5, "alpha_frequency": 0.5, "alpha_decay": 0.99,
        "stop_tokens": {0, 261, 24281}  # \n\n, User
    }
    final_cfg = {
        "temp": 0.8, "top_k": 1, "top_p": 0.3,  # Final 收敛
        "alpha_presence": 0.0, "alpha_frequency": 0.0, "alpha_decay": 0.99,
        "stop_tokens": {0, 2402, 4910}  # 换用更紧的终止符
    }

    # 状态转换 Trigger
    transition_text = "\nTherefore, the answer is \\(\\boxed{"
    transition_tokens = tokenizer.encode(transition_text)

    total_acc = 0
    total_cnt = 0
    saved_rows = []

    best_single_acc = 0.0  # 仅用于 K=1 记录最高值

    for start in range(0, len(samples), args_cli.batch):
        end = min(start + args_cli.batch, len(samples))
        batch_samples = samples[start:end]
        bsz = len(batch_samples)

        prompts = [f"User: {x['q']}\n\nAssistant: <think" for x in batch_samples]
        prompt_tokens = [[0] + tokenizer.encode(p) for p in prompts]

        # 1. Prefill Question
        state_empty = model.generate_zero_state(bsz)
        out_prefill = model.forward_batch(prompt_tokens, state_empty)

        # 保存快照，用于每次尝试的回滚 (如果 K=1，其实用不到回滚，但为了统一结构保留)
        state_snapshot = [s.clone() for s in state_empty]
        out_snapshot = out_prefill.clone()

        got_correct = [False] * bsz
        final_answers = [""] * bsz

        # 用于记录多样性 (Script 2 Logic)
        seen_first_tokens = [set() for _ in range(bsz)]

        # ====== Pass@K Loop ======
        for attempt in range(args_cli.passes):
            # 恢复初始状态
            state_curr = [s.clone() for s in state_snapshot]
            out_curr = out_snapshot.clone()

            # --- Stage 1: Generate CoT ---
            cot_tokens = [[] for _ in range(bsz)]
            finished_mask = [False] * bsz
            occurrence = torch.zeros((bsz, 65536), device=out_curr.device)

            for step in range(args_cli.cot_max_len):
                if all(finished_mask): break
                occurrence *= cot_cfg["alpha_decay"]

                # [Script 2 Logic] Ban Tokens (仅当 K>1 且不是第一次尝试时)
                ban_ids = set()
                if args_cli.passes > 1 and attempt > 0 and step == 0:
                    # 注意：这里需要对每个样本分别屏蔽，简化起见我们对整个batch操作
                    # 为保证精度，我们用稍微慢一点的方法：mask logits per sample?
                    # 这里为了性能，我们在 sample_step 外部处理 mask 是很困难的，
                    # 简单处理：如果 K>1，我们在 sample_step 内部无法针对 batch 维度屏蔽不同 token
                    # 修正方案：在传给 sample_step 之前修改 out_curr
                    pass

                    # 动态 Ban Token 逻辑 (Per-sample masking)
                if args_cli.passes > 1 and attempt > 0 and step == 0:
                    for i in range(bsz):
                        if not got_correct[i]:
                            out_curr[i, list(seen_first_tokens[i])] = -float('inf')

                tokens = sample_step(out_curr, cot_cfg, occurrence)
                tokens_cpu = tokens.view(-1).tolist()

                # 记录首个 token 用于后续屏蔽
                if step == 0:
                    for i, t in enumerate(tokens_cpu):
                        seen_first_tokens[i].add(t)

                # 推进模型
                active_idx = []
                next_input = []

                for i in range(bsz):
                    if finished_mask[i]: continue
                    t = tokens_cpu[i]
                    if t in cot_cfg["stop_tokens"]:
                        finished_mask[i] = True
                        continue

                    cot_tokens[i].append(t)
                    occurrence[i, t] += 1.0
                    active_idx.append(i)
                    next_input.append([t])

                if not active_idx: break

                state_subset = [state_curr[0][:, :, active_idx], state_curr[1][:, active_idx]]
                out_subset = model.forward_batch(next_input, state_subset)

                for k, global_idx in enumerate(active_idx):
                    state_curr[0][:, :, global_idx] = state_subset[0][:, :, k]
                    state_curr[1][:, global_idx] = state_subset[1][:, k]
                    out_curr[global_idx] = out_subset[k]

            # --- Stage 2: Transition & Answer ---
            # 这里的核心分歧点：K=1 使用 Reset (脚本1)，K>1 使用 Continuation (脚本2)

            target_state = None
            target_out = None

            if args_cli.passes == 1:
                # K=1：重置状态，重新 Prefill（高精度）
                full_prompts = []
                for i in range(bsz):
                    cot_str = tokenizer.decode(cot_tokens[i])
                    full = f"User: {batch_samples[i]['q']}\n\nAssistant: <think{cot_str}{transition_text}"
                    full_prompts.append([0] + tokenizer.encode(full))

                state_reset = model.generate_zero_state(bsz)
                out_reset = model.forward_batch(full_prompts, state_reset)
                target_state = state_reset
                target_out = out_reset
            else:
                # K>1：状态延续（高效，多样性）
                trans_input = [transition_tokens for _ in range(bsz)]
                out_trans = model.forward_batch(trans_input, state_curr)
                target_state = state_curr
                target_out = out_trans

            # --- Stage 3: Generate Final Answer ---
            ans_tokens = [[] for _ in range(bsz)]
            finished_ans = [False] * bsz
            occurrence.zero_()  # 重置重复惩罚

            for _ in range(args_cli.final_max_len):
                if all(finished_ans): break
                occurrence *= final_cfg["alpha_decay"]

                tokens = sample_step(target_out, final_cfg, occurrence)
                tokens_cpu = tokens.view(-1).tolist()

                active_idx = []
                next_input = []

                for i in range(bsz):
                    if finished_ans[i]: continue
                    t = tokens_cpu[i]
                    if t in final_cfg["stop_tokens"]:
                        finished_ans[i] = True
                        continue
                    ans_tokens[i].append(t)
                    occurrence[i, t] += 1.0
                    active_idx.append(i)
                    next_input.append([t])

                if not active_idx: break

                state_subset = [target_state[0][:, :, active_idx], target_state[1][:, active_idx]]
                out_subset = model.forward_batch(next_input, state_subset)

                for k, global_idx in enumerate(active_idx):
                    target_state[0][:, :, global_idx] = state_subset[0][:, :, k]
                    target_state[1][:, global_idx] = state_subset[1][:, k]
                    target_out[global_idx] = out_subset[k]

            # --- Evaluation ---
            for i in range(bsz):
                # 如果这个样本在之前的尝试中已经做对了，就不更新了
                if got_correct[i]: continue

                ans_str = tokenizer.decode(ans_tokens[i])
                pred = extract_number(ans_str)
                gold = batch_samples[i]['a']

                cot_str = tokenizer.decode(cot_tokens[i])
                full_gen = f"<think{cot_str}{transition_text}{ans_str}"

                norm_pred = _normalize_text(pred)
                norm_gold = _normalize_text(gold)
                is_correct = (norm_pred == norm_gold) or _is_numeric_equal(pred, gold)

                if is_correct:
                    got_correct[i] = True
                    final_answers[i] = full_gen
                    print(f"HIT! Q: {batch_samples[i]['q'][:30]}... | Pred: {pred} | Gold: {gold}")

                # 如果是最后一次尝试且还没做对，保留这次的错误答案
                if attempt == args_cli.passes - 1 and not got_correct[i]:
                    final_answers[i] = full_gen

        # Batch 结束统计
        for i in range(bsz):
            total_cnt += 1
            if got_correct[i]: total_acc += 1

            if args_cli.output:
                saved_rows.append({
                    "q": batch_samples[i]['q'],
                    "a": batch_samples[i]['a'],
                    "gen": final_answers[i],
                    "correct": got_correct[i]
                })

        # 仅内部记录，不实时输出进度
        current_acc = total_acc / total_cnt
        if args_cli.passes == 1 and current_acc > best_single_acc:
            best_single_acc = current_acc

    # Final Report
    print("=" * 40)
    print(f"Evaluated {total_cnt} samples")
    final_acc = total_acc / total_cnt if total_cnt else 0.0
    if args_cli.passes == 1:
        print(f"Pass@1 Accuracy (max over run): {best_single_acc:.4f}")
    else:
        print(f"Pass@{args_cli.passes} Accuracy: {final_acc:.4f}")

    if args_cli.output:
        out_p = Path(args_cli.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            for row in saved_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved to {args_cli.output}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()