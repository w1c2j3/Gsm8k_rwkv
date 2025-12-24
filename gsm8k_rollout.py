import argparse
import json
import os
import re
import types
import urllib.request
from pathlib import Path

import torch

from reference.rwkv7 import RWKV_x070
from reference.utils import TRIE_TOKENIZER


def parse_args():
    parser = argparse.ArgumentParser(description="GSM8K batch rollout for RWKV-7 g1a models")
    parser.add_argument(
        "--model",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "rwkv7-g1a-0.4b-20250905-ctx4096"),
        help="Path prefix to rwkv7 checkpoint (without trailing .pth)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="eval/gsm8k_test.jsonl",
        help="Path to GSM8K jsonl (question/answer). Ignored if --question provided. If missing, downloads official test split.",
    )
    parser.add_argument("--question", type=str, default=None, help="Single question string (overrides --input)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for generation")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples (0 = all)")
    parser.add_argument("--passes", type=int, default=1, help="Pass@k attempts per question (>=1)")
    parser.add_argument("--output", type=str, default="", help="Optional jsonl path to save predictions")

    # two-stage sampling config
    parser.add_argument("--cot_max_len", type=int, default=512, help="Max tokens for CoT stage")
    parser.add_argument("--final_max_len", type=int, default=64, help="Max tokens for final answer stage")
    return parser.parse_args()


def _normalize_gold_answer(raw):
    if raw is None:
        return None

    text = str(raw)
    text = text.replace(",", "")
    match = re.search(r"[-+]?[0-9]*\.?[0-9]+", text)
    return match.group(0) if match else text.strip()


def load_questions(args):
    target_path = Path(args.input)
    if not target_path.exists():
        print(f"{target_path} not found, downloading official GSM8K test split...")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
        urllib.request.urlretrieve(url, target_path)
        print(f"downloaded to {target_path}")

    if args.question:
        return [{"q": args.question, "a": None}]

    fixes = {
        "Mr. Finnegan has 3 tanks with a capacity of 7000 gallons, 5000 gallons, and 3000 gallons, respectively. If he fills the first tank up to 3/4 full, the second tank with water up to 4/5 of its capacity, and the third tank up to half of its capacity, how many gallons in total are in the tanks?": 10750,
    }

    data = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            q = obj.get("question") or obj.get("problem")
            a_raw = fixes.get(q, obj.get("answer"))
            if isinstance(a_raw, str) and "####" in a_raw:
                a_raw = a_raw.split("####")[-1].strip()
            a_clean = _normalize_gold_answer(a_raw)
            data.append({"q": q, "a": a_clean})
            if args.limit and len(data) >= args.limit:
                break
    return data


def build_prompt(question: str) -> str:
    return f"User: {question}\n\nAssistant: <think"


def apply_filters(text: str, filters_cfg):
    for flt in filters_cfg:
        steps = flt.get("filter", [])
        cur = [text]
        for step in steps:
            func = step.get("function")
            if func == "regex":
                pattern = step.get("regex_pattern")
                group_select = step.get("group_select")
                if not pattern:
                    continue
                tmp = []
                for t in cur:
                    matches = re.findall(pattern, t)
                    if not matches:
                        continue
                    for m in matches:
                        if isinstance(m, tuple):
                            if group_select is not None and 0 <= group_select < len(m):
                                val = m[group_select]
                            else:
                                val = next((x for x in m if x), m[0])
                        else:
                            val = m
                        if val != "":
                            tmp.append(val)
                cur = tmp
            elif func == "take_first":
                if cur:
                    return cur[-1]
        if cur:
            return cur[-1]
    return None


def extract_number(text: str, filters_cfg=None):
    for cue in ["<|endoftext|>", "\nQ:", "</s>", "<|im_end|>"]:
        if cue in text:
            text = text.split(cue, 1)[0]
    text_no_comma = text.replace(",", "")
    if "####" in text_no_comma:
        text_no_comma = text_no_comma.split("####")[-1]
    if "The answer is" in text_no_comma:
        text_no_comma = text_no_comma.split("The answer is", 1)[-1]
    elif "Answer:" in text_no_comma:
        text_no_comma = text_no_comma.split("Answer:", 1)[-1]
    if filters_cfg:
        res = apply_filters(text_no_comma, filters_cfg)
        if res:
            return res.strip()
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", text_no_comma)
    return matches[-1] if matches else None


def _normalize_text(value: str) -> str:
    if value is None:
        return ""
    value = str(value)
    # collapse spaces/nbsp and strip commas for numeric equivalence
    value = value.replace(",", "").replace("\\ ", " ").replace("\u00a0", " ")
    value = re.sub(r"\s+", " ", value.strip())
    return value


def _is_numeric_equal(a: str, b: str, tol: float = 1e-6) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def _prepare_model_path(raw: str) -> str:
    p = Path(raw)
    if p.suffix == ".pth":
        p = p.with_suffix("")
    return str(p)


def sample_top_k_top_p_with_penalty(
    logits: torch.Tensor,
    *,
    temp: float,
    top_k: int,
    top_p: float,
    occurrence: torch.Tensor,
    alpha_presence: float,
    alpha_frequency: float,
    stop_tokens: set[int],
    ban_tokens: tuple[int, ...] = (),
):
    with torch.no_grad():
        logits = logits.float()
        if occurrence is not None:
            logits = logits - alpha_presence - occurrence * alpha_frequency
        if ban_tokens:
            logits[:, list(ban_tokens)] = -float("inf")
        if temp is not None and temp <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        if temp and temp != 1.0:
            logits = logits / temp

        if top_k and top_k > 0 and top_k < logits.size(-1):
            logits, topk_idx = torch.topk(logits, top_k, dim=-1)
        else:
            topk_idx = None

        probs = torch.softmax(logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

        if 0.0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum > top_p
            mask[:, 0] = False
            filtered = sorted_probs.masked_fill(mask, 0.0)
            denom = filtered.sum(dim=-1, keepdim=True)
            fallback = denom.squeeze(-1) <= 0
            filtered = torch.where(denom > 0, filtered / denom.clamp(min=1e-9), filtered)
            choice = torch.multinomial(filtered, num_samples=1).squeeze(-1)
            local_idx = torch.gather(sorted_idx, -1, choice.unsqueeze(-1)).squeeze(-1)
            if torch.any(fallback):
                greedy_idx = torch.argmax(sorted_probs[fallback], dim=-1, keepdim=False)
                local_idx[fallback] = torch.gather(sorted_idx[fallback], -1, greedy_idx.unsqueeze(-1)).squeeze(-1)
        else:
            prob_sum = probs.sum(dim=-1, keepdim=True)
            fallback = prob_sum.squeeze(-1) <= 0
            probs = torch.where(prob_sum > 0, probs / prob_sum.clamp(min=1e-9), probs)
            local_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
            if torch.any(fallback):
                local_idx[fallback] = torch.argmax(probs[fallback], dim=-1)

        if topk_idx is not None:
            local_idx = torch.gather(topk_idx, -1, local_idx.unsqueeze(-1)).squeeze(-1)
        return local_idx.unsqueeze(-1)


def main():
    args_cli = parse_args()
    if not args_cli.output:
        args_cli.output = "rollout_output.jsonl"
    model_path = _prepare_model_path(args_cli.model)
    args = types.SimpleNamespace()
    args.vocab_size = 65536
    args.head_size = 64
    args.MODEL_NAME = model_path

    print(f"loading model: {args.MODEL_NAME}")
    model = RWKV_x070(args)
    tokenizer = TRIE_TOKENIZER("reference/rwkv_vocab_v20230424.txt")

    samples = load_questions(args_cli)
    prompts = [build_prompt(x["q"]) for x in samples]
    # max_len 不再使用，改为分阶段长度
    print(f"total samples: {len(prompts)} | batch size: {args_cli.batch} | cot_max_len: {args_cli.cot_max_len} | final_max_len: {args_cli.final_max_len}")

    total_acc = 0
    total_cnt = 0
    saved_rows = []

    # 官方默认采样参数（与 FreeResponsePipeline 对齐）
    cot_cfg = {
        "temp": 0.3,
        "top_k": 50,
        "top_p": 0.3,
        "alpha_presence": 0.5,
        "alpha_frequency": 0.5,
        "alpha_decay": 0.99,
        "stop_tokens": {0, 261, 24281},
    }
    final_cfg = {
        "temp": 0.8,
        "top_k": 1,
        "top_p": 0.3,   # Final 收敛
        "alpha_presence": 0.0,
        "alpha_frequency": 0.0,
        "alpha_decay": 0.99,
        "stop_tokens": {0, 2402, 4910},
    }

    for start in range(0, len(prompts), args_cli.batch):
        end = min(start + args_cli.batch, len(prompts))
        cur_prompts = prompts[start:end]
        cur_items = samples[start:end]
        bsz = len(cur_prompts)

        # pad_zero: 在 prompt 前补 token 0
        prompt_tokens = [[0] + tokenizer.encode(p) for p in cur_prompts]

        state_prefill = model.generate_zero_state(bsz)
        out_prefill = model.forward_batch(prompt_tokens, state_prefill)
        state_snapshot = [x.clone() for x in state_prefill]

        got_correct = [False] * bsz
        first_sample_text = [""] * bsz
        first_hit_text = [""] * bsz

        for attempt in range(max(1, args_cli.passes)):
            state_cot = [x.clone() for x in state_snapshot]
            out_cot = out_prefill

            cot_tokens = [[] for _ in range(bsz)]
            cot_finished = [False] * bsz
            cot_occ = torch.zeros((bsz, args.vocab_size), device=out_cot.device, dtype=torch.float32)

            for _ in range(args_cli.cot_max_len):
                if all(cot_finished):
                    break
                cot_occ *= cot_cfg["alpha_decay"]
                token = sample_top_k_top_p_with_penalty(
                    out_cot,
                    temp=cot_cfg["temp"],
                    top_k=cot_cfg["top_k"],
                    top_p=cot_cfg["top_p"],
                    occurrence=cot_occ,
                    alpha_presence=cot_cfg["alpha_presence"],
                    alpha_frequency=cot_cfg["alpha_frequency"],
                    stop_tokens=cot_cfg["stop_tokens"],
                ).tolist()
                # 避免污染已完成样本：仅对未完成样本继续推进 state
                active_idx = []
                active_tokens = []
                for i in range(bsz):
                    if cot_finished[i] or got_correct[i]:
                        cot_finished[i] = True
                        continue
                    token_id = token[i][0]
                    if token_id in cot_cfg["stop_tokens"]:
                        cot_finished[i] = True
                        continue
                    cot_tokens[i].append(token_id)
                    cot_occ[i, token_id] += 1.0
                    active_idx.append(i)
                    active_tokens.append([token_id])
                if active_idx:
                    # slice states for active samples only
                    state_view = [state_cot[0][:, :, active_idx], state_cot[1][:, active_idx]]
                    out_active = model.forward_batch(active_tokens, state_view)
                    # write back state and logits
                    for k, i in enumerate(active_idx):
                        state_cot[0][:, :, i] = state_view[0][:, :, k]
                        state_cot[1][:, i] = state_view[1][:, k]
                        out_cot[i] = out_active[k]

            # final stage
            final_prompts_tokens = []
            for i in range(bsz):
                cot_text = tokenizer.decode(cot_tokens[i], utf8_errors="ignore")
                final_prompt = f"User: {cur_items[i]['q']}\n\nAssistant: <think{cot_text}\nTherefore, the answer is \\(\\boxed{{"
                final_prompts_tokens.append([0] + tokenizer.encode(final_prompt))

            state_final = model.generate_zero_state(bsz)
            out_final = model.forward_batch(final_prompts_tokens, state_final)

            final_tokens = [[] for _ in range(bsz)]
            final_finished = [False] * bsz
            final_occ = torch.zeros((bsz, args.vocab_size), device=out_final.device, dtype=torch.float32)
            for _ in range(args_cli.final_max_len):
                if all(final_finished):
                    break
                final_occ *= final_cfg["alpha_decay"]
                token = sample_top_k_top_p_with_penalty(
                    out_final,
                    temp=final_cfg["temp"],
                    top_k=final_cfg["top_k"],
                    top_p=final_cfg["top_p"],
                    occurrence=final_occ,
                    alpha_presence=final_cfg["alpha_presence"],
                    alpha_frequency=final_cfg["alpha_frequency"],
                    stop_tokens=final_cfg["stop_tokens"],
                ).tolist()
                active_idx = []
                active_tokens = []
                for i in range(bsz):
                    if got_correct[i] or final_finished[i]:
                        final_finished[i] = True
                        continue
                    token_id = token[i][0]
                    if token_id in final_cfg["stop_tokens"]:
                        final_finished[i] = True
                        continue
                    final_tokens[i].append(token_id)
                    final_occ[i, token_id] += 1.0
                    active_idx.append(i)
                    active_tokens.append([token_id])
                if active_idx:
                    state_view = [state_final[0][:, :, active_idx], state_final[1][:, active_idx]]
                    out_active = model.forward_batch(active_tokens, state_view)
                    for k, i in enumerate(active_idx):
                        state_final[0][:, :, i] = state_view[0][:, :, k]
                        state_final[1][:, i] = state_view[1][:, k]
                        out_final[i] = out_active[k]

            for idx in range(bsz):
                # decode with truncation at stop tokens to avoid bleed
                def _decode_until_stop(tokens_list, stop_set):
                    for j, t in enumerate(tokens_list):
                        if t in stop_set:
                            tokens_list = tokens_list[:j]
                            break
                    return tokenizer.decode(tokens_list, utf8_errors="ignore")

                gen_cot = _decode_until_stop(cot_tokens[idx], cot_cfg["stop_tokens"])
                gen_final = _decode_until_stop(final_tokens[idx], final_cfg["stop_tokens"])
                gen_text = f"{gen_cot}\nTherefore, the answer is \\(\\boxed{{{gen_final}"

                if attempt == 0:
                    first_sample_text[idx] = gen_text

                if got_correct[idx]:
                    continue

                pred = extract_number(gen_final)
                gold = cur_items[idx]["a"]
                pred_norm = _normalize_text(pred)
                gold_norm = _normalize_text(gold)
                if pred_norm and gold_norm and (pred_norm == gold_norm):
                    got_correct[idx] = True
                    first_hit_text[idx] = gen_text

        for idx in range(bsz):
            total_cnt += 1
            total_acc += 1 if got_correct[idx] else 0

            gen_text = first_hit_text[idx] if got_correct[idx] and first_hit_text[idx] else first_sample_text[idx]
            final_only = gen_text.split("Therefore, the answer is ", 1)
            final_segment = final_only[1] if len(final_only) > 1 else gen_text
            pred = extract_number(final_segment)
            gold = cur_items[idx]["a"]
            pred_norm = _normalize_text(pred)
            gold_norm = _normalize_text(gold)
            correct_exact = pred_norm == gold_norm

            if total_cnt <= 5:
                print("-" * 40)
                print(f"Q: {cur_items[idx]['q']}")
                print(f"Gold: {gold} | Pred: {pred} | exact={correct_exact}")
                print(f"Generated: ...{gen_text[:200].replace(chr(10), ' ')}...")

            if args_cli.output:
                saved_rows.append(
                    {
                        "question": cur_items[idx]["q"],
                        "gold": gold,
                        "pred": pred,
                        "pred_norm": pred_norm,
                        "gold_norm": gold_norm,
                        "correct": got_correct[idx],
                        "correct_exact": correct_exact,
                        "gen": gen_text,
                    }
                )

    acc = total_acc / total_cnt
    print("=" * 80)
    print(f"GSM8K rollout done. samples={total_cnt} acc={acc:.3f}")

    if args_cli.output and saved_rows:
        out_path = Path(args_cli.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for row in saved_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"saved predictions to {out_path}")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
