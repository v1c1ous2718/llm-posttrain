"""Self-training (expert iteration, ReST-EM style) on GSM8K: the model learns from its own verified answers.

Design, fixed before running:
  start      ckpt/sft_0438 (the same start as the GRPO / PPO runs)
  rounds     3
  generate   round r uses the round r-1 model (round 1: the start). 1,600 GSM8K-train prompts per round,
             disjoint across rounds (seed 0 permutation), 8 samples each at T = 1, max 320 tokens.
             1,600 x 8 = 12,800 samples = the sampling budget of 100 GRPO steps (16 prompts x 8).
  filter     keep completions whose "#### N" matches the gold answer; truncate after the answer;
             drop exact duplicates; at most 4 per prompt.
  train      ReST-EM: fine-tune from the fixed start (not the previous round) on this round's data,
             same recipe as the original SFT (AdamW8bit, lr 1e-5, batch 32, warmup 12, cosine), 2 epochs.
  evaluate   pt.evaluate on val and test; test is reported. Generation-side diversity is logged per round.

  uv run python -m pt.selftrain gen   <round>     # writes runs/selftrain/round<r>_data.json
  uv run python -m pt.selftrain train <round>     # writes ckpt/selftrain_r<r>
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time

from .data import MODEL, PROMPT, ROOT, ANS, load, reward

START = ROOT / "ckpt" / "sft_0438"
OUT = ROOT / "runs" / "selftrain"
N_PROMPTS, K, CAP, ROUNDS = 1600, 8, 4, 3


def model_for(r: int):
    return START if r == 0 else ROOT / "ckpt" / f"selftrain_r{r}"


def prompts_for(r: int) -> list[dict]:
    data = load("train")
    idx = list(range(len(data)))
    random.Random(0).shuffle(idx)
    return [data[i] for i in idx[(r - 1) * N_PROMPTS: r * N_PROMPTS]]


def gen(r: int):
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    from vllm import LLM, SamplingParams
    from .evaluate import STOP
    OUT.mkdir(parents=True, exist_ok=True)
    data = prompts_for(r)
    llm = LLM(str(model_for(r - 1)), dtype="bfloat16", gpu_memory_utilization=0.6, max_model_len=1024, seed=r)
    t0 = time.time()
    outs = llm.generate([PROMPT.format(q=d["q"]) for d in data],
                        SamplingParams(n=K, temperature=1.0, top_p=1.0, max_tokens=320, stop=STOP, seed=r),
                        use_tqdm=False)
    kept, solved, distinct_ans, distinct_txt, lens = [], 0, [], [], []
    for d, o in zip(data, outs):
        texts = [c.text for c in o.outputs]
        lens += [len(c.token_ids) for c in o.outputs]
        answers = [m.group(1) if (m := ANS.search(t)) else None for t in texts]
        distinct_ans.append(len(set(answers)))
        distinct_txt.append(len(set(texts)))
        good, seen = [], set()
        for t in texts:
            if reward(t, d["gold"]) == 1.0:
                sol = t[:ANS.search(t).end()].strip()
                if sol not in seen:
                    seen.add(sol); good.append(sol)
        solved += bool(good)
        kept += [{"q": d["q"], "sol": s, "gold": d["gold"]} for s in good[:CAP]]
    stats = {"round": r, "generator": str(model_for(r - 1).name), "prompts": len(data), "samples": len(data) * K,
             "sample_acc": sum(reward(c.text, d["gold"]) for d, o in zip(data, outs) for c in o.outputs) / (len(data) * K),
             "prompts_solved": solved / len(data), "kept": len(kept),
             "mean_distinct_answers": sum(distinct_ans) / len(data),
             "mean_distinct_texts": sum(distinct_txt) / len(data),
             "mean_len_tokens": sum(lens) / len(lens), "gen_time_s": time.time() - t0}
    (OUT / f"round{r}_data.json").write_text(json.dumps(kept))
    (OUT / f"round{r}_gen.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats), flush=True)


def train(r: int, epochs=2, bs=32, micro=4, lr=1e-5, warmup=12, seed=0):
    import bitsandbytes as bnb
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .sft import batches
    torch.manual_seed(seed)
    data = json.loads((OUT / f"round{r}_data.json").read_text())
    steps = max(warmup + 1, math.ceil(epochs * len(data) / bs))
    tok = AutoTokenizer.from_pretrained(START)
    model = AutoModelForCausalLM.from_pretrained(START, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / (steps - warmup))))
    it = batches(tok, data, micro, seed)
    log, t0 = [], time.time()
    for step in range(1, steps + 1):
        tot, ntok = 0.0, 0
        for _ in range(bs // micro):
            inp, lab, att = (t.cuda() for t in next(it))
            logits = model(input_ids=inp, attention_mask=att).logits[:, :-1].float()
            tgt = lab[:, 1:]
            loss_sum = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                                                         ignore_index=-100, reduction="sum")
            n = (tgt != -100).sum()
            (loss_sum / n / (bs // micro)).backward()
            tot += loss_sum.item(); ntok += n.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        log.append({"step": step, "loss": tot / ntok, "time": time.time() - t0})
        if step % 20 == 0 or step == 1:
            print(f"selftrain r{r} step {step}/{steps} loss {tot/ntok:.4f} ({time.time()-t0:.0f}s)", flush=True)
    d = model_for(r)
    model.save_pretrained(d, safe_serialization=True); tok.save_pretrained(d)
    (OUT / f"round{r}_train.json").write_text(json.dumps({"examples": len(data), "steps": steps, "epochs": epochs,
                                                          "train_time_s": time.time() - t0,
                                                          "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
                                                          "log": log}))


if __name__ == "__main__":
    {"gen": gen, "train": train}[sys.argv[1]](int(sys.argv[2]))
