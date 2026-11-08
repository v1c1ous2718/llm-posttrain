"""Full-parameter SFT of Qwen2.5-0.5B on GSM8K solutions. Loss on completion tokens only.
Saves HF checkpoints at chosen optimizer steps so each one can be evaluated and used as an RL start."""
import json
import math
import random
import sys
import time

import bitsandbytes as bnb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import MODEL, PROMPT, ROOT, load

CKPT = ROOT / "ckpt"


def batches(tok, data, bs, seed, max_len=512):
    rng = random.Random(seed)
    order = list(range(len(data)))
    while True:
        rng.shuffle(order)
        for i in range(0, len(order) - bs + 1, bs):
            ids, labels = [], []
            for j in order[i:i + bs]:
                p = tok(PROMPT.format(q=data[j]["q"]))["input_ids"]
                c = tok(data[j]["sol"])["input_ids"] + [tok.eos_token_id]
                x = (p + c)[:max_len]
                ids.append(x)
                labels.append(([-100] * len(p) + c)[:max_len])
            L = max(map(len, ids))
            inp = torch.full((bs, L), tok.pad_token_id)
            lab = torch.full((bs, L), -100)
            att = torch.zeros((bs, L), dtype=torch.long)
            for k, (x, y) in enumerate(zip(ids, labels)):
                inp[k, :len(x)] = torch.tensor(x); lab[k, :len(y)] = torch.tensor(y); att[k, :len(x)] = 1
            yield inp, lab, att


def train(steps=438, bs=32, micro=4, lr=1e-5, warmup=12, save_at=(25, 50, 100, 200, 438), seed=0):
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / (steps - warmup))))
    data = load("train")
    it = batches(tok, data, micro, seed)
    log, t0 = [], time.time()
    for step in range(1, steps + 1):
        tot, ntok = 0.0, 0
        for _ in range(bs // micro):
            inp, lab, att = (t.cuda() for t in next(it))
            out = model(input_ids=inp, attention_mask=att)
            logits = out.logits[:, :-1].float()
            tgt = lab[:, 1:]
            loss_sum = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                                                         ignore_index=-100, reduction="sum")
            n = (tgt != -100).sum()
            (loss_sum / n / (bs // micro)).backward()
            tot += loss_sum.item(); ntok += n.item()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        log.append({"step": step, "loss": tot / ntok, "grad_norm": gn, "lr": sched.get_last_lr()[0],
                    "examples": step * bs, "time": time.time() - t0})
        if step % 10 == 0 or step == 1:
            print(f"sft step {step} loss {tot/ntok:.4f} gn {gn:.2f} ({time.time()-t0:.0f}s)", flush=True)
        if step in save_at:
            d = CKPT / f"sft_{step:04d}"
            model.save_pretrained(d, safe_serialization=True); tok.save_pretrained(d)
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs" / "sft_log.json").write_text(json.dumps(log))


if __name__ == "__main__":
    train()
