"""On-policy RL on GSM8K with a verifiable reward: GRPO and PPO from scratch, sharing everything else.

Per RL step:
  1. rollout: B prompts x G samples from vLLM (colocated, weights synced from the policy every step)
  2. reward : 1 if "#### <answer>" matches the gold number, else 0
  3. old / ref log-probs with the HF policy and frozen reference model
  4. advantages
       GRPO: (r - group mean) / (group std + eps), same value for every token of a sample; KL(k3) in the loss
       PPO : critic V(s_t), per-token reward = -beta * (logp_old - logp_ref) plus R on the last token, GAE
  5. one pass over the rollout in `n_mb` minibatches, clipped ratio objective (one optimizer step per minibatch)

Usage: python -m pt.rl <algo> <start_ckpt> <run_name> [steps]
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"   # keep the engine in-process so weights can be pushed into it
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

import bitsandbytes as bnb
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from .data import PROMPT, ROOT, extract, load, reward

STOP = ["\nQuestion:", "Question:"]


@dataclass
class Cfg:
    algo: str = "grpo"           # grpo | ppo
    start: str = ""
    name: str = ""
    steps: int = 150
    B: int = 16                  # prompts per step
    G: int = 8                   # samples per prompt (PPO also uses 8 so both see 128 samples per step)
    max_new: int = 320
    temp: float = 1.0
    lr: float = 1e-6
    critic_lr: float = 1e-5
    beta: float = 0.04           # KL coefficient
    clip: float = 0.2
    n_mb: int = 4                # optimizer steps per rollout
    micro: int = 4               # micro-batch for forward/backward
    gae_lambda: float = 0.95
    critic_train_layers: int = 6 # PPO critic: only top-k transformer layers + value head are trained (memory)
    eval_every: int = 25
    seed: int = 0
    vllm_util: float = 0.22


def selective_logprobs(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """log p(ids) from logits via fused cross-entropy (fp32 over the vocab)."""
    b, L, V = logits.shape
    return -F.cross_entropy(logits.reshape(-1, V).float(), ids.reshape(-1), reduction="none").view(b, L)


def token_logprobs(model, ids, att):
    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1]
    return selective_logprobs(logits, ids[:, 1:])          # [b, L-1], position t predicts token t+1


class Critic(nn.Module):
    def __init__(self, path, train_layers):
        super().__init__()
        self.body = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).model
        self.head = nn.Linear(self.body.config.hidden_size, 1, dtype=torch.bfloat16)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        for p in self.body.parameters():
            p.requires_grad_(False)
        for layer in self.body.layers[-train_layers:]:
            for p in layer.parameters():
                p.requires_grad_(True)
        for p in self.body.norm.parameters():
            p.requires_grad_(True)
        self.body.gradient_checkpointing_enable()

    def forward(self, ids, att):
        h = self.body(input_ids=ids, attention_mask=att).last_hidden_state
        return self.head(h).squeeze(-1).float()[:, :-1]     # value of the state before token t+1


class Trainer:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        self.out = ROOT / "runs" / cfg.name
        self.out.mkdir(parents=True, exist_ok=True)
        # vLLM first, so it reserves its slice before the training models allocate
        self.llm = LLM(cfg.start, dtype="bfloat16", gpu_memory_utilization=cfg.vllm_util, max_model_len=1024,
                       seed=cfg.seed, enforce_eager=False)
        self.tok = AutoTokenizer.from_pretrained(cfg.start)
        self.policy = AutoModelForCausalLM.from_pretrained(cfg.start, torch_dtype=torch.bfloat16).cuda()
        self.policy.gradient_checkpointing_enable(); self.policy.config.use_cache = False
        self.ref = AutoModelForCausalLM.from_pretrained(cfg.start, torch_dtype=torch.bfloat16).cuda().eval()
        for p in self.ref.parameters():
            p.requires_grad_(False)
        self.opt = bnb.optim.AdamW8bit(self.policy.parameters(), lr=cfg.lr, betas=(0.9, 0.99), weight_decay=0.0)
        self.critic = None
        if cfg.algo == "ppo":
            self.critic = Critic(cfg.start, cfg.critic_train_layers).cuda()
            self.copt = bnb.optim.AdamW8bit([p for p in self.critic.parameters() if p.requires_grad],
                                            lr=cfg.critic_lr, weight_decay=0.0)
        self.train_data = load("train")
        self.val = load("val")
        self.log = []

    # ---- vLLM weight sync -------------------------------------------------------------
    def sync(self):
        model = self.llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
        model.load_weights((n, p.data) for n, p in self.policy.named_parameters())
        self.llm.reset_prefix_cache()

    # ---- rollout ------------------------------------------------------------------------
    def rollout(self):
        c = self.cfg
        batch = random.sample(self.train_data, c.B)
        prompts = [PROMPT.format(q=d["q"]) for d in batch]
        outs = self.llm.generate(prompts, SamplingParams(n=c.G, temperature=c.temp, max_tokens=c.max_new,
                                                         stop=STOP, seed=random.randrange(1 << 30)), use_tqdm=False)
        seqs, rewards, fmt, groups = [], [], [], []
        for gi, (o, d, p) in enumerate(zip(outs, batch, prompts)):
            p_ids = self.tok(p)["input_ids"]
            for s in o.outputs:
                ids = list(s.token_ids)
                if s.finish_reason == "stop" and s.stop_reason is not None:   # stopped on a stop string
                    ids = self.tok(s.text)["input_ids"]
                ids = ids[:c.max_new]
                seqs.append((p_ids, ids))
                rewards.append(reward(s.text, d["gold"]))
                fmt.append(extract(s.text) is not None)
                groups.append(gi)
        return seqs, torch.tensor(rewards), fmt, torch.tensor(groups)

    def pad(self, seqs):
        L = max(len(p) + len(cpl) for p, cpl in seqs)
        n = len(seqs)
        ids = torch.full((n, L), self.tok.pad_token_id or 0)
        att = torch.zeros((n, L), dtype=torch.long)
        cmask = torch.zeros((n, L - 1))          # aligned with logprob positions
        for i, (p, cpl) in enumerate(seqs):
            x = p + cpl
            ids[i, :len(x)] = torch.tensor(x); att[i, :len(x)] = 1
            cmask[i, len(p) - 1:len(x) - 1] = 1
        return ids.cuda(), att.cuda(), cmask.cuda()

    @torch.no_grad()
    def forward_all(self, model_fn, ids, att):
        return torch.cat([model_fn(ids[i:i + self.cfg.micro], att[i:i + self.cfg.micro])
                          for i in range(0, len(ids), self.cfg.micro)])

    # ---- advantages ---------------------------------------------------------------------
    def grpo_adv(self, r, groups, cmask):
        mean = torch.zeros_like(r); std = torch.zeros_like(r)
        for g in groups.unique():
            m = groups == g
            mean[m] = r[m].mean(); std[m] = r[m].std()
        a = (r - mean) / (std + 1e-4)
        return a.cuda().unsqueeze(1) * cmask

    def ppo_adv(self, r, old_lp, ref_lp, values, cmask):
        c = self.cfg
        kl = (old_lp - ref_lp) * cmask
        tok_r = -c.beta * kl
        last = cmask.cumsum(1).argmax(1)                     # index of the final completion token
        tok_r[torch.arange(len(r)), last] += r.cuda()
        adv = torch.zeros_like(tok_r)
        gae = torch.zeros(len(r), device=r.device if r.is_cuda else "cuda")
        values = values * cmask
        T = tok_r.size(1)
        for t in reversed(range(T)):
            nv = values[:, t + 1] if t + 1 < T else torch.zeros_like(gae)
            nv = nv * cmask[:, t + 1] if t + 1 < T else nv
            delta = tok_r[:, t] + nv - values[:, t]
            gae = (delta + c.gae_lambda * gae) * cmask[:, t]
            adv[:, t] = gae
        ret = adv + values
        m = cmask.bool()
        adv = (adv - adv[m].mean()) / (adv[m].std() + 1e-6) * cmask
        return adv, ret

    # ---- one RL step --------------------------------------------------------------------
    def step(self, it):
        c = self.cfg
        t0 = time.time()
        seqs, r, fmt, groups = self.rollout()
        t_gen = time.time() - t0
        ids, att, cmask = self.pad(seqs)
        t1 = time.time()
        old_lp = self.forward_all(lambda a, b: token_logprobs(self.policy, a, b), ids, att)
        ref_lp = self.forward_all(lambda a, b: token_logprobs(self.ref, a, b), ids, att)
        if c.algo == "grpo":
            adv = self.grpo_adv(r, groups, cmask); ret = values_old = None
        else:
            values_old = self.forward_all(self.critic, ids, att)
            adv, ret = self.ppo_adv(r, old_lp, ref_lp, values_old, cmask)

        t_lp = time.time() - t1
        n = len(seqs)
        perm = torch.randperm(n)
        stats = {"clipfrac": [], "kl": [], "vloss": []}
        self.policy.train()
        for mb in perm.chunk(c.n_mb):
            for chunk in mb.chunk(max(1, len(mb) // c.micro)):
                m = cmask[chunk]
                lp = token_logprobs(self.policy, ids[chunk], att[chunk])
                ratio = torch.exp(lp - old_lp[chunk])
                a = adv[chunk]
                pg = -torch.min(ratio * a, ratio.clamp(1 - c.clip, 1 + c.clip) * a)
                if c.algo == "grpo":
                    d = ref_lp[chunk] - lp
                    kl = torch.exp(d) - d - 1                        # k3 estimator, >= 0
                    per_tok = pg + c.beta * kl
                    loss = ((per_tok * m).sum(1) / m.sum(1).clamp(min=1)).mean()   # sequence-mean (GRPO paper)
                    stats["kl"].append(((kl * m).sum() / m.sum()).item())
                else:
                    loss = (pg * m).sum() / m.sum()
                    stats["kl"].append((((old_lp[chunk] - ref_lp[chunk]) * m).sum() / m.sum()).item())
                (loss * len(chunk) / len(mb)).backward()
                stats["clipfrac"].append(((((ratio - 1).abs() > c.clip).float() * m).sum() / m.sum()).item())
                if c.algo == "ppo":
                    v = self.critic(ids[chunk], att[chunk])
                    vclip = values_old[chunk] + (v - values_old[chunk]).clamp(-c.clip, c.clip)
                    vl = torch.max((v - ret[chunk]) ** 2, (vclip - ret[chunk]) ** 2)
                    vloss = 0.5 * (vl * m).sum() / m.sum()
                    (vloss * len(chunk) / len(mb)).backward()
                    stats["vloss"].append(vloss.item())
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.opt.step(); self.opt.zero_grad(set_to_none=True)
            if c.algo == "ppo":
                torch.nn.utils.clip_grad_norm_([p for p in self.critic.parameters() if p.requires_grad], 1.0)
                self.copt.step(); self.copt.zero_grad(set_to_none=True)
        self.sync()

        gm = torch.stack([r[groups == g].mean() for g in groups.unique()])
        rec = {"step": it, "reward": r.mean().item(), "format": sum(fmt) / len(fmt),
               "frac_mixed": ((gm > 0) & (gm < 1)).float().mean().item(),
               "frac_all_wrong": (gm == 0).float().mean().item(),
               "len": float(cmask.sum(1).mean()), "kl": sum(stats["kl"]) / len(stats["kl"]),
               "clipfrac": sum(stats["clipfrac"]) / len(stats["clipfrac"]),
               "vloss": (sum(stats["vloss"]) / len(stats["vloss"])) if stats["vloss"] else None,
               "t_gen": t_gen, "t_lp": t_lp, "t_step": time.time() - t0,
               "mem_gb": torch.cuda.max_memory_allocated() / 1e9}
        return rec

    @torch.no_grad()
    def eval_val(self):
        prompts = [PROMPT.format(q=d["q"]) for d in self.val]
        outs = self.llm.generate(prompts, SamplingParams(temperature=0, max_tokens=self.cfg.max_new, stop=STOP),
                                 use_tqdm=False)
        return sum(reward(o.outputs[0].text, d["gold"]) for o, d in zip(outs, self.val)) / len(self.val)

    def run(self):
        c = self.cfg
        evals = [{"step": 0, "val_greedy": self.eval_val()}]
        print(f"[{c.name}] step 0 val {evals[-1]['val_greedy']:.3f}", flush=True)
        for it in range(1, c.steps + 1):
            rec = self.step(it)
            self.log.append(rec)
            if it % 5 == 0 or it <= 3:
                print(f"[{c.name}] step {it} R {rec['reward']:.3f} fmt {rec['format']:.2f} mixed {rec['frac_mixed']:.2f} "
                      f"len {rec['len']:.0f} kl {rec['kl']:.4f} clip {rec['clipfrac']:.3f} "
                      f"gen {rec['t_gen']:.1f}s lp {rec['t_lp']:.1f}s step {rec['t_step']:.1f}s mem {rec['mem_gb']:.1f}GB", flush=True)
            if it % c.eval_every == 0:
                evals.append({"step": it, "val_greedy": self.eval_val()})
                print(f"[{c.name}] step {it} val {evals[-1]['val_greedy']:.3f}", flush=True)
            (self.out / "log.json").write_text(json.dumps({"cfg": asdict(c), "log": self.log, "evals": evals}))
        d = ROOT / "ckpt" / c.name
        self.policy.save_pretrained(d, safe_serialization=True); self.tok.save_pretrained(d)


if __name__ == "__main__":
    algo, start, name = sys.argv[1:4]
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 150
    extra = {"lr": float(sys.argv[5])} if len(sys.argv) > 5 else {}
    Trainer(Cfg(algo=algo, start=start, name=name, steps=steps, **extra)).run()
