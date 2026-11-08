"""vLLM evaluation of a checkpoint: format rate, greedy pass@1, sampled pass@k, per-prompt solve rate."""
import json
import os
import sys
from math import comb
from pathlib import Path

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
from vllm import LLM, SamplingParams

from .data import PROMPT, ROOT, extract, load, reward

STOP = ["\nQuestion:", "Question:"]


def pass_at_k(n, c, k):
    return 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)


def evaluate(model_path: str, splits=("val", "test"), k=8, max_tokens=320, gpu_util=0.6, out=None):
    llm = LLM(model_path, dtype="bfloat16", gpu_memory_utilization=gpu_util, max_model_len=1024,
              seed=0, enforce_eager=False)
    res = {"model": model_path}
    for split in splits:
        data = load(split)
        prompts = [PROMPT.format(q=d["q"]) for d in data]
        greedy = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=max_tokens, stop=STOP), use_tqdm=False)
        g = [o.outputs[0].text for o in greedy]
        samp = llm.generate(prompts, SamplingParams(n=k, temperature=1.0, top_p=1.0, max_tokens=max_tokens,
                                                    stop=STOP, seed=0), use_tqdm=False)
        solve = [sum(reward(c.text, d["gold"]) for c in o.outputs) / k for o, d in zip(samp, data)]
        r = {
            "n": len(data),
            "greedy_acc": sum(reward(t, d["gold"]) for t, d in zip(g, data)) / len(data),
            "greedy_format": sum(extract(t) is not None for t in g) / len(data),
            "sample_acc": sum(solve) / len(data),                       # = pass@1 at T=1
            "sample_format": sum(extract(c.text) is not None for o in samp for c in o.outputs) / (k * len(data)),
            f"pass@{k}": sum(pass_at_k(k, round(s * k), k) for s in solve) / len(data),
            "frac_all_wrong": sum(s == 0 for s in solve) / len(data),
            "frac_all_right": sum(s == 1 for s in solve) / len(data),
            "frac_mixed": sum(0 < s < 1 for s in solve) / len(data),     # prompts that give GRPO a gradient
            "mean_len_tokens": sum(len(c.token_ids) for o in samp for c in o.outputs) / (k * len(data)),
            "examples": [{"q": d["q"], "gold": d["gold"], "out": t} for t, d in list(zip(g, data))[:3]],
        }
        res[split] = r
        print(f"[{Path(model_path).name}] {split}: greedy {r['greedy_acc']:.3f} fmt {r['greedy_format']:.3f} | "
              f"T=1 acc {r['sample_acc']:.3f} pass@{k} {r[f'pass@{k}']:.3f} mixed {r['frac_mixed']:.3f} "
              f"len {r['mean_len_tokens']:.0f}", flush=True)
    if out:
        Path(out).write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    evaluate(sys.argv[1], out=sys.argv[2] if len(sys.argv) > 2 else None)
