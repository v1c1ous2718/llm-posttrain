"""GSM8K splits and the answer checker (the verifiable reward).

  sft/rl prompts : GSM8K train minus 473 held-out  (7000)
  val            : 473 from GSM8K train (seed 0)   -> all selection
  test           : GSM8K test (1319)               -> reported only
"""
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODEL = "Qwen/Qwen2.5-0.5B"

PROMPT = "Question: {q}\nAnswer: Let's think step by step.\n"
# Completion target format (SFT): reasoning lines, then "#### <number>"


def clean_solution(sol: str) -> str:
    sol = re.sub(r"<<[^>]*>>", "", sol)          # drop calculator annotations
    return sol.strip()


def gold_number(sol: str) -> str:
    return norm(sol.split("####")[-1])


def norm(s: str) -> str:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


ANS = re.compile(r"####\s*(-?[\d,]*\.?\d+)")


def extract(completion: str):
    """Final answer if the completion follows the format, else None."""
    m = ANS.search(completion)
    return norm(m.group(1)) if m else None


def reward(completion: str, gold: str) -> float:
    a = extract(completion)
    return 1.0 if a is not None and a == gold else 0.0


def build():
    DATA.mkdir(exist_ok=True)
    tr = load_dataset("openai/gsm8k", "main", split="train")
    te = load_dataset("openai/gsm8k", "main", split="test")
    idx = list(range(len(tr)))
    random.Random(0).shuffle(idx)
    val_idx, train_idx = set(idx[:473]), idx[473:]

    def rec(r):
        return {"q": r["question"], "sol": clean_solution(r["answer"]), "gold": gold_number(r["answer"])}
    out = {"train": [rec(tr[i]) for i in train_idx], "val": [rec(tr[i]) for i in sorted(val_idx)],
           "test": [rec(r) for r in te]}
    for k, v in out.items():
        (DATA / f"{k}.json").write_text(json.dumps(v))
        print(k, len(v))


def load(split):
    return json.loads((DATA / f"{split}.json").read_text())


if __name__ == "__main__":
    build()
