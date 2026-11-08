"""Figures for the SFT->RL report. Reads runs/eval_*.json, runs/<rl run>/log.json, runs/sft_log.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parents[1] / "runs"
F = Path(__file__).resolve().parents[1] / "report" / "figs"
F.mkdir(parents=True, exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "axes.axisbelow": True, "lines.linewidth": 2, "savefig.dpi": 200,
                     "savefig.bbox": "tight", "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
SFT_STEPS = [0, 25, 50, 100, 200, 438]
STARTS = [("grpo_from_base", 0), ("grpo_from_sft0025", 25), ("grpo_from_sft0100", 100), ("grpo_from_sft0438", 438)]


def j(p):
    p = R / p
    return json.loads(p.read_text()) if p.exists() else None


def ev(step):
    return j("eval_base.json") if step == 0 else j(f"eval_sft_{step:04d}.json")


def save(fig, n):
    fig.savefig(F / n); plt.close(fig); print("wrote", n)


def fig_sft_diagnostics():
    es = [(s, ev(s)) for s in SFT_STEPS if ev(s)]
    if len(es) < 2: return
    x = [s for s, _ in es]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    series = [("greedy_format", "format compliance (greedy)", C[3]), ("greedy_acc", "greedy accuracy", C[0]),
              ("pass@8", "pass@8 (T = 1)", C[2]), ("frac_mixed", "prompts with 0 < solve rate < 1", C[1])]
    for k, lab, c in series:
        y = [e["val"][k] * 100 for _, e in es]
        ax.plot(x, y, "o-", color=c, label=lab, ms=5)
    ax.set_xscale("symlog", linthresh=25); ax.set_xticks(x, [str(s) for s in x])
    ax.set_xlabel("SFT optimizer steps (batch 32; 438 = 2 epochs)"); ax.set_ylabel("% of val prompts")
    ax.set_ylim(-3, 103); ax.legend(frameon=False, fontsize=8.5, loc="lower right", bbox_to_anchor=(1, 0.08))
    ax.set_title("What SFT gives the RL stage (val, 8 samples per prompt)", loc="left", fontsize=11)
    save(fig, "sft_diagnostics.png")


def smooth(y, k=5):
    y = np.asarray(y, float)
    return np.convolve(y, np.ones(k) / k, mode="valid") if len(y) >= k else y


def fig_grpo_curves():
    runs = [(n, s, j(f"{n}/log.json")) for n, s in STARTS]
    runs = [r for r in runs if r[2]]
    if not runs: return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
    for i, (n, s, L) in enumerate(runs):
        lab = "base" if s == 0 else f"SFT {s} steps"
        st = [r["step"] for r in L["log"]]
        for ax, key in zip(axes[:2], ["reward", "frac_mixed"]):
            y = smooth([r[key] for r in L["log"]])
            ax.plot(st[len(st) - len(y):], np.array(y) * 100, color=C[i], label=lab)
        e = L["evals"]
        axes[2].plot([x["step"] for x in e], [x["val_greedy"] * 100 for x in e], "o-", color=C[i], label=lab, ms=4)
    axes[0].set_title("training reward (5-step mean, %)", loc="left", fontsize=10)
    axes[1].set_title("prompts with mixed group outcome (%)", loc="left", fontsize=10)
    axes[2].set_title("val greedy accuracy (%)", loc="left", fontsize=10)
    for ax in axes: ax.set_xlabel("GRPO step")
    axes[2].legend(frameon=False, fontsize=8.5)
    save(fig, "grpo_curves.png")


def fig_gain():
    rows = []
    for n, s in STARTS:
        a, b = ev(s), j(f"eval_{n}.json")
        if a and b:
            rows.append((s, a["test"]["greedy_acc"] * 100, b["test"]["greedy_acc"] * 100))
    if not rows: return
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    x = np.arange(len(rows))
    ax.bar(x - 0.18, [r[1] for r in rows], 0.34, color=C[0], label="before RL")
    ax.bar(x + 0.18, [r[2] for r in rows], 0.34, color=C[1], label="after 100 GRPO steps")
    for i, r in enumerate(rows):
        ax.text(i + 0.18, r[2] + 1, f"{r[2]-r[1]:+.1f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, ["base" if r[0] == 0 else f"SFT {r[0]}" for r in rows])
    ax.set_ylabel("test greedy accuracy (%)"); ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(r[2] for r in rows) * 1.12)
    ax.legend(frameon=False, fontsize=9, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2)
    ax.set_title("GSM8K test (1,319): accuracy before vs after GRPO", loc="left", fontsize=11, pad=22)
    save(fig, "grpo_gain.png")


def fig_ppo_vs_grpo():
    runs = [("grpo_from_sft0438", "GRPO, lr 1e-6", C[0], "--"), ("ppo_from_sft0438", "PPO, lr 1e-6", C[1], "--"),
            ("grpo_from_sft0438_lr5e-6", "GRPO, lr 5e-6", C[0], "-"), ("ppo_from_sft0438_lr5e-6", "PPO, lr 5e-6", C[1], "-")]
    runs = [(j(f"{n}/log.json"), lab, c, ls) for n, lab, c, ls in runs if j(f"{n}/log.json")]
    if not runs: return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
    for L, lab, c, ls in runs:
        st = [r["step"] for r in L["log"]]
        y = smooth([r["reward"] for r in L["log"]], 9)
        axes[0].plot(st[len(st) - len(y):], np.array(y) * 100, color=c, ls=ls, label=lab)
        e = L["evals"]
        axes[1].plot([x["step"] for x in e], [x["val_greedy"] * 100 for x in e], "o", color=c, ls=ls, label=lab, ms=4)
        y = smooth([r["kl"] for r in L["log"]], 9)
        axes[2].plot(st[len(st) - len(y):], y, color=c, ls=ls, label=lab)
    axes[0].set_title("training reward (9-step mean, %)", loc="left", fontsize=10)
    axes[1].set_title("val greedy accuracy (%)", loc="left", fontsize=10)
    axes[2].set_title("KL to reference (per token)", loc="left", fontsize=10)
    for ax in axes: ax.set_xlabel("RL step (start: SFT 438)")
    axes[1].legend(frameon=False, fontsize=8.5, handlelength=3.5)
    save(fig, "ppo_vs_grpo.png")


def fig_sft_loss():
    L = j("sft_log.json")
    if not L: return
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    y = smooth([r["loss"] for r in L], 9)
    ax.plot(np.arange(len(y)) + 5, y, color=C[0])
    for s in SFT_STEPS[1:]:
        ax.axvline(s, color=INK2, lw=0.8, ls=":")
    ax.set_xlabel("SFT step (dotted: saved checkpoints)"); ax.set_ylabel("token loss")
    ax.set_title("SFT loss (9-step mean)", loc="left", fontsize=11)
    save(fig, "sft_loss.png")


def fig_selftrain():
    rs = [("SFT 438", j("eval_sft_0438.json"))] + [(f"self-train r{r}", j(f"eval_selftrain_r{r}.json")) for r in (1, 2, 3)]
    if any(e is None for _, e in rs): return
    g = j("eval_grpo_from_sft0438_lr5e-6.json")["test"]
    gens = [json.loads((R / "selftrain" / f"round{r}_gen.json").read_text()) for r in (1, 2, 3)]
    x = np.arange(len(rs))
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for k, lab, c in [("greedy_acc", "greedy accuracy", C[0]), ("sample_acc", "T=1 pass@1", C[1]), ("pass@8", "pass@8", C[2])]:
        y = [e["test"][k] * 100 for _, e in rs]
        a.plot(x, y, "o-", color=c, label=lab)
        a.axhline(g[k] * 100, color=c, ls=":", lw=1.3)
    a.set_xticks(x, [n for n, _ in rs], fontsize=8.5); a.set_ylabel("GSM8K test (%)")
    a.set_title("self-training vs GRPO (dotted: GRPO lr 5e-6, 100 steps)", fontsize=9.5)
    a.legend(fontsize=8, frameon=False, loc="center", bbox_to_anchor=(0.5, 0.6), ncol=3)
    r = [1, 2, 3]
    b.plot(r, [s["mean_distinct_answers"] for s in gens], "o-", color=C[4], label="distinct answers / 8 samples")
    b.set_ylabel("distinct answers per prompt", color=C[4]); b.set_xticks(r, [f"round {i}" for i in r])
    b2 = b.twinx(); b2.grid(False)
    b2.plot(r, [s["prompts_solved"] * 100 for s in gens], "s--", color=C[3], label="prompts solved (%)")
    b2.set_ylabel("train prompts with >= 1 correct (%)", color=C[3])
    b.set_title("generation side (train prompts, 8 samples each)", fontsize=9.5)
    fig.tight_layout(); save(fig, "selftrain.png")


if __name__ == "__main__":
    fig_sft_diagnostics(); fig_sft_loss(); fig_grpo_curves(); fig_gain(); fig_ppo_vs_grpo(); fig_selftrain()
