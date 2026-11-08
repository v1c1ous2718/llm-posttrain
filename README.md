# When should SFT hand over to RL? SFT → GRPO / PPO on GSM8K

Qwen2.5-0.5B **base**, full fine-tuning on one RTX 3060 (12 GB).

GRPO and PPO are written from scratch in PyTorch. Rollouts come from vLLM running in the same process, and the policy weights are synced to vLLM after every step. This extends NTU GenAI-ML 2025 HW07, which used DPO, by switching to online RL.

Full report (in Chinese): [`report/report.md`](report/report.md). Evaluation JSONs and training logs are in `runs/`.

Key findings (GSM8K test set, 1,319 problems):

- **SFT mainly teaches the answer format.** 25 SFT steps take format compliance from 5.9% to 98%, and the share of prompts with both right and wrong samples (where GRPO gets gradient) from 2% to 49%. Seventeen times more SFT adds about 2 points.
- **RL from the base model starves.** GRPO starting from base reaches 12.4%. 25 SFT steps alone reach 35.1%.
- **GRPO from SFT at lr 5e-6:**
  - Greedy accuracy: 37.1% → 43.0%.
  - Pass@1 at T = 1: 17.3% → 29.4%.
  - Pass@8: 54.7% → 66.2%.
- **PPO at the same settings:** greedy 36.9%. It took 32% longer per step and used 1.6 GB more memory, because of the critic.
- **Self-training (ReST-EM style, 3 rounds):** fine-tune on the model's own correct answers only, with 12,800 samples per round, the same as 100 GRPO steps.
  - Pass@1 at T = 1 rises every round: 17.3% → 21.1% → 24.1% → 25.4%.
  - Greedy accuracy stays flat at 36.8–39.0%. Wrong answers give it no learning signal.
  - Pass@8 keeps rising (54.7% → 62.2%), so there is no sign of collapse within 3 rounds.
  - Round 1 is behind GRPO at the same sampling budget: greedy 36.8% vs 43.0%.

Caveats:
- Each configuration ran with one seed.
- The lr 5e-6 runs were added after the pre-registered lr 1e-6 runs barely moved the policy.
- Self-training was also added later. It used one seed and the SFT learning rate (1e-5).
- The PPO critic trains only its top 6 layers from a zero-initialized head.
- The validation split, taken from GSM8K train, scores 13–15 points above test, so only test is reported.

```bash
uv sync
uv run python -m pt.data
uv run python -m pt.sft
./scripts/run_all.sh                       # pre-registered evals, GRPO x4, PPO
uv run python -m pt.rl grpo ckpt/sft_0438 grpo_from_sft0438_lr5e-6 100 5e-6
uv run python -m pt.rl ppo  ckpt/sft_0438 ppo_from_sft0438_lr5e-6 100 5e-6
./scripts/selftrain.sh                     # self-training: 3 rounds of generate -> filter -> SFT -> eval
uv run python scripts/make_figs.py
```
