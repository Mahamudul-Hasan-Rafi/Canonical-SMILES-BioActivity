# V3 reproduction + ablation study

Standalone scripts (the notebook `bioactivity_dl 8_2_C_v3_re.ipynb` and `models/` are never modified).
Run everything with the project venv: `E:\ML\BioActivity\.venv\Scripts\python.exe`.

| Script | What it does | Time |
|---|---|---|
| `check_saved_models.py` | Loads your saved `models/kfold_v3_fold*.pth` / `best_finetuned_model_v3__.pth`, proves the port is exact, re-scores the test set 10× | ~1 min |
| `run_experiments.py --plan all --workers 3` | Retrains full V3 (3 seeds, both notebook protocols) + 15 ablations (5-fold CV) | several hours, resumable |
| `baselines.py` | RF / XGBoost / LogReg on the same folds (CPU) | ~1 min |
| `data_checks.py` | Duplicates, test-vs-train similarity, scaffold overlap | ~1 min |
| `summarize.py` | Builds `results/summary.md`, `ablation_table.csv`, `reproduction_table.csv`, `ablation_delta_auc.png` | seconds |

## What changed vs the notebook (and why the numbers are still comparable)

* **Dynamic padding** instead of 512 tokens: identical outputs (max |Δlogit| = 7e-7 in fp32), ~3× less compute.
* **MoLFormer attention patch**: removes a blocking `torch.equal` check and draws the random
  attention features on the CPU; the output is bit-identical given the same features, and the
  feature distribution is unchanged.
* **Fused AdamW**: same update rule, fewer kernels.
* **Several training processes share the GPU**, each capped at ≤25 % of VRAM (on Windows
  going over 24 GB silently spills to system RAM and everything crawls).
* **Thermal guard**: every worker pauses at 78 °C GPU and resumes at 70 °C (`--hot/--cool`).
  `logs/gpu.csv` records temperature/power every minute.
* **MoLFormer is pinned** to revision `7b12d94` and loaded offline. The Hub's current revision
  requires a newer `transformers` and fails to import with 4.57.
* Each fold model scores the test set with **its own** descriptor scaler (the notebook used the
  last fold's scaler for all five; the effect is < 0.001 on every metric).

Early-stopping, epochs, patience, LR schedule, LLRD, focal loss, sampler, augmentation and all
hyper-parameters are the notebook's (`optuna_results_v3__.json`).

Restarting `run_experiments.py` skips finished jobs (`results/jobs/*.json`).
