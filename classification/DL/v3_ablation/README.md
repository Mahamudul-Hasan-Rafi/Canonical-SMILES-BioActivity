# V3 verification, ablation and extended study

Standalone study of `../bioactivity_dl 8_2_C_v3_re.ipynb` (HDAC1 activity classification,
ChEMBL IC50 data). The notebook and `../models/` are never modified.
**Start with [`results/REPORT.md`](results/REPORT.md).**

Run everything with the project venv: `E:\ML\BioActivity\.venv\Scripts\python.exe`.

## Layout

| File | Role |
|---|---|
| `data.py` | dataset loading + cached ECFP / MACCS / descriptor features |
| `splits.py` | the notebook's random split and a Bemis-Murcko scaffold split, frozen to `splits/*.npz` |
| `backbones.py` | SMILES encoders (MoLFormer, ChemBERTa-77M-MLM/MTR, ChemBERTa-zinc), each pinned to a Hub revision, loaded offline |
| `core.py` | V3 model (backbone-generic, ablation switches), training loop, prediction |
| `experiments.py` | registry: every job = (protocol, split, backbone, hp set, variant, seed, fold) |
| `runner.py` | parallel, resumable, thermally guarded job runner (`--queue`, `--exp`, `--smoke`, `--status`) |
| `report.py`, `metrics.py` | builds `results/REPORT.md` (notebook metrics, paired DeLong + Holm) |
| `metrics_table.py` | accuracy / balanced accuracy / precision / recall / F1 of every experiment |
| `baselines.py` | RF / XGBoost / LogReg on the same folds (CPU) |
| `explain_v3.py`, `explain_figures.py` | explainability of the published ensemble (exact modality Shapley, SHAP, IG, LIME, TreeSHAP, deletion test, applicability domain) |
| `descriptor_stats.py` | descriptor vs activity tests (Mann-Whitney, Welch/Student t, z, scaffold-clustered logistic regression) |
| `check_saved_models.py` | re-scores the notebook's saved models; proves the port is exact |
| `data_checks.py` | duplicates, test-vs-train similarity |
| `v3_core.py`, `notebook_classes.py`, `migrate_legacy.py` | frozen legacy code (reference for `tests/test_core_parity.py`), the notebook's classes for unpickling, and the importer of the first run's results |
| `tests/test_core_parity.py` | proves `core.py` == `v3_core.py` (bit-identical init, forward and training) |

Results: `results/store/` (one `.npz` + `.json` per trained model, with config hash),
`results/baselines/`, `results/explain/`, `results/store_extra/` (runs outside the final design),
`results/archive/` (first-run layout). All of `results/`, `cache/`, `splits/`, `logs/` are git-ignored.

## Reproduce

```
python runner.py --smoke                      # 1-epoch check of every configuration
python runner.py --queue --workers 3          # all new experiments (skips finished jobs)
python baselines.py --split random; python baselines.py --split scaffold
python explain_v3.py                          # explainability (published models)
python report.py; python metrics_table.py; python descriptor_stats.py
```

## Differences from the notebook (none change the maths)

* Dynamic padding instead of 512 tokens (outputs identical; ~3x less compute).
* MoLFormer attention patch: no blocking `torch.equal` check; random attention features drawn
  on the CPU (bit-identical output given the same features; same distribution).
* Fused AdamW, TF32, best checkpoint kept in RAM, several training processes share the GPU
  (each capped at 25 % VRAM; Windows otherwise spills to system RAM).
* SMILES augmentation is seeded (RDKit's `doRandom` ignores every seed, so the notebook's
  augmentation was never reproducible).
* Each fold model scores the test set with its own descriptor scaler (the notebook reused the
  last fold's; effect < 0.001).
* MoLFormer pinned to revision `7b12d94`; the Hub's current revision needs a newer `transformers`.
* Thermal guard: `--hot/--cool` (default 78/70 °C; the final runs used 88/80 as an emergency-only stop).
