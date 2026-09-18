# Draft plan: next experiments on V3 (for review — nothing started yet)

Status when drafted (19 Sep 2026, 04:40): reproduction finished (18 jobs); ablation running
(28 of 93 jobs done, estimated finish around 12:00–13:00). Hyperparameter optimization is **not** re-run
anywhere. Every experiment uses `optuna_results_v3__.json`, except the "untuned" one, which by
definition doesn't.

## 0. Ground rules for the codebase and the long run

1. **Never edit a file the live run imports** (`v3_core.py`, `run_experiments.py`). Workers restart
   every 8 jobs and would silently pick up half-edited code. All new work goes into new modules
   until the ablation finishes.
2. **One registry, one runner.** After the ablation finishes, restructure into:
   - `splits.py`: notebook random split + scaffold split, saved once to `splits/*.npz`, so every
     model sees identical indices.
   - `backbones.py`: MoLFormer (pinned revision 7b12d94) and ChemBERTa models (pinned revisions)
     behind one interface: tokenizer, hidden size, layer list for freezing and layer-wise LR decay (LLRD).
   - `experiments.py`: every experiment is a named list of jobs (split × variant × hyperparameters ×
     backbone × seed × fold). This is the single source of truth.
   - `run_experiments.py --exp <name>`: the same parallel runner, with the temperature guard and memory cap.
   - Results go to `results/<experiment>/jobs/`. The finished ablation moves to `results/ablation_random/`.
3. **Every result file records its exact configuration + a hash of the code.** The runner only skips a
   job if a finished result with the *same* hash exists, so results from different code versions
   can never mix.
4. **Smoke test before every long run.** Each experiment first runs 1 fold × 1 epoch (~1–2 min),
   so a bug costs minutes, not hours.
5. **One queue in priority order.** Results are usable as each experiment finishes, and the queue
   can resume after a crash or reboot.
6. Suggested: commit `v3_ablation/` to git on a branch (only if you say yes).

## 1. Scaffold-split comparison

Question: does V3 still work on **new chemotypes**? The current random split is easy: 11% of test
molecules have a fingerprint identical to a training molecule, 80% have a close analogue
(Tanimoto ≥ 0.7), and 68% share a training scaffold.

- **Split:** Bemis–Murcko scaffolds (2,402 unique), randomized and label-stratified by scaffold group
  (`StratifiedGroupKFold`).
  - About 15% of molecules form a held-out test set whose scaffolds never appear in training.
  - 5-fold scaffold-grouped CV runs on the remaining ~85%.
  - Duplicates and stereoisomers share a scaffold, so they automatically stay on the same side.
- **Models (same protocol as notebook cells 17–18):**
  - V3 full, 3 seeds.
  - V3 fingerprints-only (no MoLFormer), 3 seeds.
  - MoLFormer-only, 1 seed.
  - RF and XGBoost baselines.
  - Later, the best ChemBERTa backbone, if it's competitive.
- **Output:** a random-vs-scaffold table showing how much each model loses on unseen scaffolds.
  This is the strongest possible evidence for the pretrained-language-model claim.

## 2. Untuned V3 (before hyperparameter optimization)

- The notebook's own pre-HPO fallback config (cell 15): 8 heads, hidden 512, dropout 0.2, 3 classifier
  layers, 3 cross-modal layers, batch 32, lr 5e-6, wd 1e-4.
- Random split, 5-fold CV + ensemble test, 3 seeds.
- **Output:** tuned vs untuned, with the seed spread, answering "how much did Optuna buy?"

## 3. Non-balanced V3 (no class-balancing tricks)

- **(a) No sampler:** V3 trained on the natural 79/21 class ratio (focal loss α=0.5 is already neutral).
  Seed 42 already exists from the ablation; 2 more seeds.
- **(b) Fully vanilla:** no sampler + plain BCE instead of focal loss, 3 seeds.
- Same 5-fold CV + ensemble test. Metrics are reported both at the OOF MCC-optimal threshold (as in
  the notebook) and at a fixed 0.5 threshold, because imbalance mostly shows up at a fixed threshold.
  Calibration (Brier score, reliability curve) is included.

## 4. Backbone swap: ChemBERTa family in place of MoLFormer

| Backbone | Architecture | Status |
|---|---|---|
| DeepChem/ChemBERTa-77M-MLM | RoBERTa, 3 layers, hidden 384 | used in your `1_2 - ChemMTM` notebook; weights not cached |
| DeepChem/ChemBERTa-77M-MTR | RoBERTa, 3 layers, hidden 384 | used in `8_2 - ChemMTR`; only tokenizer/config cached |
| seyonec/ChemBERTa-zinc-base-v1 | RoBERTa, 6 layers, hidden 768 | used in `8_2 - ChemZinc`; not cached |
| optional: seyonec/PubChem10M_SMILES_BPE_450k, HUBioDataLab/SELFormer | RoBERTa | if you want more contemporaries |

- **Downloads need your approval.** Rough sizes: about 15 MB each for the two 77M models and about
  180 MB for zinc-base (confirmed before downloading). Revisions are pinned so a Hub update can't
  break anything.
- **Kept identical to V3:** fusion width 768, ECFP/MACCS/descriptor branches, cross-modal Transformer,
  gates, classifier, loss, sampler, augmentation, and all tuned hyperparameters.
- **Adapted per backbone:**
  - A linear 384→768 projection is added when the hidden size is 384.
  - The bottom third of layers is frozen (MoLFormer: 4 of 12, ChemBERTa-77M: 1 of 3, zinc-base: 2 of 6).
  - Layer-wise LR decay stays at 0.95.
  - Layer mixing uses the last min(4, n_layers) layers.
- 5-fold CV + ensemble test, 3 seeds each.
- **Caveat stated in the report:** the hyperparameters were tuned for MoLFormer, so ChemBERTa is
  compared under MoLFormer's settings (no re-tuning, as instructed).

## 5. Explainability

This data looks like HDAC-inhibitor data: 3,041 molecules (55%) carry a hydroxamic-acid zinc-binding
group (ZBG), and 84% of those are active. The checks below are built around that. Everything
explains **your published models** (`models/kfold_v3_fold1-5.pth`), read-only.

1. **Which modality matters:**
   - Gate activations per modality.
   - Cross-modal attention maps.
   - Occlusion: replace one modality with its training mean and measure the change in probability and AUC.
2. **Atom-level attributions:**
   - Integrated Gradients on MoLFormer token embeddings, mapped to atoms.
   - Integrated Gradients on ECFP bits, mapped to atom environments via bitInfo.
   - Averaged over several random-feature draws, since MoLFormer's attention is stochastic.
   - Rendered as RDKit heat maps for about 12 test molecules (TP / TN / FP / FN).
3. **Chemistry sanity check (quantitative):** share of attribution on ZBG vs linker vs cap, for actives
   vs inactives; and which substructures push predictions toward active or inactive across the test set.
4. **Faithfulness test:** delete the top-k attributed tokens or bits and measure the probability drop
   vs deleting random ones (deletion curves).
5. **Cross-check:** TreeSHAP on the XGBoost ECFP model. Do both models point at the same substructures?
6. **Applicability domain:** error rate vs similarity to the training set, plus a calibration curve.

## 6. Order, time, heat

| # | Job | Heavy GPU jobs | Estimated wall-clock time |
|---|---|---|---|
| — | finish current ablation | 65 left | ~8 h (ETA 12:00–13:00) |
| 0 | restructure + smoke tests | — | ~30 min |
| 1 | scaffold split | ~25 | ~3 h |
| 2 | non-balanced (a) + (b) | 25 | ~3.3 h |
| 3 | untuned | 15 | ~2.2 h |
| 4 | ChemBERTa ×3 backbones | 45 (small models) | ~2.7 h |
| 5 | explainability | light | ~1 h |

- **Total after the ablation:** about 12 h, so everything ends roughly 20 h from now.
- **The biggest speed lever is the temperature limit.** At 78 °C the workers pause ~40–50% of the
  time. An 80–82 °C limit (still below the 3090's own ~83 °C throttle point) would save roughly
  25–35%. Your call.
- **Final deliverable:** one report (tables + figures) covering the reproduction, ablation, scaffold
  split, untuned, non-balanced, backbones and explainability, plus the CSVs behind every number.

## Decisions needed from you

1. Untuned config: the notebook's fallback config (recommended) or the class defaults?
2. Non-balanced: both (a) and (b) (recommended), or only (a)?
3. Approve downloading the three ChemBERTa models? Add PubChem10M / SELFormer?
4. Temperature limit: keep 78 °C, or allow 80–82 °C?
5. Scaffold split: randomized stratified (recommended) or deterministic (largest scaffolds to train,
   harder)?
6. Is the target HDAC? (This decides the chemistry checks in explainability.)
7. OK to commit the code to a git branch?
