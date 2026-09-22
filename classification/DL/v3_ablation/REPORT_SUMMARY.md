# HDAC1 bioactivity model — consolidated results

Dataset: 5,494 ChEMBL HDAC1 (CHEMBL325) molecules with IC50; active if pIC50 >= 6, inactive if <= 5,
the 1-10 uM band excluded. Two evaluation splits, each with 5-fold cross-validation on the
development set and a held-out test set used only for final reporting.

| split | development | test | test molecules sharing a Murcko scaffold with training |
|---|---|---|---|
| random (as in the original notebook) | 4,669 | 825 | **68.1 %** |
| scaffold (StratifiedGroupKFold on Murcko scaffolds) | 4,759 | 735 | **0 %** |

Median maximum Tanimoto of a test molecule to the training set: 0.806 (random), 0.731 (scaffold).

---

## 1. Proposed model

**Classification** — rank-average, fixed weights 2 : 1

* 2 x **LightGBM** on ECFP4 1024 binary + 6 RO5 descriptors (600 rounds, 63 leaves, lr 0.05),
  refitted per fold: 5 fitted models
* 1 x **V5-MT**, the multi-view network, 3 seeds x 5 folds: 15 networks

20 fitted models in total. The score is converted to a probability by per-fold Platt scaling
(fitted on the other four folds); the decision threshold is fitted out-of-fold (MCC-optimal).
Ranking metrics use the raw rank-average, since per-fold calibrators are not comparable across
folds.

**Potency** — 1 : 1 average of a LightGBM regressor (ECFP4 2048 count/chirality + MACCS +
6 descriptors) and the V8-R2 network's predicted pIC50.

### V5-MT

Five views fused to 768 dimensions: ChemBERTa-77M-MLM (bottom 4 layers frozen, weighted sum of the
top 4, CLS+mean+attention pooling), count/chirality ECFP4 2048 with 10 % feature dropout, a D-MPNN
graph encoder, MACCS 167, and 6 RO5 descriptors; modality-type embeddings, a 2-layer cross-modal
Transformer, a per-modality gate, concatenation to 768. A classifier head gives the decision; an
auxiliary head predicts pIC50 with loss weight 0.1. **pIC50 is a training target only — it is
excluded from the model inputs by construction and is not required at prediction time.**

---

## 2. Headline results

| | random OOF | random test | scaffold OOF | scaffold test |
|---|---|---|---|---|
| accuracy | 0.9381 | 0.9370 | 0.9153 | 0.9238 |
| balanced accuracy | 0.8992 | 0.9002 | 0.8577 | 0.8708 |
| precision / recall | 0.9548 / 0.9671 | 0.9574 / 0.9632 | 0.9364 / 0.9576 | 0.9390 / 0.9652 |
| specificity | 0.8313 | 0.8372 | 0.7577 | 0.7764 |
| F1 | 0.9609 | 0.9603 | 0.9469 | 0.9519 |
| ROC-AUC | 0.9660 | 0.9707 | 0.9483 | 0.9667 |
| AUPRC | 0.9881 | 0.9920 | 0.9830 | 0.9900 |
| MCC | 0.8127 | 0.8075 | 0.7392 | 0.7707 |
| Brier (calibrated) | 0.0517 | 0.0523 | 0.0676 | 0.0577 |
| errors | 289 | 52 | 403 | 56 |
| potency RMSE | 0.6215 | 0.5788 | 0.7601 | 0.6573 |

### Selective prediction (abstention cut fixed out-of-fold)

| coverage | random OOF | random test | scaffold OOF | scaffold test |
|---|---|---|---|---|
| 100 % | 0.9381 | 0.9370 | 0.9153 | 0.9238 |
| **95 %** | **0.9538** | **0.9537** | 0.9312 | **0.9498** |
| 90 % | 0.9650 | 0.9650 | 0.9468 | 0.9550 |
| 85 % | 0.9713 | 0.9727 | 0.9582 | 0.9694 |

---

## 3. Does each component earn its place? (out-of-fold)

| comparison | random dAUC (p) | scaffold dAUC (p) |
|---|---|---|
| vs **LightGBM alone** | +0.0040 (**p < 1e-4**) | +0.0033 (**p = 0.0014**) |
| vs **V5-MT alone** | +0.0033 (**p = 0.022**) | +0.0111 (**p < 1e-4**) |
| vs **V3 (published)** | +0.0036 (**p = 0.0066**) | +0.0057 (**p = 0.0051**) |
| vs **CheMeleon** (Burns 2025) | +0.0094 (**p = 0.0001**) | +0.0087 (**p = 0.0084**) |
| vs **Chemprop v2** (Heid 2024) | +0.0330 (**p < 1e-4**) | +0.0457 (**p < 1e-4**) |

McNemar on per-molecule errors, out-of-fold: vs V3 p = 0.0050 (random) and 0.0350 (scaffold);
vs CheMeleon p = 0.0150 (random); vs Chemprop p < 1e-4 (both).

Both components are necessary: the combination significantly outperforms **each of its own parts**
on both splits.

---

## 4. Against published methods (their implementations, our folds, 3 seeds)

| | ours | CheMeleon | Chemprop |
|---|---|---|---|
| random OOF AUC | **0.9660** | 0.9566 | 0.9331 |
| scaffold OOF AUC | **0.9483** | 0.9396 | 0.9027 |
| random OOF potency RMSE | **0.6215** | 0.6349 | 0.7174 |
| scaffold OOF potency RMSE | 0.7601 | **0.7583** | 0.8564 |
| random OOF cliff RMSE | **0.6743** | 0.6930 | 0.7610 |

Potency: clearly ahead of Chemprop (-0.096 RMSE, p < 1e-4 on both splits); ahead of CheMeleon on
the random split (-0.0135, p = 0.029) and level on scaffold (+0.0018, p = 0.77). None of
CheMeleon's test-set advantages are statistically significant (p = 0.07 - 0.66).

Classical baselines on the same folds, nested CV: SVM 0.9427 OOF AUC, Tanimoto k-NN 0.9240,
k-NN (cosine) 0.9099 on the random split.

---

## 5. Activity cliffs (MoleculeACE protocol)

Cliff compounds — potency differing by >= 1 log unit from a molecule >= 0.9 similar by ECFP
Tanimoto, Murcko-scaffold Tanimoto or SMILES Levenshtein similarity — are **57.7 % of this
dataset**.

| | error rate (all) | error rate (cliff) | penalty |
|---|---|---|---|
| proposed, random OOF | 0.0619 | 0.0783 | +0.0164 |
| proposed, scaffold OOF | 0.0847 | 0.1065 | +0.0218 |

Our cliff RMSE (0.6743 random OOF) is the best of any model tested, ahead of CheMeleon's 0.6930.

---

## 6. Ablations of the deep branch (seed 42, 5 folds, change in OOF MCC)

| ablation | random | scaffold |
|---|---|---|
| - ECFP view | **-0.0658** | **-0.0629** |
| - SMILES view | -0.0171 | -0.0049 |
| - descriptor view | -0.0164 | -0.0043 |
| - MACCS view | -0.0048 | -0.0089 |
| - D-MPNN graph view | -0.0048 | +0.0072 |
| - cross-modal Transformer (+ type emb, gate) | +0.0023 | -0.0050 |
| - gate only | -0.0002 | -0.0054 |
| - auxiliary pIC50 head | -0.0077 | -0.0057 |
| binary ECFP1024 instead of count/chiral 2048 | -0.0052 | -0.0038 |

ECFP dominates. The cross-modal Transformer and the gate are inert on both splits, reproducing the
same finding on the earlier MoLFormer stack. The D-MPNN branch contributes nothing measurable.

---

## 7. Data characterisation

* class balance 78.7 % active / 21.3 % inactive; pIC50 mean 6.665, sd 1.308
* every variable is non-normal (D'Agostino K2 and Shapiro-Wilk p < 1e-27), which is why
  non-parametric tests are used throughout
* descriptor collinearity: TPSA/NumHAcceptors r = 0.83, VIF 9.60 and 4.94. Descriptors are retained
  (dropping TPSA and MolWt costs 0.004-0.013 MCC); individual descriptor attributions are therefore
  not interpreted in isolation
* feature integrity verified: 0 mismatches against fresh RDKit recomputation for 300 molecules
  (ECFP, MACCS, descriptors), no non-finite values, no all-zero fingerprints
* **24 structures are duplicated with conflicting activity labels**; 8 appear on both sides of the
  random split. They act as unlearnable noise and depress test accuracy by about 0.7 %. The
  scaffold split has none.
* 14 molecules sit inside the excluded 5 < pIC50 < 6 band; 31 rows disagree with the label rule
* only 20 of 4,669 development molecules contradict their own label, so the label ceiling is 99.6 %

---

## 8. Claims that are supported

1. The proposed model significantly outperforms the published V3 model out-of-fold on both splits.
2. It significantly outperforms each of its own components — LightGBM alone and V5-MT alone — on
   both splits.
3. It significantly outperforms Chemprop v2 and CheMeleon on classification, out-of-fold, on both
   splits, with their authors' implementations run on our folds.
4. On potency it is ahead of CheMeleon on the random split and level on the scaffold split; both are
   far ahead of Chemprop.
5. 95 % accuracy is achievable through selective prediction: 95.4 % at 95 % coverage (random).
6. Classification accuracy is bounded by potency precision; the label ceiling is 99.6 %.
7. The attention-based fusion contributes nothing measurable; performance comes from multi-view
   complementarity plus potency-aware training.

## 9. Claims to avoid

* "95 % accuracy" without stating the 95 % coverage condition.
* "We beat the state of the art on potency" — it is a tie on the scaffold split and on both test sets.
* "Architectural gains transfer to novel scaffolds" — the retrieval module (V8-R2) significantly
  degrades there (-0.011 AUC, p = 0.002), which is why the proposed classifier uses V5-MT.
* Any claim resting on the held-out test sets alone; at n = 735-825 the standard error is +-0.84 %.

## 10. Limitations to disclose

1. **pIC50 as auxiliary supervision.** In V5-MT, pIC50 is a training target only, never a model
   input, and is not required at prediction time; all compared methods received the same
   supervision. Removing it costs 0.006-0.008 MCC. (The potency model V8-R2 does read the measured
   pIC50 of retrieved *training* analogues at inference — the same category as a k-NN using training
   labels; retrieval is fold-safe and verified by tests/test_v6_retrieval.py.)
2. **One scaffold partition**, with 3 training seeds.
3. **Ablations are single-seed**; differences below ~0.005 MCC are not individually meaningful.
4. **The 2 : 1 weight was prespecified, not tuned**; a sweep over w = 0.1-0.5 changes OOF MCC by
   less than 0.006.
5. Hyperparameters are inherited from the original V3 Optuna study; a full 40-trial replication for
   the new architecture produced nothing better under cross-validation.
6. 24 duplicated structures with conflicting labels are retained for continuity with the original
   dataset.

## 11. Where the numbers live

`results/final_model_{random,scaffold}.md` (the proposed model, end to end),
`results/external_{random,scaffold}.md` (published comparators),
`results/moleculeace_{random,scaffold}.md` (cliff protocol),
`results/data_stats.md` (dataset and feature audit),
`results/descriptor_stats.md` (descriptor significance),
`results/metrics_all.md` (every experiment), `ARCHITECTURE.md`.
