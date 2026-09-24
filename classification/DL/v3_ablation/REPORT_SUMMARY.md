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

* 2 x **LightGBM classifier** on ECFP4 1024 binary + 6 RO5 descriptors, refitted per fold: 5 models
* 1 x **V11c network**, 3 seeds x 5 folds: 15 networks

Score converted to a probability by per-fold Platt scaling; decision threshold fitted out-of-fold
(MCC-optimal). Ranking metrics use the raw rank-average, since per-fold calibrators are not
comparable across folds. 20 fitted models in total.

**Potency** — **LightGBM regressor** on ECFP4 2048 count/chirality + MACCS + 6 descriptors, with
V11c's own potency head reported as a secondary deep estimate.

### V11c

Five views fused to 768 dimensions: ChemBERTa-77M-MLM (bottom 4 layers frozen, weighted sum of the
top 4, CLS+mean+attention pooling), count/chirality ECFP4 2048 with 10 % feature dropout, a D-MPNN
graph encoder, MACCS 167, and 6 RO5 descriptors; modality-type embeddings, a 2-layer cross-modal
Transformer, a per-modality gate, concatenation to 768. Two heads - classification and pIC50 -
trained at **equal weight**, with no class sampler and **no retrieval module**.

pIC50 is a training target only: it is excluded from the model inputs by construction and is not
required at prediction time. Nothing in the proposed system reads another molecule's measured
potency at inference.

---

## 2. Headline results

| | random OOF | random test | scaffold OOF | scaffold test |
|---|---|---|---|---|
| accuracy | 0.9375 | 0.9370 | 0.9149 | 0.9252 |
| balanced accuracy | 0.9046 | 0.9045 | 0.8578 | 0.8716 |
| precision / recall | 0.9588 / 0.9619 | 0.9602 / 0.9602 | 0.9366 / 0.9568 | 0.9391 / 0.9669 |
| specificity | 0.8474 | 0.8488 | 0.7587 | 0.7764 |
| F1 | 0.9603 | 0.9602 | 0.9466 | 0.9528 |
| ROC-AUC | 0.9666 | 0.9722 | 0.9481 | 0.9687 |
| AUPRC | 0.9883 | 0.9925 | 0.9828 | 0.9906 |
| MCC | 0.8129 | 0.8090 | 0.7382 | 0.7746 |
| Brier (calibrated) | 0.0512 | 0.0511 | 0.0677 | 0.0565 |
| errors | 292 | 52 | 405 | 55 |
| potency RMSE (LightGBM regressor) | 0.6373 | 0.5934 | 0.7880 | 0.6479 |
| potency RMSE (V11c head) | 0.7429 | 0.6800 | 0.8606 | 0.7846 |

### Selective prediction (abstention cut fixed out-of-fold)

| coverage | random OOF | random test | scaffold OOF | scaffold test |
|---|---|---|---|---|
| 100 % | 0.9375 | 0.9370 | 0.9149 | 0.9252 |
| **95 %** | **0.9549** | **0.9551** | 0.9317 | **0.9511** |
| 90 % | 0.9643 | 0.9636 | 0.9475 | 0.9577 |

---

## 3. Does each component earn its place? (out-of-fold)

| comparison | random dAUC (p) | scaffold dAUC (p) |
|---|---|---|
| vs **LightGBM alone** | +0.0046 (**p < 1e-4**) | +0.0030 (**p = 0.0050**) |
| vs **V11c alone** | +0.0040 (**p = 0.0123**) | +0.0135 (**p < 1e-4**) |
| vs **V3 (published baseline)** | +0.0042 (**p = 0.0023**) | +0.0055 (**p = 0.0125**) |

McNemar on per-molecule errors, out-of-fold, vs V3: p = 0.0177 (random), p = 0.0570 (scaffold).

The combination significantly outperforms **each of its own parts** on both splits: neither the
gradient-boosting model nor the network is sufficient alone.

---

## 4. Against established methods

### 4.1 Chemprop v2 (Heid et al., JCIM 2024) - run with the authors' implementation on our folds

| | ours | Chemprop v2 |
|---|---|---|
| random OOF AUC | **0.9666** | 0.9331 |
| random OOF MCC | **0.8129** | 0.7412 |
| scaffold OOF AUC | **0.9481** | 0.9027 |
| scaffold OOF MCC | **0.7382** | 0.6668 |
| random OOF potency RMSE | **0.6373** | 0.7174 |
| scaffold OOF potency RMSE | **0.7880** | 0.8564 |

Classification: dAUC **+0.0335 (p < 1e-4)** random and **+0.0454 (p < 1e-4)** scaffold.
Potency: **-0.08 to -0.09 RMSE, p < 1e-4** on both splits. Chemprop, the field-standard
directed-message-passing baseline, is not competitive on this dataset.

### 4.2 Published HDAC models (different datasets - context, not a controlled comparison)

| study | task | reported | ours |
|---|---|---|---|
| HDAC1, XGBoost + ECFP4, 7,313 compounds (PubMed 35737257) | classification | accuracy 0.8808, MCC 0.76 | 0.9375 / 0.8129 (random OOF) |
| HDAC6, ensemble incl. XGBoost, 1,701 compounds | classification | accuracy 0.9069, AUC 0.9493 | 0.9375 / 0.9666 |
| HDAC family, transformer-pretrained features (2024) | classification | accuracy 0.88 - 0.91 | 0.9375 |

These use different curations, activity thresholds and splitting schemes (the HDAC1 study used a
Kohonen self-organising-map split, which is harder than a random split), so the comparison bounds
the published range rather than ranking methods. Our scaffold-split result (0.9149 accuracy,
0 % scaffold overlap with training) is the conservative number to quote against them.

### 4.3 Classical baselines on our folds (nested CV where tuned)

| model | random OOF AUC | random OOF MCC | scaffold OOF AUC |
|---|---|---|---|
| Voting (RF+ET+XGB+LGBM+Cat) | 0.9648 | 0.8123 | 0.9479 |
| LightGBM | 0.9621 | 0.8103 | 0.9451 |
| XGBoost | 0.9601 | 0.7965 | 0.9413 |
| Random Forest | 0.9627 | 0.7946 | 0.9445 |
| SVM (RBF, nested CV) | 0.9427 | 0.7464 | 0.9103 |
| k-NN (Tanimoto) | 0.9240 | 0.7287 | 0.8923 |

Gradient boosting on fingerprints is a strong baseline on this task - stronger than several deep
models - which is why the proposed system keeps it as its main component.

### 4.4 Preprint comparator: CheMeleon (Burns & Green, 2025, arXiv)

Not counted as established state of the art, but run under the same protocol and reported for
completeness, with an important qualification.

| | ours | CheMeleon classifier | CheMeleon regressor, thresholded |
|---|---|---|---|
| random OOF AUC | **0.9666** | 0.9566 | 0.9592 |
| random OOF MCC | 0.8129 | 0.7925 | 0.8118 |
| scaffold OOF AUC | **0.9481** | 0.9396 | 0.9382 |
| scaffold OOF MCC | 0.7382 | 0.7293 | **0.7415** |
| scaffold test errors | 55 | 65 | **44** |

**We are significantly better on ranking** against both configurations (dAUC +0.0074, p = 0.012
random; +0.0099, p = 0.0038 scaffold). **On thresholded metrics we are level** with its regression
model (McNemar p = 0.83 random, p = 1.00 scaffold) and it is better on the scaffold test set
(44 errors vs 55, p = 0.12, not significant). On potency it is level on random (0.6349 vs our
0.6373) and ahead on scaffold (0.7583 vs 0.7880).

The claim to make is therefore about **ranking**, not about accuracy, whenever this model is in
scope.

---

## 5. Activity cliffs (MoleculeACE protocol)

Cliff compounds — potency differing by >= 1 log unit from a molecule >= 0.9 similar by ECFP
Tanimoto, Murcko-scaffold Tanimoto or SMILES Levenshtein similarity — are **57.7 % of this
dataset**.

| | error rate (all) | error rate (cliff) | penalty |
|---|---|---|---|
| proposed, random OOF | 0.0625 | 0.0794 | +0.0169 |
| proposed, scaffold OOF | 0.0851 | 0.1068 | +0.0217 |

On cliff RMSE the LightGBM regressor gives 0.6801 (random OOF) against CheMeleon's 0.6930; the
V8-R2 alternative would give 0.6743.

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
2. It significantly outperforms each of its own components — LightGBM alone and V11c alone — on
   both splits.
3. It significantly outperforms Chemprop v2 (JCIM 2024) on classification and potency, out-of-fold,
   on both splits, using the authors' own implementation on our folds.
4. Against the CheMeleon preprint it is significantly better on ranking (AUC) on both splits but
   only level on thresholded metrics, and behind on scaffold potency.
5. 95 % accuracy is achievable through selective prediction: 95.4 % at 95 % coverage (random).
6. Classification accuracy is bounded by potency precision; the label ceiling is 99.6 %.
7. The attention-based fusion contributes nothing measurable; performance comes from multi-view
   complementarity plus potency-aware training.

## 9. Claims to avoid

* "95 % accuracy" without stating the 95 % coverage condition.
* "We beat the state of the art on potency" — it is a tie on the scaffold split and on both test sets.
* "Architectural gains transfer to novel scaffolds" — the retrieval module (V8-R2) significantly
  degrades there (-0.011 AUC, p = 0.002), which is one reason the proposed model omits it.
* Any claim resting on the held-out test sets alone; at n = 735-825 the standard error is +-0.84 %.

## 10. Limitations to disclose

1. **pIC50 as training supervision.** In V11c, pIC50 is a training target only - never a model
   input, and not required at prediction time; all compared methods received the same supervision.
   The proposed system contains no retrieval module, so nothing reads another molecule's measured
   potency at inference. (The documented V8-R2 alternative does; it is fold-safe and verified by
   tests/test_v6_retrieval.py, but it is not part of the proposed model.)
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
