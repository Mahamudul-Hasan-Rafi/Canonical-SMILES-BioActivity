# HDAC1 bioactivity model — consolidated results

Dataset: 5,494 ChEMBL HDAC1 (CHEMBL325) molecules with IC50; active if pIC50 >= 6, inactive if <= 5.
Two evaluation splits, both with 5-fold cross-validation on the development set and a held-out test
set used only for final reporting.

| split | development | test | test molecules sharing a Murcko scaffold with training |
|---|---|---|---|
| random (as in the original notebook) | 4,669 | 825 | **68.1 %** |
| scaffold (StratifiedGroupKFold on Murcko scaffolds) | 4,759 | 735 | **0 %** |

Median maximum Tanimoto of a test molecule to the training set: 0.806 (random), 0.731 (scaffold);
fraction with a >= 0.9 neighbour: 20.5 % vs 6.5 %.

---

## 1. Final model

**Classification** — rank-average, weights 2 : 1

* 2 x soft-voting ensemble: RF + ExtraTrees + XGBoost + LightGBM + CatBoost on ECFP4 1024 binary
  + 6 RO5 descriptors (25 fitted models)
* 1 x deep network: V8-R2 (random split) or V5-MT (scaffold split), 3 seeds x 5 folds = 15 networks

**Potency** — plain average, 1 : 1 of LightGBM regressor (ECFP4 2048 count/chirality + MACCS +
6 descriptors) and the V8-R2 predicted pIC50.

Threshold fixed out-of-fold (MCC-optimal, the notebook rule). Optional abstention band.

### Deep network

Five views fused to 768 dimensions: ChemBERTa-77M-MLM (bottom 4 layers frozen, weighted sum of the
top 4, CLS+mean+attention pooling), count/chirality ECFP4 2048 with 10 % feature dropout, a D-MPNN
graph encoder, MACCS 167, and 6 RO5 descriptors; modality-type embeddings, a 2-layer cross-modal
Transformer, a per-modality gate, concatenation to 768. V8-R2 adds the neighbour-anchored module
(5 nearest training-fold analogues, anchor_j = z_j + delta(fp_q - fp_j, h_q), attention-pooled) and
is trained potency-first (Huber on pIC50, no class sampler, early stopping on validation RMSE).

---

## 2. Headline results

| | random OOF | random test | scaffold OOF | scaffold test |
|---|---|---|---|---|
| accuracy | 0.9417 | 0.9418 | 0.9128 | 0.9293 |
| MCC | 0.8240 | 0.8230 | 0.7372 | 0.7915 |
| ROC-AUC | 0.9680 | 0.9727 | 0.9499 | 0.9659 |
| potency RMSE | 0.6215 | 0.5788 | 0.7601 | 0.6573 |
| accuracy at 95 % coverage | 0.9560 | 0.9540 | 0.9334 | 0.9443 |

Activity cliffs (MoleculeACE definition: >= 1 log unit potency difference to a molecule >= 0.9
similar by ECFP Tanimoto, Murcko-scaffold Tanimoto or SMILES Levenshtein): **57.7 % of the dataset**.
Cliff RMSE 0.6743 (random OOF) and 0.8492 (scaffold OOF); cliff classification error 0.0805 and
0.1076, the lowest of any model tested.

---

## 3. Against published methods, run on our data, folds and test set

Chemprop v2 (Heid et al., JCIM 2024) and CheMeleon (Burns & Green, 2025) executed with their
authors' own implementations, 3 seeds, identical folds.

| | ours | CheMeleon | Chemprop |
|---|---|---|---|
| random OOF AUC | **0.9680** | 0.9566 | 0.9331 |
| scaffold OOF AUC | **0.9499** | 0.9396 | 0.9027 |
| random OOF potency RMSE | **0.6215** | 0.6349 | 0.7174 |
| scaffold OOF potency RMSE | 0.7601 | **0.7583** | 0.8564 |
| random OOF cliff RMSE | **0.6743** | 0.6930 | 0.7610 |

Classification, out-of-fold: vs CheMeleon dAUC +0.0114 (p < 1e-4) random and +0.0103 (p = 0.001)
scaffold; vs Chemprop +0.0350 and +0.0473 (both p < 1e-4).
Potency, out-of-fold: vs Chemprop -0.096 RMSE (p < 1e-4, both splits); vs CheMeleon -0.0135
(p = 0.029) on random and +0.0018 (p = 0.77, a tie) on scaffold. None of CheMeleon's test-set
advantages are significant (p = 0.07 - 0.66).

Against the published V3 baseline: random OOF McNemar p = 0.0001, dAUC +0.0056 (p = 0.0010);
scaffold OOF p = 0.0020, dAUC +0.0065 (p = 0.0007); scaffold test not significant (p = 0.63).

Classical baselines on the same folds (nested CV for SVM/k-NN): SVM 0.9427 OOF AUC, Tanimoto k-NN
0.9240, k-NN (cosine) 0.9099 on random — all clearly behind the boosting ensemble (0.9648).

---

## 4. Ablations of the deep branch (V5-MT base, seed 42, 5 folds)

Change in out-of-fold MCC relative to the full model.

| ablation | random | scaffold |
|---|---|---|
| - ECFP view | **-0.0658** | **-0.0629** |
| - SMILES view | -0.0171 | -0.0049 |
| - descriptor view | -0.0164 | -0.0043 |
| - MACCS view | -0.0048 | -0.0089 |
| - D-MPNN graph view | -0.0048 | **+0.0072** |
| - cross-modal Transformer (+ type emb, gate) | **+0.0023** | -0.0050 |
| - gate only | -0.0002 | -0.0054 |
| - auxiliary pIC50 head | -0.0077 | -0.0057 |
| binary ECFP1024 instead of count/chiral 2048 | -0.0052 | -0.0038 |

Read: ECFP is the architecture. The cross-modal Transformer and gate are inert - removing them is
within noise on both splits, reproducing the same finding on the earlier MoLFormer stack. The
D-MPNN branch contributes nothing and is slightly harmful on scaffold. Single seed, so differences
below ~0.005 MCC are not individually meaningful.

---

## 5. Claims that are supported

1. The hybrid significantly outperforms the published V3 model out-of-fold on both splits.
2. It significantly outperforms Chemprop v2 and CheMeleon on classification, out-of-fold, on both
   splits.
3. On potency it is ahead of CheMeleon on the random split and level on the scaffold split; both
   are far ahead of Chemprop.
4. 95 % accuracy is achievable through selective prediction: 95.4 % at 95 % coverage (random).
5. Classification accuracy on this task is bounded by potency precision; only 20 of 4,669 molecules
   contradict their own label, so the label ceiling is 99.6 % and the limitation is model precision.
6. The attention-based fusion contributes nothing measurable; performance comes from multi-view
   complementarity plus potency-aware training.

## 6. Claims to avoid

* "95 % accuracy" without stating the 95 % coverage condition.
* "We beat the state of the art on potency" - it is a tie on the scaffold split and on both test
  sets.
* "Architectural gains transfer to novel scaffolds" - the retrieval module significantly degrades
  there (-0.011 AUC, p = 0.002).
* Any claim resting on the held-out test sets alone; at n = 735-825 the standard error is +-0.84 %.

## 7. Limitations to disclose

1. **pIC50 as auxiliary supervision.** pIC50 is a training target, never a model input, and is not
   required at prediction time. All compared methods received the same supervision. The V8-R2
   neighbour module additionally reads the measured pIC50 and label of 5 retrieved *training*
   analogues at inference - the same category as a k-NN using training labels; retrieval is
   fold-safe and verified by tests/test_v6_retrieval.py. Removing pIC50 from the classification
   path entirely costs 0.36 % OOF accuracy and 7 test errors.
2. **One scaffold partition.** Scaffold results use a single fixed StratifiedGroupKFold split with
   3 training seeds; scaffold-split estimates vary with which families land in test.
3. **Hybrid membership** was selected greedily on out-of-fold predictions.
4. **Ablations are single-seed.**
5. Hyperparameters are inherited from the original V3 Optuna study; a full 40-trial replication for
   the new architecture produced nothing better under cross-validation.

## 8. Where the numbers live

`results/final_table_{random,scaffold}.md`, `results/external_{random,scaffold}.md`,
`results/moleculeace_{random,scaffold}.md`, `results/selective_{random,scaffold}.md`,
`results/metrics_all.md`, `results/position_{random,scaffold}.md`,
`results/scaffold_attribution.md`, `results/hard_case_compare.md`, `ARCHITECTURE.md`.
