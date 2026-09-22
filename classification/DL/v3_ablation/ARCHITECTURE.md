# Architecture

The final system is a **hybrid**: a gradient-boosting ensemble and a deep multi-view network,
combined. Both branches are described below, then the combination, then the evidence for why the
system is built this way.

Input is one molecule as a SMILES string. Output is a class decision (active / inactive) and a
predicted potency (pIC50).

---

## 1. Final configuration (what to present and deploy)

### Classification

```
rank-average, fixed weights 2 : 1
  +-- 2 x  LightGBM        on ECFP4 1024 binary + 6 RO5 descriptors
  |                        600 rounds, 63 leaves, lr 0.05; 5 fitted models
  +-- 1 x  V5-MT network   five views fused to 768; 3 seeds x 5 folds = 15 networks
-> per-fold Platt scaling -> probability
-> threshold fitted out-of-fold (MCC-optimal)
-> active / inactive          [+ optional abstention band]
```

20 fitted models in total. Ranking metrics use the raw rank-average; the calibrated probability is
used where a probability is required (Brier, deployment), because per-fold Platt calibrators are
not comparable across folds.

### Potency

```
plain average, weights 1 : 1
  +-- LightGBM regressor   on ECFP4 2048 count + chirality + MACCS + 6 descriptors
  +-- V8-R2 predicted pIC50
```

### Measured performance

| | random OOF | random test | scaffold OOF | scaffold test |
|---|---|---|---|---|
| accuracy | 0.9381 | 0.9370 | 0.9153 | 0.9238 |
| balanced accuracy | 0.8992 | 0.9002 | 0.8577 | 0.8708 |
| ROC-AUC | 0.9660 | 0.9707 | 0.9483 | 0.9667 |
| MCC | 0.8127 | 0.8075 | 0.7392 | 0.7707 |
| Brier (calibrated) | 0.0517 | 0.0523 | 0.0676 | 0.0577 |
| accuracy at 95 % coverage | 0.9538 | 0.9537 | 0.9312 | 0.9498 |
| potency RMSE | 0.6215 | 0.5788 | 0.7601 | 0.6573 |
| potency RMSE on cliffs | 0.6743 | 0.6352 | 0.8492 | 0.6543 |

### Both components are necessary (out-of-fold delta AUC)

| vs | random | scaffold |
|---|---|---|
| LightGBM alone | +0.0040 (p < 1e-4) | +0.0033 (p = 0.0014) |
| V5-MT alone | +0.0033 (p = 0.022) | +0.0111 (p < 1e-4) |
| V3 (published) | +0.0036 (p = 0.0066) | +0.0057 (p = 0.0051) |
| CheMeleon (Burns 2025) | +0.0094 (p = 0.0001) | +0.0087 (p = 0.0084) |
| Chemprop v2 (Heid 2024) | +0.0330 (p < 1e-4) | +0.0457 (p < 1e-4) |

The 2 : 1 weight was prespecified, not tuned: sweeping the deep weight from 0.1 to 0.5 changes
out-of-fold MCC by less than 0.006. Equal weights (1 : 1) would weaken the evidence that the deep
branch contributes, because the deep model is the weaker classifier and its value is diversity.

Activity cliffs (MoleculeACE definition) are 57.7 % of this dataset; the model's cliff error rate
is 0.0783 against 0.0619 overall (random, out-of-fold).

---

## 2. Classical branch

Features (the feature set from the ML notebook, 1030 columns): Morgan radius 2, 1024 bits, binary,
plus MolWt, MolLogP, TPSA, NumHDonors, NumHAcceptors, NumRotatableBonds.

| learner | configuration |
|---|---|
| RandomForest | 500 trees, class_weight balanced_subsample, min_samples_leaf 1 |
| ExtraTrees | 500 trees, class_weight balanced_subsample |
| XGBoost | 600 rounds, depth 6, lr 0.05, subsample 0.8, colsample 0.5, hist |
| LightGBM | 600 rounds, 63 leaves, lr 0.05, subsample 0.8 (freq 1), colsample 0.5 |
| CatBoost | 600 iterations, depth 6, lr 0.05, rsm 0.5 |

The proposed model uses **LightGBM alone** as the classical branch: it was the strongest single
learner out-of-fold on both splits (MCC 0.8103 random, 0.7272 scaffold), and pairing it with the
network matches or beats the five-learner voting ensemble with half the fitted models (20 vs 40).
Soft voting over all five (probability average; verified identical to sklearn's VotingClassifier)
remains available and is reported as an alternative. **No scale_pos_weight on the boosters** — that setting is what
produced the 0.896 artefact in the original notebook.

Stacking (logistic meta-learner over the same five, inner 3-fold CV) was tried and is *worse* than
plain voting (OOF MCC 0.7223 vs 0.7366 on the scaffold split) at four times the cost.

---

## 3. Deep branch

Five views of the molecule, fused to 768 dimensions.

| # | view | encoder |
|---|---|---|
| 1 | SMILES | ChemBERTa-77M-MLM (12 layers, h=384, bottom 4 frozen), learned weighted sum of the top 4 layers, (CLS + mean + attention)/3 pooling, 384 to 768 |
| 2 | ECFP4 | 2048 bits, counts + chirality, log1p, 10 % feature dropout; 2048-512-768 |
| 3 | graph | D-MPNN, hidden 300, depth 3, directed bond messages (atom 44 / bond 14 features); 300 to 768 |
| 4 | MACCS | 167-256-768 |
| 5 | descriptors | 6 RO5 properties, scaler fitted on the training fold only; 6-64-768 |

Fusion: modality-type embeddings, then a 2-layer cross-modal Transformer (4 heads, pre-norm), a
per-modality sigmoid gate, and concatenation of 5x768 projected to 768.

**Ablation verdict: the cross-modal Transformer and the gate contribute nothing measurable.** The
`concat_fusion` variant, which deletes the Transformer, scores at or above the full model. What
does the work is having five complementary views at all. The machinery is kept only so config
hashes and bit-exact parity with the published V3 model remain valid.

Neighbour-anchored module (V6 onward): the 5 nearest **training-fold** analogues are retrieved by
Tanimoto similarity (leakage-checked for every fold of both splits by tests/test_v6_retrieval.py).
Each analogue j contributes `anchor_j = z_j + delta(fp_query - fp_j, h_query)` — start from a
measured potency and predict only the change caused by the structural difference. Attention pools
the analogues into a context vector plus a pooled anchor, added back into the fused representation;
a 0.1-weighted auxiliary loss pushes each anchor toward the true potency.

Heads: classifier 768-256-128-1, and a potency head 768-256-1 whose output maps to a logit as
`2 * (pIC50 - 5.5)`.

Two deep variants are used:

* **V5-MT** — classification-trained, 0.1-weighted auxiliary potency head. Used on novel chemotypes.
* **V8-R2** — potency-first: Huber loss on pIC50, **no class sampler**, early stopping on validation
  RMSE, plus the neighbour-anchored delta. Used where analogues exist.

Training: AdamW, LLRD 0.95, from-scratch modules at 10x base LR, lr 9.8917e-6, weight decay
2.1248e-5, warmup then cosine, batch 32, gradient accumulation 2, seeded SMILES augmentation, max
50 epochs with patience 12. 3 seeds x 5 folds = 15 networks, 23-24 M trainable parameters each.

---

## 4. Why this configuration

**Feature assignment is empirical, not aesthetic.** Giving the classical branch the deep branch's
richer features (2048 count + chirality + MACCS) helps the regressor (hybrid potency RMSE 0.6267
to 0.6207, paired bootstrap p = 0.0098) but not the classifier, and on the scaffold split it makes
classification significantly *worse* (McNemar p = 0.019, dAUC -0.0025, p = 0.037). Each branch gets
the representation its task benefits from.

**The deep branch earns its place through diversity, not accuracy.** When both branches were given
identical features the hybrid gain disappeared (p = 0.89). LightGBM alone beats the deep regressor
on potency (0.6373 vs 0.6596 OOF) and the voting ensemble beats every individual deep model on OOF
accuracy. The hybrid is better than either because the two fail on different molecules.

### What each component is worth

| component | verdict |
|---|---|
| ECFP branch | essential; largest single ablation drop |
| count + chiral 2048 FP vs binary 1024 | helps regression, not classification |
| D-MPNN graph branch | small gain, complementary to ECFP |
| descriptors, focal loss, LLRD, augmentation, multi-layer pooling | 0.3 - 1 % each |
| cross-modal Transformer + gate | no measurable contribution |
| auxiliary pIC50 head | +0.4 % accuracy, +0.02 MCC |
| neighbour-anchored delta | cuts activity-cliff errors 85 to 72; hurts on novel scaffolds |
| backbone (MoLFormer vs ChemBERTa) | interchangeable: 55 scaffold-test errors each, ChemBERTa with 2.3x fewer parameters |
| EMA + test-time augmentation + multi-sample dropout (V9) | no gain; slightly worse on scaffold |
| voting vs stacking | voting wins, at a quarter of the cost |

**Accuracy is limited by potency precision, not by architecture.** Accuracy tracks potency RMSE
predictively (projected 0.9351, observed 0.9338 for V8-R2). Only 20 of 4,669 molecules contradict
their own label, so the ceiling is 99.6 % — but reaching 95 % at full coverage needs RMSE around
0.60, and four architectural rounds bought about 0.05 each. That gap is a data problem (HDAC
isoform multi-task, ChEMBL-wide potency pre-training), not a layer problem. The supported route to
95 % today is abstention: 95.4 % accuracy at 95 % coverage on the random split.

---

## 5. Alternatives built, measured and NOT adopted

Each was implemented and evaluated under the same protocol; each is kept in the repository so the
decision is reproducible, and none is part of the proposed model.

| alternative | result | why rejected |
|---|---|---|
| 9-member NNLS potency stack (`stack_potency.py`) | random OOF RMSE 0.6215 -> 0.6148, scaffold 0.7601 -> 0.7580 | 0.0067 RMSE for 85 fitted models instead of 20, plus fold-wise fitted weights that add a leakage surface to defend. The 2-member hybrid already beats CheMeleon on random OOF (p = 0.029). |
| V10, auxiliary RDKit-descriptor head (`reg_delta_desc`) | random OOF RMSE 0.6598 vs 0.6596 baseline | no effect. CheMeleon's descriptor signal needs its 1M-molecule pretraining corpus; as an auxiliary task on 4.7k molecules it does nothing. |
| MoLFormer-encoder potency model | 0.6583 alone, largest scaffold stack weight (0.241) | improves nothing outside the stack, and the stack is not adopted. |
| RDKit-204 / Mordred-1325 descriptor blocks as booster features | LGBMReg random OOF 0.6373 -> 0.6585 -> 0.7045 | raw descriptors dilute the boosters. Confirms their value lies in pretraining, not as input features. |
| Full Optuna HPO replication (40 trials per branch) | no configuration beat the inherited hyperparameters under cross-validation | the notebook's proxy objective (50 % data, 25 epochs) does not predict full-regime performance. |
| V9: weight EMA + test-time augmentation + multi-sample dropout | random OOF AUC -0.0012; scaffold -0.0033 (p = 0.030) | no gain, slightly worse on scaffold. |
| Uni-Mol, MoleculeACE packages | not run | their installs force numpy 2 / rdkit-pypi 2022.9.5, which would invalidate every cached feature. MoleculeACE's model suite is already covered by our baselines; its cliff protocol is implemented directly instead. |

---

## 6. Caveats to state in any write-up

1. Hybrid membership was selected greedily on out-of-fold predictions — mild on 4,759 molecules,
   but it is a selection step.
2. On the scaffold test set the advantage over V3 is not significant (p = 0.63); claim improved
   ranking on unseen scaffolds, not improved accuracy.
3. LightGBM alone on 1030 features is 0.5 - 1.5 points behind and trains in 10 seconds against
   about 25 GPU-minutes. For a production triage tool that is the better trade.

---

## 7. Reports

`results/external_{random,scaffold}.md` (against published methods),
`results/moleculeace_{random,scaffold}.md` (activity-cliff protocol),
`results/stack_potency_{random,scaffold}.md` (the stack alternative),
`results/position_{random,scaffold}.md` (standing on both axes), `results/final_table_*.md`
(headline ensembles), `results/metrics_all.md` (every experiment, both splits),
`results/reg_report.md` (potency), `results/hybrid_*.md` (threshold and hybrid selection),
`results/selective_*.md` (accuracy vs coverage), `results/scaffold_attribution.md` (backbone vs
modules), `results/ml_scaffold_ECFPDesc.md` (classical models), `results/hard_case_compare.md`
(error taxonomy), `REPORT.md` (full study).
