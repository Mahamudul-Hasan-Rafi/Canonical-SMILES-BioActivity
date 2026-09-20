# Architecture of the model family

Input is one molecule as a SMILES string. The network builds several independent views of that
molecule, fuses them into one 768-dimensional vector, and reads out either a class logit
(active / inactive) or a predicted pIC50. The design principle: a SMILES language model, a
fingerprint, a graph network and physicochemical descriptors fail on *different* molecules, so
combining them covers more chemistry than any one of them.

All models share the training machinery in `core.py`; they differ only through `RunCfg`
switches, and every configuration is content-hashed so results can never be silently mixed.

## 1. The views

| # | view | encoder | present in |
|---|---|---|---|
| 1 | SMILES | MoLFormer-XL (h=768) or ChemBERTa-77M-MLM (h=384), bottom third frozen, learned weighted sum of the top 4 layers, (CLS + mean + attention)/3 pooling | all |
| 2 | ECFP4 | 1024 binary (V3) or 2048 count + chirality, log1p, 10 % feature dropout (V4-A onward); 2048->512->768 | all |
| 3 | molecular graph | D-MPNN, hidden 300, depth 3, directed bond messages (atom 44 / bond 14 features); 300->768 | V4-C onward |
| 4 | MACCS keys | 167 -> 256 -> 768 | all |
| 5 | RO5 descriptors | 6 standardised properties (scaler fitted on the training fold only); 6 -> 64 -> 768 | all |

## 2. Fusion

Modality vectors become a short token sequence, receive modality-type embeddings, pass through a
2-layer cross-modal Transformer (4 heads, pre-norm), then a per-modality sigmoid gate, then
concatenation (n_mod x 768) projected back to 768.

**Ablation verdict: the cross-modal Transformer and the gate contribute nothing measurable.**
`concat_fusion`, which deletes the Transformer, scores at or above the full model. What does the
work is having several complementary views concatenated at all. The machinery is retained only so
that config hashes and bit-exact parity with the published V3 model remain valid.

## 3. Heads

* Classification head 768 -> 256 -> 128 -> 1, sigmoid.
* Potency head 768 -> 256 -> 1 predicting a standardised z; pIC50 = z * sigma + mu.
* Regression-first models derive the class from potency: `logit = 2 * (pIC50 - 5.5)`. The mapping
  is monotone, so thresholding the probability equals thresholding the predicted potency.

## 4. Neighbour-anchored module (V6 / V7 / V8)

Differentiable matched-molecular-pair reasoning. For a query, the 5 nearest **training-fold**
analogues are retrieved by Tanimoto similarity (leakage-checked for every fold of both splits by
`tests/test_v6_retrieval.py`). Each analogue j contributes

    anchor_j = z_j + delta(fp_query - fp_j, h_query)

i.e. start from a measured potency and predict only the change caused by the structural
difference. Attention pools the analogues into a context vector plus a pooled anchor, added back
into the fused representation; a 0.1-weighted auxiliary loss pushes each individual anchor toward
the true potency so the module cannot be ignored.

This module reduces activity-cliff errors (85 -> 72 out-of-fold) and is the only component that
degrades significantly under scaffold splitting (-0.011 AUC, p = 0.002), because scaffold
splitting removes the close analogues it depends on.

## 5. Training

| element | setting |
|---|---|
| loss | focal (alpha 0.75, gamma 2.0) + 0.05 label smoothing; Huber on pIC50 for regression-first |
| imbalance | WeightedRandomSampler (V3 / V5); removed in V8, where it distorts the potency distribution |
| optimiser | AdamW, LLRD 0.95 per layer, from-scratch modules at 10x base LR |
| LR / decay | 9.8917e-6 / 2.1248e-5 (tuned once, never re-tuned), warmup -> cosine |
| batch | 32, gradient accumulation 2 |
| augmentation | seeded SMILES enumeration |
| stopping | max 50 epochs, patience 12; best epoch by validation AUROC, or validation RMSE for V8 |
| evaluation | 5-fold CV -> 5 models; 3 seeds -> 15-model ensemble; threshold = OOF MCC-optimal |

## 6. The models

| model | composition | trainable | use when |
|---|---|---|---|
| V3 (published) | MoLFormer + 4 views + xattn fusion | 48.8 M | baseline; safe on novel scaffolds |
| V4 baseline | **identical to V3 with ChemBERTa in place of MoLFormer** (plus a 384->768 projection) | 21.4 M | backbone control only |
| V4-A / V4-C | + count/chiral 2048 FP / + D-MPNN graph branch | ~23 M | ablation steps |
| V5-MT | V4-C + 0.1-weighted auxiliary pIC50 head | 23.1 M | general-purpose best classifier |
| V8-R2 | V5-MT stack trained potency-first + neighbour-anchored delta | 24.1 M | analogue-rich chemistry; best potency (OOF RMSE 0.66) |
| blend | rank-average of V5-MT and V8-R2 | - | best overall on the random split |

## 7. What each component is worth

| component | verdict |
|---|---|
| ECFP branch | essential; largest single ablation drop |
| count + chiral 2048 FP vs binary 1024 | small consistent gain |
| D-MPNN graph branch | small gain, complementary to ECFP |
| descriptors, focal loss, LLRD, augmentation, multi-layer pooling | 0.3 - 1 % each |
| cross-modal Transformer + gate | no measurable contribution |
| auxiliary pIC50 head | +0.4 % accuracy, +0.02 MCC |
| neighbour-anchored delta | cuts cliff errors; hurts on novel scaffolds |
| backbone (MoLFormer vs ChemBERTa) | interchangeable: 55 scaffold-test errors each, ChemBERTa with 2.3x fewer parameters |

The gains come from multi-view complementarity and potency-aware training, not from the
attention-based fusion the original design foregrounds.

## 8. Reports

`results/final_table_random.md`, `results/final_table_scaffold.md` (headline ensembles),
`results/metrics_all.md` (every experiment, both splits), `results/reg_report.md` (potency),
`results/v8_analysis.md` (blend), `results/scaffold_attribution.md` (backbone vs modules),
`results/hard_case_compare.md` (error taxonomy), `REPORT.md` (full study).
