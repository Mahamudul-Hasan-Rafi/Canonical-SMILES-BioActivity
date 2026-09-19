"""
Performance enhancements that need no retraining, built from stored predictions:

  * seed ensemble   : average of the 3 V3 seeds (3 x 5 = 15 models)
  * DL + XGBoost    : equal-weight blend of the V3 seed ensemble and XGBoost(ECFP+MACCS+Desc)
  * DL ensemble     : V3 x 3 seeds + the 3 ChemBERTa backbones

Protocol as in the notebook: threshold = OOF MCC-optimal, applied to the test ensemble.
Significance: paired DeLong on OOF (4,669 molecules) and on test (825 molecules).
Also validates metrics.delong against a paired bootstrap. Writes results/enhancements.md.
"""
import os
import sys

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M
import report as R


def main():
    st = R.Store()
    rng = np.random.default_rng(0)
    runs = [st.cv(seed=s) for s in R.SEEDS]
    y_oof, y_te = runs[0]["oof_y"], runs[0]["test_y"]
    L = ["# Enhancements without retraining\n"]

    # DeLong implementation check against a paired bootstrap
    L += ["## DeLong implementation check (paired bootstrap, 2,000 resamples, OOF)\n",
          "| comparison | Δ AUC | DeLong SE | bootstrap SE | DeLong p | bootstrap p |", "|---|---|---|---|---|---|"]
    ref = runs[0]
    pos, neg = np.where(y_oof == 1)[0], np.where(y_oof == 0)[0]
    for v in ["no_ecfp", "smiles_only", "no_smiles"]:
        a, b = st.cv(seed=42, variant=v)["oof"], ref["oof"]
        d, p = M.delong(y_oof, a, b)
        ds = []
        for _ in range(2000):
            i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            ds.append(roc_auc_score(y_oof[i], a[i]) - roc_auc_score(y_oof[i], b[i]))
        ds = np.array(ds)
        pb = 2 * min((ds >= 0).mean(), (ds <= 0).mean())
        L.append(f"| {v} vs full | {d:+.4f} | {abs(d) / stats.norm.isf(p / 2):.4f} | {ds.std():.4f} | {p:.4f} | {pb:.4f} |")
    L.append("")

    seed_oof = np.mean([r["oof"] for r in runs], 0)
    seed_te = np.mean([r["ens"] for r in runs], 0)
    z = np.load(os.path.join(R.RES, "baselines", "random", "XGBoost__ECFP+MACCS+Desc.npz"))
    xo, xt = z["oof"], z["test_probs"].mean(0)
    cb = [st.cv(seed=42, backbone=b) for b in R.E.NEW_BACKBONES]
    dl_oof = np.mean([r["oof"] for r in runs + cb], 0)
    dl_te = np.mean([r["ens"] for r in runs + cb], 0)
    # V4 (ChemBERTa-MLM backbone) seed ensembles + XGBoost on the same upgraded fingerprints (V4-D)
    def seed_ens(**kw):
        rr = [st.cv(seed=s_, **kw) for s_ in R.SEEDS]
        rr = [r for r in rr if r is not None]
        return (np.mean([r["oof"] for r in rr], 0), np.mean([r["ens"] for r in rr], 0)) if rr else (None, None)
    a_oof, a_te = seed_ens(backbone="chemberta_mlm", variant="fp_upgrade")
    c_oof, c_te = seed_ens(backbone="chemberta_mlm", variant="graph_branch")
    z2 = np.load(os.path.join(R.RES, "baselines", "random", "XGBoost__ECFPc2048+MACCS+Desc.npz"))
    x2o, x2t = z2["oof"], z2["test_probs"].mean(0)
    v4 = []
    if c_oof is not None:
        v4 = [("V4-C seed ensemble (15 models)", c_oof, c_te),
              ("V4-A + V4-C seed ensemble (30 models)", 0.5 * (a_oof + c_oof), 0.5 * (a_te + c_te)),
              ("XGBoost on ECFPc2048+MACCS+Desc", x2o, x2t),
              ("V4-D: 0.5 x V4-C ensemble + 0.5 x XGBoost(ECFPc2048)", 0.5 * c_oof + 0.5 * x2o, 0.5 * c_te + 0.5 * x2t),
              ("V4-D': 0.5 x (V4-A+V4-C) ensemble + 0.5 x XGBoost(ECFPc2048)",
               0.25 * (a_oof + c_oof) + 0.5 * x2o, 0.25 * (a_te + c_te) + 0.5 * x2t)]
    cands = [("V3, single seed (42), 5 models", runs[0]["oof"], runs[0]["ens"]),
             ("V3 seed ensemble (3 seeds x 5 folds = 15 models)", seed_oof, seed_te),
             ("XGBoost on ECFP+MACCS+Desc", xo, xt),
             ("DL ensemble: V3 x3 seeds + 3 ChemBERTa", dl_oof, dl_te),
             ("Blend: 0.5 x V3 seed ensemble + 0.5 x XGBoost", 0.5 * seed_oof + 0.5 * xo, 0.5 * seed_te + 0.5 * xt),
             ("Blend: 0.5 x DL ensemble + 0.5 x XGBoost", 0.5 * dl_oof + 0.5 * xo, 0.5 * dl_te + 0.5 * xt)] + v4
    L += ["## Results (held-out test set, notebook threshold protocol)\n",
          "| model | OOF AUC | test AUC | test MCC | test BalAcc | test Acc | test Precision | test Recall | test F1 | threshold |",
          "|" + "---|" * 10]
    for name, o, t in cands:
        thr = M.mcc_thr(y_oof, o)
        m = M.thr_metrics(y_te, t, thr)
        L.append(f"| {name} | {roc_auc_score(y_oof, o):.4f} | {m['ROC-AUC']:.4f} | {m['MCC']:.4f} | {m['Balanced Acc']:.4f} | "
                 f"{m['Accuracy']:.4f} | {m['Precision']:.4f} | {m['Sensitivity']:.4f} | {m['F1']:.4f} | {thr:.3f} |")
    L += ["", "## Paired DeLong tests\n", "| comparison | Δ OOF AUC | p OOF | Δ test AUC | p test |", "|---|---|---|---|---|"]
    blend = (0.5 * seed_oof + 0.5 * xo, 0.5 * seed_te + 0.5 * xt)
    pairs = [("seed ensemble vs single-seed V3", (seed_oof, seed_te), (runs[0]["oof"], runs[0]["ens"])),
             ("V3+XGB blend vs V3 seed ensemble", blend, (seed_oof, seed_te)),
             ("V3+XGB blend vs XGBoost alone", blend, (xo, xt))]
    if c_oof is not None:
        v4d = (0.5 * c_oof + 0.5 * x2o, 0.5 * c_te + 0.5 * x2t)
        pairs += [("V4-C seed ensemble vs V3 (MoLFormer) seed ensemble", (c_oof, c_te), (seed_oof, seed_te)),
                  ("V4-D vs V3 (MoLFormer) seed ensemble", v4d, (seed_oof, seed_te)),
                  ("V4-D vs single-seed V3 (as in the notebook)", v4d, (runs[0]["oof"], runs[0]["ens"])),
                  ("V4-D vs XGBoost(ECFPc2048) alone", v4d, (x2o, x2t)),
                  ("V4-D vs previous best blend (V3 ens + XGB)", v4d, blend)]
    for name, a, b in pairs:
        do, po = M.delong(y_oof, a[0], b[0])
        dt, pt = M.delong(y_te, a[1], b[1])
        L.append(f"| {name} | {do:+.4f} | {po:.1e} | {dt:+.4f} | {pt:.3f} |")
    L += ["", "Blend weights were fixed at 0.5 a priori (not tuned). OOF predictions of the deep models are slightly "
          "optimistic (their folds also chose the early-stopping epoch); XGBoost's are not, which biases OOF "
          "comparisons in favour of the deep models, never against them.\n"]
    with open(os.path.join(R.RES, "enhancements.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
