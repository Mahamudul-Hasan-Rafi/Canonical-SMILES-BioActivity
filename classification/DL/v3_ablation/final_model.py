"""
The proposed model, evaluated end to end.

Classification : rank-average of LightGBM (ECFP4 1024 + 6 descriptors) and the V5-MT network,
                 fixed weights 2 : 1, threshold fitted out-of-fold (MCC-optimal).
Potency        : 1 : 1 average of the LightGBM regressor and the V8-R2 predicted pIC50, reported
                 alongside the LightGBM regressor alone so the deep contribution is visible.

Everything is scored on the same folds and the same held-out test set as every other model in the
study, including the published comparators. Writes results/final_model_<split>.md
"""
import argparse
import os
import sys

import numpy as np
from scipy.stats import binomtest, rankdata
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
import data
import metrics as M
import moleculeace_cliffs as MC
import report as R

KEYS = ["Accuracy", "Balanced Acc", "Precision", "Sensitivity", "Specificity", "F1",
        "ROC-AUC", "AUPRC", "MCC", "Brier"]
NAME = {"Sensitivity": "Recall", "Specificity": "Specificity"}
W_DEEP = 1.0 / 3.0            # 2 : 1, classical : deep


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="random", choices=["random", "scaffold"])
    args = ap.parse_args()
    sp = args.split
    st = R.Store()
    dev, te = st.splits[sp]["dev"], st.splits[sp]["test"]
    y, yt = st.y[dev], st.y[te]
    smiles = list(data.get_features(st.df)["smiles"])
    pic = data.get_pic50(smiles)
    p_dev, p_te = pic[dev], pic[te]
    cliff = MC.cliff_flags(smiles, pic)
    c_dev, c_te = cliff[dev], cliff[te]

    def npz(nm):
        p = os.path.join(R.RES, "baselines", sp, nm + ".npz")
        return np.load(p) if os.path.exists(p) else None

    def deep(variant, backbone="chemberta_mlm"):
        rr = [r for r in (st.cv(split=sp, seed=s, backbone=backbone, variant=variant) for s in R.SEEDS) if r]
        return (np.mean([r["oof"] for r in rr], 0), np.mean([r["ens"] for r in rr], 0)) if rr else None

    ref_name = "PROPOSED: LightGBM + V11c (2:1)"
    z = npz("LightGBM__ECFP+Desc")
    LG = (z["oof"], z["test_probs"].mean(0))
    V5 = deep("mt_w10_ns")            # V11c: balanced multi-task, no retrieval
    FINAL = ((1 - W_DEEP) * rankdata(LG[0]) / len(y) + W_DEEP * rankdata(V5[0]) / len(y),
             (1 - W_DEEP) * rankdata(LG[1]) / len(yt) + W_DEEP * rankdata(V5[1]) / len(yt))

    # the rank-average is a uniform score, not a probability: calibrate it to probabilities with
    # per-fold Platt scaling fitted on the OTHER folds, so no molecule calibrates its own score
    from sklearn.linear_model import LogisticRegression
    fold_of = np.full(len(dev), -1)
    for f, (_, f_vl) in enumerate(st.splits[sp]["folds"]):
        fold_of[f_vl] = f
    cal_o = np.zeros(len(dev))
    cal_t = []
    for f in range(5):
        tr = fold_of != f
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(FINAL[0][tr].reshape(-1, 1), y[tr])
        cal_o[~tr] = lr.predict_proba(FINAL[0][~tr].reshape(-1, 1))[:, 1]
        cal_t.append(lr.predict_proba(FINAL[1].reshape(-1, 1))[:, 1])
    CAL = (cal_o, np.mean(cal_t, 0))
    # Ranking and the decision use the raw rank-average: each fold has its own Platt calibrator, so
    # mixing calibrated scores across folds would perturb the global out-of-fold ranking. The
    # calibrated probabilities are used only where a probability is required (Brier, deployment).

    models = {ref_name: FINAL,
              "LightGBM alone": LG,
              "V11c alone": V5,
              "V5-MT alone (aux 0.1)": deep("graph_mt"),
              "V3 (published)": deep("full", backbone="molformer")}
    for nm, f in (("CheMeleon [Burns 2025]", "CheMeleon__graph"), ("Chemprop [Heid 2024]", "Chemprop__graph")):
        zz = npz(f)
        if zz is not None:
            models[nm] = (zz["oof"], zz["test_probs"].mean(0))
    models = {k: v for k, v in models.items() if v is not None}
    thr = {k: M.mcc_thr(y, v[0]) for k, v in models.items()}

    L = [f"# Proposed model — {sp} split\n",
         "**Classification** rank-average of LightGBM (ECFP4 1024 binary + 6 RO5 descriptors) and the "
         "V11c multi-view network (balanced multi-task, no retrieval), fixed weights 2 : 1, "
         "converted to probabilities by per-fold Platt "
         "scaling. Threshold fitted out-of-fold (MCC-optimal). "
         f"20 fitted models. Development {len(dev):,} molecules, held-out test {len(te)}.\n",
         "## Classification\n",
         "| model | " + " | ".join(NAME.get(k, k) for k in KEYS) + " | errors |",
         "|" + "---|" * (2 + len(KEYS))]
    for proto, pick, yy in (("out-of-fold", 0, y), ("held-out test", 1, yt)):
        L.append(f"| **{proto}** |" + " |" * (len(KEYS) + 1))
        for k, v in models.items():
            m = dict(M.thr_metrics(yy, v[pick], thr[k]))
            if k == ref_name:                     # report the calibrated Brier for the proposed model
                from sklearn.metrics import brier_score_loss
                m["Brier"] = brier_score_loss(yy, CAL[pick])
            err = int(((v[pick] > thr[k]).astype(int) != yy).sum())
            L.append(f"| {k} | " + " | ".join(f"{m[x]:.4f}" for x in KEYS) + f" | {err} |")

    ref = ref_name
    L += ["", "## Does each component earn its place?\n",
          "| comparison | set | McNemar b / c | p | delta AUC | p |", "|---|---|---|---|---|---|"]
    for other in ("LightGBM alone", "V11c alone", "V5-MT alone (aux 0.1)", "V3 (published)",
                  "CheMeleon [Burns 2025]", "Chemprop [Heid 2024]"):
        if other not in models:
            continue
        for lab, pick, yy in (("OOF", 0, y), ("test", 1, yt)):
            ea = (models[ref][pick] > thr[ref]).astype(int) != yy
            eb = (models[other][pick] > thr[other]).astype(int) != yy
            n01, n10 = int((ea & ~eb).sum()), int((~ea & eb).sum())
            p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
            d, pd_ = M.delong(yy, models[ref][pick], models[other][pick])
            L.append(f"| proposed vs {other} | {lab} | {n01} / {n10} | {p:.4f} | {d:+.4f} | {pd_:.4f} |")
    L.append("\nb = molecules only the proposed model gets wrong, c = only the other. "
             "Positive delta AUC = proposed is better.\n")

    # ---------- selective prediction ----------
    o, t = models[ref]
    do, dt = np.abs(o - thr[ref]), np.abs(t - thr[ref])
    L += ["## Selective prediction (abstention cut fixed out-of-fold)\n",
          "| coverage | OOF accuracy | test accuracy | test n |", "|---|---|---|---|"]
    for cov in (1.0, 0.95, 0.9, 0.85, 0.8):
        cut = np.quantile(do, 1 - cov)
        mo, mt = do >= cut, dt >= cut
        L.append(f"| {cov:.0%} | {accuracy_score(y[mo], (o[mo] > thr[ref]).astype(int)):.4f} | "
                 f"{accuracy_score(yt[mt], (t[mt] > thr[ref]).astype(int)):.4f} | {int(mt.sum())} |")

    # ---------- potency ----------
    pot = {}
    zz = npz("LGBMReg__ECFPc2048+MACCS+Desc")
    if zz is not None:
        pot["LightGBM regressor"] = (zz["oof_pic50"], zz["test_pic50"].mean(0))
    import experiments as _E
    try:
        oofp = np.zeros(len(dev)); tests = []
        pos = {int(v): k for k, v in enumerate(dev)}
        for sd in R.SEEDS:
            t = []
            for f in range(5):
                zz = np.load(os.path.join(R.RES, "store",
                                          _E.job_id(_E.job("cv", sp, "chemberta_mlm", "tuned",
                                                           "mt_w10_ns", sd, f)) + ".npz"))
                oofp[[pos[int(i)] for i in zz["val_idx"]]] = zz["val_pic50"]
                t.append(zz["test_pic50"])
            tests.append(np.mean(t, 0))
        pot["V11c potency head"] = (oofp, np.mean(tests, 0))
    except Exception:
        pass
    r8 = deep("reg_first_delta")
    if r8:
        pot["V8-R2 (deep, alternative)"] = (core.probs_to_pic50(r8[0]), core.probs_to_pic50(r8[1]))
    if "LightGBM regressor" in pot and "V8-R2 (deep, alternative)" in pot:
        pot["LightGBM-reg + V8-R2 (alternative)"] = tuple(
            np.mean([pot["LightGBM regressor"][i], pot["V8-R2 (deep, alternative)"][i]], 0) for i in (0, 1))
    for nm, f in (("CheMeleon-reg [Burns 2025]", "CheMeleon-reg__graph"),
                  ("Chemprop-reg [Heid 2024]", "Chemprop-reg__graph")):
        zz = npz(f)
        if zz is not None:
            pot[nm] = (zz["oof_pic50"], zz["test_pic50"].mean(0))
    if pot:
        L += ["", "## Potency (pIC50, log units)\n",
              "| model | OOF RMSE | OOF RMSE_cliff | OOF MAE | test RMSE | test RMSE_cliff |",
              "|---|---|---|---|---|---|"]
        for k in sorted(pot, key=lambda k: rmse(pot[k][0] - p_dev)):
            e, et = pot[k][0] - p_dev, pot[k][1] - p_te
            L.append(f"| {k} | {rmse(e):.4f} | {rmse(e[c_dev]):.4f} | {np.abs(e).mean():.4f} | "
                     f"{rmse(et):.4f} | {rmse(et[c_te]):.4f} |")

    # ---------- cliff behaviour ----------
    L += ["", "## Activity cliffs (MoleculeACE definition)\n",
          f"{cliff.mean():.1%} of the dataset are cliff compounds "
          f"({int(c_dev.sum()):,} of {len(dev):,} development, {int(c_te.sum())} of {len(te)} test).\n",
          "| model | OOF error (all) | OOF error (cliff) | penalty |", "|---|---|---|---|"]
    for k, v in models.items():
        err = (v[0] > thr[k]).astype(int) != y
        L.append(f"| {k} | {err.mean():.4f} | {err[c_dev].mean():.4f} | {err[c_dev].mean() - err.mean():+.4f} |")

    out = os.path.join(R.RES, f"final_model_{sp}.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
