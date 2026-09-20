"""
Where do we stand? Two comparisons, kept strictly apart.

  1. Like-for-like: every model on the same folds and the same held-out test set of this
     dataset. This is the only rigorous ranking.
  2. Published reference points from the literature, which use different datasets, activity
     thresholds and splits, so they bound the range rather than rank the models.

Reports classification accuracy and potency RMSE side by side, on both splits.
Writes results/position_<split>.md.
"""
import argparse
import os
import sys

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
import data
import hybrid as H
import metrics as M
import report as R

REG = ["LGBMReg", "XGBReg", "RFReg", "CatReg"]


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="random", choices=["random", "scaffold"])
    args = ap.parse_args()
    sp = args.split
    st = R.Store()
    dev, te = st.splits[sp]["dev"], st.splits[sp]["test"]
    y = st.y[dev]
    pic = data.get_pic50(st.df["canonical_smiles"].tolist())
    p_dev, p_te = pic[dev], pic[te]

    def npz(nm):
        return np.load(os.path.join(R.RES, "baselines", sp, nm + "__ECFP+Desc.npz"))

    def ens(**kw):
        rr = [r for r in (st.cv(split=sp, seed=s, **kw) for s in R.SEEDS) if r is not None]
        return (np.mean([r["oof"] for r in rr], 0), np.mean([r["ens"] for r in rr], 0), rr[0]["test_y"]) if rr else None

    # ---------- potency ----------
    pot = {}
    for nm in REG:
        z = npz(nm)
        pot[nm] = (z["oof_pic50"], z["test_pic50"].mean(0))
    r = ens(backbone="chemberta_mlm", variant="reg_first_delta")
    if r is not None:
        pot["V8-R2 (deep)"] = (core.probs_to_pic50(r[0]), core.probs_to_pic50(r[1]))
    # greedy potency ensemble, chosen on OOF RMSE
    members, best = [], 1e9
    while len(members) < 6:
        cand = [(rmse(np.mean([pot[m][0] for m in members + [k]], 0) - p_dev), k) for k in pot]
        v, k = min(cand)
        if v >= best - 1e-5:
            break
        best, members = v, members + [k]
    pot["HYBRID potency (" + " + ".join(members) + ")"] = (np.mean([pot[m][0] for m in members], 0),
                                                           np.mean([pot[m][1] for m in members], 0))

    # ---------- classification ----------
    cls = {}
    for nm in ("Voting", "LightGBM", "CatBoost", "XGBoost"):
        z = npz(nm)
        cls[nm] = (z["oof"], z["test_probs"].mean(0))
    for nm, kw in (("V3 (published)", {}), ("V5-MT", dict(backbone="chemberta_mlm", variant="graph_mt")),
                   ("V8-R2", dict(backbone="chemberta_mlm", variant="reg_first_delta"))):
        r = ens(**kw)
        if r is not None:
            cls[nm] = (r[0], r[1])
    yt = st.y[te]
    hyb = (["Voting", "Voting", "V8-R2"] if sp == "random" else ["Voting", "Voting", "V5-MT", "CatBoost"])
    cls["HYBRID ensemble"] = (np.mean([rankdata(cls[m][0]) / len(y) for m in hyb], 0),
                              np.mean([rankdata(cls[m][1]) / len(yt) for m in hyb], 0))

    L = [f"# Where the hybrid stands ({sp} split)\n",
         f"Same folds, same held-out test set ({len(y):,} development / {len(yt)} test molecules) for every "
         "row. Classification threshold fixed out-of-fold; potency in log units.\n",
         "## Potency (pIC50 RMSE, lower is better)\n",
         "| model | OOF RMSE | test RMSE | OOF MAE | vs our best |", "|---|---|---|---|---|"]
    bo = min(rmse(v[0] - p_dev) for v in pot.values())
    for k, v in sorted(pot.items(), key=lambda kv: rmse(kv[1][0] - p_dev)):
        e, et = v[0] - p_dev, v[1] - p_te
        L.append(f"| {k} | **{rmse(e):.4f}** | {rmse(et):.4f} | {np.abs(e).mean():.4f} | "
                 f"{rmse(e) - bo:+.4f} |")
    L += ["", "## Classification accuracy\n",
          "| model | OOF acc | test acc | OOF MCC | test MCC | test AUC | vs our best (test acc) |",
          "|---|---|---|---|---|---|---|"]
    thr = {k: H.acc_thr(y, v[0]) for k, v in cls.items()}
    ba = max(accuracy_score(yt, (v[1] > thr[k]).astype(int)) for k, v in cls.items())
    for k, v in sorted(cls.items(), key=lambda kv: -accuracy_score(y, (kv[1][0] > thr[kv[0]]).astype(int))):
        po, pt = (v[0] > thr[k]).astype(int), (v[1] > thr[k]).astype(int)
        L.append(f"| {k} | **{accuracy_score(y, po):.4f}** | {accuracy_score(yt, pt):.4f} | "
                 f"{matthews_corrcoef(y, po):.4f} | {matthews_corrcoef(yt, pt):.4f} | "
                 f"{roc_auc_score(yt, v[1]):.4f} | {accuracy_score(yt, pt) - ba:+.4f} |")
    with open(os.path.join(R.RES, f"position_{sp}.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
