"""
Selective prediction: accuracy as a function of coverage.

The model abstains on the molecules it is least sure about (distance of the score from the
decision threshold) and is scored only on the rest. Coverage and the abstention rule are fixed
on the out-of-fold predictions; the held-out test set is scored once at those fixed choices.
This is how a screening model is actually deployed, and it is the only honest way to reach a
high accuracy on this dataset without more potency data.
Writes results/selective_<split>.md.
"""
import argparse
import glob
import os
import sys

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import accuracy_score, matthews_corrcoef

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hybrid as H
import report as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="random", choices=["random", "scaffold"])
    args = ap.parse_args()
    sp = args.split
    st = R.Store()
    y = st.y[st.splits[sp]["dev"]]

    def load(nm):
        z = np.load(os.path.join(R.RES, "baselines", sp, nm + "__ECFP+Desc.npz"))
        return z["oof"], z["test_probs"].mean(0), z["test_labels"].astype(int)

    def ens(**kw):
        rr = [r for r in (st.cv(split=sp, seed=s, **kw) for s in R.SEEDS) if r is not None]
        return np.mean([r["oof"] for r in rr], 0), np.mean([r["ens"] for r in rr], 0), rr[0]["test_y"]
    parts = ([load("Voting"), load("Voting")] +
             ([ens(backbone="chemberta_mlm", variant="reg_first_delta")] if sp == "random"
              else [ens(backbone="chemberta_mlm", variant="graph_mt"), load("CatBoost")]))
    yt = parts[0][2]
    o = np.mean([rankdata(p[0]) / len(y) for p in parts], 0)
    t = np.mean([rankdata(p[1]) / len(yt) for p in parts], 0)
    thr = H.acc_thr(y, o)

    L = [f"# Selective prediction: accuracy vs coverage ({sp} split)\n",
         "Hybrid ensemble. The model abstains on the molecules whose score lies closest to the "
         "decision threshold; the abstention cut for each coverage level is taken from the "
         "out-of-fold scores and applied unchanged to the test set.\n",
         "| coverage | OOF accuracy | OOF MCC | OOF n | test accuracy | test MCC | test n | test errors |",
         "|---|---|---|---|---|---|---|---|"]
    do, dt = np.abs(o - thr), np.abs(t - thr)
    for cov in (1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5):
        cut = np.quantile(do, 1 - cov)
        mo, mt = do >= cut, dt >= cut
        po, pt = (o[mo] > thr).astype(int), (t[mt] > thr).astype(int)
        L.append(f"| {cov:.0%} | **{accuracy_score(y[mo], po):.4f}** | {matthews_corrcoef(y[mo], po):.4f} | "
                 f"{mo.sum():,} | **{accuracy_score(yt[mt], pt):.4f}** | {matthews_corrcoef(yt[mt], pt):.4f} | "
                 f"{mt.sum()} | {int((pt != yt[mt]).sum())} |")
    L += ["", "Test coverage differs slightly from the nominal level because the cut is fixed "
          "out-of-fold rather than re-fitted on the test scores - that is the point.\n"]
    with open(os.path.join(R.RES, f"selective_{sp}.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
