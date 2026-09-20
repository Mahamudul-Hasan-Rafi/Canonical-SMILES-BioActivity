"""
System-level route to higher accuracy, with every choice made out-of-fold.

Two levers that need no new training:
  1. the decision threshold. Every table so far uses the OOF MCC-optimal threshold (the notebook
     rule). MCC and accuracy do not peak at the same place on an imbalanced set, so the
     accuracy-optimal OOF threshold is reported alongside it.
  2. a hybrid ensemble. Deep and gradient-boosting models fail on different molecules; members
     are added greedily while the OOF accuracy improves (rank-averaged, so scales are comparable).

The held-out test set is scored once at the end, at the threshold fixed out-of-fold.
Writes results/hybrid_<split>.md.
"""
import argparse
import glob
import os
import sys

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M
import report as R


def acc_thr(y, p):
    """Threshold maximising accuracy, scanned over the midpoints of the sorted scores."""
    o = np.unique(p)
    cand = (o[:-1] + o[1:]) / 2 if len(o) > 1 else o
    if len(cand) > 4000:
        cand = np.quantile(cand, np.linspace(0, 1, 4000))
    accs = [accuracy_score(y, (p > t).astype(int)) for t in cand]
    return float(cand[int(np.argmax(accs))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="random", choices=["random", "scaffold"])
    ap.add_argument("--sets", default="ECFP+Desc")
    args = ap.parse_args()
    sp = args.split
    st = R.Store()
    y = st.y[st.splits[sp]["dev"]]
    cand = {}
    for path in sorted(glob.glob(os.path.join(R.RES, "baselines", sp, "*.npz"))):
        nm = os.path.basename(path)[:-4]
        if nm.split("__")[-1] not in [x.strip() for x in args.sets.split(",")] or nm.startswith("LogReg"):
            continue
        z = np.load(path)
        cand[nm.split("__")[0]] = (z["oof"], z["test_probs"].mean(0), z["test_labels"].astype(int))

    def ens(**kw):
        rr = [r for r in (st.cv(split=sp, seed=s, **kw) for s in R.SEEDS) if r is not None]
        return (np.mean([r["oof"] for r in rr], 0), np.mean([r["ens"] for r in rr], 0), rr[0]["test_y"]) if rr else None
    for nm, kw in [("V3", {}), ("V5-MT", dict(backbone="chemberta_mlm", variant="graph_mt")),
                   ("V7", dict(backbone="chemberta_mlm", variant="graph_mt_delta_res")),
                   ("V8-R2", dict(backbone="chemberta_mlm", variant="reg_first_delta")),
                   ("V9", dict(backbone="chemberta_mlm", variant="v9"))]:
        r = ens(**kw)
        if r is not None:
            cand[nm] = r
    yt = list(cand.values())[0][2]
    rk = {k: (rankdata(v[0]) / len(y), rankdata(v[1]) / len(v[1])) for k, v in cand.items()}

    def score(members):
        o = np.mean([rk[m][0] for m in members], 0)
        t = np.mean([rk[m][1] for m in members], 0)
        return o, t

    # greedy forward selection with replacement, on OOF accuracy
    members, best = [], -1.0
    while True:
        gains = []
        for k in cand:
            o, _ = score(members + [k])
            gains.append((accuracy_score(y, (o > acc_thr(y, o)).astype(int)), k))
        g, k = max(gains)
        if g <= best + 1e-6 or len(members) >= 8:
            break
        best, members = g, members + [k]

    L = [f"# Raising accuracy without new representations ({sp} split)\n",
         "Thresholds and ensemble membership are chosen on the out-of-fold predictions "
         f"({len(y):,} molecules); the test set ({len(yt)}) is scored once at those fixed choices.\n",
         "## Single models: MCC-optimal vs accuracy-optimal threshold (both fixed out-of-fold)\n",
         "| model | OOF acc @MCC-thr | OOF acc @acc-thr | test acc @MCC-thr | test acc @acc-thr | test MCC @acc-thr |",
         "|---|---|---|---|---|---|"]
    for k, v in sorted(cand.items(), key=lambda kv: -accuracy_score(y, (kv[1][0] > acc_thr(y, kv[1][0])).astype(int))):
        tm, ta = M.mcc_thr(y, v[0]), acc_thr(y, v[0])
        L.append(f"| {k} | {accuracy_score(y, (v[0] > tm).astype(int)):.4f} | "
                 f"{accuracy_score(y, (v[0] > ta).astype(int)):.4f} | "
                 f"{accuracy_score(yt, (v[1] > tm).astype(int)):.4f} | "
                 f"{accuracy_score(yt, (v[1] > ta).astype(int)):.4f} | "
                 f"{matthews_corrcoef(yt, (v[1] > ta).astype(int)):.4f} |")

    o, t = score(members)
    ta = acc_thr(y, o)
    L += ["", "## Greedy hybrid ensemble (rank-averaged)\n",
          "members, in the order they were added: " + ", ".join(members) + "\n",
          "| set | accuracy | balanced acc | MCC | ROC-AUC | errors |", "|---|---|---|---|---|---|"]
    for lab, p, yy in (("out-of-fold", o, y), ("held-out test", t, yt)):
        pr = (p > ta).astype(int)
        L.append(f"| {lab} | **{accuracy_score(yy, pr):.4f}** | "
                 f"{M.thr_metrics(yy, p, ta)['Balanced Acc']:.4f} | {matthews_corrcoef(yy, pr):.4f} | "
                 f"{roc_auc_score(yy, p):.4f} | {int((pr != yy).sum())} |")
    L += ["", f"Accuracy-optimal OOF threshold = {ta:.4f} (rank scale).",
          "Raising accuracy by moving the threshold trades away balanced accuracy and recall on the "
          "minority class; both are reported so the trade is visible.\n"]
    with open(os.path.join(R.RES, f"hybrid_{sp}.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
