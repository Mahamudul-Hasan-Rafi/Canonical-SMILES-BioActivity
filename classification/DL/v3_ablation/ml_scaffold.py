"""
Classical ML and ensemble models on the scaffold split, on exactly the folds and held-out test
set used by the deep models, scored under the same protocol (threshold = OOF MCC-optimal).

Every model in results/baselines/<split>/ is reported, together with the deep models, so the
comparison is like-for-like. The best classical model is then tested against V3 with McNemar's
exact test and paired DeLong. Writes results/ml_<split>.md.
"""
import argparse
import glob
import os
import sys

import numpy as np
from scipy.stats import binomtest, rankdata
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M
import report as R

KEYS = ["Accuracy", "Balanced Acc", "Precision", "Sensitivity", "Specificity", "F1", "ROC-AUC", "MCC"]
NAME = {"Sensitivity": "Recall", "Specificity": "Spec."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="scaffold", choices=["random", "scaffold"])
    ap.add_argument("--sets", default="", help="only report baselines built on these feature sets")
    args = ap.parse_args()
    sp = args.split
    st = R.Store()
    y = st.y[st.splits[sp]["dev"]]
    mods = {}
    want = [x.strip() for x in args.sets.split(",")] if args.sets else None
    for path in sorted(glob.glob(os.path.join(R.RES, "baselines", sp, "*.npz"))):
        if want and os.path.basename(path)[:-4].split("__")[-1] not in want:
            continue
        z = np.load(path)
        mods[os.path.basename(path)[:-4].replace("__", " on ")] = (z["oof"], z["test_probs"].mean(0),
                                                                   z["test_labels"].astype(int))

    def ens(**kw):
        rr = [r for r in (st.cv(split=sp, seed=s, **kw) for s in R.SEEDS) if r is not None]
        return (np.mean([r["oof"] for r in rr], 0), np.mean([r["ens"] for r in rr], 0), rr[0]["test_y"]) if rr else None
    deep = {"[DL] V3 (MoLFormer)": ens(),
            "[DL] V4 baseline (ChemBERTa)": ens(backbone="chemberta_mlm"),
            "[DL] V5-MT": ens(backbone="chemberta_mlm", variant="graph_mt"),
            "[DL] V8-R2": ens(backbone="chemberta_mlm", variant="reg_first_delta")}
    deep = {k: v for k, v in deep.items() if v is not None}
    if "[DL] V5-MT" in deep and "[DL] V8-R2" in deep:
        a, b = deep["[DL] V5-MT"], deep["[DL] V8-R2"]
        deep["[DL] V5-MT + V8-R2 blend"] = ((rankdata(a[0]) + rankdata(b[0])) / (2 * len(y)),
                                            (rankdata(a[1]) + rankdata(b[1])) / (2 * len(a[1])), a[2])
    mods.update(deep)
    thr = {k: M.mcc_thr(y, v[0]) for k, v in mods.items()}
    order = sorted(mods, key=lambda k: -matthews_corrcoef(y, (mods[k][0] > thr[k]).astype(int)))
    yt = mods[order[0]][2]

    L = [f"# Classical ML, ensembles and deep models on the {sp} split\n",
         f"Same 5 folds and held-out test set ({len(y):,} development / {len(yt)} test molecules); "
         "threshold = out-of-fold MCC-optimal for every model. Sorted by OOF MCC.\n",
         "## Out-of-fold\n",
         "| model | " + " | ".join(NAME.get(k, k) for k in KEYS) + " | errors |", "|" + "---|" * (2 + len(KEYS))]
    for k in order:
        m = M.thr_metrics(y, mods[k][0], thr[k])
        err = int(((mods[k][0] > thr[k]).astype(int) != y).sum())
        L.append(f"| {k} | " + " | ".join(f"{m[x]:.4f}" for x in KEYS) + f" | {err} |")
    L += ["", "## Held-out test set\n",
          "| model | " + " | ".join(NAME.get(k, k) for k in KEYS) + " | errors |", "|" + "---|" * (2 + len(KEYS))]
    for k in order:
        m = M.thr_metrics(yt, mods[k][1], thr[k])
        err = int(((mods[k][1] > thr[k]).astype(int) != yt).sum())
        L.append(f"| {k} | " + " | ".join(f"{m[x]:.4f}" for x in KEYS) + f" | {err} |")

    ref = "[DL] V3 (MoLFormer)"
    if ref in mods:
        best = [k for k in order if not k.startswith("[DL]")][:3]
        L += ["", f"## Best classical models vs {ref}\n",
              "| model | set | McNemar b / c | p | delta AUC | p |", "|---|---|---|---|---|---|"]
        for k in best:
            for lab, pick, yy in (("OOF", 0, y), ("test", 1, yt)):
                ea = (mods[k][pick] > thr[k]).astype(int) != yy
                eb = (mods[ref][pick] > thr[ref]).astype(int) != yy
                n01, n10 = int((ea & ~eb).sum()), int((~ea & eb).sum())
                p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
                d, pd_ = M.delong(yy, mods[k][pick], mods[ref][pick])
                L.append(f"| {k} | {lab} | {n01} / {n10} | {p:.3f} | {d:+.4f} | {pd_:.3f} |")
        L += ["", "b = molecules only the classical model gets wrong, c = only V3. "
              "Negative delta AUC = worse than V3.\n"]
    suffix = "_" + args.sets.replace("+", "") if args.sets else ""
    with open(os.path.join(R.RES, f"ml_{sp}{suffix}.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
