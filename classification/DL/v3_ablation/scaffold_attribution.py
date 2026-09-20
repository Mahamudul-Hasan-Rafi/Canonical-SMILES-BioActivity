"""
Scaffold split: is the loss of the newer models caused by the BACKBONE swap
(MoLFormer -> ChemBERTa) or by the ADDED MODULES (graph branch, pIC50 head, retrieval)?

The chain is walked one change at a time, each step tested against the step before it:

  V3 (MoLFormer, V3 architecture)
    -> V4 baseline (ChemBERTa, same V3 architecture)      = backbone effect
    -> V5-MT       (+ count FP, D-MPNN, pIC50 head)       = added-module effect
    -> V8-R2       (regression-first + retrieval delta)    = regression-first effect

3-seed ensembles, threshold = OOF MCC-optimal. McNemar on per-molecule errors and paired
DeLong on the AUCs. Writes results/scaffold_attribution.md.
"""
import os
import sys

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M
import report as R

SP = "scaffold"
CHAIN = [("V3 (MoLFormer)", {}),
         ("V4 baseline (ChemBERTa)", dict(backbone="chemberta_mlm")),
         ("V5-MT (+ graph + pIC50 head)", dict(backbone="chemberta_mlm", variant="graph_mt")),
         ("V8-R2 (regression-first)", dict(backbone="chemberta_mlm", variant="reg_first_delta"))]
STEPS = [(1, 0, "backbone swap: MoLFormer -> ChemBERTa"),
         (2, 1, "added modules: count FP + D-MPNN + pIC50 head"),
         (3, 2, "regression-first + retrieval delta")]


def main():
    st = R.Store()
    dev = st.splits[SP]["dev"]
    y = st.y[dev]
    mods, names = {}, []
    for name, kw in CHAIN:
        rr = [r for r in (st.cv(split=SP, seed=s, **kw) for s in R.SEEDS) if r is not None]
        if not rr:
            print(f"missing: {name}")
            continue
        mods[name] = (np.mean([r["oof"] for r in rr], 0), np.mean([r["ens"] for r in rr], 0), rr[0]["test_y"])
        names.append(name)
    thr = {k: M.mcc_thr(y, v[0]) for k, v in mods.items()}
    yt = mods[names[0]][2]

    L = ["# Scaffold split: backbone or modules?\n",
         f"3-seed ensembles on the scaffold split ({len(dev):,} development / {len(yt)} test molecules), "
         "threshold = OOF MCC-optimal.\n",
         "| model | OOF AUC | OOF Acc | OOF MCC | test AUC | test Acc | test MCC | test errors |",
         "|---|---|---|---|---|---|---|---|"]
    for k in names:
        o, t, _ = mods[k]
        po, pt = (o > thr[k]).astype(int), (t > thr[k]).astype(int)
        L.append(f"| {k} | {roc_auc_score(y, o):.4f} | {accuracy_score(y, po):.4f} | {matthews_corrcoef(y, po):.4f} | "
                 f"{roc_auc_score(yt, t):.4f} | {accuracy_score(yt, pt):.4f} | {matthews_corrcoef(yt, pt):.4f} | "
                 f"{int((pt != yt).sum())} |")

    L += ["", "## One change at a time (each row tests a model against the row above it)\n",
          "| change | set | McNemar b / c | p | delta AUC | p |", "|---|---|---|---|---|---|"]
    for i, j, label in STEPS:
        if i >= len(names) or j >= len(names):
            continue
        a, b = names[i], names[j]
        for lab, pick, yy in (("OOF", 0, y), ("test", 1, yt)):
            ea = (mods[a][pick] > thr[a]).astype(int) != yy
            eb = (mods[b][pick] > thr[b]).astype(int) != yy
            n01, n10 = int((ea & ~eb).sum()), int((~ea & eb).sum())
            p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
            d, pd_ = M.delong(yy, mods[a][pick], mods[b][pick])
            L.append(f"| {label} | {lab} | {n01} / {n10} | {p:.3f} | {d:+.4f} | {pd_:.3f} |")
    L += ["", "b = molecules only the newer model gets wrong, c = only the previous one. "
          "Negative delta AUC = the change hurts.\n"]
    with open(os.path.join(R.RES, "scaffold_attribution.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
