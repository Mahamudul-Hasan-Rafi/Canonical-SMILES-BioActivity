"""
Head-to-head: V5-MT vs the neighbour-anchored V6 models.

  * full metric table for the 3-seed ensembles (OOF 4,669 and test 825), notebook threshold rule
  * McNemar exact test on the per-molecule errors (test and OOF, ensembles and per seed)
  * paired DeLong on the AUCs (test and OOF)
Writes results/v5_vs_v6.md.
"""
import os
import sys

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M
import report as R

KEYS = ["Accuracy", "Balanced Acc", "Precision", "Sensitivity", "F1", "ROC-AUC", "MCC", "Brier"]
NAME = {"Sensitivity": "Recall"}


def mcnemar(a, b):
    """a, b: boolean error vectors. Returns (only-a-wrong, only-b-wrong, exact p)."""
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    return n01, n10, (binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0)


def main():
    st = R.Store()
    models = {"V5-MT": dict(backbone="chemberta_mlm", variant="graph_mt"),
              "V6-K": dict(backbone="chemberta_mlm", variant="graph_mt_knn"),
              "V6-R": dict(backbone="chemberta_mlm", variant="graph_mt_delta"),
              "V7": dict(backbone="chemberta_mlm", variant="graph_mt_delta_res")}
    runs = {k: [st.cv(seed=s, **kw) for s in R.SEEDS] for k, kw in models.items()}
    y_oof, y_te = runs["V5-MT"][0]["oof_y"], runs["V5-MT"][0]["test_y"]
    ens = {k: (np.mean([r["oof"] for r in v], 0), np.mean([r["ens"] for r in v], 0)) for k, v in runs.items()}
    thr = {k: M.mcc_thr(y_oof, o) for k, (o, _) in ens.items()}
    err_te = {k: (t > thr[k]).astype(int) != y_te for k, (_, t) in ens.items()}
    err_oof = {k: (o > thr[k]).astype(int) != y_oof for k, (o, _) in ens.items()}

    L = ["# V5-MT vs V6 (neighbour-anchored): head-to-head\n",
         "3-seed ensembles (15 models each), threshold = OOF MCC-optimal per model (notebook rule).\n",
         "## Held-out test set (825 molecules)\n",
         "| metric | " + " | ".join(models) + " |", "|" + "---|" * (1 + len(models))]
    mt = {k: M.thr_metrics(y_te, ens[k][1], thr[k]) for k in models}
    mo = {k: M.thr_metrics(y_oof, ens[k][0], thr[k]) for k in models}
    for key in KEYS:
        L.append(f"| {NAME.get(key, key)} | " + " | ".join(f"{mt[k][key]:.4f}" for k in models) + " |")
    L.append(f"| errors (of 825) | " + " | ".join(str(int(err_te[k].sum())) for k in models) + " |")
    L += ["", "## Out-of-fold (4,669 development molecules)\n",
          "| metric | " + " | ".join(models) + " |", "|" + "---|" * (1 + len(models))]
    for key in KEYS:
        L.append(f"| {NAME.get(key, key)} | " + " | ".join(f"{mo[k][key]:.4f}" for k in models) + " |")
    L.append(f"| errors (of 4,669) | " + " | ".join(str(int(err_oof[k].sum())) for k in models) + " |")

    L += ["", "## McNemar exact test (per-molecule errors, ensembles)\n",
          "| comparison | set | only A wrong | only B wrong | both wrong | p |", "|---|---|---|---|---|---|"]
    for a, b in [("V6-K", "V5-MT"), ("V6-R", "V5-MT"), ("V6-R", "V6-K"), ("V7", "V5-MT"), ("V7", "V6-R")]:
        for label, ea, eb in [("test", err_te[a], err_te[b]), ("OOF", err_oof[a], err_oof[b])]:
            n01, n10, p = mcnemar(ea, eb)
            L.append(f"| A = {a} vs B = {b} | {label} | {n01} | {n10} | {int((ea & eb).sum())} | {p:.3f} |")
    L += ["", "## McNemar per seed (test set, single 5-fold model each)\n",
          "| comparison | seed | only A wrong | only B wrong | p |", "|---|---|---|---|---|"]
    for a, b in [("V6-K", "V5-MT"), ("V6-R", "V5-MT"), ("V7", "V5-MT")]:
        for i, s in enumerate(R.SEEDS):
            ra, rb = runs[a][i], runs[b][i]
            ea = (ra["ens"] > M.mcc_thr(y_oof, ra["oof"])).astype(int) != y_te
            eb = (rb["ens"] > M.mcc_thr(y_oof, rb["oof"])).astype(int) != y_te
            n01, n10, p = mcnemar(ea, eb)
            L.append(f"| A = {a} vs B = {b} | {s} | {n01} | {n10} | {p:.3f} |")
    L += ["", "## Paired DeLong on the AUCs (ensembles)\n",
          "| comparison | Δ test AUC | p test | Δ OOF AUC | p OOF |", "|---|---|---|---|---|"]
    for a, b in [("V6-K", "V5-MT"), ("V6-R", "V5-MT"), ("V6-R", "V6-K"), ("V7", "V5-MT"), ("V7", "V6-R")]:
        dt, pt = M.delong(y_te, ens[a][1], ens[b][1])
        do, po = M.delong(y_oof, ens[a][0], ens[b][0])
        L.append(f"| {a} vs {b} | {dt:+.4f} | {pt:.3f} | {do:+.4f} | {po:.3f} |")
    L += ["", "McNemar compares the models on the same molecules: n01/n10 are the molecules only one model gets "
          "wrong, and only those carry information. With ~50 test errors each and large overlap, the test set can "
          "only detect large differences; the OOF comparison (4,669 molecules) is the more sensitive one.\n"]
    with open(os.path.join(R.RES, "v5_vs_v6.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
