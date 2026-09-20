"""
Phase 3 of the hard-case research: do the V6 architectures capture the hard examples?

Per error category (defined from the data by hard_cases.py, independent of any model) this
compares out-of-fold error rates and within-category AUC of several models on the SAME 4,669
development molecules, each at its own OOF MCC-optimal threshold (notebook protocol).
McNemar's exact test compares each model's per-molecule errors against V5-MT on the cliff and
noise-band subsets. Test-set numbers are reported once, at the end, for the final models only.
Writes results/hard_case_compare.md.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M
import report as R


def mcnemar(err_a, err_b):
    b, c = int((err_a & ~err_b).sum()), int((~err_a & err_b).sum())
    return b, c, (binomtest(b, b + c, 0.5).pvalue if b + c else 1.0)


def main():
    st = R.Store()
    t = pd.read_csv(os.path.join(R.RES, "hard_cases.csv"))
    y = t["y"].values

    def ens(**kw):
        rr = [r for r in (st.cv(seed=sd, **kw) for sd in R.SEEDS) if r is not None]
        if not rr:
            return None
        return np.mean([r["oof"] for r in rr], 0), np.mean([r["ens"] for r in rr], 0), rr[0]["test_y"]
    models = {"V3 (MoLFormer, published)": ens(), "V5-MT": ens(backbone="chemberta_mlm", variant="graph_mt"),
              "V6-K (analogue context)": ens(backbone="chemberta_mlm", variant="graph_mt_knn"),
              "V6-R (anchored delta)": ens(backbone="chemberta_mlm", variant="graph_mt_delta"),
              "V7 (gated correction)": ens(backbone="chemberta_mlm", variant="graph_mt_delta_res")}
    z = np.load(os.path.join(R.RES, "baselines", "random", "XGBoost__ECFPc2048+MACCS+Desc.npz"))
    models["XGBoost (V4 fingerprints)"] = (z["oof"], z["test_probs"].mean(0), z["test_labels"].astype(int))
    models = {k: v for k, v in models.items() if v is not None}
    # sanity: hard_cases.csv rows are in dev order == Store OOF order
    assert (y == st.cv(seed=42)["oof_y"]).all()

    cats = {"all": np.ones(len(t), bool),
            "activity cliffs (any)": t["cliff"].values,
            "noise band (0-0.5 from cut-off)": ((t.margin >= 0) & (t.margin < 0.5)).values,
            "stereo twins": t["stereo_twin"].values,
            "clear cases (margin >= 0.5, no cliff)": ((t.margin >= 0.5) & ~t["cliff"]).values,
            "label contradicts pIC50": (t.margin < 0).values}
    errs = {}
    L = ["# Do the V6 architectures capture the hard cases? (out-of-fold, 4,669 development molecules)\n",
         "Error rate per category at each model's own OOF MCC-optimal threshold; AUC within the category "
         "(both classes present).\n",
         "| category | n | " + " | ".join(f"{m} err" for m in models) + " |", "|" + "---|" * (2 + len(models))]
    for name, (o, _, _) in models.items():
        errs[name] = (o > M.mcc_thr(y, o)).astype(int) != y
    for c, m in cats.items():
        L.append(f"| {c} | {m.sum()} | " + " | ".join(f"{errs[k][m].mean():.3f} ({errs[k][m].sum()})" for k in models) + " |")
    L += ["", "| category | " + " | ".join(f"{m} AUC" for m in models) + " |", "|" + "---|" * (1 + len(models))]
    for c, m in cats.items():
        if len(set(y[m])) == 2:
            L.append(f"| {c} | " + " | ".join(f"{roc_auc_score(y[m], models[k][0][m]):.4f}" for k in models) + " |")
    ref = "V5-MT"
    if ref in errs:
        L += ["", f"## McNemar exact test vs {ref} (per-molecule errors: b = only this model wrong, c = only {ref} wrong)\n",
              "| model | subset | b | c | p |", "|---|---|---|---|---|"]
        for k in models:
            if k == ref:
                continue
            for c in ["all", "activity cliffs (any)", "noise band (0-0.5 from cut-off)"]:
                b, cc, p = mcnemar(errs[k][cats[c]], errs[ref][cats[c]])
                L.append(f"| {k} | {c} | {b} | {cc} | {p:.3g} |")
    # held-out test, once
    L += ["", "## Held-out test set (825 molecules; used once, after the architecture choices)\n",
          "| model | test AUC | test accuracy | test errors |", "|---|---|---|---|"]
    for k, (o, te_p, te_y) in models.items():
        thr = M.mcc_thr(y, o)
        L.append(f"| {k} | {roc_auc_score(te_y, te_p):.4f} | {accuracy_score(te_y, te_p > thr):.4f} | "
                 f"{int(((te_p > thr).astype(int) != te_y).sum())} |")
    with open(os.path.join(R.RES, "hard_case_compare.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
