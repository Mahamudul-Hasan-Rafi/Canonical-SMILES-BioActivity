"""
Final head-to-head table for the study: every headline model as the 3-seed ensemble
(15 fold-models each), threshold = OOF MCC-optimal, i.e. the notebook's protocol.
Regression models are scored through their predicted potency. Writes results/final_table.md.
"""
import os
import sys

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M
import report as R

KEYS = ["Accuracy", "Balanced Acc", "Precision", "Sensitivity", "Specificity", "F1", "ROC-AUC", "AUPRC", "MCC"]
NAME = {"Sensitivity": "Recall", "Specificity": "Spec."}


def main():
    st = R.Store()
    y = st.y[st.splits["random"]["dev"]]

    def ens(**kw):
        rr = [r for r in (st.cv(seed=s, **kw) for s in R.SEEDS) if r is not None]
        return (np.mean([r["oof"] for r in rr], 0), np.mean([r["ens"] for r in rr], 0), rr[0]["test_y"]) if rr else None

    mods = {"V3 published (MoLFormer)": ens(),
            "V5-MT (ChemBERTa + graph + pIC50 head)": ens(backbone="chemberta_mlm", variant="graph_mt"),
            "V6-R (neighbour-anchored delta)": ens(backbone="chemberta_mlm", variant="graph_mt_delta"),
            "V7 (gated analogue correction)": ens(backbone="chemberta_mlm", variant="graph_mt_delta_res"),
            "V8-R2 (regression-first + delta)": ens(backbone="chemberta_mlm", variant="reg_first_delta")}
    a, b = mods["V5-MT (ChemBERTa + graph + pIC50 head)"], mods["V8-R2 (regression-first + delta)"]
    mods["V5-MT + V8-R2 (rank blend)"] = ((rankdata(a[0]) + rankdata(b[0])) / (2 * len(y)),
                                          (rankdata(a[1]) + rankdata(b[1])) / (2 * len(a[1])), a[2])
    z = np.load(os.path.join(R.RES, "baselines", "random", "XGBoost__ECFPc2048+MACCS+Desc.npz"))
    mods["XGBoost (same features)"] = (z["oof"], z["test_probs"].mean(0), z["test_labels"].astype(int))
    mods = {k: v for k, v in mods.items() if v is not None}

    thr = {k: M.mcc_thr(y, v[0]) for k, v in mods.items()}
    L = ["# Final metrics - all headline models\n",
         "3-seed ensembles (15 fold-models each); threshold = out-of-fold MCC-optimal, as in the notebook. "
         "Precision / recall / F1 are for the active class.\n"]
    for proto, title, pick in (("test", "Held-out test set (825 molecules)", 1),
                               ("oof", "Out-of-fold (4,669 development molecules)", 0)):
        L += [f"## {title}\n", "| model | " + " | ".join(NAME.get(k, k) for k in KEYS) + " | errors |",
              "|" + "---|" * (2 + len(KEYS))]
        for k, v in mods.items():
            yy = v[2] if proto == "test" else y
            m = M.thr_metrics(yy, v[pick], thr[k])
            err = int(((v[pick] > thr[k]).astype(int) != yy).sum())
            L.append(f"| {k} | " + " | ".join(f"{m[x]:.4f}" for x in KEYS) + f" | {err} |")
        L.append("")
    with open(os.path.join(R.RES, "final_table.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
