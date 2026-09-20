"""
Final head-to-head table for the study: every headline model as the 3-seed ensemble
(15 fold-models each), threshold = OOF MCC-optimal, i.e. the notebook's protocol.
Regression models are scored through their predicted potency. Writes results/final_table.md.
"""
import argparse
import os
import sys

import numpy as np
from scipy.stats import binomtest, rankdata
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M
import report as R

KEYS = ["Accuracy", "Balanced Acc", "Precision", "Sensitivity", "Specificity", "F1", "ROC-AUC", "AUPRC", "MCC"]
NAME = {"Sensitivity": "Recall", "Specificity": "Spec."}


def potency_and_tests(st, sp, dev, y, mods, thr):
    """pIC50 precision of the regression model, plus blend-vs-V5-MT significance tests."""
    import core
    import data
    import metrics as M
    pic = data.get_pic50(st.df["canonical_smiles"].tolist())
    p_dev, p_te = pic[dev], pic[st.splits[sp]["test"]]
    out = ["## Potency prediction (V8-R2, the only model trained to regress pIC50)", ""]
    reg = [k for k in mods if k.startswith("V8-R2")]
    if reg:
        o, t, _ = mods[reg[0]]
        eo, et = core.probs_to_pic50(o) - p_dev, core.probs_to_pic50(t) - p_te
        out += ["| set | RMSE | MAE |", "|---|---|---|",
                f"| out-of-fold | {np.sqrt(np.mean(eo ** 2)):.4f} | {np.abs(eo).mean():.4f} |",
                f"| held-out test | {np.sqrt(np.mean(et ** 2)):.4f} | {np.abs(et).mean():.4f} |", ""]
    a, b = "V5-MT (ChemBERTa + graph + pIC50 head)", "V5-MT + V8-R2 (rank blend)"
    if a in mods and b in mods:
        out += ["## Is the blend better than V5-MT alone?", "",
                "| set | McNemar b / c | p | delta AUC (DeLong) | p |", "|---|---|---|---|---|"]
        for label, pick, yy in (("out-of-fold", 0, y), ("held-out test", 1, mods[a][2])):
            ea = (mods[b][pick] > thr[b]).astype(int) != yy
            eb = (mods[a][pick] > thr[a]).astype(int) != yy
            n01, n10 = int((ea & ~eb).sum()), int((~ea & eb).sum())
            pm = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
            d, pd_ = M.delong(yy, mods[b][pick], mods[a][pick])
            out.append(f"| {label} | {n01} / {n10} | {pm:.4f} | {d:+.4f} | {pd_:.4f} |")
        out += ["", "b = molecules only the blend gets wrong, c = only V5-MT gets wrong.", ""]
    return out



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="random", choices=["random", "scaffold"])
    args = ap.parse_args()
    sp = args.split
    st = R.Store()
    dev = st.splits[sp]["dev"]
    y = st.y[dev]

    def ens(**kw):
        rr = [r for r in (st.cv(split=sp, seed=s, **kw) for s in R.SEEDS) if r is not None]
        return (np.mean([r["oof"] for r in rr], 0), np.mean([r["ens"] for r in rr], 0), rr[0]["test_y"]) if rr else None

    mods = {"V3 published (MoLFormer)": ens(),
            "V5-MT (ChemBERTa + graph + pIC50 head)": ens(backbone="chemberta_mlm", variant="graph_mt"),
            "V6-R (neighbour-anchored delta)": ens(backbone="chemberta_mlm", variant="graph_mt_delta"),
            "V7 (gated analogue correction)": ens(backbone="chemberta_mlm", variant="graph_mt_delta_res"),
            "V8-R2 (regression-first + delta)": ens(backbone="chemberta_mlm", variant="reg_first_delta")}
    a, b = mods["V5-MT (ChemBERTa + graph + pIC50 head)"], mods["V8-R2 (regression-first + delta)"]
    mods["V5-MT + V8-R2 (rank blend)"] = ((rankdata(a[0]) + rankdata(b[0])) / (2 * len(y)),
                                          (rankdata(a[1]) + rankdata(b[1])) / (2 * len(a[1])), a[2])
    z = np.load(os.path.join(R.RES, "baselines", sp, "XGBoost__ECFPc2048+MACCS+Desc.npz"))
    mods["XGBoost (same features)"] = (z["oof"], z["test_probs"].mean(0), z["test_labels"].astype(int))
    mods = {k: v for k, v in mods.items() if v is not None}

    thr = {k: M.mcc_thr(y, v[0]) for k, v in mods.items()}
    L = ["# Final metrics - all headline models\n",
         "3-seed ensembles (15 fold-models each); threshold = out-of-fold MCC-optimal, as in the notebook. "
         "Precision / recall / F1 are for the active class.\n"]
    n_te, n_dev = len(st.splits[sp]["test"]), len(dev)
    for proto, title, pick in (("test", f"Held-out test set ({n_te:,} molecules)", 1),
                               ("oof", f"Out-of-fold ({n_dev:,} development molecules)", 0)):
        L += [f"## {title}\n", "| model | " + " | ".join(NAME.get(k, k) for k in KEYS) + " | errors |",
              "|" + "---|" * (2 + len(KEYS))]
        for k, v in mods.items():
            yy = v[2] if proto == "test" else y
            m = M.thr_metrics(yy, v[pick], thr[k])
            err = int(((v[pick] > thr[k]).astype(int) != yy).sum())
            L.append(f"| {k} | " + " | ".join(f"{m[x]:.4f}" for x in KEYS) + f" | {err} |")
        L.append("")
    L += potency_and_tests(st, sp, dev, y, mods, thr)
    with open(os.path.join(R.RES, f"final_table_{sp}.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
