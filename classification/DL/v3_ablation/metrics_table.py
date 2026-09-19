"""
Accuracy / balanced accuracy / precision / recall / F1 for every finished experiment
on the notebook's (random) split, under the notebook's two evaluation protocols:

  test : 5-model ensemble on the 825 held-out molecules, threshold = OOF MCC-optimal (cell 18)
  cv   : out-of-fold predictions, per-fold metrics at the OOF max-F1 threshold, mean over folds (cell 17)

Precision / recall / F1 are for the ACTIVE class (label 1), as in the notebook.
Writes results/metrics_all.{csv,md}.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
import experiments as E
import report as R

KEYS = ["Accuracy", "Balanced Acc", "Precision", "Sensitivity", "F1", "ROC-AUC", "MCC"]
NAMES = {"Sensitivity": "Recall"}


def main():
    st = R.Store()
    rows = []

    def add(group, label, kw, seeds):
        runs = [(s, st.cv(seed=s, **kw)) for s in seeds]
        runs = [(s, r) for s, r in runs if r is not None]
        if not runs:
            return
        for proto in ("test", "cv"):
            vals = {k: [r[proto][k] for _, r in runs] for k in KEYS}
            row = {"group": group, "experiment": label, "protocol": proto, "seeds": len(runs)}
            for k in KEYS:
                m, s = np.mean(vals[k]), (np.std(vals[k], ddof=1) if len(runs) > 1 else np.nan)
                row[NAMES.get(k, k)] = m
                row[NAMES.get(k, k) + " sd"] = s
            rows.append(row)

    add("Reproduction", "V3 full (notebook config)", {}, E.REPRO_SEEDS)
    add("Ablation (seed 42)", "full: V3 reference for the ablation rows", {}, [42])
    for v in E.ABLATION_ORDER:
        if v == "no_sampler":
            continue          # reported under the class-imbalance group
        add("Ablation (seed 42)", f"{v}: {core.VARIANTS[v][0]}", dict(variant=v), [42])
    add("Class imbalance", "V3 (sampler + focal loss)", {}, E.REPRO_SEEDS)
    add("Class imbalance", "(a) no sampler, focal loss", dict(variant="no_sampler"), E.REPRO_SEEDS)
    add("Class imbalance", "(b) no sampler, plain BCE", dict(variant="vanilla"), E.REPRO_SEEDS)
    add("Tuning", "Untuned (notebook fallback config)", dict(hp_set="untuned"), E.REPRO_SEEDS)
    for b in E.NEW_BACKBONES:
        add("Backbone (seed 42)", f"{b}", dict(backbone=b), [42])
    add("V4 (ChemBERTa-MLM)", "V3 with ChemBERTa-77M-MLM (V4 baseline)", dict(backbone="chemberta_mlm"), E.REPRO_SEEDS)
    for v in E.V4_VARIANTS:
        add("V4 (ChemBERTa-MLM)", core.VARIANTS[v][0], dict(backbone="chemberta_mlm", variant=v), E.REPRO_SEEDS)
    for v in E.V5_VARIANTS:
        add("V5 (pIC50-aware)", core.VARIANTS[v][0], dict(backbone="chemberta_mlm", variant=v), E.REPRO_SEEDS)
    for v in E.V6_VARIANTS:
        add("V6 (neighbour-anchored)", core.VARIANTS[v][0], dict(backbone="chemberta_mlm", variant=v), E.REPRO_SEEDS)
    for name in ["RF__ECFP", "XGBoost__ECFP", "RF__ECFP+MACCS+Desc", "XGBoost__ECFP+MACCS+Desc",
                 "LogReg__ECFP+MACCS+Desc", "RF__ECFPc2048+MACCS+Desc", "XGBoost__ECFPc2048+MACCS+Desc"]:
        r = st.baseline("random", name)
        if r is None:
            continue
        for proto in ("test", "cv"):
            row = {"group": "Classical baseline", "experiment": name.replace("__", " on "), "protocol": proto, "seeds": 1}
            for k in KEYS:
                row[NAMES.get(k, k)] = r[proto][k]
                row[NAMES.get(k, k) + " sd"] = np.nan
            rows.append(row)

    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(R.RES, "metrics_all.csv"), index=False)
    cols = ["Accuracy", "Balanced Acc", "Precision", "Recall", "F1", "ROC-AUC", "MCC"]

    def cell(r, c):
        return f"{r[c]:.4f}" if np.isnan(r[c + ' sd']) else f"{r[c]:.4f} ± {r[c + ' sd']:.4f}"
    L = ["# Final metrics - all experiments on the notebook split\n"]
    for proto, title in (("test", "Held-out test set (825 molecules): 5-model ensemble, OOF MCC-optimal threshold"),
                         ("cv", "5-fold cross-validation (out-of-fold, mean over folds, OOF max-F1 threshold)")):
        L += [f"## {title}\n", "| group | experiment | seeds | " + " | ".join(cols) + " |", "|" + "---|" * (3 + len(cols))]
        for _, r in t[t.protocol == proto].iterrows():
            L.append(f"| {r['group']} | {r['experiment']} | {r['seeds']} | " + " | ".join(cell(r, c) for c in cols) + " |")
        L.append("")
    L.append("Precision, recall and F1 refer to the active class. Recall of the inactive class = specificity "
             "(see REPORT.md). ± = sd over training seeds where 3 seeds exist.\n")
    with open(os.path.join(R.RES, "metrics_all.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
