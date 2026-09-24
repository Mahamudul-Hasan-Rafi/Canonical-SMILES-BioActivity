"""
Every experiment in the study, in one table.

Scans results/store for every trained configuration and results/baselines for every classical and
published-comparator model, scores them all under the same protocol (threshold = out-of-fold
MCC-optimal), and adds the proposed hybrid. Multi-seed configurations are reported as the ensemble
of their seeds, matching how the proposed model is built.

Writes results/all_metrics.{csv,md}
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
import data
import metrics as M
import report as R

KEYS = ["Accuracy", "Balanced Acc", "Precision", "Sensitivity", "Specificity", "F1",
        "ROC-AUC", "AUPRC", "MCC"]
NICE = {"Sensitivity": "Recall", "Specificity": "Specificity"}


def main():
    st = R.Store()
    pic = data.get_pic50(list(data.get_features(st.df)["smiles"]))
    rows = []

    # what has actually been trained, per split
    combos = defaultdict(set)
    for f in glob.glob(os.path.join(R.RES, "store", "*.json")):
        j = json.load(open(f))["job"]
        if j["protocol"] != "cv":
            continue
        combos[(j["split"], j["backbone"], j["hp_set"], j["variant"])].add(j["seed"])

    for sp in ("random", "scaffold"):
        dev, te = st.splits[sp]["dev"], st.splits[sp]["test"]
        y, yt = st.y[dev], st.y[te]
        p_dev = pic[dev]

        def add(group, name, oof, test, n_seeds, rmse_oof=None):
            thr = M.mcc_thr(y, oof)
            r = {"split": sp, "group": group, "model": name, "seeds": n_seeds}
            for proto, p, yy in (("OOF", oof, y), ("test", test, yt)):
                m = M.thr_metrics(yy, p, thr)
                for k in KEYS:
                    r[f"{proto} {NICE.get(k, k)}"] = m[k]
                r[f"{proto} errors"] = int(((p > thr).astype(int) != yy).sum())
            r["potency RMSE (OOF)"] = rmse_oof
            rows.append(r)

        # ---- deep models ----
        for (s2, bb, hp, var), seeds in sorted(combos.items()):
            if s2 != sp:
                continue
            rr = [r for r in (st.cv(split=sp, seed=sd, backbone=bb, hp_set=hp, variant=var)
                              for sd in sorted(seeds)) if r]
            if not rr:
                continue
            oof = np.mean([r["oof"] for r in rr], 0)
            test = np.mean([r["ens"] for r in rr], 0)
            rm = None
            if core.VARIANTS.get(var, ("", {}))[1].get("task") == "reg":
                rm = float(np.sqrt(np.mean((core.probs_to_pic50(oof) - p_dev) ** 2)))
            tag = f"{var} [{bb}]" + ("" if hp == "tuned" else f" ({hp})")
            grp = ("deep: proposed component" if var in ("mt_w10_ns", "graph_mt") else
                   "deep: ablation" if var.startswith("mt_") else
                   "deep: regression" if rm is not None else "deep: other")
            add(grp, tag, oof, test, len(rr), rm)

        # ---- classical + published ----
        for path in sorted(glob.glob(os.path.join(R.RES, "baselines", sp, "*.npz"))):
            base = os.path.basename(path)[:-4]
            z = np.load(path)
            if "oof" not in z:
                continue
            rm = (float(np.sqrt(np.mean((z["oof_pic50"] - p_dev) ** 2))) if "oof_pic50" in z else None)
            grp = ("published comparator" if base.lower().startswith(("chemprop", "chemeleon"))
                   else "classical")
            add(grp, base.replace("__", " on "), z["oof"], z["test_probs"].mean(0), 1, rm)

        # ---- the proposed hybrid ----
        zc = np.load(os.path.join(R.RES, "baselines", sp, "LightGBM__ECFP+Desc.npz"))
        LG = (zc["oof"], zc["test_probs"].mean(0))
        rr = [r for r in (st.cv(split=sp, seed=sd, backbone="chemberta_mlm", variant="mt_w10_ns")
                          for sd in R.SEEDS) if r]
        if rr:
            D = (np.mean([r["oof"] for r in rr], 0), np.mean([r["ens"] for r in rr], 0))
            add("PROPOSED", "LightGBM + V11c (2:1)",
                (2 / 3) * rankdata(LG[0]) / len(y) + (1 / 3) * rankdata(D[0]) / len(y),
                (2 / 3) * rankdata(LG[1]) / len(yt) + (1 / 3) * rankdata(D[1]) / len(yt), 3,
                float(np.sqrt(np.mean((np.load(os.path.join(
                    R.RES, "baselines", sp, "LGBMReg__ECFPc2048+MACCS+Desc.npz"))["oof_pic50"] - p_dev) ** 2))))

    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(R.RES, "all_metrics.csv"), index=False)
    order = ["PROPOSED", "deep: proposed component", "classical", "published comparator",
             "deep: regression", "deep: ablation", "deep: other"]
    cols = ["Accuracy", "Balanced Acc", "Precision", "Recall", "Specificity", "F1", "ROC-AUC", "MCC"]
    L = ["# Every experiment, one table\n",
         "Threshold = out-of-fold MCC-optimal for every model. Multi-seed configurations are the "
         "ensemble of their seeds. Potency RMSE is shown only for models that predict pIC50.\n"]
    for sp in ("random", "scaffold"):
        sub = t[t.split == sp]
        L += [f"## {sp} split\n"]
        for proto in ("OOF", "test"):
            L += [f"### {proto}\n",
                  "| group | model | seeds | " + " | ".join(cols) + " | errors | potency RMSE |",
                  "|" + "---|" * (5 + len(cols))]
            for g in order:
                gs = sub[sub.group == g].sort_values(f"{proto} MCC", ascending=False)
                for _, r in gs.iterrows():
                    rm = "" if pd.isna(r["potency RMSE (OOF)"]) else f"{r['potency RMSE (OOF)']:.4f}"
                    L.append(f"| {g} | {r['model']} | {r['seeds']} | " +
                             " | ".join(f"{r[f'{proto} {c}']:.4f}" for c in cols) +
                             f" | {int(r[f'{proto} errors'])} | {rm} |")
            L.append("")
    with open(os.path.join(R.RES, "all_metrics.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print(f"{len(t)} rows written to results/all_metrics.md")
    print(t.groupby(["split", "group"]).size().to_string())


if __name__ == "__main__":
    main()
