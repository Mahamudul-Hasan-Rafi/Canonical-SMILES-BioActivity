"""
V8 regression-first study: how precisely can the network predict pIC50, and what does that
precision buy in classification accuracy?

Primary metric: out-of-fold RMSE in pIC50 units on the 4,669 development molecules.
Secondary: MAE, Spearman, RMSE inside the near-cut-off band and on activity cliffs, and the
classification metrics obtained by thresholding the predicted potency (notebook OOF-MCC rule).
Differences in RMSE are tested with a paired bootstrap over molecules (same molecules, both
models). The held-out test set is reported once, at the end.
Writes results/reg_report.md.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
import data
import metrics as M
import report as R

MODELS = {"V5-REG (previous protocol)": dict(backbone="chemberta_mlm", variant="graph_reg"),
          "V8-R1 regression-first": dict(backbone="chemberta_mlm", variant="reg_first"),
          "V8-R2 + anchored delta": dict(backbone="chemberta_mlm", variant="reg_first_delta")}
B = 10000


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def boot(ea, eb, rng):
    """Paired bootstrap on the RMSE difference (a - b) over molecules."""
    n = len(ea)
    idx = rng.integers(0, n, size=(B, n))
    d = np.sqrt(np.square(ea)[idx].mean(1)) - np.sqrt(np.square(eb)[idx].mean(1))
    obs = rmse(ea) - rmse(eb)
    p = 2 * min((d >= 0).mean(), (d <= 0).mean())
    return obs, np.percentile(d, [2.5, 97.5]), min(1.0, float(p))


def main():
    st = R.Store()
    dev, test_idx = st.splits["random"]["dev"], st.splits["random"]["test"]
    pic = data.get_pic50(st.df["canonical_smiles"].tolist())
    p_dev, p_te = pic[dev], pic[test_idx]
    y_dev = st.y[dev]
    hc = pd.read_csv(os.path.join(R.RES, "hard_cases.csv"))
    band = ((hc.margin >= 0) & (hc.margin < 0.5)).values
    cliff = hc["cliff"].values

    got = {}
    for name, kw in MODELS.items():
        runs = [st.cv(seed=s, **kw) for s in R.SEEDS]
        runs = [r for r in runs if r is not None]
        if not runs:
            continue
        oof = np.mean([core.probs_to_pic50(r["oof"]) for r in runs], 0)
        te = np.mean([core.probs_to_pic50(r["ens"]) for r in runs], 0)
        per_seed = [rmse(core.probs_to_pic50(r["oof"]) - p_dev) for r in runs]
        got[name] = dict(oof=oof, te=te, y_te=runs[0]["test_y"], n=len(runs), per_seed=per_seed,
                         oof_prob=np.mean([r["oof"] for r in runs], 0),
                         te_prob=np.mean([r["ens"] for r in runs], 0))
    if not got:
        print("no regression runs found")
        return

    L = ["# V8: regression-first models - potency precision (out-of-fold, 4,669 molecules)\n",
         "All models predict pIC50; the class decision is the predicted potency thresholded by the "
         "notebook's OOF-MCC rule. Ensembles average the predicted pIC50 over training seeds.\n",
         "| model | seeds | OOF RMSE | per-seed RMSE | MAE | Spearman | RMSE near cut-off | RMSE on cliffs |",
         "|---|---|---|---|---|---|---|---|"]
    for k, g in got.items():
        e = g["oof"] - p_dev
        L.append(f"| {k} | {g['n']} | **{rmse(e):.4f}** | " +
                 ", ".join(f"{v:.3f}" for v in g["per_seed"]) +
                 f" | {np.abs(e).mean():.4f} | {spearmanr(g['oof'], p_dev).statistic:.4f} | "
                 f"{rmse(e[band]):.4f} ({band.sum()}) | {rmse(e[cliff]):.4f} ({cliff.sum()}) |")

    ref = "V5-REG (previous protocol)"
    if ref in got and len(got) > 1:
        rng = np.random.default_rng(0)
        L += ["", f"## Paired bootstrap on the RMSE difference vs {ref} (10,000 resamples)\n",
              "| model | delta RMSE | 95 % CI | p |", "|---|---|---|---|"]
        for k, g in got.items():
            if k == ref:
                continue
            o, ci, p = boot(g["oof"] - p_dev, got[ref]["oof"] - p_dev, rng)
            L.append(f"| {k} | {o:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {p:.4f} |")
        L.append("\nNegative = more precise than the previous protocol.\n")

    L += ["", "## Classification obtained from the predicted potency (out-of-fold)\n",
          "| model | ROC-AUC | Accuracy | Balanced Acc | MCC |", "|---|---|---|---|---|"]
    thr = {}
    for k, g in got.items():
        thr[k] = M.mcc_thr(y_dev, g["oof_prob"])
        pred = (g["oof_prob"] > thr[k]).astype(int)
        L.append(f"| {k} | {roc_auc_score(y_dev, g['oof_prob']):.4f} | {accuracy_score(y_dev, pred):.4f} | "
                 f"{M.thr_metrics(y_dev, g['oof_prob'], thr[k])['Balanced Acc']:.4f} | "
                 f"{matthews_corrcoef(y_dev, pred):.4f} |")

    L += ["", "## Held-out test set (825 molecules, reported once)\n",
          "| model | test RMSE | test MAE | test ROC-AUC | test Accuracy | errors |", "|---|---|---|---|---|---|"]
    for k, g in got.items():
        e = g["te"] - p_te
        pred = (g["te_prob"] > thr[k]).astype(int)
        L.append(f"| {k} | {rmse(e):.4f} | {np.abs(e).mean():.4f} | {roc_auc_score(g['y_te'], g['te_prob']):.4f} | "
                 f"{accuracy_score(g['y_te'], pred):.4f} | {int((pred != g['y_te']).sum())} |")

    best = min(got, key=lambda k: rmse(got[k]["oof"] - p_dev))
    L += ["", "## Does the accuracy follow the potency precision?\n",
          "A molecule is classified correctly when its predicted potency lands on the correct side of the "
          "model's own decision threshold t. With a normal potency error of the measured RMSE, "
          "P(correct) = Phi(s / RMSE), where s is the signed distance of the MEASURED pIC50 from t, "
          "positive when the measurement agrees with the label. Molecules whose measurement contradicts "
          "their label enter with a negative s, so the projection accounts for them.\n",
          "| model | OOF RMSE | threshold t (pIC50) | projected accuracy | observed OOF accuracy |",
          "|---|---|---|---|---|"]
    from scipy.stats import norm
    for k, g in got.items():
        s = rmse(g["oof"] - p_dev)
        t = float(core.probs_to_pic50(thr[k]))
        sd = np.where(y_dev == 1, p_dev - t, t - p_dev)
        proj = float(np.mean(norm.cdf(sd / s)))
        acc = accuracy_score(y_dev, (g["oof_prob"] > thr[k]).astype(int))
        L.append(f"| {k} | {s:.4f} | {t:.3f} | {proj:.4f} | {acc:.4f} |")
    L.append(f"\nBest potency model: **{best}** (OOF RMSE {rmse(got[best]['oof'] - p_dev):.4f}).\n")

    with open(os.path.join(R.RES, "reg_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
