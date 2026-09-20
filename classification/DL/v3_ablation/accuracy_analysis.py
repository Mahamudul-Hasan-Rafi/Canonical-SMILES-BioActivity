"""
Can the notebook split reach 95 % test accuracy? Analysis on stored predictions only.

  1. ceiling    : errors of the best models by distance of the measured pIC50 from the
                  class cut-off (active >= 6, inactive <= 5); the accuracy reachable if every
                  error outside the +-0.5 log-unit measurement-noise band were fixed
  2. stacking   : logistic-regression meta-model over the OOF predictions of all base models
                  (+ a Tanimoto kNN model aimed at activity cliffs), fitted on OOF only
  3. threshold  : MCC-optimal (notebook) vs accuracy-optimal threshold, both chosen on OOF
  4. selective  : accuracy on the molecules the model is most confident about
                  (accuracy-coverage curve; abstain on the rest)
The test set is never used for fitting or threshold selection. Writes results/accuracy_analysis.md.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data
import metrics as M
import report as R
import splits


def acc_thr(y, p):
    ths = np.linspace(0.05, 0.95, 901)
    return float(ths[int(np.argmax([accuracy_score(y, (p > t).astype(int)) for t in ths]))])


def knn_probs(fps, dev, folds, test, k=7):
    """Similarity-weighted kNN (Tanimoto, ECFP4-2048 chiral): OOF over the CV folds + test."""
    from rdkit import DataStructs
    y = data.load_df()["bioactivity"].values
    sim_dev = np.array([DataStructs.BulkTanimotoSimilarity(fps[i], [fps[j] for j in dev]) for i in dev])
    sim_te = np.array([DataStructs.BulkTanimotoSimilarity(fps[i], [fps[j] for j in dev]) for i in test])
    ydev = y[dev]

    def vote(S, pool):
        out = np.zeros(len(S))
        for r in range(len(S)):
            s = S[r, pool]
            top = np.argsort(-s)[:k]
            w = s[top] ** 3 + 1e-6
            out[r] = (w * ydev[pool][top]).sum() / w.sum()
        return out
    oof = np.zeros(len(dev))
    for tr, vl in folds:
        oof[vl] = vote(sim_dev[vl], tr)
    return oof, vote(sim_te, np.arange(len(dev)))


def main():
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    st = R.Store()
    df = data.load_df()
    s = splits.get_split("random", df)
    dev, te, folds = s["dev"], s["test"], s["folds"]
    y = df["bioactivity"].values
    y_oof, y_te = y[dev], y[te]

    # measured potency for the ceiling analysis
    raw = pd.read_excel(r"E:\ML\BioActivity\Dataset\bioactivity_dataset.xlsx")
    raw = raw[np.isfinite(raw.pIC50)]
    p_med = df["canonical_smiles"].map(raw.groupby("canonical_smiles").pIC50.median()).values
    margin = np.where(y == 1, p_med - 6, 5 - p_med)          # < 0: label contradicts its own pIC50

    # base models (OOF over the 5 folds, test = mean of fold models)
    bases = {}

    def add_seed_ens(name, **kw):
        rr = [r for r in (st.cv(seed=sd, **kw) for sd in R.SEEDS) if r is not None]
        if rr:
            bases[name] = (np.mean([r["oof"] for r in rr], 0), np.mean([r["ens"] for r in rr], 0))
    add_seed_ens("V3 MoLFormer (3 seeds)")
    add_seed_ens("V4-A (3 seeds)", backbone="chemberta_mlm", variant="fp_upgrade")
    add_seed_ens("V4-C (3 seeds)", backbone="chemberta_mlm", variant="graph_branch")
    add_seed_ens("V3 ChemBERTa-MLM (3 seeds)", backbone="chemberta_mlm")
    add_seed_ens("V5-MT (3 seeds)", backbone="chemberta_mlm", variant="graph_mt")
    add_seed_ens("V5-REG (3 seeds)", backbone="chemberta_mlm", variant="graph_reg")
    add_seed_ens("V6-K (3 seeds)", backbone="chemberta_mlm", variant="graph_mt_knn")
    add_seed_ens("V6-R (3 seeds)", backbone="chemberta_mlm", variant="graph_mt_delta")
    add_seed_ens("V7 (3 seeds)", backbone="chemberta_mlm", variant="graph_mt_delta_res")
    for b in ["XGBoost__ECFPc2048+MACCS+Desc", "RF__ECFPc2048+MACCS+Desc", "XGBoost__ECFP+MACCS+Desc",
              "RF__ECFP+MACCS+Desc"]:
        z = np.load(os.path.join(R.RES, "baselines", "random", b + ".npz"))
        bases[b.replace("__", " on ")] = (z["oof"], z["test_probs"].mean(0))
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(x)) for x in df["canonical_smiles"]]
    bases["kNN Tanimoto (k=7)"] = knn_probs(fps, dev, folds, te)

    # stacking: logistic regression on logit(OOF); its own OOF via the same folds
    def logit(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))
    names = list(bases)
    Xo = np.column_stack([logit(bases[n][0]) for n in names])
    Xt = np.column_stack([logit(bases[n][1]) for n in names])
    st_oof = np.zeros(len(dev))
    for tr, vl in folds:
        st_oof[vl] = LogisticRegression(C=1.0, max_iter=2000).fit(Xo[tr], y_oof[tr]).predict_proba(Xo[vl])[:, 1]
    meta = LogisticRegression(C=1.0, max_iter=2000).fit(Xo, y_oof)
    bases["STACK: logistic regression over all models"] = (st_oof, meta.predict_proba(Xt)[:, 1])
    v4d = (0.5 * bases["V4-C (3 seeds)"][0] + 0.5 * bases["XGBoost on ECFPc2048+MACCS+Desc"][0],
           0.5 * bases["V4-C (3 seeds)"][1] + 0.5 * bases["XGBoost on ECFPc2048+MACCS+Desc"][1])
    bases["V4-D (V4-C ens + XGBoost, 0.5/0.5)"] = v4d

    L = ["# Can we reach 95 % test accuracy? (notebook split, 825 test molecules)\n",
         "Thresholds are chosen on out-of-fold predictions only; the test set is never used for fitting.\n",
         "## Accuracy by model and threshold rule\n",
         "| model | OOF AUC | test AUC | test acc (MCC-opt thr, notebook) | test acc (accuracy-opt thr) | test errors | "
         "errors with pIC50 within 0.5 of cut-off | ceiling if all other errors fixed |", "|" + "---|" * 8]
    best = None
    for n, (o, t) in bases.items():
        tm, ta = M.mcc_thr(y_oof, o), acc_thr(y_oof, o)
        a_m, a_a = accuracy_score(y_te, t > tm), accuracy_score(y_te, t > ta)
        err = (t > ta).astype(int) != y_te
        noise = err & (margin[te] < 0.5)
        ceil = 1 - noise.sum() / len(te)
        L.append(f"| {n} | {roc_auc_score(y_oof, o):.4f} | {roc_auc_score(y_te, t):.4f} | {a_m:.4f} | {a_a:.4f} | "
                 f"{err.sum()} | {noise.sum()} | {ceil:.4f} |")
        if best is None or roc_auc_score(y_oof, o) > best[1]:
            best = (n, roc_auc_score(y_oof, o), o, t, ta)
    w = dict(zip(names, meta.coef_[0]))
    L += ["", "Stacker weights (logit scale): " + ", ".join(f"{k} {v:+.2f}" for k, v in w.items()) + "\n"]

    # selective prediction on the best-OOF model
    n, _, o, t, ta = best
    conf = np.abs(t - ta)
    L += [f"## Selective prediction: {n} (abstain on the least confident molecules)\n",
          "| coverage | molecules predicted | test accuracy | errors | abstained molecules within 0.5 log of cut-off |",
          "|---|---|---|---|---|"]
    order = np.argsort(-conf)
    for cov in [1.0, 0.95, 0.9, 0.85, 0.8, 0.75]:
        k = int(round(cov * len(te)))
        keep = order[:k]
        drop = order[k:]
        acc = accuracy_score(y_te[keep], t[keep] > ta)
        near = (margin[te][drop] < 0.5).mean() if len(drop) else float("nan")
        L.append(f"| {cov:.0%} | {k} | {acc:.4f} | {int(((t[keep] > ta).astype(int) != y_te[keep]).sum())} | "
                 f"{near:.0%} |" if len(drop) else f"| {cov:.0%} | {k} | {acc:.4f} | "
                 f"{int(((t[keep] > ta).astype(int) != y_te[keep]).sum())} | — |")
    # abstention thresholds must come from OOF to be legitimate: report the OOF-calibrated version too
    conf_oof = np.abs(o - ta)
    L += ["", "OOF-calibrated rule (legitimate for deployment): confidence cut-off chosen on OOF so that 90 % of "
          "OOF molecules are predicted; applied unchanged to test:"]
    cut = np.quantile(conf_oof, 0.10)
    keep = conf >= cut
    L.append(f"- test coverage {keep.mean():.1%}, test accuracy on predicted molecules "
             f"{accuracy_score(y_te[keep], t[keep] > ta):.4f} (OOF accuracy at the same rule "
             f"{accuracy_score(y_oof[conf_oof >= cut], o[conf_oof >= cut] > ta):.4f})\n")
    with open(os.path.join(R.RES, "accuracy_analysis.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
