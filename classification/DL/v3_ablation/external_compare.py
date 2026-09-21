"""
Head-to-head against published methods, run on our dataset, our folds and our held-out test set.

Comparators (each run with its authors' own implementation):
  Chemprop v2 (D-MPNN)   Heid et al., JCIM 2024          -- the standard graph baseline
  CheMeleon              Burns & Green, 2025             -- descriptor-based foundation model
  ChemBERTa-2            Ahmad et al., 2022              -- our V4 backbone, already in the study

Everything is scored under the protocol used throughout: classification at each model's own
out-of-fold MCC-optimal threshold, potency in log units. Differences against our final system
are tested with McNemar's exact test and paired DeLong (classification) and a paired bootstrap
on the RMSE (potency).

Writes results/external_<split>.md
"""
import argparse
import os
import sys

import numpy as np
from scipy.stats import binomtest, rankdata
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
import data
import metrics as M
import report as R

B = 10000


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def boot(ea, eb, rng):
    idx = rng.integers(0, len(ea), size=(B, len(ea)))
    d = np.sqrt(np.square(ea)[idx].mean(1)) - np.sqrt(np.square(eb)[idx].mean(1))
    p = 2 * min((d >= 0).mean(), (d <= 0).mean())
    return rmse(ea) - rmse(eb), np.percentile(d, [2.5, 97.5]), min(1.0, float(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="random", choices=["random", "scaffold"])
    args = ap.parse_args()
    sp = args.split
    st = R.Store()
    dev, te = st.splits[sp]["dev"], st.splits[sp]["test"]
    y, yt = st.y[dev], st.y[te]
    pic = data.get_pic50(st.df["canonical_smiles"].tolist())
    p_dev, p_te = pic[dev], pic[te]

    def npz(nm):
        p = os.path.join(R.RES, "baselines", sp, nm + ".npz")
        return np.load(p) if os.path.exists(p) else None

    def deep(**kw):
        rr = [r for r in (st.cv(split=sp, seed=s, **kw) for s in R.SEEDS) if r is not None]
        return (np.mean([r["oof"] for r in rr], 0), np.mean([r["ens"] for r in rr], 0)) if rr else None

    # ---------------- classification ----------------
    cls, src = {}, {}
    for nm, kw in (("V3 (published)", {}), ("V5-MT (ours)", dict(backbone="chemberta_mlm", variant="graph_mt")),
                   ("V8-R2 (ours)", dict(backbone="chemberta_mlm", variant="reg_first_delta"))):
        r = deep(**kw)
        if r:
            cls[nm], src[nm] = r, "ours"
    for nm, f in (("Voting (ours)", "Voting__ECFP+Desc"), ("LightGBM (ours)", "LightGBM__ECFP+Desc"),
                  ("Chemprop [Heid 2024]", "Chemprop__graph"), ("CheMeleon [Burns 2025]", "CheMeleon__graph")):
        z = npz(f)
        if z is not None:
            cls[nm] = (z["oof"], z["test_probs"].mean(0))
            src[nm] = "external" if "[" in nm else "ours"
    hyb = ["Voting (ours)", "Voting (ours)"] + (["V8-R2 (ours)"] if sp == "random" else ["V5-MT (ours)"])
    if all(h in cls for h in hyb):
        cls["HYBRID (ours)"] = (np.mean([rankdata(cls[h][0]) / len(y) for h in hyb], 0),
                                np.mean([rankdata(cls[h][1]) / len(yt) for h in hyb], 0))
        src["HYBRID (ours)"] = "ours"
    thr = {k: M.mcc_thr(y, v[0]) for k, v in cls.items()}

    L = [f"# Against published methods, on our data ({sp} split)\n",
         f"Every model trained on the same 5 folds and scored on the same held-out test set "
         f"({len(y):,} development / {len(yt)} test molecules). Published methods use their authors' "
         "own implementations. Threshold = out-of-fold MCC-optimal for every model.\n",
         "## Classification\n",
         "| model | OOF acc | OOF MCC | OOF AUC | test acc | test MCC | test AUC | test errors |",
         "|---|---|---|---|---|---|---|---|"]
    for k in sorted(cls, key=lambda k: -matthews_corrcoef(y, (cls[k][0] > thr[k]).astype(int))):
        o, t = cls[k]
        po, pt = (o > thr[k]).astype(int), (t > thr[k]).astype(int)
        L.append(f"| {k} | {accuracy_score(y, po):.4f} | {matthews_corrcoef(y, po):.4f} | "
                 f"{roc_auc_score(y, o):.4f} | {accuracy_score(yt, pt):.4f} | {matthews_corrcoef(yt, pt):.4f} | "
                 f"{roc_auc_score(yt, t):.4f} | {int((pt != yt).sum())} |")

    # ---------------- potency ----------------
    pot = {}
    r = deep(backbone="chemberta_mlm", variant="reg_first_delta")
    if r:
        pot["V8-R2 (ours)"] = (core.probs_to_pic50(r[0]), core.probs_to_pic50(r[1]))
    for nm, f in (("LGBMReg (ours)", "LGBMReg__ECFPc2048+MACCS+Desc"),
                  ("Chemprop-reg [Heid 2024]", "Chemprop-reg__graph"),
                  ("CheMeleon-reg [Burns 2025]", "CheMeleon-reg__graph")):
        z = npz(f)
        if z is not None:
            pot[nm] = (z["oof_pic50"], z["test_pic50"].mean(0))
    if "LGBMReg (ours)" in pot and "V8-R2 (ours)" in pot:
        pot["HYBRID potency (ours)"] = (np.mean([pot["LGBMReg (ours)"][0], pot["V8-R2 (ours)"][0]], 0),
                                        np.mean([pot["LGBMReg (ours)"][1], pot["V8-R2 (ours)"][1]], 0))
    L += ["", "## Potency (pIC50, log units)\n",
          "| model | OOF RMSE | OOF MAE | test RMSE | test MAE |", "|---|---|---|---|---|"]
    for k in sorted(pot, key=lambda k: rmse(pot[k][0] - p_dev)):
        e, et = pot[k][0] - p_dev, pot[k][1] - p_te
        L.append(f"| {k} | {rmse(e):.4f} | {np.abs(e).mean():.4f} | {rmse(et):.4f} | {np.abs(et).mean():.4f} |")

    # ---------------- significance vs the best external ----------------
    ext_c = [k for k in cls if "[" in k]
    ext_p = [k for k in pot if "[" in k]
    ref = "HYBRID (ours)" if "HYBRID (ours)" in cls else None
    if ref and ext_c:
        L += ["", f"## {ref} vs published methods\n",
              "| comparison | set | McNemar b / c | p | delta AUC | p |", "|---|---|---|---|---|---|"]
        for k in ext_c:
            for lab, pick, yy in (("OOF", 0, y), ("test", 1, yt)):
                ea = (cls[ref][pick] > thr[ref]).astype(int) != yy
                eb = (cls[k][pick] > thr[k]).astype(int) != yy
                n01, n10 = int((ea & ~eb).sum()), int((~ea & eb).sum())
                p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
                d, pd_ = M.delong(yy, cls[ref][pick], cls[k][pick])
                L.append(f"| ours vs {k} | {lab} | {n01} / {n10} | {p:.4f} | {d:+.4f} | {pd_:.4f} |")
        L.append("\nb = molecules only our model gets wrong, c = only theirs. Positive delta AUC = ours better.\n")
    if "HYBRID potency (ours)" in pot and ext_p:
        rng = np.random.default_rng(0)
        L += ["", "## Potency: ours vs published (paired bootstrap on OOF RMSE)\n",
              "| comparison | delta RMSE | 95 % CI | p |", "|---|---|---|---|"]
        for k in ext_p:
            o, ci, p = boot(pot["HYBRID potency (ours)"][0] - p_dev, pot[k][0] - p_dev, rng)
            L.append(f"| ours vs {k} | {o:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {p:.4f} |")
        L.append("\nNegative = our model is more precise.\n")

    with open(os.path.join(R.RES, f"external_{sp}.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
