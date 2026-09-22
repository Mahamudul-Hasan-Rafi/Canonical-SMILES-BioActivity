"""
Potency stacking over our own models only - no external model enters the ensemble.

The current potency hybrid is a 1:1 average of LGBMReg and V8-R2. This fits weights instead,
with non-negative least squares, and does it WITHOUT leakage: for every fold, the weights are
fitted on the out-of-fold predictions of the *other* four folds and applied to this fold. Test
predictions use the average of the five fold-weight vectors. So the reported OOF RMSE is an
honest out-of-sample number, not the in-sample fit that plain stacking on OOF would give.

Members: every potency model we have trained - the deep regressors (V5-REG, V8-R1, V8-R2) and
the gradient-boosting / forest regressors on each feature block, including the descriptor
variants that are individually weaker but may still add diversity.

Reports global and cliff RMSE against CheMeleon, with a paired bootstrap.
Writes results/stack_potency_<split>.md
"""
import argparse
import glob
import os
import sys

import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
import data
import moleculeace_cliffs as MC
import report as R

B = 10000


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="random", choices=["random", "scaffold"])
    args = ap.parse_args()
    sp = args.split
    st = R.Store()
    dev, te = st.splits[sp]["dev"], st.splits[sp]["test"]
    smiles = list(data.get_features(st.df)["smiles"])
    pic = data.get_pic50(smiles)
    p_dev, p_te = pic[dev], pic[te]
    cliff = MC.cliff_flags(smiles, pic)
    c_dev, c_te = cliff[dev], cliff[te]

    # fold membership of every development molecule
    fold_of = np.full(len(dev), -1)
    for f, (_, f_vl) in enumerate(st.splits[sp]["folds"]):
        fold_of[f_vl] = f

    members, ext = {}, {}
    for nm, kw in (("V5-REG", dict(backbone="chemberta_mlm", variant="graph_reg")),
                   ("V8-R1", dict(backbone="chemberta_mlm", variant="reg_first")),
                   ("V8-R2", dict(backbone="chemberta_mlm", variant="reg_first_delta")),
                   ("V8-R2-hpo", dict(backbone="chemberta_mlm", variant="reg_delta_hpo"))):
        rr = [r for r in (st.cv(split=sp, seed=s, **kw) for s in R.SEEDS) if r is not None]
        if rr:
            members[nm] = (core.probs_to_pic50(np.mean([r["oof"] for r in rr], 0)),
                           core.probs_to_pic50(np.mean([r["ens"] for r in rr], 0)))
    for path in sorted(glob.glob(os.path.join(R.RES, "baselines", sp, "*Reg__*.npz"))):
        z = np.load(path)
        if "oof_pic50" not in z:
            continue
        base = os.path.basename(path)[:-4]
        if base.lower().startswith(("chemeleon", "chemprop")):
            continue        # published comparators are never ensemble members (Windows glob is case-insensitive)
        members[base.replace("__", " on ")] = (z["oof_pic50"], z["test_pic50"].mean(0))
    for nm, f in (("CheMeleon-reg", "CheMeleon-reg__graph"), ("Chemprop-reg", "Chemprop-reg__graph")):
        p = os.path.join(R.RES, "baselines", sp, f + ".npz")
        if os.path.exists(p):
            z = np.load(p)
            ext[nm] = (z["oof_pic50"], z["test_pic50"].mean(0))

    names = sorted(members)
    A_oof = np.column_stack([members[n][0] for n in names])
    A_te = np.column_stack([members[n][1] for n in names])

    # leakage-free stacking: weights for each fold come from the other folds only
    stack_oof = np.zeros(len(dev))
    W = []
    for f in range(5):
        tr = fold_of != f
        w, _ = nnls(A_oof[tr], p_dev[tr])
        W.append(w)
        stack_oof[~tr] = A_oof[~tr] @ w
    w_mean = np.mean(W, 0)
    stack_te = A_te @ w_mean

    simple = np.mean([members[n][0] for n in names if n in ("V8-R2",)] +
                     [members[n][0] for n in names if n.startswith("LGBMReg on ECFPc2048+MACCS+Desc")], 0)
    L = [f"# Potency stacking over our own models ({sp} split)\n",
         f"{len(names)} member models. Weights fitted per fold by non-negative least squares on the "
         "other four folds' out-of-fold predictions, so no molecule contributes to the weights used "
         "to predict it.\n",
         "## Fitted weights (mean over folds, non-zero only)\n",
         "| member | weight |", "|---|---|"]
    for n, w in sorted(zip(names, w_mean), key=lambda kv: -kv[1]):
        if w > 1e-4:
            L.append(f"| {n} | {w:.3f} |")
    L += ["", "## Result\n",
          "| model | OOF RMSE | OOF RMSE_cliff | test RMSE | test RMSE_cliff |", "|---|---|---|---|---|"]
    rows = [("STACK (ours, all own models)", stack_oof, stack_te)]
    if simple.shape == p_dev.shape:
        rows.append(("HYBRID 1:1 (previous)", simple, np.mean(
            [members[n][1] for n in names if n in ("V8-R2",)] +
            [members[n][1] for n in names if n.startswith("LGBMReg on ECFPc2048+MACCS+Desc")], 0)))
    for n in ("V8-R2", "LGBMReg on ECFPc2048+MACCS+Desc"):
        if n in members:
            rows.append((n + " (single)", members[n][0], members[n][1]))
    for n, v in ext.items():
        rows.append((n + " [published]", v[0], v[1]))
    for nm, o, t in rows:
        eo, et = o - p_dev, t - p_te
        L.append(f"| {nm} | **{rmse(eo):.4f}** | {rmse(eo[c_dev]):.4f} | {rmse(et):.4f} | {rmse(et[c_te]):.4f} |")

    if "CheMeleon-reg" in ext:
        rng = np.random.default_rng(0)
        ea, eb = stack_oof - p_dev, ext["CheMeleon-reg"][0] - p_dev
        idx = rng.integers(0, len(ea), size=(B, len(ea)))
        d = np.sqrt(np.square(ea)[idx].mean(1)) - np.sqrt(np.square(eb)[idx].mean(1))
        p = 2 * min((d >= 0).mean(), (d <= 0).mean())
        L += ["", "## Stack vs CheMeleon (paired bootstrap on OOF)\n",
              f"delta RMSE {rmse(ea) - rmse(eb):+.4f}  95 % CI "
              f"[{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]  p = {min(1.0, p):.4f}",
              "", "Negative = our stack is more precise.\n"]
    with open(os.path.join(R.RES, f"stack_potency_{sp}.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
