"""
MoleculeACE's activity-cliff protocol (van Tilborg et al., JCIM 2022) applied to every model
in this study.

The benchmark's real contribution is not its model zoo - those are SVM / RF / GBM / KNN and the
usual graph networks, all of which we already run - but its evaluation: a model is judged on
activity-cliff compounds separately from the rest, because global RMSE hides failure exactly
where medicinal chemistry needs accuracy.

Cliff definition, as in the paper: two molecules form an activity-cliff pair when their potency
differs by at least 10-fold (1 log unit) AND they are similar by ANY of
  * ECFP4 Tanimoto >= 0.9            (substructure similarity)
  * Murcko scaffold ECFP Tanimoto >= 0.9
  * SMILES Levenshtein similarity >= 0.9
A molecule is a cliff compound if it belongs to at least one such pair. Pairs are formed within
the whole dataset, so the flag is a property of the chemistry, not of any split or model.

Reports RMSE and RMSE_cliff for every potency model, and the classification error rate on cliff
compounds, on both splits. Writes results/moleculeace_<split>.md.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core
import data
import metrics as M
import report as R

SIM_CUT = 0.9
POT_CUT = 1.0
CACHE = os.path.join(R.RES, "cache", "moleculeace_cliffs.npz")


def tanimoto(X):
    """All-pairs Tanimoto for binary rows, in float32."""
    X = np.asarray(X, dtype=np.float32)
    inter = X @ X.T
    n = X.sum(1)
    union = n[:, None] + n[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        T = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
    return T.astype(np.float32)


def cliff_flags(smiles, pic):
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        if len(z["cliff"]) == len(smiles):
            return z["cliff"].astype(bool)
    from rapidfuzz import fuzz, process
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

    def fps(smis):
        out = np.zeros((len(smis), 1024), np.uint8)
        for i, s in enumerate(smis):
            m = Chem.MolFromSmiles(s) if s else None
            if m is not None:
                out[i] = np.frombuffer(bytes(gen.GetFingerprint(m).ToBitString(), "ascii"), "u1") - ord("0")
        return out

    scaf = []
    for s in smiles:
        try:
            scaf.append(MurckoScaffold.MurckoScaffoldSmiles(smiles=s, includeChirality=False))
        except Exception:
            scaf.append("")
    big_pot = np.abs(pic[:, None] - pic[None, :]) >= POT_CUT
    sim = tanimoto(fps(smiles)) >= SIM_CUT
    sim |= tanimoto(fps(scaf)) >= SIM_CUT
    lev = process.cdist(smiles, smiles, scorer=fuzz.ratio, workers=-1) / 100.0
    sim |= lev >= SIM_CUT
    np.fill_diagonal(sim, False)
    cliff = (big_pot & sim).any(1)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, cliff=cliff)
    return cliff


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="random", choices=["random", "scaffold"])
    args = ap.parse_args()
    sp = args.split
    st = R.Store()
    smiles = list(data.get_features(st.df)["smiles"])
    pic = data.get_pic50(smiles)
    cliff = cliff_flags(smiles, pic)
    dev, te = st.splits[sp]["dev"], st.splits[sp]["test"]
    y, yt = st.y[dev], st.y[te]
    c_dev, c_te = cliff[dev], cliff[te]
    p_dev, p_te = pic[dev], pic[te]

    def npz(nm):
        p = os.path.join(R.RES, "baselines", sp, nm + ".npz")
        return np.load(p) if os.path.exists(p) else None

    def deep(**kw):
        rr = [r for r in (st.cv(split=sp, seed=s, **kw) for s in R.SEEDS) if r is not None]
        return (np.mean([r["oof"] for r in rr], 0), np.mean([r["ens"] for r in rr], 0)) if rr else None

    # potency models
    pot = {}
    r = deep(backbone="chemberta_mlm", variant="reg_first_delta")
    if r:
        pot["V8-R2 (ours)"] = (core.probs_to_pic50(r[0]), core.probs_to_pic50(r[1]))
    for nm, f in (("LGBMReg (ours)", "LGBMReg__ECFPc2048+MACCS+Desc"),
                  ("LGBMReg+RDKit (ours)", "LGBMReg__ECFPc2048+MACCS+RDKit"),
                  ("LGBMReg+Mordred (ours)", "LGBMReg__ECFPc2048+MACCS+Mordred"),
                  ("Chemprop-reg [Heid 2024]", "Chemprop-reg__graph"),
                  ("CheMeleon-reg [Burns 2025]", "CheMeleon-reg__graph")):
        z = npz(f)
        if z is not None:
            pot[nm] = (z["oof_pic50"], z["test_pic50"].mean(0))
    if "LGBMReg (ours)" in pot and "V8-R2 (ours)" in pot:
        pot["HYBRID potency (ours)"] = (np.mean([pot["LGBMReg (ours)"][0], pot["V8-R2 (ours)"][0]], 0),
                                        np.mean([pot["LGBMReg (ours)"][1], pot["V8-R2 (ours)"][1]], 0))

    L = [f"# MoleculeACE activity-cliff protocol ({sp} split)\n",
         f"Cliff compounds: potency differing by >= {POT_CUT:.0f} log unit from a molecule that is at least "
         f"{SIM_CUT:.0%} similar by ECFP Tanimoto, Murcko-scaffold Tanimoto or SMILES Levenshtein "
         "similarity (van Tilborg et al., JCIM 2022).\n",
         f"Of {len(cliff):,} molecules, {cliff.sum():,} ({cliff.mean():.1%}) are cliff compounds; "
         f"{c_dev.sum():,} of {len(dev):,} development and {c_te.sum()} of {len(te)} test molecules.\n",
         "## Potency: global vs cliff RMSE\n",
         "| model | OOF RMSE | OOF RMSE_cliff | penalty | test RMSE | test RMSE_cliff |",
         "|---|---|---|---|---|---|"]
    for k in sorted(pot, key=lambda k: rmse(pot[k][0][c_dev] - p_dev[c_dev])):
        e, et = pot[k][0] - p_dev, pot[k][1] - p_te
        L.append(f"| {k} | {rmse(e):.4f} | **{rmse(e[c_dev]):.4f}** | {rmse(e[c_dev]) - rmse(e):+.4f} | "
                 f"{rmse(et):.4f} | {rmse(et[c_te]):.4f} |")

    # classification error on cliff compounds
    cls = {}
    for nm, kw in (("V5-MT (ours)", dict(backbone="chemberta_mlm", variant="graph_mt")),
                   ("V3 (published)", {})):
        rr = deep(**kw)
        if rr:
            cls[nm] = rr
    for nm, f in (("Voting (ours)", "Voting__ECFP+Desc"), ("Chemprop [Heid 2024]", "Chemprop__graph"),
                  ("CheMeleon [Burns 2025]", "CheMeleon__graph")):
        z = npz(f)
        if z is not None:
            cls[nm] = (z["oof"], z["test_probs"].mean(0))
    if cls:
        L += ["", "## Classification error rate: all compounds vs cliff compounds (out-of-fold)\n",
              "| model | error (all) | error (cliff) | penalty |", "|---|---|---|---|"]
        for k, v in sorted(cls.items(), key=lambda kv: (kv[1][0] > M.mcc_thr(y, kv[1][0])).astype(int)[c_dev].mean()):
            thr = M.mcc_thr(y, v[0])
            err = (v[0] > thr).astype(int) != y
            L.append(f"| {k} | {err.mean():.4f} | **{err[c_dev].mean():.4f}** | {err[c_dev].mean() - err.mean():+.4f} |")
    L.append("\nThe penalty column is what MoleculeACE exists to expose: the cost a model pays exactly "
             "where the structure-activity relationship is discontinuous.\n")
    with open(os.path.join(R.RES, f"moleculeace_{sp}.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
