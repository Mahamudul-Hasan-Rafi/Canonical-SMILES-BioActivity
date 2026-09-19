"""
Phase 1 of the hard-case research: forensic taxonomy of the best deep model's errors.

Uses ONLY out-of-fold predictions on the 4,669 development molecules (the held-out test set
is not looked at), so architecture decisions made from this analysis cannot leak test info.

For each development molecule (its "training side" = the 4 other CV folds):
  margin     : measured median pIC50 distance from the class cut-off (active >= 6, inactive <= 5)
  noise band : 0 <= margin < 0.5 ; contradiction : margin < 0
  cliff      : nearest training-side neighbour (ECFP4 Tanimoto >= 0.8) has the opposite label
  stereo twin: a training-side molecule with the same non-stereo graph
  OOD        : max Tanimoto to training side < 0.5
Also asks HOW the model fails: on cliff errors, does it copy the neighbour's label? In the
noise band, does predicted potency (V5-REG) still rank molecules correctly?
Writes results/hard_cases.md and results/hard_cases.csv.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data
import metrics as M
import report as R
import splits


def main():
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    st = R.Store()
    df = data.load_df()
    s = splits.get_split("random", df)
    dev, folds = s["dev"], s["folds"]
    y = df["bioactivity"].values.astype(int)
    pic = data.get_pic50(df["canonical_smiles"].values)
    yd, pd_ = y[dev], pic[dev]
    fold_of = np.zeros(len(dev), int)
    for f, (_, vl) in enumerate(folds):
        fold_of[vl] = f

    def ens(**kw):
        rr = [st.cv(seed=sd, **kw) for sd in R.SEEDS]
        return np.mean([r["oof"] for r in rr], 0)
    p_mt = ens(backbone="chemberta_mlm", variant="graph_mt")
    p_reg = ens(backbone="chemberta_mlm", variant="graph_reg")
    p_v3 = ens()
    thr = M.mcc_thr(yd, p_mt)
    pred = (p_mt > thr).astype(int)
    err = pred != yd

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    mols = [Chem.MolFromSmiles(df["canonical_smiles"][i]) for i in dev]
    fps = [gen.GetFingerprint(m) for m in mols]
    flat = [Chem.MolToSmiles(m, isomericSmiles=False) for m in mols]
    nn_sim, nn_lab, nn_pic, stereo = np.zeros(len(dev)), np.zeros(len(dev), int), np.zeros(len(dev)), np.zeros(len(dev), bool)
    for i in range(len(dev)):
        pool = np.where(fold_of != fold_of[i])[0]
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fps[i], [fps[j] for j in pool]))
        j = pool[int(np.argmax(sims))]
        nn_sim[i], nn_lab[i], nn_pic[i] = sims.max(), yd[j], pd_[j]
        stereo[i] = any(flat[k] == flat[i] for k in pool[sims >= 0.99])
    margin = np.where(yd == 1, pd_ - 6, 5 - pd_)
    cliff = (nn_sim >= 0.8) & (nn_lab != yd)
    cat = np.where(margin < 0, "label contradicts pIC50",
          np.where(cliff & (margin >= 0.5), "activity cliff (outside noise band)",
          np.where(margin < 0.5, "noise band (pIC50 within 0.5 of cut-off)",
          np.where(nn_sim < 0.5, "out-of-domain (max sim < 0.5)", "other"))))
    tab = pd.DataFrame({"idx": dev, "y": yd, "pIC50": pd_, "margin": margin, "p_v5mt": p_mt, "p_v5reg": p_reg,
                        "p_v3": p_v3, "err": err, "nn_sim": nn_sim, "nn_label": nn_lab, "nn_pIC50": nn_pic,
                        "cliff": cliff, "stereo_twin": stereo, "category": cat})
    tab.to_csv(os.path.join(R.RES, "hard_cases.csv"), index=False)

    L = ["# Hard-case taxonomy (out-of-fold, 4,669 development molecules; test set not used)\n",
         f"Model: V5-MT 3-seed ensemble, OOF MCC-optimal threshold {thr:.3f}. OOF errors: {err.sum()} "
         f"({err.mean():.1%}).\n",
         "## Where the errors are\n",
         "| category (mutually exclusive, in this priority) | molecules | errors | error rate | share of all errors |",
         "|---|---|---|---|---|"]
    for c in ["label contradicts pIC50", "noise band (pIC50 within 0.5 of cut-off)",
              "activity cliff (outside noise band)", "out-of-domain (max sim < 0.5)", "other"]:
        m = cat == c
        L.append(f"| {c} | {m.sum()} | {err[m].sum()} | {err[m].mean():.3f} | {err[m].sum() / err.sum():.1%} |")
    L.append(f"\nOverlapping flags: activity cliffs anywhere {cliff.sum()} molecules ({err[cliff].sum()} errors, "
             f"error rate {err[cliff].mean():.3f}); stereo twins {stereo.sum()} ({err[stereo].sum()} errors, "
             f"rate {err[stereo].mean():.3f}).\n")

    # how the model fails on cliffs: copying the neighbour?
    ce = cliff & err
    L += ["## How the model fails\n",
          f"- **Activity cliffs:** on {ce.sum()} cliff errors the prediction equals the nearest neighbour's label in "
          f"{(pred[ce] == nn_lab[ce]).mean():.0%} of cases. On correctly handled cliffs ({(cliff & ~err).sum()}) the "
          f"model *overrode* the neighbour. The model largely behaves as a similarity lookup; it does not reason "
          f"about the structural difference.",
          f"- **Cliff magnitude:** mean |pIC50 difference| to the neighbour = "
          f"{np.abs(pd_[cliff] - nn_pic[cliff]).mean():.2f} log units (errors {np.abs(pd_[ce] - nn_pic[ce]).mean():.2f},"
          f" correct {np.abs(pd_[cliff & ~err] - nn_pic[cliff & ~err]).mean():.2f})."]
    band = (margin >= 0) & (margin < 0.5)
    rho_band = spearmanr(p_reg[band], pd_[band]).correlation
    rho_all = spearmanr(p_reg, pd_).correlation
    L += [f"- **Noise band:** predicted potency (V5-REG) vs measured pIC50, Spearman rho = {rho_all:.2f} overall but "
          f"{rho_band:.2f} inside the band; mean |p - threshold| is {np.abs(p_mt[band] - thr).mean():.3f} in the band vs "
          f"{np.abs(p_mt[~band] - thr).mean():.3f} outside. Within the band the model has little usable signal.",
          f"- **Model agreement:** V3 (MoLFormer) and V5-MT make the same mistake on "
          f"{((p_v3 > M.mcc_thr(yd, p_v3)).astype(int) != yd)[err].mean():.0%} of V5-MT's errors -> the errors are "
          f"shared by architectures, i.e. driven by the data.\n"]
    # potential gains per category (upper bounds)
    L += ["## Upper bound on what fixing each category could give (OOF accuracy)\n",
          f"- current OOF accuracy {1 - err.mean():.4f}"]
    for c in ["activity cliff (outside noise band)", "out-of-domain (max sim < 0.5)", "other"]:
        L.append(f"- fix every '{c}' error: {1 - (err & (cat != c)).mean():.4f}")
    L.append(f"- fix all three: {1 - (err & np.isin(cat, ['label contradicts pIC50', 'noise band (pIC50 within 0.5 of cut-off)'])).mean():.4f}\n")
    with open(os.path.join(R.RES, "hard_cases.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
