"""
Dataset and feature audit.

Four parts:
  1. composition        - class balance, pIC50 distribution, the 5-6 exclusion gap, duplicates
  2. normality          - D'Agostino K2, Shapiro-Wilk, skew and kurtosis for pIC50 and every
                          descriptor, which is what justifies the non-parametric tests used in
                          descriptor_stats.py
  3. correlation        - Pearson and Spearman among descriptors, pIC50 and label, plus VIF
  4. feature integrity  - are the cached features actually correct? Recomputes ECFP / MACCS /
                          descriptors from SMILES for a random sample and compares bit-for-bit,
                          checks ranges, all-zero rows, constant columns, and cross-split
                          duplicate structures

Writes results/data_stats.md
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data
import report as R
import splits

DESC = ["MolWt", "MolLogP", "NumHDonors", "NumHAcceptors", "TPSA", "NumRotatableBonds"]


def normality(x, name):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    k2, p_k2 = stats.normaltest(x)
    sub = x if len(x) <= 5000 else np.random.default_rng(0).choice(x, 5000, replace=False)
    w, p_w = stats.shapiro(sub)
    return dict(feature=name, n=len(x), mean=x.mean(), sd=x.std(ddof=1), median=np.median(x),
                skew=stats.skew(x), kurtosis=stats.kurtosis(x), K2_p=p_k2, shapiro_p=p_w,
                normal="no" if min(p_k2, p_w) < 0.05 else "not rejected")


def main():
    df = data.load_df()
    feats = data.get_features(df)
    smiles = list(feats["smiles"])
    y = feats["labels"].astype(int)
    pic = data.get_pic50(smiles)
    L = ["# Dataset and feature audit\n"]

    # ---------- 1. composition ----------
    gap = ((pic > 5) & (pic < 6)).sum()
    contra = ((y == 1) & (pic < 6)).sum() + ((y == 0) & (pic > 5)).sum()
    dup = pd.Series(smiles).duplicated().sum()
    L += ["## 1. Composition\n",
          f"* molecules: **{len(y):,}**; active {int(y.sum()):,} ({y.mean():.1%}), "
          f"inactive {int((1 - y).sum()):,} ({1 - y.mean():.1%})",
          f"* pIC50: mean {pic.mean():.3f}, sd {pic.std(ddof=1):.3f}, median {np.median(pic):.3f}, "
          f"range {pic.min():.2f} - {pic.max():.2f}",
          f"* active pIC50 {pic[y == 1].mean():.3f} +- {pic[y == 1].std(ddof=1):.3f}; "
          f"inactive {pic[y == 0].mean():.3f} +- {pic[y == 0].std(ddof=1):.3f}",
          f"* molecules inside the excluded 5 < pIC50 < 6 band: **{gap}**",
          f"* label / pIC50 disagreements (active with pIC50 < 6 or inactive with pIC50 > 5): **{contra}**",
          f"* duplicate canonical SMILES: **{dup}**", ""]
    for name in ("random", "scaffold"):
        s = splits.get_split(name, df)
        ov = len(set(np.array(smiles)[s["dev"]]) & set(np.array(smiles)[s["test"]]))
        L.append(f"* {name} split: dev {len(s['dev']):,} ({y[s['dev']].mean():.1%} active), "
                 f"test {len(s['test']):,} ({y[s['test']].mean():.1%} active), "
                 f"identical structures on both sides: **{ov}**")
    L.append("")

    # ---------- 2. normality ----------
    rows = [normality(pic, "pIC50"), normality(pic[y == 1], "pIC50 (active)"),
            normality(pic[y == 0], "pIC50 (inactive)")]
    for i, nm in enumerate(DESC):
        rows.append(normality(feats["desc_raw"][:, i], nm))
    t = pd.DataFrame(rows)
    L += ["## 2. Normality\n",
          "D'Agostino K2 and Shapiro-Wilk (on a 5,000-molecule subsample where larger); "
          "p < 0.05 rejects normality.\n",
          "| variable | n | mean | sd | median | skew | kurtosis | K2 p | Shapiro p | normal? |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        L.append(f"| {r['feature']} | {r['n']:,} | {r['mean']:.3f} | {r['sd']:.3f} | {r['median']:.3f} | "
                 f"{r['skew']:+.2f} | {r['kurtosis']:+.2f} | {r['K2_p']:.2e} | {r['shapiro_p']:.2e} | "
                 f"{r['normal']} |")
    L.append("")

    # ---------- 3. correlation ----------
    M = np.column_stack([feats["desc_raw"], pic, y.astype(float)])
    cols = DESC + ["pIC50", "label"]
    P = np.corrcoef(M, rowvar=False)
    S = stats.spearmanr(M).statistic
    L += ["## 3. Correlation\n", "Pearson (lower triangle) / Spearman (upper triangle).\n",
          "| | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for i, c in enumerate(cols):
        cells = [f"{P[i, j]:+.2f}" if j < i else ("—" if j == i else f"*{S[i, j]:+.2f}*")
                 for j in range(len(cols))]
        L.append(f"| **{c}** | " + " | ".join(cells) + " |")
    hi = [(cols[i], cols[j], P[i, j]) for i in range(len(cols)) for j in range(i + 1, len(cols))
          if abs(P[i, j]) >= 0.7]
    L += ["", "Pairs with |Pearson r| >= 0.7: " + (", ".join(f"{a}/{b} {r:+.2f}" for a, b, r in hi) if hi else "none"), ""]
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        X = np.column_stack([np.ones(len(y)), stats.zscore(feats["desc_raw"], axis=0)])
        vif = [variance_inflation_factor(X, i + 1) for i in range(len(DESC))]
        L += ["Variance inflation factors: " + ", ".join(f"{d} {v:.2f}" for d, v in zip(DESC, vif)) +
              "  (VIF > 5 indicates collinearity)", ""]
    except Exception as exc:
        L += [f"VIF unavailable ({exc})", ""]

    # ---------- 4. feature integrity ----------
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(smiles), 300, replace=False)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    bad_fp = bad_mc = bad_ds = 0
    for i in idx:
        m = Chem.MolFromSmiles(smiles[i])
        fp = np.array(gen.GetFingerprint(m), dtype=np.float32)
        mc = np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.float32)
        ds = np.array([Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.NumHDonors(m),
                       Descriptors.NumHAcceptors(m), Descriptors.TPSA(m),
                       Descriptors.NumRotatableBonds(m)], dtype=np.float32)
        bad_fp += int(not np.array_equal(fp, feats["ecfp"][i]))
        bad_mc += int(not np.array_equal(mc, feats["maccs"][i]))
        bad_ds += int(not np.allclose(ds, feats["desc_raw"][i], atol=1e-4))
    E, Mc, D = feats["ecfp"], feats["maccs"], feats["desc_raw"]
    L += ["## 4. Feature integrity\n",
          f"* recomputed from SMILES for 300 random molecules: ECFP mismatches **{bad_fp}**, "
          f"MACCS mismatches **{bad_mc}**, descriptor mismatches **{bad_ds}**",
          f"* ECFP 1024: density {E.mean():.4f} ({E.sum(1).mean():.1f} bits set per molecule), "
          f"all-zero rows {int((E.sum(1) == 0).sum())}, never-set bits {int((E.sum(0) == 0).sum())}, "
          f"always-set bits {int((E.sum(0) == len(E)).sum())}",
          f"* MACCS 167: density {Mc.mean():.4f}, bit 0 set in {int(Mc[:, 0].sum())} molecules "
          "(RDKit MACCS is 1-indexed, so 0 is expected), "
          f"never-set bits {int((Mc.sum(0) == 0).sum())}",
          f"* descriptors: non-finite values {int((~np.isfinite(D)).sum())}, "
          f"molecules with MolWt outside 100-1500 {int(((D[:, 0] < 100) | (D[:, 0] > 1500)).sum())}, "
          f"negative TPSA {int((D[:, 4] < 0).sum())}", ""]
    L.append("| descriptor | min | median | max |")
    L.append("|---|---|---|---|")
    for i, nm in enumerate(DESC):
        L.append(f"| {nm} | {D[:, i].min():.2f} | {np.median(D[:, i]):.2f} | {D[:, i].max():.2f} |")

    out = os.path.join(R.RES, "data_stats.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
