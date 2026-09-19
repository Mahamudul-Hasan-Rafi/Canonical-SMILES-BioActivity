"""
Statistical association between each physico-chemical descriptor and the
bioactivity class (active vs inactive) on the full V3 dataset.

Per descriptor (univariate):
  * Mann-Whitney U   (primary test: no normality assumption, handles integer counts)
  * Welch t-test     (parametric check)
  * point-biserial r (= Pearson r with the 0/1 label)
  * effect sizes: rank-biserial r, Cohen's d, single-feature ROC-AUC
  * Holm-Bonferroni correction over the descriptors tested
Non-independence: molecules come in congeneric series (2,402 Murcko scaffolds), which
makes the tests above over-confident. Logistic regressions are therefore also fitted
with scaffold-cluster-robust standard errors (each scaffold family = one cluster).
Multivariable:
  * logistic regression on standardised descriptors (statsmodels) -> Wald p, odds ratio / sd,
    VIF for collinearity, likelihood-ratio test of the whole descriptor block

With n = 5,494 even negligible differences give tiny p-values, so effect sizes are
reported next to every p-value. Writes results/descriptor_stats.{csv,md}.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data
import splits

OUT = os.path.join(data.HERE, "results")


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def magnitude_auc(a):
    d = abs(a - 0.5)
    return "negligible" if d < 0.06 else "small" if d < 0.14 else "medium" if d < 0.21 else "large"


def main():
    import statsmodels.api as sm
    from sklearn.metrics import roc_auc_score
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    df = data.load_df()
    feats = data.get_features(df)
    y = df["bioactivity"].values.astype(int)
    X = pd.DataFrame(feats["desc_raw"], columns=data.DESC_NAMES)
    # the 5 descriptors stored in the Excel file (sanity: they match RDKit's)
    excel = {"MolWt": "MW", "MolLogP": "LogP", "NumHDonors": "NumHDonors", "TPSA": "TPSA",
             "NumRotatableBonds": "NumRotatableBonds"}
    agree = {k: float(np.corrcoef(X[k], df[v])[0, 1]) for k, v in excel.items()}

    rows = []
    for c in X.columns:
        a, i = X[c][y == 1], X[c][y == 0]
        U, p_mw = stats.mannwhitneyu(a, i, alternative="two-sided")
        t, p_t = stats.ttest_ind(a, i, equal_var=False)
        t_s, p_s = stats.ttest_ind(a, i, equal_var=True)                       # Student (pooled variance)
        z = (a.mean() - i.mean()) / np.sqrt(a.var(ddof=1) / len(a) + i.var(ddof=1) / len(i))
        p_z = 2 * stats.norm.sf(abs(z))                                          # large-sample two-sample z-test
        r_pb, p_pb = stats.pointbiserialr(y, X[c])
        sd_pooled = np.sqrt(((len(a) - 1) * a.var() + (len(i) - 1) * i.var()) / (len(a) + len(i) - 2))
        auc = roc_auc_score(y, X[c])
        rows.append({"descriptor": c,
                     "active median": a.median(), "inactive median": i.median(),
                     "active mean": a.mean(), "inactive mean": i.mean(),
                     "p Mann-Whitney": p_mw, "Welch t": t, "p Welch t": p_t, "Student t": t_s, "p Student t": p_s,
                     "z": z, "p z-test": p_z, "p Levene (equal variance)": stats.levene(a, i).pvalue,
                     "skew active": stats.skew(a), "skew inactive": stats.skew(i),
                     "p normality active (D'Agostino)": stats.normaltest(a).pvalue,
                     "point-biserial r": r_pb, "p point-biserial": p_pb,
                     "rank-biserial r": 2 * U / (len(a) * len(i)) - 1,
                     "Cohen d": (a.mean() - i.mean()) / sd_pooled,
                     "single-feature AUC": auc, "effect": magnitude_auc(auc)})
    t = pd.DataFrame(rows)
    t["p Mann-Whitney (Holm)"] = holm(t["p Mann-Whitney"].values)

    # scaffold-cluster-robust univariate logistic regression
    groups = pd.factorize(splits.scaffolds(df))[0]
    Zu = (X - X.mean()) / X.std()
    p_naive, p_clu, infl = [], [], []
    for c in X.columns:
        Xc = sm.add_constant(Zu[[c]])
        a = sm.Logit(y, Xc).fit(disp=0)
        b = sm.Logit(y, Xc).fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups})
        p_naive.append(a.pvalues[c])
        p_clu.append(b.pvalues[c])
        infl.append(b.bse[c] / a.bse[c])
    t["p logit naive"] = p_naive
    t["p logit scaffold-clustered"] = p_clu
    t["p scaffold-clustered (Holm)"] = holm(np.array(p_clu))
    t["SE inflation from clustering"] = infl

    # multivariable logistic regression on standardised descriptors
    Z = (X - X.mean()) / X.std()
    logit = sm.Logit(y, sm.add_constant(Z)).fit(disp=0)
    null = sm.Logit(y, np.ones((len(y), 1))).fit(disp=0)
    lr = 2 * (logit.llf - null.llf)
    p_lr = stats.chi2.sf(lr, df=Z.shape[1])
    vif = [variance_inflation_factor(sm.add_constant(Z).values, k + 1) for k in range(Z.shape[1])]
    t["logit coef (per sd)"] = logit.params[1:].values
    t["odds ratio per sd"] = np.exp(logit.params[1:].values)
    t["p Wald (multivariable)"] = logit.pvalues[1:].values
    t["VIF"] = vif
    desc_only_auc = roc_auc_score(y, logit.predict(sm.add_constant(Z)))
    logit_c = sm.Logit(y, sm.add_constant(Z)).fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups})
    t["p Wald multivariable scaffold-clustered"] = logit_c.pvalues[1:].values
    joint = logit_c.wald_test(np.eye(Z.shape[1] + 1)[1:], scalar=True)
    t.to_csv(os.path.join(OUT, "descriptor_stats.csv"), index=False)

    def fp(p):
        return "< 1e-300" if p < 1e-300 else f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"
    L = ["# Descriptor vs bioactivity class: statistical tests\n",
         f"n = {len(y)} molecules ({y.sum()} active, {(y == 0).sum()} inactive).\n",
         "## Univariate (each descriptor on its own)\n",
         "| descriptor | median active / inactive | p Mann-Whitney (Holm) | p Welch t | point-biserial r | "
         "rank-biserial r | Cohen d | single-feature AUC | effect |", "|" + "---|" * 9]
    for _, r in t.iterrows():
        L.append(f"| {r['descriptor']} | {r['active median']:.2f} / {r['inactive median']:.2f} | "
                 f"{fp(r['p Mann-Whitney (Holm)'])} | {fp(r['p Welch t'])} | {r['point-biserial r']:+.3f} | "
                 f"{r['rank-biserial r']:+.3f} | {r['Cohen d']:+.2f} | {r['single-feature AUC']:.3f} | {r['effect']} |")
    L += ["", "## Parametric tests (difference in means)\n",
          "Levene's test rejects equal variances and D'Agostino's test rejects normality for every descriptor, "
          "so Welch's t-test is the appropriate parametric test; with 4,326 / 1,168 molecules the central limit "
          "theorem makes the t / z approximation for the mean difference reliable despite the skew.\n",
          "| descriptor | Student t (p) | Welch t (p) | z (p) | Levene p | skew active / inactive |", "|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        L.append(f"| {r['descriptor']} | {r['Student t']:.2f} ({fp(r['p Student t'])}) | {r['Welch t']:.2f} ({fp(r['p Welch t'])}) | "
                 f"{r['z']:.2f} ({fp(r['p z-test'])}) | {fp(r['p Levene (equal variance)'])} | "
                 f"{r['skew active']:.2f} / {r['skew inactive']:.2f} |")
    L += ["", f"## Accounting for congeneric series (clustered by {groups.max() + 1} Murcko scaffolds)\n",
          "| descriptor | p naive (logistic) | p scaffold-clustered | Holm-adjusted | SE inflation |", "|---|---|---|---|---|"]
    for _, r in t.iterrows():
        L.append(f"| {r['descriptor']} | {fp(r['p logit naive'])} | {fp(r['p logit scaffold-clustered'])} | "
                 f"{fp(r['p scaffold-clustered (Holm)'])} | {r['SE inflation from clustering']:.2f}x |")
    L += ["", "## Multivariable logistic regression (all 6 descriptors together, standardised)\n",
          "| descriptor | odds ratio per +1 sd | p Wald | p Wald scaffold-clustered | VIF |", "|---|---|---|---|---|"]
    for _, r in t.iterrows():
        L.append(f"| {r['descriptor']} | {r['odds ratio per sd']:.3f} | {fp(r['p Wald (multivariable)'])} | "
                 f"{fp(r['p Wald multivariable scaffold-clustered'])} | {r['VIF']:.1f} |")
    L += ["", f"Likelihood-ratio test, all 6 descriptors vs intercept only: chi2 = {lr:.1f}, df = 6, p = {fp(p_lr)}; "
          f"scaffold-clustered joint Wald test chi2 = {joint.statistic:.1f}, p = {fp(float(joint.pvalue))}; "
          f"McFadden pseudo-R2 = {logit.prsquared:.3f}; in-sample ROC-AUC of descriptors alone = {desc_only_auc:.3f}.\n",
          "Effect-size guide (single-feature AUC distance from 0.5): <0.06 negligible, <0.14 small, <0.21 medium, else large.\n",
          "RDKit descriptors vs the Excel columns (Pearson r): " + ", ".join(f"{k} {v:.4f}" for k, v in agree.items()) + "\n"]
    with open(os.path.join(OUT, "descriptor_stats.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
