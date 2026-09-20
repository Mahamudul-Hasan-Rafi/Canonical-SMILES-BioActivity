"""
V8 follow-up: the regression-first model against the classification-trained ones, and the
combination of the two.

V8-R2 is trained to predict potency; V5-MT / V7 are trained to classify. They fail on
different molecules, so their rank-average is tested here as well. Everything is selected on
the out-of-fold predictions (4,669 development molecules); the held-out test set is reported
once, at the end. Comparisons use McNemar's exact test on the per-molecule errors and the
paired DeLong test on the AUCs.
Writes results/v8_analysis.md.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import binomtest, rankdata
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef
import core, data, report as R, metrics as M


class _Tee:
    """Echo the report to stdout and to results/v8_analysis.md."""

    def __init__(self, path):
        self.fh = open(path, "w", encoding="utf-8")

    def write(self, t):
        sys.__stdout__.write(t)
        self.fh.write(t)

    def flush(self):
        sys.__stdout__.flush()
        self.fh.flush()


sys.stdout = _Tee(os.path.join(R.RES, "v8_analysis.md"))
st = R.Store(); dev = st.splits["random"]["dev"]; y = st.y[dev]
pic = data.get_pic50(st.df["canonical_smiles"].tolist()); p_dev = pic[dev]
hc = pd.read_csv(os.path.join(R.RES, "hard_cases.csv"))
band = ((hc.margin >= 0) & (hc.margin < 0.5)).values; cliff = hc["cliff"].values
def ens(v):
    r = [st.cv(seed=s, backbone="chemberta_mlm", variant=v) for s in R.SEEDS]
    return np.mean([x["oof"] for x in r], 0), np.mean([x["ens"] for x in r], 0), r[0]["test_y"]
M5 = ens("graph_mt"); M7 = ens("graph_mt_delta_res"); R2 = ens("reg_first_delta")
mods = {"V5-MT": M5, "V7": M7, "V8-R2": R2}
# blend: rank-average of the OOF scores (monotone, threshold re-fit on OOF)
for nm, a, b in [("V5-MT+V8-R2", M5, R2), ("V7+V8-R2", M7, R2)]:
    o = (rankdata(a[0]) + rankdata(b[0])) / (2 * len(y))
    t = (rankdata(a[1]) + rankdata(b[1])) / (2 * len(a[1]))
    mods[nm] = (o, t, a[2])
print(f"{'model':<12} {'OOF AUC':>8} {'OOF acc':>8} {'OOF MCC':>8} {'err':>5} {'band err':>9} {'cliff err':>10}")
thr = {}; err = {}
for k, (o, t, yt) in mods.items():
    thr[k] = M.mcc_thr(y, o); pr = (o > thr[k]).astype(int); err[k] = pr != y
    print(f"{k:<12} {roc_auc_score(y,o):>8.4f} {accuracy_score(y,pr):>8.4f} {matthews_corrcoef(y,pr):>8.4f} "
          f"{err[k].sum():>5} {err[k][band].sum():>9} {err[k][cliff].sum():>10}")
print("\nMcNemar (OOF) vs V5-MT:  b = only this wrong, c = only V5-MT wrong")
for k in mods:
    if k == "V5-MT": continue
    b = int((err[k] & ~err["V5-MT"]).sum()); c = int((~err[k] & err["V5-MT"]).sum())
    print(f"  {k:<12} b={b:>3} c={c:>3} p={binomtest(b,b+c,0.5).pvalue:.4f}")
print("\nheld-out test (reported once)")
for k, (o, t, yt) in mods.items():
    pr = (t > thr[k]).astype(int)
    print(f"  {k:<12} AUC {roc_auc_score(yt,t):.4f}  acc {accuracy_score(yt,pr):.4f}  errors {int((pr!=yt).sum())}")

print("\npaired DeLong vs V5-MT")
for k in mods:
    if k == "V5-MT":
        continue
    do, po = M.delong(y, mods[k][0], mods["V5-MT"][0])
    yt = mods[k][2]
    dt, pt = M.delong(yt, mods[k][1], mods["V5-MT"][1])
    print(f"  {k:<12} OOF dAUC {do:+.4f} p {po:.4f} | test dAUC {dt:+.4f} p {pt:.4f}")
