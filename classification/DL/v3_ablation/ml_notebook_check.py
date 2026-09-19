"""
Reproduces the XGBoost pipeline of classification/ML/bioactivity_ml final copy.ipynb
(cells 15, 20, 37, 40) on the same data and split, then changes one setting at a time
to locate where its test accuracy (0.8958) differs from the ~0.93 of a plain XGBoost.

Notebook pipeline:
  features : ECFP4 1024 bits + [MW, LogP, TPSA, HBD, RotB] (Excel) + HBA (RDKit)
  split    : train_test_split 70/30 stratified (seed 42), then 50/50 (seed 42) -> 3845/824/825
  CV       : StratifiedKFold(5, seed 42) on train+val, early stopping (30 rounds, aucpr) on the
             validation fold, n_estimators for the final model = mean best_iteration
  final    : fitted on X_train only (3845) despite the "train+val" comment
  params   : scale_pos_weight = neg/pos (0.27), max_delta_step 1, min_child_weight 5, gamma 0.1,
             max_depth 4, subsample 0.8, colsample 0.8, reg_alpha 0.1, reg_lambda 1.5,
             learning_rate default; `sample_weight` passed to the constructor (not to fit)
  threshold: OOF MCC-optimal on linspace(0.1, 0.9, 500)
Writes results/ml_notebook_check.md.
"""
import os
import sys
import warnings

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data

warnings.filterwarnings("ignore")


def features(df):
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    rows = []
    for _, r in df.iterrows():
        mol = Chem.MolFromSmiles(r["canonical_smiles"])
        rows.append(list(gen.GetFingerprint(mol)) + [r["MW"], r["LogP"], r["TPSA"], r["NumHDonors"],
                                                     r["NumRotatableBonds"], Descriptors.NumHAcceptors(mol)])
    return np.array(rows, dtype=float)


NB_PARAMS = dict(objective="binary:logistic", eval_metric="aucpr", max_delta_step=1, min_child_weight=5,
                 gamma=0.1, max_depth=4, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.5,
                 random_state=42, n_jobs=6)


def run(X, y, tr, va, te, params, spw=True, early_stop=True, n_est=None, final_on="train"):
    """The notebook's CV -> threshold -> final-model procedure with a given configuration."""
    Xcv, ycv = np.vstack([X[tr], X[va]]), np.concatenate([y[tr], y[va]])
    ratio = (ycv == 0).sum() / (ycv == 1).sum()
    p = dict(params, scale_pos_weight=ratio if spw else 1.0)
    oof, iters = np.zeros(len(ycv)), []
    for f_tr, f_va in StratifiedKFold(5, shuffle=True, random_state=42).split(Xcv, ycv):
        if early_stop:
            m = XGBClassifier(**p, early_stopping_rounds=30)
            m.fit(Xcv[f_tr], ycv[f_tr], eval_set=[(Xcv[f_va], ycv[f_va])], verbose=False)
            iters.append(m.best_iteration)
        else:
            m = XGBClassifier(**p, n_estimators=n_est).fit(Xcv[f_tr], ycv[f_tr])
        oof[f_va] = m.predict_proba(Xcv[f_va])[:, 1]
    ths = np.linspace(0.10, 0.90, 500)
    thr = float(ths[np.argmax([matthews_corrcoef(ycv, (oof >= t).astype(int)) for t in ths])])
    n_final = int(np.mean(iters)) if early_stop else n_est
    Xf, yf = (X[tr], y[tr]) if final_on == "train" else (Xcv, ycv)
    ratio_f = (yf == 0).sum() / (yf == 1).sum()
    mf = XGBClassifier(**dict(p, scale_pos_weight=ratio_f if spw else 1.0), n_estimators=n_final).fit(Xf, yf)
    prob = mf.predict_proba(X[te])[:, 1]
    pred = (prob >= thr).astype(int)
    return dict(oof_auc=roc_auc_score(ycv, oof), test_auc=roc_auc_score(y[te], prob), thr=thr, trees=n_final,
                acc=accuracy_score(y[te], pred), bal=balanced_accuracy_score(y[te], pred),
                mcc=matthews_corrcoef(y[te], pred), acc05=accuracy_score(y[te], (prob >= 0.5).astype(int)))


def main():
    df = data.load_df()
    y = df["bioactivity"].values.astype(int)
    X = features(df)
    idx = np.arange(len(df))
    tr, tmp = train_test_split(idx, test_size=0.3, stratify=y, random_state=42)
    va, te = train_test_split(tmp, test_size=0.5, random_state=42)

    # does XGBoost use a sample_weight given to the constructor?
    import io
    import contextlib
    buf = io.StringIO()
    with warnings.catch_warnings(record=True) as w, contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        warnings.simplefilter("always")
        m1 = XGBClassifier(**NB_PARAMS, n_estimators=50, sample_weight=np.ones(len(tr))).fit(X[tr], y[tr])
    m0 = XGBClassifier(**NB_PARAMS, n_estimators=50).fit(X[tr], y[tr])
    same = np.allclose(m0.predict_proba(X[te]), m1.predict_proba(X[te]))
    msgs = " | ".join(str(x.message)[:120] for x in w) + buf.getvalue()[:200]

    lr = {"learning_rate": 0.3}      # xgboost default when not set (as in the notebook)
    rows = [
        ("1. Notebook final XGBoost (exact)", dict(params=dict(NB_PARAMS, **lr))),
        ("2. ... without scale_pos_weight", dict(params=dict(NB_PARAMS, **lr), spw=False)),
        ("3. ... min_child_weight 1 (with scale_pos_weight)", dict(params=dict(NB_PARAMS, **lr, min_child_weight=1))),
        ("4. ... both (2) and (3)", dict(params=dict(NB_PARAMS, **lr, min_child_weight=1), spw=False)),
        ("5. ... (4) + final model on train+val", dict(params=dict(NB_PARAMS, **lr, min_child_weight=1), spw=False,
                                                     final_on="trainval")),
        ("6. Notebook permutation-test XGBoost (cell 21 params, no weighting, 300 trees, lr 0.05, depth 6)",
         dict(params=dict(objective="binary:logistic", eval_metric="logloss", max_depth=6, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=6),
              spw=False, early_stop=False, n_est=300)),
    ]
    L = ["# Reproducing the ML notebook's XGBoost (same data, same split)\n",
         f"`sample_weight` passed to the XGBClassifier constructor changes the model: **{not same}** "
         f"(identical predictions with and without it -> it is silently ignored). Messages: `{msgs.strip()[:200] or 'none'}`\n",
         "| configuration | trees | OOF AUC | test AUC | threshold | test acc | test acc @0.5 | test bal. acc | test MCC |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, kw in rows:
        r = run(X, y, tr, va, te, **kw)
        L.append(f"| {name} | {r['trees']} | {r['oof_auc']:.4f} | {r['test_auc']:.4f} | {r['thr']:.3f} | "
                 f"{r['acc']:.4f} | {r['acc05']:.4f} | {r['bal']:.4f} | {r['mcc']:.4f} |")
        print(L[-1], flush=True)
    with open(os.path.join(data.HERE, "results", "ml_notebook_check.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\n".join(L[:2]))


if __name__ == "__main__":
    main()
