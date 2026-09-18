"""
Classical baselines on exactly the same 5 folds + held-out test set as V3.
Answers the reviewer question "does the deep model beat a fingerprint forest?".

CPU only, n_jobs capped (default 6) so the machine stays cool.
Writes results/baselines/<name>.npz with OOF + fold test probabilities.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v3_core as C

OUT = os.path.join(C.HERE, "results", "baselines")


def models(n_jobs):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    return {
        "RF": lambda: RandomForestClassifier(n_estimators=500, n_jobs=n_jobs, random_state=42,
                                             class_weight="balanced_subsample", min_samples_leaf=1),
        "XGBoost": lambda: XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8,
                                         colsample_bytree=0.5, n_jobs=n_jobs, random_state=42,
                                         tree_method="hist", eval_metric="logloss"),
        "LogReg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000,
                                                                             class_weight="balanced")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    df = C.load_df()
    feats = C.get_features(df)
    tr, va, te = C.notebook_splits(df)
    dev_idx, folds = C.notebook_folds(df, tr, va)
    y = feats["labels"].astype(int)
    feature_sets = {
        "ECFP": feats["ecfp"],
        "ECFP+MACCS+Desc": np.hstack([feats["ecfp"], feats["maccs"], feats["desc_raw"]]),
    }
    for fs_name, X in feature_sets.items():
        for m_name, make in models(args.n_jobs).items():
            name = f"{m_name}__{fs_name}"
            path = os.path.join(OUT, name + ".npz")
            if os.path.exists(path):
                continue
            t0 = time.time()
            oof = np.zeros(len(dev_idx), np.float32)
            test_probs = []
            for f_tr, f_vl in folds:
                clf = make()
                clf.fit(X[dev_idx[f_tr]], y[dev_idx[f_tr]])
                oof[f_vl] = clf.predict_proba(X[dev_idx[f_vl]])[:, 1]
                test_probs.append(clf.predict_proba(X[te])[:, 1])
            np.savez(path, oof=oof, oof_labels=y[dev_idx], test_probs=np.stack(test_probs),
                     test_labels=y[te])
            from sklearn.metrics import roc_auc_score
            print(f"{name:<28} OOF AUC {roc_auc_score(y[dev_idx], oof):.4f} | "
                  f"test ens AUC {roc_auc_score(y[te], np.mean(test_probs, 0)):.4f} | {time.time()-t0:.0f}s",
                  flush=True)


if __name__ == "__main__":
    main()
