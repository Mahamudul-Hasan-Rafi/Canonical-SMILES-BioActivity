"""
Classical baselines on exactly the same folds + held-out test set as V3, for each split.
Answers the reviewer question "does the deep model beat a fingerprint forest?".

CPU only, n_jobs capped (default 6) so the machine stays cool.
Writes results/baselines/<split>/<name>.npz with OOF + per-fold test probabilities.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data
import splits

OUT = os.path.join(data.HERE, "results", "baselines")


def models(n_jobs):
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
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
        "ExtraTrees": lambda: ExtraTreesClassifier(n_estimators=500, n_jobs=n_jobs, random_state=42,
                                                   class_weight="balanced_subsample", min_samples_leaf=1),
        "LightGBM": lambda: _lgbm(n_jobs),
        "CatBoost": lambda: _catboost(n_jobs),
    }


def _lgbm(n_jobs):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=600, num_leaves=63, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.5, n_jobs=n_jobs, random_state=42, verbose=-1)


def _catboost(n_jobs):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(iterations=600, depth=6, learning_rate=0.05, rsm=0.5, random_seed=42,
                              thread_count=n_jobs, verbose=0, allow_writing_files=False)


def ensembles(n_jobs):
    """Voting and stacking over the five tree/boosting learners (same hyperparameters as above).
    Stacking uses an internal 3-fold CV inside each training fold, so the meta-learner never sees
    predictions a base learner made on its own training data."""
    from sklearn.ensemble import StackingClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    base = lambda: [(k, models(n_jobs)[k]()) for k in ("RF", "ExtraTrees", "XGBoost", "LightGBM", "CatBoost")]
    return {
        "Voting": lambda: VotingClassifier(estimators=base(), voting="soft", n_jobs=1),
        "Stacking": lambda: StackingClassifier(estimators=base(), n_jobs=1, cv=3, stack_method="predict_proba",
                                               final_estimator=LogisticRegression(C=1.0, max_iter=3000,
                                                                                  class_weight="balanced")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--split", default="random", choices=["random", "scaffold"])
    ap.add_argument("--sets", default="", help="comma-separated feature sets to run (default: all)")
    ap.add_argument("--models", default="", help="comma-separated model names to run (default: all)")
    ap.add_argument("--ensembles", default="1", help="0 to skip the voting/stacking ensembles")
    args = ap.parse_args()
    out_dir = os.path.join(OUT, args.split)
    os.makedirs(out_dir, exist_ok=True)
    df = data.load_df()
    feats = data.get_features(df)
    s = splits.get_split(args.split, df)
    dev_idx, folds, te = s["dev"], s["folds"], s["test"]
    y = feats["labels"].astype(int)
    import v4_modules
    fp2 = v4_modules.get_v4_features(feats["smiles"])["ecfp2048c"]
    feature_sets = {
        "ECFP": feats["ecfp"],
        # exactly the ML notebook's features: Morgan r2 1024 binary + the 6 RO5 descriptors
        "ECFP+Desc": np.hstack([feats["ecfp"], feats["desc_raw"]]),
        "ECFP+MACCS+Desc": np.hstack([feats["ecfp"], feats["maccs"], feats["desc_raw"]]),
        # V4 fingerprint: count-based Morgan r2, 2048 bits, chirality (log1p counts)
        "ECFPc2048+MACCS+Desc": np.hstack([fp2, feats["maccs"], feats["desc_raw"]]),
    }
    if args.sets:
        want = [x.strip() for x in args.sets.split(",")]
        feature_sets = {k: v for k, v in feature_sets.items() if k in want}
        assert feature_sets, f"no such feature set; available: {list(feature_sets)}"
    want_m = [x.strip() for x in args.models.split(",")] if args.models else None
    for fs_name, X in feature_sets.items():
        for m_name, make in models(args.n_jobs).items():
            if want_m and m_name not in want_m:
                continue
            name = f"{m_name}__{fs_name}"
            path = os.path.join(out_dir, name + ".npz")
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
                     test_labels=y[te], dev_idx=dev_idx, test_idx=te)
            from sklearn.metrics import roc_auc_score
            print(f"{name:<28} OOF AUC {roc_auc_score(y[dev_idx], oof):.4f} | "
                  f"test ens AUC {roc_auc_score(y[te], np.mean(test_probs, 0)):.4f} | {time.time()-t0:.0f}s",
                  flush=True)

    # Voting / stacking ensembles, on each selected feature set
    for fs_name, X in (feature_sets.items() if args.ensembles != "0" else []):
        for m_name, make in ensembles(args.n_jobs).items():
            name = f"{m_name}__{fs_name}"
            path = os.path.join(out_dir, name + ".npz")
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
                     test_labels=y[te], dev_idx=dev_idx, test_idx=te)
            from sklearn.metrics import roc_auc_score
            print(f"{name:<28} OOF AUC {roc_auc_score(y[dev_idx], oof):.4f} | "
                  f"test ens AUC {roc_auc_score(y[te], np.mean(test_probs, 0)):.4f} | {time.time()-t0:.0f}s", flush=True)

    # V5 counterpart: XGBoost regressing pIC50 (training folds only); score = sigmoid(2 * (pIC50_hat - 5.5))
    from xgboost import XGBRegressor
    pic = data.get_pic50(feats["smiles"])
    # potency regressors on every selected feature set (score = sigmoid(2 * (pIC50_hat - 5.5)))
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor as _XGBR
    pic_all = data.get_pic50(feats["smiles"])
    regs = {
        "XGBReg": lambda: _XGBR(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.5, n_jobs=args.n_jobs, random_state=42, tree_method="hist"),
        "LGBMReg": lambda: LGBMRegressor(n_estimators=600, num_leaves=63, learning_rate=0.05, subsample=0.8,
                                         subsample_freq=1, colsample_bytree=0.5, n_jobs=args.n_jobs,
                                         random_state=42, verbose=-1),
        "CatReg": lambda: CatBoostRegressor(iterations=600, depth=6, learning_rate=0.05, rsm=0.5, random_seed=42,
                                            thread_count=args.n_jobs, verbose=0, allow_writing_files=False),
        "RFReg": lambda: RandomForestRegressor(n_estimators=500, n_jobs=args.n_jobs, random_state=42),
    }
    sig = lambda v: 1 / (1 + np.exp(-2 * (v - 5.5)))
    for fs_name, X in feature_sets.items():
        for r_name, make in regs.items():
            name = f"{r_name}__{fs_name}"
            path = os.path.join(out_dir, name + ".npz")
            if os.path.exists(path):
                continue
            t0 = time.time()
            oof = np.zeros(len(dev_idx), np.float32)
            oof_pic = np.zeros(len(dev_idx), np.float32)
            test_probs, test_pic = [], []
            for f_tr, f_vl in folds:
                reg = make().fit(X[dev_idx[f_tr]], pic_all[dev_idx[f_tr]])
                pv, pt = reg.predict(X[dev_idx[f_vl]]), reg.predict(X[te])
                oof[f_vl], oof_pic[f_vl] = sig(pv), pv
                test_probs.append(sig(pt))
                test_pic.append(pt)
            np.savez(path, oof=oof, oof_pic50=oof_pic, oof_labels=y[dev_idx], test_probs=np.stack(test_probs),
                     test_pic50=np.stack(test_pic), test_labels=y[te], dev_idx=dev_idx, test_idx=te)
            rmse = float(np.sqrt(np.mean((oof_pic - pic_all[dev_idx]) ** 2)))
            rmse_t = float(np.sqrt(np.mean((np.mean(test_pic, 0) - pic_all[te]) ** 2)))
            print(f"{name:<28} OOF RMSE {rmse:.4f} | test RMSE {rmse_t:.4f} | {time.time()-t0:.0f}s", flush=True)

    if "ECFPc2048+MACCS+Desc" not in feature_sets:
        return
    X = feature_sets["ECFPc2048+MACCS+Desc"]
    name = "XGBoostReg__ECFPc2048+MACCS+Desc"
    path = os.path.join(out_dir, name + ".npz")
    if not os.path.exists(path):
        t0 = time.time()
        oof = np.zeros(len(dev_idx), np.float32)
        test_probs = []
        sig = lambda p: 1 / (1 + np.exp(-2 * (p - 5.5)))
        for f_tr, f_vl in folds:
            reg = XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5,
                               n_jobs=args.n_jobs, random_state=42, tree_method="hist").fit(X[dev_idx[f_tr]], pic[dev_idx[f_tr]])
            oof[f_vl] = sig(reg.predict(X[dev_idx[f_vl]]))
            test_probs.append(sig(reg.predict(X[te])))
        np.savez(path, oof=oof, oof_labels=y[dev_idx], test_probs=np.stack(test_probs), test_labels=y[te],
                 dev_idx=dev_idx, test_idx=te)
        from sklearn.metrics import roc_auc_score
        print(f"{name:<28} OOF AUC {roc_auc_score(y[dev_idx], oof):.4f} | "
              f"test ens AUC {roc_auc_score(y[te], np.mean(test_probs, 0)):.4f} | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
