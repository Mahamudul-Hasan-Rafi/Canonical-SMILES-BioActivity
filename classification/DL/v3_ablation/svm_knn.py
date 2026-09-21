"""
SVM and k-NN baselines with nested cross-validation.

Protocol: the same 5 outer folds and held-out test set as every other model. Inside each outer
training fold a GridSearchCV (3-fold, scored by ROC-AUC) picks the hyperparameters, the winner is
refitted on that whole training fold, and it predicts the held-out fold and the test set. The
search therefore never sees the fold it is scored on, and the test set is never involved in
tuning. Per-fold winners are printed so the stability of the choice is visible.

Models
  SVM            RBF / linear support vector machine on ECFP4 1024 + 6 descriptors (standardised)
  KNN            k nearest neighbours on the same features (standardised), distance metric tuned
  KNN-Tanimoto   k nearest neighbours on the binary fingerprint alone with the Jaccard/Tanimoto
                 metric - the canonical chemistry choice, and the one a reviewer will expect

Writes results/baselines/<split>/<name>__<featureset>.npz in the same format as baselines.py.

  python svm_knn.py --split random
  python svm_knn.py --split scaffold
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


def specs(n_jobs):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    svm = Pipeline([("sc", StandardScaler()), ("clf", SVC(class_weight="balanced", cache_size=1000))])
    svm_grid = {"clf__kernel": ["rbf"], "clf__C": [1.0, 10.0, 100.0], "clf__gamma": ["scale", 1e-3, 1e-4]}
    knn = Pipeline([("sc", StandardScaler()), ("clf", KNeighborsClassifier(n_jobs=n_jobs))])
    knn_grid = {"clf__n_neighbors": [1, 3, 5, 9, 15, 25], "clf__weights": ["uniform", "distance"],
                "clf__metric": ["euclidean", "cosine"]}
    tan = KNeighborsClassifier(metric="jaccard", algorithm="brute", n_jobs=n_jobs)
    tan_grid = {"n_neighbors": [1, 3, 5, 9, 15, 25], "weights": ["uniform", "distance"]}
    return {
        "SVM": (svm, svm_grid, "ECFP+Desc", False),
        "KNN": (knn, knn_grid, "ECFP+Desc", False),
        "KNN-Tanimoto": (tan, tan_grid, "ECFP", True),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="random", choices=["random", "scaffold"])
    ap.add_argument("--n-jobs", type=int, default=6)
    args = ap.parse_args()
    from sklearn.base import clone
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.metrics import roc_auc_score

    out_dir = os.path.join(OUT, args.split)
    os.makedirs(out_dir, exist_ok=True)
    df = data.load_df()
    feats = data.get_features(df)
    s = splits.get_split(args.split, df)
    dev_idx, folds, te = s["dev"], s["folds"], s["test"]
    y = feats["labels"].astype(int)
    X_sets = {"ECFP+Desc": np.hstack([feats["ecfp"], feats["desc_raw"]]),
              "ECFP": feats["ecfp"]}

    for name, (est, grid, fs, boolean) in specs(args.n_jobs).items():
        path = os.path.join(out_dir, f"{name}__{fs}.npz")
        if os.path.exists(path):
            print(f"{name:<14} cached")
            continue
        X = X_sets[fs].astype(bool) if boolean else X_sets[fs]
        t0 = time.time()
        oof = np.zeros(len(dev_idx), np.float32)
        test_probs, chosen = [], []
        for f_tr, f_vl in folds:
            tr = dev_idx[f_tr]
            gs = GridSearchCV(clone(est), grid, scoring="roc_auc", n_jobs=args.n_jobs,
                              cv=StratifiedKFold(3, shuffle=True, random_state=42), refit=False)
            gs.fit(X[tr], y[tr])
            chosen.append(gs.best_params_)
            best = clone(est).set_params(**gs.best_params_)
            if name == "SVM":                      # probabilities only for the refit, not the search
                best.set_params(clf__probability=True, clf__random_state=42)
            best.fit(X[tr], y[tr])
            oof[f_vl] = best.predict_proba(X[dev_idx[f_vl]])[:, 1]
            test_probs.append(best.predict_proba(X[te])[:, 1])
        np.savez(path, oof=oof, oof_labels=y[dev_idx], test_probs=np.stack(test_probs),
                 test_labels=y[te], dev_idx=dev_idx, test_idx=te)
        print(f"{name:<14} OOF AUC {roc_auc_score(y[dev_idx], oof):.4f} | "
              f"test ens AUC {roc_auc_score(y[te], np.mean(test_probs, 0)):.4f} | "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
        for i, c in enumerate(chosen):
            print(f"                 fold {i}: {c}", flush=True)


if __name__ == "__main__":
    main()
