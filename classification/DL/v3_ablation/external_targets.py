"""
External validation: our potency model, unchanged, on MoleculeACE benchmark targets.

MoleculeACE (van Tilborg et al., JCIM 2022) ships 30 single-target ChEMBL datasets with a
prescribed train/test split and a per-molecule activity-cliff flag. This runs our models on five
of them under the benchmark's own protocol - their split, their cliff labels - so the question
"does the architecture generalise beyond HDAC1?" gets an answer that is not ours to frame.

Models, all trained from scratch on each target:
  SVR on ECFP        the benchmark's strongest classical baseline family
  LightGBM regressor our classical branch
  V8-R2              our deep potency network (retrieval + anchored delta)
  1 : 1 hybrid       the combination we propose

Reported: RMSE and RMSE_cliff on the held-out test molecules.

  python external_targets.py --targets CHEMBL204_Ki CHEMBL2147_Ki --seeds 42
Writes results/external_targets.md
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as _data

DATA = os.path.join(_data.HERE, "results", "mace_data")
CACHE = os.path.join(_data.HERE, "results", "mace_cache")
Y = "y [pEC50/pKi]"


def featurise(smiles):
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(_data.featurize_one, smiles, chunksize=64))
    return (np.stack([r[0] for r in res]).astype(np.float32),
            np.stack([r[1] for r in res]).astype(np.float32),
            np.stack([r[2] for r in res]).astype(np.float32))


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def run_target(target, seeds, epochs):
    import torch
    import backbones
    import core
    import gpu
    import v4_modules
    df = pd.read_csv(os.path.join(DATA, target + ".csv"))
    smiles = df["smiles"].tolist()
    y = df[Y].values.astype(np.float32)
    cliff = df["cliff_mol"].values.astype(bool)
    is_test = (df["split"] == "test").values
    tr_all, te = np.where(~is_test)[0], np.where(is_test)[0]

    ecfp, maccs, desc = featurise(smiles)
    feats = {"smiles": np.array(smiles, dtype=object), "ecfp": ecfp, "maccs": maccs,
             "desc_raw": desc, "pic50": y,
             # labels are used only for the AUROC that train_one logs; the objective is RMSE
             "labels": (y >= np.median(y)).astype(np.float32)}

    # per-target caches so the HDAC1 feature caches are never touched
    os.makedirs(CACHE, exist_ok=True)
    v4_modules.V4_CACHE = os.path.join(CACHE, f"{target}_v4.npz")
    v4_modules.SIM_CACHE = os.path.join(CACHE, f"{target}_sim.npy")

    out = {}
    X = np.hstack([ecfp, desc])
    from lightgbm import LGBMRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    out["SVR on ECFP"] = make_pipeline(StandardScaler(), SVR(C=10.0, gamma="scale")).fit(
        ecfp[tr_all], y[tr_all]).predict(ecfp[te])
    out["LightGBM regressor"] = LGBMRegressor(n_estimators=600, num_leaves=63, learning_rate=0.05,
                                              subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                                              n_jobs=6, random_state=42, verbose=-1).fit(
        X[tr_all], y[tr_all]).predict(X[te])

    # V8-R2: 10 % of the training set is held out for early stopping, the test set is untouched
    preds = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(tr_all))
        n_val = max(20, int(0.1 * len(tr_all)))
        vl, tr = tr_all[perm[:n_val]], tr_all[perm[n_val:]]
        cfg = core.make_cfg("reg_first_delta", backbone="chemberta_mlm", max_epochs=epochs)
        r = core.train_one(cfg, feats, tr, vl, {"test": te}, seed, torch.device("cuda"),
                           backbones.load_tokenizer("chemberta_mlm"),
                           guard=gpu.ThermalGuard(hot_c=88, cool_c=80))
        preds.append(core.probs_to_pic50(r["test_probs"]))
    out["V8-R2 (deep)"] = np.mean(preds, 0)
    out["HYBRID 1:1"] = np.mean([out["LightGBM regressor"], out["V8-R2 (deep)"]], 0)
    return {k: (rmse(v - y[te]), rmse(v[cliff[te]] - y[te][cliff[te]])) for k, v in out.items()}, \
        dict(n=len(df), train=len(tr_all), test=len(te), cliff_frac=float(cliff.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["CHEMBL204_Ki", "CHEMBL2147_Ki", "CHEMBL218_EC50",
                                                     "CHEMBL4005_Ki", "CHEMBL233_Ki"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()
    import core
    core.setup_torch()
    rows, meta = {}, {}
    t0 = time.time()
    for t in args.targets:
        rows[t], meta[t] = run_target(t, args.seeds, args.epochs)
        print(f"{t}: " + "  ".join(f"{k} {v[0]:.3f}/{v[1]:.3f}" for k, v in rows[t].items())
              + f"   ({(time.time() - t0) / 60:.1f} min)", flush=True)

    models = list(next(iter(rows.values())))
    L = ["# External validation on MoleculeACE targets\n",
         "Our models trained from scratch on each target using the benchmark's own train/test split "
         "and its own activity-cliff labels. Each cell is RMSE (all test molecules) / RMSE_cliff.\n",
         "| target | n | test | cliffs | " + " | ".join(models) + " |",
         "|" + "---|" * (4 + len(models))]
    for t in args.targets:
        m = meta[t]
        L.append(f"| {t} | {m['n']:,} | {m['test']} | {m['cliff_frac']:.0%} | " +
                 " | ".join(f"{rows[t][k][0]:.3f} / {rows[t][k][1]:.3f}" for k in models) + " |")
    L.append("| **mean** | | | | " +
             " | ".join(f"**{np.mean([rows[t][k][0] for t in args.targets]):.3f}** / "
                        f"{np.mean([rows[t][k][1] for t in args.targets]):.3f}" for k in models) + " |")
    L += ["", f"Seeds: {args.seeds}. For reference, the MoleculeACE paper reports an average RMSE of "
          "0.675 for its strongest classical baseline (ECFP + SVM) across all 30 targets.\n"]
    with open(os.path.join(_data.HERE, "results", "external_targets.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
