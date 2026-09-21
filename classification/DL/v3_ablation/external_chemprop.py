"""
Published comparator #1: Chemprop v2 (D-MPNN), Heid et al., JCIM 2024.

Runs the authors' own package on OUR dataset, OUR folds and OUR held-out test set, for both
tasks: bioactivity classification and pIC50 regression. Predictions are written in the same
.npz format as every other baseline, so the existing analysis scripts treat it like any other
model.

Protocol, matched to the rest of the study:
  * the 5 outer folds of the chosen split; the model trains on 4 folds and predicts the 5th
  * 10 % of each training fold is held out for early stopping (the evaluation fold is never
    used for model selection)
  * the held-out test set is predicted by every fold model and averaged, as for our models
  * optional multiple seeds, averaged the same way our 3-seed ensembles are

  python external_chemprop.py --split random --task cls --seeds 42
  python external_chemprop.py --split scaffold --task reg --seeds 42 7 2024

Writes results/baselines/<split>/Chemprop[-reg]__graph.npz
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data
import splits


def foundation_mp():
    """CheMeleon message-passing block (Burns & Green 2025), loaded exactly as chemprop's
    --from-foundation chemeleon does: the Zenodo checkpoint's hyper-parameters and weights."""
    import os
    import torch
    from chemprop.nn import BondMessagePassing
    ck = torch.load(os.path.expanduser("~/.chemprop/chemeleon_mp.pt"), weights_only=True)
    mp = BondMessagePassing(**ck["hyper_parameters"])
    mp.load_state_dict(ck["state_dict"])
    return mp


def run_fold(smis, y_all, tr, vl, te, task, seed, epochs, workers=0, foundation=False):
    import torch
    import lightning.pytorch as pl
    from chemprop import data as cdata, featurizers, models, nn as cnn

    pl.seed_everything(seed, workers=True)
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()

    def make(idx, shuffle=False, scaler=None, train=False):
        pts = [cdata.MoleculeDatapoint.from_smi(smis[i], y=np.array([y_all[i]], float)) for i in idx]
        ds = cdata.MoleculeDataset(pts, featurizer=feat)
        sc = None
        if train and task == "reg":
            sc = ds.normalize_targets()
        elif scaler is not None and task == "reg":
            ds.normalize_targets(scaler)
        return cdata.build_dataloader(ds, shuffle=shuffle, num_workers=workers), sc

    # inner validation split for early stopping, taken from the training folds only
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(tr))
    n_val = max(1, int(0.1 * len(tr)))
    inner_vl, inner_tr = tr[perm[:n_val]], tr[perm[n_val:]]

    tr_loader, scaler = make(inner_tr, shuffle=True, train=True)
    vl_loader, _ = make(inner_vl, scaler=scaler)
    ev_loader, _ = make(vl)
    te_loader, _ = make(te)

    mp = foundation_mp() if foundation else cnn.BondMessagePassing()
    agg = cnn.MeanAggregation()
    if task == "cls":
        ffn = cnn.BinaryClassificationFFN(input_dim=mp.output_dim)
        metrics = [cnn.metrics.BinaryAUROC()]
    else:
        transform = cnn.UnscaleTransform.from_standard_scaler(scaler) if scaler is not None else None
        ffn = cnn.RegressionFFN(input_dim=mp.output_dim, output_transform=transform)
        metrics = [cnn.metrics.RMSE()]
    model = models.MPNN(mp, agg, ffn, batch_norm=True, metrics=metrics)

    trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1,
                         max_epochs=epochs, enable_checkpointing=False, enable_progress_bar=False,
                         logger=False, enable_model_summary=False,
                         callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min")])
    trainer.fit(model, tr_loader, vl_loader)

    def predict(loader):
        out = trainer.predict(model, loader)
        return np.concatenate([o.numpy().ravel() for o in out])
    return predict(ev_loader), predict(te_loader)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="random", choices=["random", "scaffold"])
    ap.add_argument("--task", default="cls", choices=["cls", "reg"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--foundation", action="store_true", help="start from the CheMeleon foundation model")
    args = ap.parse_args()

    df = data.load_df()
    feats = data.get_features(df)
    smis = list(feats["smiles"])
    labels = feats["labels"].astype(int)
    pic = data.get_pic50(smis)
    y_all = labels.astype(float) if args.task == "cls" else pic
    s = splits.get_split(args.split, df)
    dev, folds, te = s["dev"], s["folds"], s["test"]

    oof_seeds, test_seeds = [], []
    t0 = time.time()
    for seed in args.seeds:
        oof = np.zeros(len(dev), np.float32)
        tests = []
        for k, (f_tr, f_vl) in enumerate(folds):
            ev, tt = run_fold(smis, y_all, dev[f_tr], dev[f_vl], te, args.task, seed + k, args.epochs,
                              foundation=args.foundation)
            oof[f_vl] = ev
            tests.append(tt)
            print(f"  seed {seed} fold {k} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
        oof_seeds.append(oof)
        test_seeds.append(np.mean(tests, 0))
    oof = np.mean(oof_seeds, 0)
    test = np.mean(test_seeds, 0)

    stem = "CheMeleon" if args.foundation else "Chemprop"
    name = stem if args.task == "cls" else stem + "-reg"
    out_dir = os.path.join(data.HERE, "results", "baselines", args.split)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}__graph.npz")
    if args.task == "cls":
        np.savez(path, oof=oof, oof_labels=labels[dev], test_probs=np.stack(test_seeds),
                 test_labels=labels[te], dev_idx=dev, test_idx=te)
        from sklearn.metrics import roc_auc_score
        print(f"{name}: OOF AUC {roc_auc_score(labels[dev], oof):.4f} | "
              f"test AUC {roc_auc_score(labels[te], test):.4f}")
    else:
        sig = 1 / (1 + np.exp(-2 * (oof - 5.5)))
        sig_t = 1 / (1 + np.exp(-2 * (test - 5.5)))
        np.savez(path, oof=sig, oof_pic50=oof, oof_labels=labels[dev],
                 test_probs=np.stack([1 / (1 + np.exp(-2 * (t - 5.5))) for t in test_seeds]),
                 test_pic50=np.stack(test_seeds), test_labels=labels[te], dev_idx=dev, test_idx=te)
        print(f"{name}: OOF RMSE {np.sqrt(np.mean((oof - pic[dev]) ** 2)):.4f} | "
              f"test RMSE {np.sqrt(np.mean((test - pic[te]) ** 2)):.4f}")
    print(f"written {path} | {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
