"""
Hyperparameter search for the NEW deep branch.

The tuned hyperparameters in HP_SETS were optimised once, for the V3 architecture with the
MoLFormer encoder (hidden 768). The deep branch now used in the hybrid is a different network:
ChemBERTa (hidden 384, 2.3x fewer parameters), a D-MPNN branch, a potency head and a retrieval
module. Its learning rate and dropout were inherited, never fitted.

Searched on CV fold 0 of the notebook split, validation fold only - the held-out test set is
never touched. V5-MT is scored by validation AUROC, V8-R2 by validation pIC50 RMSE, each by the
criterion it is trained on. Writes results/hpo_pilot.json.
"""
import itertools
import json
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def run(job):
    variant, lr, dropout = job
    import numpy as np
    import torch
    import backbones
    import core
    import data
    import gpu
    import splits
    torch.cuda.set_per_process_memory_fraction(0.22)
    core.setup_torch()
    df = data.load_df()
    feats = data.get_features(df)
    s = splits.get_split("random", df)
    tr, vl = s["dev"][s["folds"][0][0]], s["dev"][s["folds"][0][1]]
    cfg = core.make_cfg(variant, backbone="chemberta_mlm", max_epochs=40,
                        hp_override={"learning_rate": lr, "dropout": dropout})
    out = core.train_one(cfg, feats, tr, vl, {}, 7, torch.device("cuda"),
                         backbones.load_tokenizer("chemberta_mlm"), guard=gpu.ThermalGuard(hot_c=88, cool_c=80))
    pic = data.get_pic50(feats["smiles"])[vl]
    return dict(variant=variant, lr=lr, dropout=dropout, auc=out["best_val_auc"],
                rmse=core.rmse_from_probs(out["val_probs"], pic), epoch=out["best_epoch"],
                minutes=out["train_seconds"] / 60)


def main():
    lrs = [9.8917e-6, 2e-5, 4e-5]
    drops = [0.1, 0.2]
    variants = sys.argv[1:] or ["graph_mt", "reg_first_delta"]
    jobs = [(v, lr, d) for v in variants for lr, d in itertools.product(lrs, drops)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(3) as pool:
        res = pool.map(run, jobs)
    for v in variants:
        rows = [r for r in res if r["variant"] == v]
        key = "rmse" if v.startswith("reg") else "auc"
        rows.sort(key=lambda r: (r["rmse"] if key == "rmse" else -r["auc"]))
        print(f"\n{v}  (ranked by val {key})")
        for r in rows:
            print(f"   lr {r['lr']:.2e}  dropout {r['dropout']:.1f} | val AUC {r['auc']:.4f} | "
                  f"val RMSE {r['rmse']:.4f} | best epoch {r['epoch']:>2} | {r['minutes']:.1f} min")
    path = os.path.join(HERE, "results", "hpo_pilot.json")
    old = json.load(open(path)) if os.path.exists(path) else []
    with open(path, "w") as fh:
        json.dump(old + res, fh, indent=1)
    print(f"\npilot done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
