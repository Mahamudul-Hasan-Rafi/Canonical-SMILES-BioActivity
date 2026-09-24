"""
Read out V5-MT's auxiliary potency head.

The proposed classifier is LightGBM + V5-MT. V5-MT already carries a pIC50 head (auxiliary, loss
weight 0.1) whose output is discarded during normal scoring, so the same two components can serve
both tasks: LightGBM for the class decision and potency, V5-MT for both as well. This retrains the
15 networks per split and stores their predicted pIC50 alongside a LightGBM regressor, so the
potency model needs no extra architecture beyond what the classifier already contains.

Nothing in results/store is touched; predictions go to results/potency_v5mt_<split>.npz.

  python v5mt_potency.py --split random --workers 3
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def run(job):
    split, seed, fold = job
    import torch
    import backbones
    import core
    import data
    import gpu
    import splits
    torch.cuda.set_per_process_memory_fraction(0.3)
    core.setup_torch()
    df = data.load_df()
    feats = data.get_features(df)
    s = splits.get_split(split, df)
    tr, vl = s["dev"][s["folds"][fold][0]], s["dev"][s["folds"][fold][1]]
    cfg = core.make_cfg("graph_mt", backbone="chemberta_mlm")
    out = core.train_one(cfg, feats, tr, vl, {"test": s["test"]}, seed, torch.device("cuda"),
                         backbones.load_tokenizer("chemberta_mlm"),
                         guard=gpu.ThermalGuard(hot_c=88, cool_c=80), return_pic50=True)
    return dict(split=split, seed=seed, fold=fold, val_idx=s["folds"][fold][1],
                val_pic50=out["val_pic50"], test_pic50=out["test_pic50"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="random", choices=["random", "scaffold"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 2024])
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    import data
    import splits
    df = data.load_df()
    s = splits.get_split(args.split, df)
    jobs = [(args.split, sd, f) for sd in args.seeds for f in range(5)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        res = pool.map(run, jobs)

    oof_seeds, test_seeds = [], []
    for sd in args.seeds:
        oof = np.zeros(len(s["dev"]), np.float32)
        tests = []
        for r in res:
            if r["seed"] != sd:
                continue
            oof[r["val_idx"]] = r["val_pic50"]
            tests.append(r["test_pic50"])
        oof_seeds.append(oof)
        test_seeds.append(np.mean(tests, 0))
    oof, test = np.mean(oof_seeds, 0), np.mean(test_seeds, 0)

    pic = data.get_pic50(list(data.get_features(df)["smiles"]))
    rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))
    path = os.path.join(HERE, "results", f"potency_v5mt_{args.split}.npz")
    np.savez(path, oof_pic50=oof, test_pic50=np.stack(test_seeds), dev_idx=s["dev"], test_idx=s["test"])
    print(f"V5-MT auxiliary head, {args.split}: OOF RMSE {rmse(oof - pic[s['dev']]):.4f} | "
          f"test RMSE {rmse(test - pic[s['test']]):.4f} | {(time.time() - t0) / 60:.1f} min")
    print("written", path)


if __name__ == "__main__":
    main()
