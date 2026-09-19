"""
Pilot: learning-rate multiplier for the V4 modules trained from scratch (D-MPNN, token
fusion). V3's tuned lr (9.9e-6) was chosen for fine-tuning a pretrained encoder; the new
modules may need more. Compares new_lr_mult 1 vs 10 on CV fold 0 of the notebook split
(validation fold only; the held-out test set is never used), seed 7, up to 30 epochs.
Writes results/v4_pilot_lr.json.
"""
import json
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def run(args):
    variant, mult = args
    import torch
    import backbones
    import core
    import data
    import gpu
    import splits
    torch.cuda.set_per_process_memory_fraction(0.2)
    core.setup_torch()
    df = data.load_df()
    feats = data.get_features(df)
    s = splits.get_split("random", df)
    tr, vl = s["dev"][s["folds"][0][0]], s["dev"][s["folds"][0][1]]
    cfg = core.make_cfg(variant, backbone="chemberta_mlm", max_epochs=30, new_lr_mult=mult)
    out = core.train_one(cfg, feats, tr, vl, {}, 7, torch.device("cuda"), backbones.load_tokenizer("chemberta_mlm"),
                         guard=gpu.ThermalGuard(hot_c=88, cool_c=80))
    return dict(variant=variant, mult=mult, best_val_auc=out["best_val_auc"], best_epoch=out["best_epoch"],
                curve=[h[2] for h in out["history"]], minutes=out["train_seconds"] / 60)


def main():
    jobs = [(v, float(m)) for v, m in (a.split(":") for a in sys.argv[1:])] or [
        ("token_fusion", 1.0), ("token_fusion", 10.0), ("graph_branch", 1.0), ("graph_branch", 10.0)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(4) as pool:
        res = pool.map(run, jobs)
    for r in res:
        c = r["curve"]
        print(f"{r['variant']:<13} lr x{r['mult']:<4} best val AUC {r['best_val_auc']:.4f} at epoch {r['best_epoch']:>2} | "
              f"AUC at ep 5/10/20: {c[4]:.4f} / {c[9]:.4f} / {c[min(19, len(c) - 1)]:.4f} | {r['minutes']:.1f} min")
    path = os.path.join(HERE, "results", "v4_pilot_lr.json")
    old = json.load(open(path)) if os.path.exists(path) else []
    with open(path, "w") as fh:
        json.dump(old + res, fh, indent=1)
    print(f"pilot done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
