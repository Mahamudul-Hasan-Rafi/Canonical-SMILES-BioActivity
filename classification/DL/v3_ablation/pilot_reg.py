"""
Pilot for the regression-first study (V8): which training protocol gives the most precise
pIC50 prediction? Compares, on CV fold 0 of the notebook split only (validation fold; the
held-out test set is never touched), seed 7:

  A  huber + class sampler + best epoch by AUC   -- the V5-REG protocol, the baseline
  B  mse   + no sampler     + best epoch by RMSE -- V8-R1, all three changes
  C  mse   + class sampler  + best epoch by RMSE -- isolates the sampler
  D  huber + no sampler     + best epoch by RMSE -- isolates the loss

Primary number: validation RMSE in pIC50 units. Writes results/reg_pilot.json.
"""
import json
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

GRID = {"A_v5reg":      dict(reg_loss="huber", sampler=True,  select_metric="auc"),
        "B_all3":       dict(reg_loss="mse",   sampler=False, select_metric="rmse"),
        "C_sampler_on": dict(reg_loss="mse",   sampler=True,  select_metric="rmse"),
        "D_huber":      dict(reg_loss="huber", sampler=False, select_metric="rmse")}


def run(spec):
    name, fold = spec.split("@") if "@" in spec else (spec, "0")
    fold = int(fold)
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
    tr, vl = s["dev"][s["folds"][fold][0]], s["dev"][s["folds"][fold][1]]
    cfg = core.make_cfg("graph_reg", backbone="chemberta_mlm", max_epochs=30, **GRID[name])
    out = core.train_one(cfg, feats, tr, vl, {}, 7, torch.device("cuda"),
                         backbones.load_tokenizer("chemberta_mlm"), guard=gpu.ThermalGuard(hot_c=88, cool_c=80))
    pic = data.get_pic50(feats["smiles"])[vl]
    pred = core.probs_to_pic50(out["val_probs"])
    return dict(name=name, fold=fold, **GRID[name], rmse=core.rmse_from_probs(out["val_probs"], pic),
                mae=float(abs(pred - pic).mean()), auc=out["best_val_auc"], epoch=out["best_epoch"],
                epochs_run=out["epochs_run"], minutes=out["train_seconds"] / 60)


def main():
    jobs = sys.argv[1:] or list(GRID)
    t0 = time.time()
    with mp.get_context("spawn").Pool(min(4, len(jobs))) as pool:
        res = pool.map(run, jobs)
    res.sort(key=lambda r: r["rmse"])
    print(f"{'config':<13} {'loss':<6} {'sampler':<8} {'select':<7} {'val RMSE':>9} {'MAE':>7} {'AUC':>7} {'ep':>4} {'min':>6}")
    for r in res:
        print(f"{r['name']}@{r['fold']:<9} {r['reg_loss']:<6} {str(r['sampler']):<8} {r['select_metric']:<7} "
              f"{r['rmse']:>9.4f} {r['mae']:>7.4f} {r['auc']:>7.4f} {r['epoch']:>4} {r['minutes']:>6.1f}")
    path = os.path.join(HERE, "results", "reg_pilot.json")
    old = json.load(open(path)) if os.path.exists(path) else []
    with open(path, "w") as fh:
        json.dump(old + res, fh, indent=1)
    print(f"pilot done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
