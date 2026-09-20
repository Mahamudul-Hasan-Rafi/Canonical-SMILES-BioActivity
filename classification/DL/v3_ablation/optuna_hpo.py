"""
Full Optuna study for the NEW deep branch, replicating the protocol of the V3 notebook.

Same as the notebook's V3 HPO (cell 10):
  * TPESampler(seed=42) + HyperbandPruner(min_resource=5, max_resource=25, reduction_factor=3)
  * 40 trials, 50 % subsample of the training data, MAX_EPOCHS 25, patience 5
  * the eight architecture/optimiser parameters it searched, with the same ranges

Two deliberate amendments, both because the architecture changed:
  * the learning-rate ceiling is raised from 2e-5 to 5e-5 (ChemBERTa is hidden 384 with 2.3x
    fewer trainable parameters than MoLFormer, so its optimum could sit above the old ceiling)
  * the parameters the new modules introduced are searched too: new_lr_mult (from-scratch
    modules), aux_pic50 / delta_w (potency losses) and retrieval k

Objective: validation ROC-AUC for the classification variant (as in the notebook), validation
pIC50 RMSE for the regression-first variant, which is what that branch is trained for.
Searched on CV fold 0 of the notebook split; the held-out test set is never touched.

  python optuna_hpo.py graph_mt          # classification branch
  python optuna_hpo.py reg_first_delta   # potency branch

Writes results/optuna_<variant>.json (same shape as optuna_results_v3__.json).
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

N_TRIALS = 40
MAX_EPOCHS = 25          # notebook: MAX_EPOCHS = 25
PATIENCE = 5             # notebook: patience = 5
SUBSAMPLE = 0.5          # notebook: train_df.sample(frac=0.5, random_state=42)


def space(trial, variant):
    """The notebook's eight parameters (ranges unchanged except the LR ceiling), plus the
    parameters the new modules introduced."""
    hp = {
        "num_heads": trial.suggest_categorical("num_heads", [4, 8]),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [256, 512]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.4, step=0.1),
        "num_classifier_layers": trial.suggest_int("num_classifier_layers", 2, 4, step=1),
        "n_cross_layers": trial.suggest_int("n_cross_layers", 2, 3),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
    }
    extra = {"new_lr_mult": trial.suggest_categorical("new_lr_mult", [1.0, 3.0, 10.0, 30.0])}
    if variant == "graph_mt":
        extra["aux_pic50"] = trial.suggest_float("aux_pic50", 0.03, 0.5, log=True)
    else:
        extra["delta_w"] = trial.suggest_float("delta_w", 0.03, 0.5, log=True)
        extra["retrieval"] = trial.suggest_categorical("retrieval", [3, 5, 10])
    return hp, extra


def main():
    import numpy as np
    import optuna
    import torch
    from optuna.pruners import HyperbandPruner
    from optuna.samplers import TPESampler
    import backbones
    import core
    import data
    import gpu
    import splits

    variant = sys.argv[1] if len(sys.argv) > 1 else "graph_mt"
    minimise = variant.startswith("reg")
    torch.cuda.set_per_process_memory_fraction(0.45)
    core.setup_torch()
    df = data.load_df()
    feats = data.get_features(df)
    s = splits.get_split("random", df)
    tr_full, vl = s["dev"][s["folds"][0][0]], s["dev"][s["folds"][0][1]]
    rng = np.random.default_rng(42)
    tr = np.sort(rng.choice(tr_full, size=int(len(tr_full) * SUBSAMPLE), replace=False))
    pic_vl = data.get_pic50(feats["smiles"])[vl]
    tok = backbones.load_tokenizer("chemberta_mlm")
    guard = gpu.ThermalGuard(hot_c=88, cool_c=80)
    t0 = time.time()

    def objective(trial):
        hp, extra = space(trial, variant)
        cfg = core.make_cfg(variant, backbone="chemberta_mlm", max_epochs=MAX_EPOCHS,
                            patience=PATIENCE, hp_override=hp, **extra)
        state = {"best": None}

        def on_epoch(ep, auc, score):
            val = -score if minimise else auc          # score = -rmse for the regression variant
            state["best"] = val if state["best"] is None else (min if minimise else max)(state["best"], val)
            trial.report(val, ep)
            return not trial.should_prune()

        out = core.train_one(cfg, feats, tr, vl, {}, 7, torch.device("cuda"), tok,
                             guard=guard, on_epoch=on_epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return (core.rmse_from_probs(out["val_probs"], pic_vl) if minimise else out["best_val_auc"])

    study = optuna.create_study(
        direction="minimize" if minimise else "maximize",
        sampler=TPESampler(seed=42),
        pruner=HyperbandPruner(min_resource=5, max_resource=MAX_EPOCHS, reduction_factor=3))
    study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True)

    done = [t for t in study.trials if t.value is not None]
    print(f"\n{variant}: {len(done)} completed, {len(study.trials) - len(done)} pruned, "
          f"{(time.time() - t0) / 60:.1f} min")
    print(f"best value: {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k:<24}: {v}")
    path = os.path.join(HERE, "results", f"optuna_{variant}.json")
    with open(path, "w") as fh:
        json.dump({"variant": variant, "objective": "rmse" if minimise else "auc",
                   "best_params": study.best_params, "best_value": study.best_value,
                   "n_trials": len(study.trials), "n_completed": len(done),
                   "minutes": (time.time() - t0) / 60,
                   "trials": [{"number": t.number, "value": t.value, "state": str(t.state),
                               "params": t.params} for t in study.trials]}, fh, indent=1)
    print(f"written: {path}")


if __name__ == "__main__":
    main()
