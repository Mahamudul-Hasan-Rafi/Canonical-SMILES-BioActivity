"""
Step 1 of the verification — no training, only the models you already saved.

  A. Port check      : my V3Ablatable (loaded with your saved weights) gives the
                       same logits as your pickled FineTunedBERTaECFP_v3.
  B. Padding check   : dynamic padding == padding to 512 (same logits).
  C. Re-evaluation   : re-scores models/kfold_v3_fold*.pth and
                       models/best_finetuned_model_v3__.pth on the test set,
                       10 times each (MoLFormer redraws random attention features
                       on every forward pass, so the test score is itself a
                       random variable) and compares with the notebook's printout.

Writes results/saved_model_check.json
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v3_core as C
from v3_core import *  # noqa  (class names the pickles need live in __main__)

import __main__
from notebook_classes import FineTunedBERTaECFP_v3, FocalLoss as _NBFocal  # noqa
__main__.FineTunedBERTaECFP_v3 = FineTunedBERTaECFP_v3

from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
                             average_precision_score, matthews_corrcoef, precision_score,
                             recall_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer

MODELS = os.path.join(os.path.dirname(C.HERE), "models")
OUT = os.path.join(C.HERE, "results")
os.makedirs(OUT, exist_ok=True)
DEV = torch.device("cuda")
C.setup_torch()


def metrics(y, p, thr):
    pred = (p > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {"Accuracy": accuracy_score(y, pred), "Balanced Acc": balanced_accuracy_score(y, pred),
            "Precision": precision_score(y, pred, zero_division=0), "Recall": recall_score(y, pred),
            "F1": f1_score(y, pred), "ROC-AUC": roc_auc_score(y, p), "AUPRC": average_precision_score(y, p),
            "MCC": matthews_corrcoef(y, pred), "Specificity": tn / (tn + fp)}


def port_from_saved(nb_model):
    """Build V3Ablatable with the notebook model's hyper-params and copy weights."""
    hp = dict(C.BEST_PARAMS)
    hp["num_heads"] = nb_model.cross_modal.layers[0].self_attn.num_heads
    hp["n_cross_layers"] = len(nb_model.cross_modal.layers)
    lin = [m for m in nb_model.classifier if isinstance(m, torch.nn.Linear)]
    hp["num_classifier_layers"] = len(lin) - 1
    hp["hidden_dim"] = lin[0].out_features
    cfg = C.RunCfg(hp=hp, n_pool_layers=nb_model.n_pool_layers)
    m = C.build_model(cfg)
    missing, unexpected = m.load_state_dict(nb_model.state_dict(), strict=False)
    return m, missing, unexpected


def main():
    t_start = time.time()
    df = C.load_df()
    feats = C.get_features(df)
    tok = C.load_tokenizer()
    C.register_remote_code()
    tr, va, te = C.notebook_splits(df)
    dev_idx, folds = C.notebook_folds(df, tr, va)
    collate_dyn = C.make_collate(tok.pad_token_id)
    collate_512 = C.make_collate(tok.pad_token_id, fixed_len=512)
    report = {}

    # ── A + B on fold-1 model ────────────────────────────────────────────────
    nb = torch.load(os.path.join(MODELS, "kfold_v3_fold1.pth"), weights_only=False, map_location=DEV).eval()
    port, missing, unexpected = port_from_saved(nb)
    port = port.to(DEV).eval()
    sc = StandardScaler().fit(feats["desc_raw"][dev_idx[folds[0][0]]])
    ds = C.V3Dataset(feats, te[:128], tok, sc)
    b_dyn, b_512 = collate_dyn([ds[i] for i in range(len(ds))]), collate_512([ds[i] for i in range(len(ds))])

    def logits(model, b):
        torch.manual_seed(0)   # same random-feature draw for both
        with torch.no_grad():
            return model(**{k: v.to(DEV) for k, v in b.items() if k != "labels"}).float().cpu()
    l_nb512, l_port512, l_portdyn = logits(nb, b_512), logits(port, b_512), logits(port, b_dyn)
    report["port_check"] = {
        "missing_keys": list(missing), "unexpected_keys": list(unexpected),
        "max_abs_diff_notebook_vs_port": float((l_nb512 - l_port512).abs().max()),
        "max_abs_diff_pad512_vs_dynamic": float((l_port512 - l_portdyn).abs().max()),
        "dynamic_len": int(b_dyn["input_ids"].shape[1]),
    }
    print("Port / padding check:", report["port_check"])

    # speed of one forward: 512 vs dynamic
    def bench(b, n=20):
        bb = {k: v.to(DEV) for k, v in b.items() if k != "labels"}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(3):
                port(**bb)
            torch.cuda.synchronize(); t = time.time()
            for _ in range(n):
                port(**bb)
            torch.cuda.synchronize()
        return (time.time() - t) / n
    report["forward_ms_128mols"] = {"pad512": bench(b_512) * 1e3, "dynamic": bench(b_dyn) * 1e3}
    print("Forward time (ms, 128 molecules):", report["forward_ms_128mols"])
    del nb, port
    torch.cuda.empty_cache()

    # ── C: re-evaluate the saved models ──────────────────────────────────────
    y_te = feats["labels"][te].astype(int)
    N_REP = 10
    fold_probs = {}          # (fold, rep) -> probs with own scaler
    fold_probs_last = {}     # probs with last fold's scaler (what the notebook did)
    scalers = [StandardScaler().fit(feats["desc_raw"][dev_idx[f_tr]]) for f_tr, _ in folds]
    for f in range(5):
        m = torch.load(os.path.join(MODELS, f"kfold_v3_fold{f+1}.pth"), weights_only=False, map_location=DEV).eval()
        ds_own = C.V3Dataset(feats, te, tok, scalers[f])
        ds_last = C.V3Dataset(feats, te, tok, scalers[-1])
        for r in range(N_REP):
            fold_probs[(f, r)] = C.predict(m, ds_own, DEV, collate_dyn, eval_seed=r)
            fold_probs_last[(f, r)] = C.predict(m, ds_last, DEV, collate_dyn, eval_seed=r)
        print(f"fold {f+1} scored")
        del m
        torch.cuda.empty_cache()

    NB_KF_THR = 0.430
    ens_last = [metrics(y_te, np.mean([fold_probs_last[(f, r)] for f in range(5)], 0), NB_KF_THR) for r in range(N_REP)]
    ens_own = [metrics(y_te, np.mean([fold_probs[(f, r)] for f in range(5)], 0), NB_KF_THR) for r in range(N_REP)]

    def summarise(lst):
        return {k: {"mean": float(np.mean([d[k] for d in lst])), "std": float(np.std([d[k] for d in lst])),
                    "min": float(np.min([d[k] for d in lst])), "max": float(np.max([d[k] for d in lst]))}
                for k in lst[0]}
    report["kfold_ensemble_test_notebook_scaler"] = summarise(ens_last)
    report["kfold_ensemble_test_own_scaler"] = summarise(ens_own)
    report["kfold_ensemble_notebook_reported"] = {
        "Accuracy": 0.9273, "Balanced Acc": 0.8855, "Precision": 0.9513, "Recall": 0.9571, "F1": 0.9542,
        "ROC-AUC": 0.9658, "AUPRC": 0.9897, "MCC": 0.7778, "Specificity": 0.8140}

    # OOF saved by the notebook -> recompute CV numbers & thresholds
    oof_p = np.load(os.path.join(os.path.dirname(C.HERE), "oof_probs_full.npy"))
    oof_y = np.load(os.path.join(os.path.dirname(C.HERE), "oof_lbls_full.npy"))
    thrs = np.linspace(0.10, 0.90, 500)
    mcc = [matthews_corrcoef(oof_y, (oof_p > t).astype(int)) for t in thrs]
    report["oof_recheck"] = {"oof_auc": float(roc_auc_score(oof_y, oof_p)),
                             "mcc_opt_thr": float(thrs[int(np.argmax(mcc))]), "mcc_max": float(np.max(mcc)),
                             "labels_match_dev_order": bool((oof_y == feats["labels"][dev_idx].astype(int)).all())}

    # single V3 model (notebook cell 16: threshold hard-coded 0.6; val-opt was 0.574)
    single = torch.load(os.path.join(MODELS, "best_finetuned_model_v3__.pth"), weights_only=False, map_location=DEV).eval()
    sc_tr = StandardScaler().fit(feats["desc_raw"][tr])
    ds_te = C.V3Dataset(feats, te, tok, sc_tr)
    s06, s574 = [], []
    for r in range(N_REP):
        p = C.predict(single, ds_te, DEV, collate_dyn, eval_seed=r)
        s06.append(metrics(y_te, p, 0.6)); s574.append(metrics(y_te, p, 0.574))
    report["single_test_thr0.6"] = summarise(s06)
    report["single_test_thr0.574"] = summarise(s574)
    report["single_notebook_reported"] = {"Accuracy": 0.9297, "Balanced Acc": 0.8999, "F1": 0.9554,
                                          "ROC-AUC": 0.9570, "AUPRC": 0.9858, "MCC": 0.7899, "Specificity": 0.8488}
    report["seconds"] = time.time() - t_start

    with open(os.path.join(OUT, "saved_model_check.json"), "w") as f:
        json.dump(report, f, indent=2)
    for k in ["kfold_ensemble_test_notebook_scaler", "kfold_ensemble_test_own_scaler",
              "single_test_thr0.6", "single_test_thr0.574"]:
        print("\n" + k)
        for m_, s in report[k].items():
            print(f"  {m_:<13} {s['mean']:.4f} ± {s['std']:.4f}  [{s['min']:.4f}, {s['max']:.4f}]")
    print("\nOOF recheck:", report["oof_recheck"])
    print(f"done in {report['seconds']:.0f}s")


if __name__ == "__main__":
    main()
