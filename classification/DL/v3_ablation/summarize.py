"""
Aggregates results/jobs + results/baselines into tables and a figure.

Metrics follow the notebook exactly:
  * CV table  : per-fold metrics on OOF predictions at the OOF max-F1 threshold (cell 17)
  * Test      : mean of the 5 fold models' test probabilities, threshold = OOF
                MCC-optimal on linspace(0.1, 0.9, 500) (cell 18)
  * Single    : train/val/test protocol (cell 15/16), threshold 0.6 and val max-F1
Significance of an ablation vs the full model:
  * seed noise : spread of the full model across 3 training seeds
  * DeLong     : paired test of the two ensemble test-set ROC-AUCs
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, cohen_kappa_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v3_core as C

RES = os.path.join(C.HERE, "results")
JOBS = os.path.join(RES, "jobs")

NB_CV = {"Accuracy": 0.9308, "Balanced Acc": 0.8821, "F1": 0.9565, "ROC-AUC": 0.9570, "AUPRC": 0.9851,
         "MCC": 0.7896, "Sensitivity": 0.9671, "Specificity": 0.7972, "Brier": 0.0595, "OOF AUC": 0.9549}
NB_TEST = {"Accuracy": 0.9273, "Balanced Acc": 0.8855, "F1": 0.9542, "ROC-AUC": 0.9658, "AUPRC": 0.9897,
           "MCC": 0.7778, "Sensitivity": 0.9571, "Specificity": 0.8140}
NB_SINGLE = {"Accuracy": 0.9297, "Balanced Acc": 0.8999, "F1": 0.9554, "ROC-AUC": 0.9570, "AUPRC": 0.9858,
             "MCC": 0.7899, "Sensitivity": 0.9510, "Specificity": 0.8488}


def thr_metrics(y, p, thr):
    pred = (p > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"Accuracy": accuracy_score(y, pred), "Balanced Acc": balanced_accuracy_score(y, pred),
            "Precision": precision_score(y, pred, zero_division=0), "F1": f1_score(y, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y, p), "AUPRC": average_precision_score(y, p),
            "MCC": matthews_corrcoef(y, pred), "Sensitivity": tp / max(1, tp + fn),
            "Specificity": tn / max(1, tn + fp), "Cohen k": cohen_kappa_score(y, pred),
            "Brier": brier_score_loss(y, p)}


def f1_thr(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.where((pr + rc) == 0, 1, pr + rc)
    return float(th[np.argmax(f1[:-1])])


def mcc_thr(y, p):
    ths = np.linspace(0.10, 0.90, 500)
    return float(ths[int(np.argmax([matthews_corrcoef(y, (p > t).astype(int)) for t in ths]))])


# ── DeLong (Sun & Xu 2014 fast version) ──────────────────────────────────────
def _midrank(x):
    j = np.argsort(x); z = x[j]; n = len(x); t = np.zeros(n); i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1; i = k
    out = np.empty(n); out[j] = t
    return out


def delong(y, p1, p2):
    y = np.asarray(y).astype(int)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    m, n = len(pos), len(neg)
    aucs, v10, v01 = [], [], []
    for p in (p1, p2):
        tx, ty, tz = _midrank(p[pos]), _midrank(p[neg]), _midrank(np.concatenate([p[pos], p[neg]]))
        aucs.append((tz[:m].sum() - m * (m + 1) / 2) / (m * n))
        v10.append((tz[:m] - tx) / n)
        v01.append(1 - (tz[m:] - ty) / m)
    s = np.cov(np.vstack(v10)) / m + np.cov(np.vstack(v01)) / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(max(var, 1e-12))
    return float(aucs[0] - aucs[1]), float(2 * stats.norm.sf(abs(z)))


# ── loading ──────────────────────────────────────────────────────────────────
def load_cv(variant, seed, n_dev):
    fs = [os.path.join(JOBS, f"cv__{variant}__s{seed}__f{f}.npz") for f in range(5)]
    if not all(os.path.exists(f) for f in fs):
        return None
    oof, oof_y = np.zeros(n_dev, np.float32), np.zeros(n_dev, int)
    fold_of, test = np.zeros(n_dev, int), []
    epochs, secs = [], []
    for f, path in enumerate(fs):
        z = np.load(path)
        meta = json.load(open(path.replace(".npz", ".json")))
        vi = np.array(meta["val_idx"])
        pos = np.searchsorted(DEV_SORTED, vi)
        pos = DEV_ORDER[pos]
        oof[pos], oof_y[pos], fold_of[pos] = z["val_probs"], z["val_labels"], f
        test.append(z["test_probs"])
        test_y = z["test_labels"].astype(int)
        epochs.append(meta["best_epoch"]); secs.append(meta["train_seconds"])
    return dict(oof=oof, oof_y=oof_y, fold_of=fold_of, test=np.stack(test), test_y=test_y,
                best_epochs=epochs, minutes=sum(secs) / 60)


def evaluate_cv(r):
    """Notebook cell 17 + 18 on one CV run."""
    oof, y, fo = r["oof"], r["oof_y"], r["fold_of"]
    thr_f1 = f1_thr(y, oof)
    per_fold = [thr_metrics(y[fo == f], oof[fo == f], thr_f1) for f in range(5)]
    cv = {k: float(np.mean([d[k] for d in per_fold])) for k in per_fold[0]}
    cv_sd = {k: float(np.std([d[k] for d in per_fold], ddof=1)) for k in per_fold[0]}
    cv["OOF AUC"] = float(roc_auc_score(y, oof))
    thr_mcc = mcc_thr(y, oof)
    ens = r["test"].mean(0)
    test = thr_metrics(r["test_y"], ens, thr_mcc)
    return dict(cv=cv, cv_sd=cv_sd, test=test, thr_mcc=thr_mcc, ens=ens,
                fold_auc=[roc_auc_score(y[fo == f], oof[fo == f]) for f in range(5)])


def fmt(v, d=4):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def main():
    global DEV_ORDER, DEV_SORTED
    df = C.load_df()
    tr, va, te = C.notebook_splits(df)
    dev_idx, folds = C.notebook_folds(df, tr, va)
    DEV_ORDER = np.argsort(dev_idx)
    DEV_SORTED = dev_idx[DEV_ORDER]
    n_dev = len(dev_idx)
    lines = ["# V3 reproduction and ablation study\n"]

    # ── 1. Reproduction ──────────────────────────────────────────────────────
    seeds = sorted({int(p.split("__s")[1].split("__")[0]) for p in glob.glob(os.path.join(JOBS, "cv__full__s*__f0.npz"))})
    full = {s: evaluate_cv(r) for s in seeds if (r := load_cv("full", s, n_dev)) is not None}
    keys_cv = ["OOF AUC", "ROC-AUC", "Accuracy", "Balanced Acc", "F1", "MCC", "AUPRC", "Sensitivity", "Specificity", "Brier"]
    keys_te = ["ROC-AUC", "AUPRC", "Accuracy", "Balanced Acc", "F1", "MCC", "Sensitivity", "Specificity"]
    repro_rows = []
    if full:
        lines.append("## 1. Reproduction of the notebook (full V3, best Optuna hyper-parameters)\n")
        lines.append("### 5-fold CV on train+val (OOF, notebook cell 17)\n")
        hdr = "| Metric | Notebook | " + " | ".join(f"seed {s}" for s in full) + " | mean ± sd over seeds |"
        lines += [hdr, "|" + "---|" * (3 + len(full))]
        for k in keys_cv:
            vals = [full[s]["cv"][k] for s in full]
            lines.append(f"| {k} | {fmt(NB_CV.get(k))} | " + " | ".join(fmt(v) for v in vals)
                         + f" | {np.mean(vals):.4f} ± {np.std(vals, ddof=1) if len(vals) > 1 else 0:.4f} |")
            repro_rows.append({"protocol": "cv", "metric": k, "notebook": NB_CV.get(k),
                               **{f"seed{s}": v for s, v in zip(full, vals)}})
        lines.append("\n### 5-fold ensemble on the held-out test set (notebook cell 18)\n")
        lines += [hdr.replace("mean ± sd over seeds", "mean ± sd over seeds"), "|" + "---|" * (3 + len(full))]
        for k in keys_te:
            vals = [full[s]["test"][k] for s in full]
            lines.append(f"| {k} | {fmt(NB_TEST.get(k))} | " + " | ".join(fmt(v) for v in vals)
                         + f" | {np.mean(vals):.4f} ± {np.std(vals, ddof=1) if len(vals) > 1 else 0:.4f} |")
            repro_rows.append({"protocol": "test_ensemble", "metric": k, "notebook": NB_TEST.get(k),
                               **{f"seed{s}": v for s, v in zip(full, vals)}})
        lines.append("\nThresholds (OOF MCC-optimal): " + ", ".join(f"seed {s}: {full[s]['thr_mcc']:.3f}" for s in full)
                     + "  (notebook: 0.430)\n")

    singles = {}
    for path in sorted(glob.glob(os.path.join(JOBS, "single__full__s*.npz"))):
        s = int(path.split("__s")[1].split(".")[0])
        z = np.load(path)
        meta = json.load(open(path.replace(".npz", ".json")))
        t_f1 = f1_thr(z["val_labels"], z["val_probs"])
        singles[s] = dict(m06=thr_metrics(z["test_labels"].astype(int), z["test_probs"], 0.6),
                          mval=thr_metrics(z["test_labels"].astype(int), z["test_probs"], t_f1),
                          thr=t_f1, val_auc=meta["best_val_auc"], ep=meta["best_epoch"])
    if singles:
        lines.append("### Single model, train/val/test protocol (notebook cells 15-16)\n")
        lines += ["| Metric | Notebook (thr 0.6) | " + " | ".join(f"seed {s} (0.6 / val-F1 thr)" for s in singles) + " |",
                  "|" + "---|" * (2 + len(singles))]
        lines.append(f"| best val AUROC | 0.9589 | " + " | ".join(f"{singles[s]['val_auc']:.4f} (ep {singles[s]['ep']})" for s in singles) + " |")
        for k in keys_te:
            lines.append(f"| {k} | {fmt(NB_SINGLE.get(k))} | " + " | ".join(
                f"{singles[s]['m06'][k]:.4f} / {singles[s]['mval'][k]:.4f}" for s in singles) + " |")
            repro_rows.append({"protocol": "single_test_thr0.6", "metric": k, "notebook": NB_SINGLE.get(k),
                               **{f"seed{s}": singles[s]["m06"][k] for s in singles}})
        lines.append("")
    if repro_rows:
        pd.DataFrame(repro_rows).to_csv(os.path.join(RES, "reproduction_table.csv"), index=False)

    # ── 2. Ablation ──────────────────────────────────────────────────────────
    abl_rows = []
    ref_seed = 42
    if ref_seed in full:
        ref = full[ref_seed]
        noise = {k: np.std([full[s]["test"][k] for s in full], ddof=1) if len(full) > 1 else np.nan
                 for k in ["ROC-AUC", "MCC", "AUPRC"]}
        noise_oof = np.std([full[s]["cv"]["OOF AUC"] for s in full], ddof=1) if len(full) > 1 else np.nan
        for v, (desc, _) in C.ABLATIONS.items():
            r = load_cv(v, ref_seed, n_dev)
            if r is None:
                continue
            e = evaluate_cv(r)
            d_auc, p_delong = (0.0, 1.0) if v == "full" else delong(r["test_y"], e["ens"], ref["ens"])
            fold_d = np.array(e["fold_auc"]) - np.array(ref["fold_auc"])
            p_fold = 1.0 if v == "full" else float(stats.ttest_rel(e["fold_auc"], ref["fold_auc"]).pvalue)
            abl_rows.append({
                "variant": v, "description": desc,
                "OOF AUC": e["cv"]["OOF AUC"], "CV MCC": e["cv"]["MCC"],
                "Test AUC": e["test"]["ROC-AUC"], "Test AUPRC": e["test"]["AUPRC"],
                "Test MCC": e["test"]["MCC"], "Test BalAcc": e["test"]["Balanced Acc"],
                "Test Spec": e["test"]["Specificity"],
                "dOOF AUC": e["cv"]["OOF AUC"] - ref["cv"]["OOF AUC"],
                "dTest AUC": e["test"]["ROC-AUC"] - ref["test"]["ROC-AUC"],
                "dTest MCC": e["test"]["MCC"] - ref["test"]["MCC"],
                "fold dAUC mean": float(fold_d.mean()), "fold dAUC sd": float(fold_d.std(ddof=1)),
                "p paired folds": p_fold, "p DeLong test": p_delong,
                "GPU min (5 folds)": r["minutes"], "best epochs": r["best_epochs"],
            })

    # classical baselines, same folds
    for path in sorted(glob.glob(os.path.join(RES, "baselines", "*.npz"))):
        z = np.load(path)
        name = os.path.basename(path)[:-4]
        oof, y = z["oof"], z["oof_labels"].astype(int)
        fold_of = np.zeros(len(y), int)
        for f, (_, vl) in enumerate(folds):
            fold_of[vl] = f
        e = evaluate_cv(dict(oof=oof, oof_y=y, fold_of=fold_of, test=z["test_probs"], test_y=z["test_labels"].astype(int)))
        row = {"variant": "baseline:" + name, "description": name.replace("__", " on "),
               "OOF AUC": e["cv"]["OOF AUC"], "CV MCC": e["cv"]["MCC"],
               "Test AUC": e["test"]["ROC-AUC"], "Test AUPRC": e["test"]["AUPRC"], "Test MCC": e["test"]["MCC"],
               "Test BalAcc": e["test"]["Balanced Acc"], "Test Spec": e["test"]["Specificity"]}
        if ref_seed in full:
            ref = full[ref_seed]
            row["dOOF AUC"] = e["cv"]["OOF AUC"] - ref["cv"]["OOF AUC"]
            row["dTest AUC"] = e["test"]["ROC-AUC"] - ref["test"]["ROC-AUC"]
            row["dTest MCC"] = e["test"]["MCC"] - ref["test"]["MCC"]
            fold_d = np.array(e["fold_auc"]) - np.array(ref["fold_auc"])
            row["fold dAUC mean"], row["fold dAUC sd"] = float(fold_d.mean()), float(fold_d.std(ddof=1))
            row["p paired folds"] = float(stats.ttest_rel(e["fold_auc"], ref["fold_auc"]).pvalue)
            row["p DeLong test"] = delong(z["test_labels"].astype(int), e["ens"], ref["ens"])[1]
        abl_rows.append(row)

    if abl_rows:
        tab = pd.DataFrame(abl_rows)
        tab.to_csv(os.path.join(RES, "ablation_table.csv"), index=False)
        lines.append("## 2. Ablation (5-fold CV, seed 42, same folds for every row)\n")
        if ref_seed in full and len(full) > 1:
            lines.append(f"Seed-to-seed noise of the full model (sd over {len(full)} seeds): "
                         f"OOF AUC ±{noise_oof:.4f}, test AUC ±{noise['ROC-AUC']:.4f}, test MCC ±{noise['MCC']:.4f}. "
                         "Differences smaller than ~2× these are not distinguishable from training noise.\n")
        cols = ["variant", "OOF AUC", "dOOF AUC", "Test AUC", "dTest AUC", "Test MCC", "dTest MCC",
                "Test Spec", "fold dAUC mean", "p paired folds", "p DeLong test"]
        lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, rw in tab.iterrows():
            cells = []
            for c in cols:
                v = rw.get(c)
                if c == "variant":
                    cells.append(str(v))
                elif c.startswith("d") or c.startswith("fold"):
                    cells.append("—" if pd.isna(v) else f"{v:+.4f}")
                elif c.startswith("p "):
                    cells.append("—" if pd.isna(v) else f"{v:.3f}")
                else:
                    cells.append(fmt(v))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        make_figure(tab, noise.get("ROC-AUC") if ref_seed in full else None,
                    noise_oof if ref_seed in full else None)

    with open(os.path.join(RES, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))


def make_figure(tab, noise_test, noise_oof):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = tab[tab.variant != "full"].copy()
    if t.empty or "dOOF AUC" not in t:
        return
    t = t.sort_values("dOOF AUC")
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.38 * len(t) + 1.8), sharey=True)
    for ax, col, nz, title in [(axes[0], "dOOF AUC", noise_oof, "Δ OOF ROC-AUC vs full V3 (5-fold CV)"),
                               (axes[1], "dTest AUC", noise_test, "Δ test ROC-AUC vs full V3 (5-model ensemble)")]:
        colors = ["#2a7ab9" if str(v).startswith("baseline") else ("#c0392b" if d < 0 else "#27ae60")
                  for v, d in zip(t.variant, t[col])]
        ax.barh(t.variant, t[col], color=colors)
        if nz is not None and not np.isnan(nz):
            ax.axvspan(-2 * nz, 2 * nz, color="grey", alpha=0.18, label="±2 sd seed noise (full model)")
            ax.legend(loc="lower right", fontsize=8)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RES, "ablation_delta_auc.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
