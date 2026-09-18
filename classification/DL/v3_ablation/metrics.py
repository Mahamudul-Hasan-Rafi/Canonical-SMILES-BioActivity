"""Metric definitions shared by report.py (identical to the notebook's cells 16-18)."""
import numpy as np
from scipy import stats
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, cohen_kappa_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_recall_curve, precision_score, roc_auc_score)


def thr_metrics(y, p, thr):
    pred = (p > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"Accuracy": accuracy_score(y, pred), "Balanced Acc": balanced_accuracy_score(y, pred),
            "Precision": precision_score(y, pred, zero_division=0), "F1": f1_score(y, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y, p), "AUPRC": average_precision_score(y, p),
            "MCC": matthews_corrcoef(y, pred), "Sensitivity": tp / max(1, tp + fn),
            "Specificity": tn / max(1, tn + fp), "Cohen k": cohen_kappa_score(y, pred),
            "Brier": brier_score_loss(y, p), "ECE": ece(y, p)}


def ece(y, p, bins=10):
    """Expected calibration error (equal-width bins)."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def f1_thr(y, p):
    """Notebook cell 17: threshold maximising F1 on the OOF predictions."""
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.where((pr + rc) == 0, 1, pr + rc)
    return float(th[np.argmax(f1[:-1])])


def mcc_thr(y, p):
    """Notebook cell 18: threshold maximising MCC on linspace(0.1, 0.9, 500)."""
    ths = np.linspace(0.10, 0.90, 500)
    return float(ths[int(np.argmax([matthews_corrcoef(y, (p > t).astype(int)) for t in ths]))])


def evaluate_cv(oof, oof_y, fold_of, test_probs, test_y):
    """Notebook cells 17+18 for one 5-fold run.
    cv   : per-fold metrics at the OOF max-F1 threshold (mean over folds) + OOF AUC
    test : mean of the fold models' test probabilities at the OOF MCC-optimal threshold
    test@0.5 : same ensemble at a fixed 0.5 threshold (shows raw calibration / imbalance)"""
    folds = sorted(set(fold_of.tolist()))
    t_f1 = f1_thr(oof_y, oof)
    per_fold = [thr_metrics(oof_y[fold_of == f], oof[fold_of == f], t_f1) for f in folds]
    cv = {k: float(np.mean([d[k] for d in per_fold])) for k in per_fold[0]}
    cv["OOF AUC"] = float(roc_auc_score(oof_y, oof))
    t_mcc = mcc_thr(oof_y, oof)
    ens = test_probs.mean(0)
    return dict(cv=cv, test=thr_metrics(test_y, ens, t_mcc), test05=thr_metrics(test_y, ens, 0.5),
                thr_mcc=t_mcc, ens=ens, test_y=test_y,
                fold_auc=[float(roc_auc_score(oof_y[fold_of == f], oof[fold_of == f])) for f in folds])


# ── DeLong test for two correlated ROC AUCs (Sun & Xu 2014) ──────────────────
def _midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def delong(y, p1, p2):
    """Returns (AUC1 - AUC2, two-sided p)."""
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


def mean_sd(vals):
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    if not vals:
        return np.nan, np.nan
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
