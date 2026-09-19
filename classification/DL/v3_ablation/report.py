"""
Builds results/REPORT.md (+ CSVs + figures) from results/store, results/baselines
and results/explain. Works on partial results: missing runs are shown as '—'.

  python report.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data
import experiments as E
import metrics as M
import splits
import backbones
import core

RES = os.path.join(HERE, "results")
STORE = os.path.join(RES, "store")
SEEDS = E.REPRO_SEEDS
KEYS = ["ROC-AUC", "AUPRC", "MCC", "Balanced Acc", "Sensitivity", "Specificity", "F1", "Brier"]

NB_CV = {"OOF AUC": 0.9549, "ROC-AUC": 0.9570, "Accuracy": 0.9308, "Balanced Acc": 0.8821, "F1": 0.9565,
         "MCC": 0.7896, "AUPRC": 0.9851, "Sensitivity": 0.9671, "Specificity": 0.7972, "Brier": 0.0595}
NB_TEST = {"ROC-AUC": 0.9658, "AUPRC": 0.9897, "Accuracy": 0.9273, "Balanced Acc": 0.8855, "F1": 0.9542,
           "MCC": 0.7778, "Sensitivity": 0.9571, "Specificity": 0.8140}
NB_SINGLE = {"ROC-AUC": 0.9570, "AUPRC": 0.9858, "Accuracy": 0.9297, "Balanced Acc": 0.8999, "F1": 0.9554,
             "MCC": 0.7899, "Sensitivity": 0.9510, "Specificity": 0.8488}


class Store:
    def __init__(self):
        self.df = data.load_df()
        self.y = self.df["bioactivity"].values.astype(int)
        self.splits = {n: splits.get_split(n, self.df) for n in ("random", "scaffold")}
        self._cache = {}

    def cv(self, split="random", variant="full", backbone="molformer", hp_set="tuned", seed=42):
        key = (split, variant, backbone, hp_set, seed)
        if key in self._cache:
            return self._cache[key]
        s = self.splits[split]
        dev = s["dev"]
        pos_of = {int(v): k for k, v in enumerate(dev)}
        oof, fold_of = np.full(len(dev), np.nan), np.full(len(dev), -1)
        test, test_y, minutes, ntr = [], None, 0.0, None
        for f in range(5):
            jid = E.job_id(E.job("cv", split, backbone, hp_set, variant, seed, f))
            path = os.path.join(STORE, jid + ".npz")
            if not os.path.exists(path.replace(".npz", ".json")):
                self._cache[key] = None
                return None
            z = np.load(path)
            meta = json.load(open(path.replace(".npz", ".json")))
            pos = [pos_of[int(i)] for i in z["val_idx"]]
            oof[pos], fold_of[pos] = z["val_probs"], f
            test.append(z["test_probs"])
            test_y = z["test_labels"].astype(int)
            minutes += meta["train_seconds"] / 60
            ntr = meta["n_trainable"]
        r = M.evaluate_cv(oof, self.y[dev], fold_of, np.stack(test), test_y)
        r.update(minutes=minutes, n_trainable=ntr, oof=oof, oof_y=self.y[dev])
        self._cache[key] = r
        return r

    def single(self, seed, split="random"):
        jid = E.job_id(E.job("single", split, seed=seed))
        path = os.path.join(STORE, jid + ".npz")
        if not os.path.exists(path):
            return None
        z = np.load(path)
        meta = json.load(open(path.replace(".npz", ".json")))
        yv, yt = z["val_labels"].astype(int), z["test_labels"].astype(int)
        return dict(m06=M.thr_metrics(yt, z["test_probs"], 0.6),
                    mval=M.thr_metrics(yt, z["test_probs"], M.f1_thr(yv, z["val_probs"])),
                    val_auc=meta["best_val_auc"], ep=meta["best_epoch"])

    def baseline(self, split, name):
        path = os.path.join(RES, "baselines", split, name + ".npz")
        if not os.path.exists(path):
            return None
        z = np.load(path)
        s = self.splits[split]
        fold_of = np.zeros(len(s["dev"]), int)
        for f, (_, vl) in enumerate(s["folds"]):
            fold_of[vl] = f
        return M.evaluate_cv(z["oof"], z["oof_labels"].astype(int), fold_of, z["test_probs"], z["test_labels"].astype(int))


def f4(v):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}"


def ms(vals, d=4):
    m, s = M.mean_sd(vals)
    if np.isnan(m):
        return "—"
    return f"{m:.{d}f} ± {s:.{d}f}" if len(vals) > 1 else f"{m:.{d}f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out + [""]


def seeds_runs(st, **kw):
    return [r for r in (st.cv(seed=s, **kw) for s in SEEDS) if r is not None]


# ─────────────────────────────────────────────────────────────────────────────
def sec_repro(st):
    L = ["## 1. Reproduction of the notebook (full V3, tuned hyper-parameters, random split)\n"]
    runs = {s: st.cv(seed=s) for s in SEEDS}
    runs = {s: r for s, r in runs.items() if r}
    if not runs:
        return L + ["_not available yet_\n"]
    L.append("### 5-fold CV (out-of-fold, notebook cell 17)\n")
    hdr = ["Metric", "Notebook"] + [f"seed {s}" for s in runs] + ["mean ± sd"]
    L += table(hdr, [[k, f4(NB_CV.get(k))] + [f4(r["cv"][k]) for r in runs.values()] + [ms([r["cv"][k] for r in runs.values()])]
                     for k in ["OOF AUC"] + KEYS])
    L.append("### 5-model ensemble on the held-out test set (notebook cell 18)\n")
    L += table(hdr, [[k, f4(NB_TEST.get(k))] + [f4(r["test"][k]) for r in runs.values()] + [ms([r["test"][k] for r in runs.values()])]
                     for k in KEYS[:-1]])
    L.append("Thresholds (OOF MCC-optimal): " + ", ".join(f"seed {s}: {r['thr_mcc']:.3f}" for s, r in runs.items())
             + " (notebook 0.430)\n")
    sg = {s: st.single(s) for s in SEEDS}
    sg = {s: r for s, r in sg.items() if r}
    if sg:
        L.append("### Single model, train/val/test (notebook cells 15-16), threshold 0.6\n")
        L += table(["Metric", "Notebook"] + [f"seed {s}" for s in sg] + ["mean ± sd"],
                   [["best val AUROC", "0.9589"] + [f4(r["val_auc"]) for r in sg.values()] + [ms([r["val_auc"] for r in sg.values()])]]
                   + [[k, f4(NB_SINGLE.get(k))] + [f4(r["m06"][k]) for r in sg.values()] + [ms([r["m06"][k] for r in sg.values()])]
                      for k in KEYS[:-1]])
    return L


def sec_ablation(st):
    L = ["## 2. Ablation (random split, 5-fold CV, seed 42, identical folds)\n"]
    ref = st.cv(seed=42)
    if ref is None:
        return L + ["_not available yet_\n"]
    full3 = seeds_runs(st)
    n_oof = np.std([r["cv"]["OOF AUC"] for r in full3], ddof=1) if len(full3) > 1 else np.nan
    n_test = np.std([r["test"]["ROC-AUC"] for r in full3], ddof=1) if len(full3) > 1 else np.nan
    res = {v: st.cv(variant=v, seed=42) for v in E.ABLATION_ORDER}
    done = [v for v in E.ABLATION_ORDER if res[v] is not None]
    # paired DeLong vs the full model: on the OOF predictions (4,669 dev molecules, every variant
    # predicts every molecule from a model that never trained on it) and on the 825 test molecules
    oof_d = {v: M.delong(ref["oof_y"], res[v]["oof"], ref["oof"]) for v in done}
    te_d = {v: M.delong(ref["test_y"], res[v]["ens"], ref["ens"]) for v in done}
    p_oof = dict(zip(done, M.holm([oof_d[v][1] for v in done]))) if done else {}
    p_te = dict(zip(done, M.holm([te_d[v][1] for v in done]))) if done else {}
    L.append(
        f"Evidence columns: **▲/▼** = OOF-AUC change larger than 2 sd of the full model's training-seed noise "
        f"(sd over {len(full3)} seeds: OOF AUC ±{n_oof:.4f}, test AUC ±{n_test:.4f}); **†** = paired DeLong "
        f"on the 4,669 out-of-fold predictions significant after Holm correction over the {len(done)} variants "
        f"(p < 0.05); **\*** = paired DeLong on the 825 held-out test molecules significant after Holm correction.\n")
    L.append("Caveats: the held-out test set is the clean, headline evaluation. The OOF predictions are secondary "
             "evidence: each validation fold was also used for early stopping (notebook protocol), so OOF scores are "
             "slightly optimistic; the bias affects all variants alike, so paired comparisons remain informative. "
             "DeLong covers test-molecule sampling only, not training randomness (hence the seed-noise column). "
             "On 825 test molecules the SE of an AUC difference between these highly correlated variants is "
             "~0.0016-0.0032, so differences below ~0.003-0.006 cannot be resolved there.\n")
    rows = [["full", core.VARIANTS["full"][0], f4(ref["cv"]["OOF AUC"]), "—", "—", "—", f4(ref["test"]["ROC-AUC"]),
             "—", "—", f4(ref["test"]["MCC"]), ""]]
    csv = []
    for v in E.ABLATION_ORDER:
        r = res[v]
        if r is None:
            rows.append([v, core.VARIANTS[v][0]] + ["—"] * 9)
            continue
        d_oof, p_oof_raw = oof_d[v]
        d_te, p_te_raw = te_d[v]
        se_oof = abs(d_oof) / max(stats_norm_isf(p_oof_raw / 2), 1e-12)
        ev = ""
        if abs(d_oof) > 2 * n_oof:
            ev += "▲" if d_oof > 0 else "▼"
        ev += ("†" if p_oof[v] < 0.05 else "") + ("\*" if p_te[v] < 0.05 else "")
        rows.append([v, core.VARIANTS[v][0], f4(r["cv"]["OOF AUC"]), f"{d_oof:+.4f}", f"{se_oof:.4f}",
                     fp(p_oof[v]), f4(r["test"]["ROC-AUC"]), f"{d_te:+.4f}", fp(p_te[v]), f4(r["test"]["MCC"]), ev])
        csv.append(dict(variant=v, oof_auc=r["cv"]["OOF AUC"], d_oof_auc=d_oof, se_d_oof=se_oof,
                        p_oof_delong=p_oof_raw, p_oof_delong_holm=p_oof[v], test_auc=r["test"]["ROC-AUC"],
                        d_test_auc=d_te, p_test_delong=p_te_raw, p_test_delong_holm=p_te[v],
                        test_mcc=r["test"]["MCC"], cv_mcc=r["cv"]["MCC"], gpu_minutes=r["minutes"],
                        n_trainable=r["n_trainable"]))
    L += table(["variant", "what changes", "OOF AUC", "Δ OOF", "SE Δ OOF", "p OOF DeLong (Holm)", "test AUC",
                "Δ test", "p test DeLong (Holm)", "test MCC", "evidence"], rows)
    pd.DataFrame(csv).to_csv(os.path.join(RES, "ablation_table.csv"), index=False)
    return L


def stats_norm_isf(q):
    from scipy import stats
    return stats.norm.isf(q)


def fp(p):
    return "<1e-4" if p < 1e-4 else f"{p:.4f}" if p < 0.001 else f"{p:.3f}"


def sec_scaffold(st):
    L = ["## 3. Supplementary: random split vs scaffold split\n",
         "_Robustness check only; every other section uses the notebook's split._\n",
         "Scaffold split: Bemis-Murcko scaffolds; test = 735 molecules whose scaffolds never occur in training; "
         "5-fold CV grouped by scaffold on the rest.\n"]
    rows, csv = [], []
    models = [("V3 full", dict(variant="full")), ("V3 without SMILES branch", dict(variant="no_smiles")),
              ("MoLFormer only", dict(variant="smiles_only"))]
    for name, kw in models:
        rr = seeds_runs(st, split="random", **kw)
        sr = seeds_runs(st, split="scaffold", **kw)
        rows.append([name, len(sr) or "—", ms([r["cv"]["OOF AUC"] for r in rr]), ms([r["cv"]["OOF AUC"] for r in sr]),
                     ms([r["test"]["ROC-AUC"] for r in rr]), ms([r["test"]["ROC-AUC"] for r in sr]),
                     ms([r["test"]["MCC"] for r in sr])])
        csv.append(dict(model=name, random_oof=M.mean_sd([r["cv"]["OOF AUC"] for r in rr])[0],
                        scaffold_oof=M.mean_sd([r["cv"]["OOF AUC"] for r in sr])[0],
                        random_test=M.mean_sd([r["test"]["ROC-AUC"] for r in rr])[0],
                        scaffold_test=M.mean_sd([r["test"]["ROC-AUC"] for r in sr])[0]))
    for b in ["RF__ECFP", "XGBoost__ECFP", "RF__ECFP+MACCS+Desc", "XGBoost__ECFP+MACCS+Desc", "LogReg__ECFP+MACCS+Desc"]:
        r, s = st.baseline("random", b), st.baseline("scaffold", b)
        rows.append([b.replace("__", " on "), "(deterministic)", f4(r["cv"]["OOF AUC"]) if r else "—",
                     f4(s["cv"]["OOF AUC"]) if s else "—", f4(r["test"]["ROC-AUC"]) if r else "—",
                     f4(s["test"]["ROC-AUC"]) if s else "—", f4(s["test"]["MCC"]) if s else "—"])
        csv.append(dict(model=b, random_oof=r["cv"]["OOF AUC"] if r else np.nan, scaffold_oof=s["cv"]["OOF AUC"] if s else np.nan,
                        random_test=r["test"]["ROC-AUC"] if r else np.nan, scaffold_test=s["test"]["ROC-AUC"] if s else np.nan))
    L += table(["model", "scaffold seeds", "random OOF AUC", "scaffold OOF AUC", "random test AUC", "scaffold test AUC",
                "scaffold test MCC"], rows)
    pd.DataFrame(csv).to_csv(os.path.join(RES, "scaffold_table.csv"), index=False)
    return L


def sec_variants(st, title, entries, note=""):
    """Compare configurations over 3 seeds: entries = [(label, kwargs)]."""
    L = [title + "\n"] + ([note + "\n"] if note else [])
    rows, ref = [], seeds_runs(st)
    for label, kw in entries:
        rr = seeds_runs(st, **kw)
        if not rr:
            rows.append([label, 0] + ["—"] * 8)
            continue
        rows.append([label, len(rr), ms([r["cv"]["OOF AUC"] for r in rr]), ms([r["test"]["ROC-AUC"] for r in rr]),
                     ms([r["test"]["MCC"] for r in rr]), ms([r["test"]["Balanced Acc"] for r in rr]),
                     ms([r["test05"]["Sensitivity"] for r in rr], 3), ms([r["test05"]["Specificity"] for r in rr], 3),
                     ms([r["test05"]["Brier"] for r in rr]), ms([r["test05"]["ECE"] for r in rr])])
    L += table(["configuration", "seeds", "OOF AUC", "test AUC", "test MCC (opt thr)", "test BalAcc (opt thr)",
                "Sens @0.5", "Spec @0.5", "Brier", "ECE"], rows)
    L += paired_tests(st, entries)
    return L


def paired_tests(st, entries):
    """Paired DeLong of every entry vs the first (reference) entry, seed by seed (same seed =
    same folds, same test molecules), on OOF and test predictions; Holm over all comparisons."""
    ref_kw = entries[0][1]
    comps = []
    for label, kw in entries[1:]:
        for s in SEEDS:
            a, b = st.cv(seed=s, **kw), st.cv(seed=s, **ref_kw)
            if a is None or b is None:
                continue
            comps.append((label, s, M.delong(b["oof_y"], a["oof"], b["oof"]), M.delong(b["test_y"], a["ens"], b["ens"])))
    if not comps:
        return []
    h_oof = M.holm([c[2][1] for c in comps])
    h_te = M.holm([c[3][1] for c in comps])
    rows = [[lab, s, f"{o[0]:+.4f}", fp(ho), f"{t[0]:+.4f}", fp(ht)]
            for (lab, s, o, t), ho, ht in zip(comps, h_oof, h_te)]
    return [f"Paired DeLong vs **{entries[0][0]}** (same seed, same folds and test molecules; Holm-corrected "
            f"over the {len(comps)} comparisons in this table):\n"] + table(
        ["configuration", "seed", "Δ OOF AUC", "p OOF (Holm)", "Δ test AUC", "p test (Holm)"], rows)


def sec_backbones(st):
    L = sec_variants(st, "## 6. SMILES encoder: MoLFormer vs ChemBERTa family (everything else identical)",
                     [("MoLFormer-XL (47M) - V3 as published", {})]
                     + [(backbones.BACKBONES[b]["label"], dict(backbone=b)) for b in E.NEW_BACKBONES],
                     "Same V3 head, fusion, loss, sampler, augmentation and MoLFormer-tuned hyper-parameters "
                     "(no re-tuning). 384-d encoders get a linear 384→768 projection; bottom third of layers frozen.")
    rows = []
    for b in ["molformer"] + E.NEW_BACKBONES:
        rr = seeds_runs(st, backbone=b)
        if rr:
            rows.append([b, f"{rr[0]['n_trainable']/1e6:.1f}M", f"{np.mean([r['minutes'] for r in rr]):.0f}"])
    if rows:
        L += table(["backbone", "trainable params (whole model)", "wall-clock training minutes per 5-fold run*"], rows)
        L.append("*Includes thermal pauses: the MoLFormer runs were trained under a 78 °C GPU cap that paused "
                 "training ~40-50% of the time, the ChemBERTa runs without it, so the times are not directly comparable.\n")
    return L


def sec_explain():
    p = os.path.join(RES, "explain", "explain_summary.json")
    L = ["## 7. Explainability (published 5-fold ensemble)\n"]
    if not os.path.exists(p):
        return L + ["_not available yet_\n"]
    s = json.load(open(p))
    if "modality" in s:
        mo = s["modality"]
        L += table(["modality", "mean |Shapley| (logit)", "share", "test AUC when removed", "mean gate"],
                   [[k, f4(mo["mean_abs_phi"][k]), f"{mo['share_of_total'][k]*100:.0f}%", f4(mo["occlusion_auc"][k]),
                     f"{mo['mean_gate'][k]:.3f}"] for k in mo["mean_abs_phi"]])
        L.append(f"Test AUC with all modalities: {mo['full_auc']:.4f}\n")
        L.append("The mean gate is ~0.500 for every modality by construction: V3's gate is "
                 "sigmoid(LayerNorm(W·x)), and LayerNorm centres each token to mean 0 before the sigmoid. The gate "
                 "can therefore only re-weight dimensions within a modality, never scale a whole modality up or "
                 "down, which is consistent with the no_gate ablation showing no benefit.\n")
    for key, label in [("v3_vs_xgboost", "V3 vs XGBoost TreeSHAP"), ("lime", "LIME"), ("deletion_aopc", "Deletion test (area, lower = more faithful)"),
                       ("regions", "Hydroxamate ZBG / linker / cap")]:
        if key in s:
            L.append(f"**{label}:** `{json.dumps(s[key])}`\n")
    L.append("Figures: " + ", ".join(f"[{os.path.basename(f)}](explain/{os.path.basename(f)})"
                                     for f in sorted(glob.glob(os.path.join(RES, "explain", "fig*.png")))) + "\n")
    return L


def main():
    st = Store()
    L = ["# V3 bioactivity model - verification, ablation and extended study\n",
         "All numbers are produced by `report.py` from the raw predictions in `results/`. "
         "Metrics and thresholds follow the notebook exactly (cells 16-18).\n"]
    L += sec_repro(st)
    L += sec_ablation(st)
    L += sec_scaffold(st)
    L += sec_variants(st, "## 4. Class-imbalance handling (random split, 3 seeds)",
                      [("V3 (WeightedRandomSampler + focal loss)", {}),
                       ("Unbalanced (a): no sampler, focal loss α=0.5", dict(variant="no_sampler")),
                       ("Unbalanced (b): no sampler, plain BCE", dict(variant="vanilla"))],
                      "'opt thr' = OOF MCC-optimal threshold (notebook protocol); '@0.5' = fixed 0.5 threshold.")
    L += sec_variants(st, "## 5. Tuned vs untuned hyper-parameters (random split, 3 seeds)",
                      [("Tuned (Optuna, notebook)", {}), ("Untuned (notebook cell-15 fallback config)", dict(hp_set="untuned"))],
                      "Untuned = 8 heads, hidden 512, dropout 0.2, 3 classifier + 3 cross-modal layers, lr 5e-6, wd 1e-4.")
    L += sec_backbones(st)
    L += sec_explain()
    L += ["## 8. Other result files\n",
          "- [descriptor_stats.md](descriptor_stats.md): statistical tests of the 6 descriptors vs activity "
          "(Mann-Whitney, Welch/Student t, z, scaffold-clustered logistic regression)",
          "- [metrics_all.md](metrics_all.md): accuracy / balanced accuracy / precision / recall / F1 of every experiment",
          "- [saved_model_check.json](saved_model_check.json): re-scoring of the notebook's saved models",
          "- [data_checks.json](data_checks.json): duplicates with conflicting labels, test-vs-train similarity",
          "- `ablation_table.csv`, `scaffold_table.csv`, `metrics_all.csv`, `descriptor_stats.csv`: numbers behind the tables\n"]
    with open(os.path.join(RES, "REPORT.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
