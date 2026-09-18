"""Figures + summary JSON for explain_v3.py outputs (results/explain/*.npz -> *.png)."""
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data
import splits

OUT = os.path.join(HERE, "results", "explain")
MOD = ["SMILES\n(MoLFormer)", "ECFP", "MACCS", "Descriptors"]
C_ACT, C_INACT = "#c0392b", "#2a7ab9"


def load(name):
    p = os.path.join(OUT, name + ".npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def heatmap_png(mol, weights, size=(360, 280)):
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import SimilarityMaps
    from PIL import Image
    w = np.asarray(weights, dtype=float)
    if np.abs(w).max() > 0:
        w = w / np.abs(w).max()
    d = Draw.MolDraw2DCairo(*size)
    SimilarityMaps.GetSimilarityMapFromWeights(mol, [float(x) for x in w], draw2d=d, colorMap="bwr", contourLines=4)
    d.FinishDrawing()
    return Image.open(io.BytesIO(d.GetDrawingText()))


def main():
    plt = plt_setup()
    from rdkit import Chem
    summary = {}
    df = data.load_df()
    feats = data.get_features(df)
    test = splits.get_split("random", df)["test"]
    pr = load("predict")
    y = pr["y"]

    # 1. modality Shapley + occlusion + gates
    mo = load("modality")
    if mo is not None:
        phi = mo["phi"]
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        ax[0].bar(MOD, np.abs(phi).mean(0), color="#555")
        ax[0].set_title("Exact Shapley: mean |contribution| (logit)")
        pos = np.arange(4)
        for cls, col, off in ((1, C_ACT, -0.18), (0, C_INACT, 0.18)):
            bp = ax[1].boxplot([phi[y == cls, i] for i in range(4)], positions=pos + off, widths=0.3,
                               patch_artist=True, showfliers=False)
            for b in bp["boxes"]:
                b.set_facecolor(col)
                b.set_alpha(0.6)
        ax[1].axhline(0, color="k", lw=0.6)
        ax[1].set_xticks(pos, MOD)
        ax[1].set_title("Signed contribution (red = active, blue = inactive)")
        ax[2].bar(MOD, mo["occl_auc"], color="#888")
        from sklearn.metrics import roc_auc_score
        ens_auc = roc_auc_score(y, pr["ens"])
        ax[2].axhline(ens_auc, color=C_ACT, ls="--", label=f"all modalities {ens_auc:.4f}")
        ax[2].set_ylim(0.85, 0.975)
        ax[2].set_title("Test AUC with the modality replaced by its mean")
        ax[2].legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "fig1_modality_shapley.png"), dpi=150)
        plt.close()
        summary["modality"] = {"mean_abs_phi": dict(zip(["smiles", "ecfp", "maccs", "desc"], np.abs(phi).mean(0).round(4).tolist())),
                               "share_of_total": dict(zip(["smiles", "ecfp", "maccs", "desc"],
                                                          (np.abs(phi).mean(0) / np.abs(phi).mean(0).sum()).round(3).tolist())),
                               "occlusion_auc": dict(zip(["smiles", "ecfp", "maccs", "desc"], mo["occl_auc"].round(4).tolist())),
                               "full_auc": round(float(ens_auc), 4),
                               "mean_gate": dict(zip(["smiles", "ecfp", "maccs", "desc"], mo["gates"].mean(0).round(3).tolist()))}

    # 2. SHAP beeswarms: descriptors + top MACCS keys
    sh = load("shap")
    if sh is not None:
        import shap
        from rdkit.Chem.MACCSkeys import smartsPatts
        fig = plt.figure(figsize=(7, 3.2))
        shap.summary_plot(sh["desc"], feats["desc_raw"][test], feature_names=data.DESC_NAMES, show=False, plot_size=None)
        plt.title("SHAP (logit) - descriptors")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "fig2a_shap_descriptors.png"), dpi=150)
        plt.close("all")
        names = [f"MACCS {k}: {smartsPatts.get(k, ('?',))[0][:28]}" for k in range(167)]
        fig = plt.figure(figsize=(8, 6))
        shap.summary_plot(sh["maccs"], feats["maccs"][test], feature_names=names, max_display=15, show=False, plot_size=None)
        plt.title("SHAP (logit) - top 15 MACCS keys")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "fig2b_shap_maccs.png"), dpi=150)
        plt.close("all")
        imp = np.abs(sh["maccs"]).mean(0)
        summary["top_maccs"] = [{"key": int(k), "smarts": smartsPatts.get(int(k), ("?",))[0],
                                 "mean_abs_shap": round(float(imp[k]), 4)} for k in np.argsort(-imp)[:10]]
        summary["desc_mean_abs_shap"] = dict(zip(data.DESC_NAMES, np.abs(sh["desc"]).mean(0).round(4).tolist()))

        # 3. top ECFP bits drawn
        from rdkit.Chem import Draw, rdFingerprintGenerator
        imp_e = np.abs(sh["ecfp"]).mean(0)
        top = np.argsort(-imp_e)[:12]
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
        fig, axes = plt.subplots(2, 6, figsize=(15, 5.4))
        bits_summary = []
        for ax, b in zip(axes.ravel(), top):
            has = feats["ecfp"][test][:, b] > 0
            signed = float(sh["ecfp"][has, b].mean()) if has.any() else 0.0
            img = None
            for i in test[has]:
                m = Chem.MolFromSmiles(feats["smiles"][i])
                ao = rdFingerprintGenerator.AdditionalOutput()
                ao.AllocateBitInfoMap()
                gen.GetFingerprint(m, additionalOutput=ao)
                bi = ao.GetBitInfoMap()
                if int(b) in bi:
                    try:
                        img = Draw.DrawMorganBit(m, int(b), dict(bi), useSVG=False)
                        break
                    except Exception:
                        continue
            if img is not None:
                ax.imshow(img)
            ax.set_title(f"bit {b}: {signed:+.2f} when present\n(in {has.mean()*100:.0f}% of test, act {y[has].mean()*100:.0f}%)",
                         fontsize=8)
            ax.axis("off")
            bits_summary.append({"bit": int(b), "mean_abs_shap": round(float(imp_e[b]), 4),
                                 "mean_shap_when_present": round(signed, 4), "prevalence": round(float(has.mean()), 3),
                                 "active_rate_when_present": round(float(y[has].mean()), 3) if has.any() else None})
        plt.suptitle("Most influential ECFP4 bits in V3 (SHAP, logit)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "fig3_top_ecfp_bits.png"), dpi=150)
        plt.close()
        summary["top_ecfp_bits"] = bits_summary

    # 4. showcase atom heat maps
    at, li = load("atoms"), load("lime")
    if at is not None:
        from PIL import Image
        lime_w = {int(p): np.asarray(w, float) for p, w in zip(li["pos"], li["weights"])} if li is not None else {}
        cols = [("combined", "IG + SHAP (all branches)"), ("atom_ig", "IG on MoLFormer"),
                ("atom_ecfp", "SHAP ECFP bits"), ("lime", "LIME (atom masking)")]
        rows = list(zip(pr["showcase_pos"], pr["showcase_kind"]))
        fig, axes = plt.subplots(len(rows), 4, figsize=(15, 3.1 * len(rows)))
        for r, (p, kind) in enumerate(rows):
            m = Chem.MolFromSmiles(feats["smiles"][test[p]])
            for c, (key, title) in enumerate(cols):
                ax = axes[r, c]
                w = lime_w.get(int(p)) if key == "lime" else np.asarray(at[key][p], float)
                if w is not None:
                    ax.imshow(heatmap_png(m, w))
                ax.axis("off")
                if r == 0:
                    ax.set_title(title, fontsize=11)
            axes[r, 0].text(-0.02, 0.5, f"{kind}\ny={y[p]}  p={pr['ens'][p]:.2f}", transform=axes[r, 0].transAxes,
                            ha="right", va="center", fontsize=10)
        plt.suptitle("Atom attributions (red = pushes towards ACTIVE, blue = towards INACTIVE); each map scaled to its max",
                     y=1.0)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "fig4_showcase_atom_maps.png"), dpi=110, bbox_inches="tight")
        plt.close()

        # 5. ZBG / linker / cap
        import pandas as pd
        reg = pd.DataFrame([json.loads(s) for s in at["regions"]])
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
        data_ = [[reg[reg.y == c][f"dens_{k}"].dropna() for c in (1, 0)] for k in ("zbg", "linker", "cap")]
        posx = np.arange(3)
        for c, col, off in ((0, C_ACT, -0.18), (1, C_INACT, 0.18)):
            bp = ax[0].boxplot([d[c] for d in data_], positions=posx + off, widths=0.3, patch_artist=True, showfliers=False)
            for b in bp["boxes"]:
                b.set_facecolor(col)
                b.set_alpha(0.6)
        ax[0].axhline(0, color="k", lw=0.6)
        ax[0].set_xticks(posx, ["Zinc-binding group\n(hydroxamate)", "Linker", "Cap"])
        ax[0].set_ylabel("attribution per atom (logit)")
        ax[0].set_title("Where the model looks - hydroxamates (red active / blue inactive)")
        tops = reg[["top_in_zbg", "top_in_linker", "top_in_cap"]].mean()
        ax[1].bar(["ZBG", "Linker", "Cap"], tops.values, color="#666")
        ax[1].set_title("Region holding the single most important atom")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "fig5_zbg_linker_cap.png"), dpi=150)
        plt.close()
        from scipy.stats import mannwhitneyu
        summary["regions"] = {
            "n_hydroxamates": int(len(reg)),
            "top_atom_region_share": {k: round(float(v), 3) for k, v in tops.items()},
            **{f"density_{k}": {"active_mean": round(float(reg[reg.y == 1][f"dens_{k}"].mean()), 4),
                                "inactive_mean": round(float(reg[reg.y == 0][f"dens_{k}"].mean()), 4),
                                "mannwhitney_p": float(mannwhitneyu(reg[reg.y == 1][f"dens_{k}"].dropna(),
                                                                    reg[reg.y == 0][f"dens_{k}"].dropna()).pvalue)}
               for k in ("zbg", "linker", "cap")}}

    # 6. deletion curves
    de = load("deletion")
    if de is not None:
        fr = de["fractions"]
        fig, ax = plt.subplots(figsize=(6, 4))
        summary["deletion_aopc"] = {}
        for name, cur in zip(de["names"], de["curves"]):
            mu = cur.mean(0)
            ax.plot(fr * 100, mu, marker="o", label=f"{name} (area {np.trapz(mu, fr):.3f})",
                    ls="--" if name == "random" else "-")
            summary["deletion_aopc"][str(name)] = round(float(np.trapz(mu, fr)), 4)
        ax.set_xlabel("% of atoms masked (most important first)")
        ax.set_ylabel("mean P(active)")
        ax.set_title(f"Faithfulness: deletion test ({de['curves'].shape[1]} predicted actives)")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "fig6_deletion.png"), dpi=150)
        plt.close()

    # 7. V3 vs XGBoost TreeSHAP
    tr = load("tree")
    if tr is not None:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].scatter(tr["xgb_bit_importance"], tr["v3_bit_importance"], s=6, alpha=0.5)
        ax[0].set_xlabel("XGBoost TreeSHAP mean |value|")
        ax[0].set_ylabel("V3 SHAP mean |value|")
        ax[0].set_title(f"ECFP bit importance: Spearman {float(tr['bit_rho']):.2f}, top-20 overlap {int(tr['top20_overlap'])}/20")
        ax[1].hist(tr["per_mol_atom_rho"], bins=30, color="#666")
        ax[1].set_title(f"Per-molecule atom-map agreement (median rho {np.median(tr['per_mol_atom_rho']):.2f})")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "fig7_v3_vs_xgboost_shap.png"), dpi=150)
        plt.close()
        summary["v3_vs_xgboost"] = {"bit_importance_spearman": round(float(tr["bit_rho"]), 3),
                                    "top20_overlap": int(tr["top20_overlap"]),
                                    "median_per_molecule_atom_spearman": round(float(np.median(tr["per_mol_atom_rho"])), 3)}
    if li is not None:
        summary["lime"] = {"n_molecules": int(len(li["pos"])),
                           "run_to_run_spearman": round(float(np.nanmean(li["stability"])), 3),
                           "median_agreement_with_IG_SHAP": round(float(np.nanmedian(li["agreement_with_combined"])), 3)}

    # 8. applicability domain + calibration
    do = load("domain")
    if do is not None:
        rows = [json.loads(s) for s in do["bins"]]
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        ax[0].bar([r["bin"] for r in rows], [r["error_rate"] for r in rows], color="#666")
        for i, r in enumerate(rows):
            ax[0].text(i, r["error_rate"], f"n={r['n']}\nAUC {r['auc']:.2f}", ha="center", va="bottom", fontsize=8)
        ax[0].set_xlabel("max Tanimoto similarity to training set")
        ax[0].set_ylabel("error rate")
        ax[0].set_title("Applicability domain")
        ax[1].plot([0, 1], [0, 1], color="grey", ls="--")
        ax[1].plot(do["calib_pred"], do["calib_frac"], marker="o")
        ax[1].set_xlabel("predicted P(active)")
        ax[1].set_ylabel("observed fraction active")
        ax[1].set_title(f"Calibration (Brier {float(do['brier']):.4f})")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "fig8_domain_calibration.png"), dpi=150)
        plt.close()
        summary["applicability_domain"] = rows
        summary["brier"] = round(float(do["brier"]), 4)

    with open(os.path.join(OUT, "explain_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=1)[:4000])


if __name__ == "__main__":
    main()
