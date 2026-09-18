"""
Explainability for the published V3 ensemble (models/kfold_v3_fold1-5.pth, read-only).

Stages (each caches its output in results/explain/, re-run with --force):
  predict   ensemble test probabilities (+ per-member / per-draw), showcase molecules
  modality  exact Shapley values over the 4 modality tokens (2^4 coalitions), gate
            activations, modality-occlusion AUC
  shap      SHAP GradientExplainer (expected gradients) on [SMILES token, ECFP bits,
            MACCS keys, descriptors] -> feature-level SHAP
  ig        Integrated Gradients on MoLFormer token embeddings -> per-atom attribution
  atoms     ECFP-bit / MACCS-key SHAP projected onto atoms; combined atom maps;
            hydroxamate ZBG / linker / cap attribution analysis
  tree      TreeSHAP on an XGBoost-ECFP model; agreement with V3's ECFP SHAP
  lime      LIME (atom masking with dummy atoms) + run-to-run stability
  deletion  faithfulness: delete top-attributed atoms vs random atoms
  domain    error rate vs similarity to training data; calibration
  figures   all plots + results/explain/explain_summary.json

MoLFormer re-draws its random attention features on every forward pass, so every
attribution is averaged over several fixed draws ("draw seeds") and all 5 models.
"""
import argparse
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import backbones
import core
import data
import gpu
import splits

OUT = os.path.join(HERE, "results", "explain")
MODELS_DIR = os.path.join(os.path.dirname(HERE), "models")
DEV = torch.device("cuda")
MOD_NAMES = ["SMILES (MoLFormer)", "ECFP", "MACCS", "Descriptors"]
ATOM_TOKEN = re.compile(r"^(\[[^\]]+\]|Br|Cl|B|C|N|O|S|P|F|I|b|c|n|o|s|p|\*)$")
ZBG_SMARTS = "[CX3](=O)[NX3][OX2H1]"          # hydroxamic acid zinc-binding group
NB_THRESHOLD = 0.430                           # notebook cell 18 (OOF MCC-optimal)
GUARD = gpu.ThermalGuard(hot_c=78, cool_c=70, every_s=30)


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def save(name, **arrays):
    np.savez_compressed(os.path.join(OUT, name + ".npz"), **arrays)


def load(name):
    z = np.load(os.path.join(OUT, name + ".npz"), allow_pickle=True)
    return {k: z[k] for k in z.files}


def exists(name):
    return os.path.exists(os.path.join(OUT, name + ".npz"))


# ─────────────────────────────────────────────────────────────────────────────
# Loading the published ensemble
# ─────────────────────────────────────────────────────────────────────────────
class Ctx:
    """Everything shared by the stages."""

    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        backbones.register_remote_code()
        import __main__
        from notebook_classes import FineTunedBERTaECFP_v3
        __main__.FineTunedBERTaECFP_v3 = FineTunedBERTaECFP_v3
        core.setup_torch()
        self.df = data.load_df()
        self.feats = data.get_features(self.df)
        self.split = splits.get_split("random", self.df)
        self.tok = backbones.load_tokenizer("molformer")
        self.test = self.split["test"]
        self.y_test = self.feats["labels"][self.test].astype(int)
        self.members = []
        for f in range(5):
            nb = torch.load(os.path.join(MODELS_DIR, f"kfold_v3_fold{f+1}.pth"), weights_only=False, map_location="cpu")
            m = core.build_model(core.make_cfg("full"))
            m.load_state_dict(nb.state_dict(), strict=True)
            m.to(DEV).eval()
            for p in m.parameters():
                p.requires_grad_(False)
            tr = self.split["dev"][self.split["folds"][f][0]]
            sc = StandardScaler().fit(self.feats["desc_raw"][tr])
            self.members.append(dict(model=m, scaler=sc, train_idx=tr))
            del nb
        log("loaded 5 published fold models (strict state_dict match)")

    # tabular inputs for dataframe indices, scaled with member k's scaler
    def tab(self, idx, k):
        sc = self.members[k]["scaler"]
        return (torch.tensor(self.feats["ecfp"][idx], device=DEV),
                torch.tensor(self.feats["maccs"][idx], device=DEV),
                torch.tensor(sc.transform(self.feats["desc_raw"][idx]).astype(np.float32), device=DEV))

    def ids(self, smiles_list):
        enc = self.tok(list(smiles_list), truncation=True, max_length=512)
        L = max(len(x) for x in enc["input_ids"])
        ids = torch.full((len(enc["input_ids"]), L), self.tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, x in enumerate(enc["input_ids"]):
            ids[i, :len(x)] = torch.tensor(x)
            mask[i, :len(x)] = 1
        return ids.to(DEV), mask.to(DEV)

    @torch.no_grad()
    def smiles_tokens(self, idx, k, draw, bs=128):
        """Projected SMILES modality token [N, D] for member k under random-feature draw `draw`."""
        m = self.members[k]["model"]
        out = []
        for i in range(0, len(idx), bs):
            GUARD.check()
            ids, mask = self.ids(self.feats["smiles"][idx[i:i + bs]])
            torch.manual_seed(100_000 * draw + i)
            out.append(m.pool(m.smiles_hidden(ids, mask), mask).float())
        return torch.cat(out)

    @torch.no_grad()
    def predict_smiles(self, smiles_list, draw=0, bs=256):
        """Ensemble P(active) for arbitrary (e.g. perturbed) SMILES; features recomputed."""
        fe = [data.featurize_one(s) for s in smiles_list]
        ecfp = np.stack([f[0] for f in fe])
        maccs = np.stack([f[1] for f in fe])
        desc = np.stack([f[2] for f in fe])
        probs = np.zeros(len(smiles_list))
        for k, mem in enumerate(self.members):
            m = mem["model"]
            for i in range(0, len(smiles_list), bs):
                ids, mask = self.ids(smiles_list[i:i + bs])
                d = torch.tensor(mem["scaler"].transform(desc[i:i + bs]).astype(np.float32), device=DEV)
                torch.manual_seed(100_000 * draw + i)
                lg = m(ids, mask, torch.tensor(ecfp[i:i + bs], device=DEV), torch.tensor(maccs[i:i + bs], device=DEV), d)
                probs[i:i + bs] += torch.sigmoid(lg.float()).cpu().numpy() / len(self.members)
        return probs


def tab_tokens(m, ecfp, maccs, desc):
    return m.ecfp_proj(ecfp), m.maccs_proj(maccs), m.desc_proj(desc)


# ─────────────────────────────────────────────────────────────────────────────
# Stage: predict
# ─────────────────────────────────────────────────────────────────────────────
def stage_predict(ctx, draws=4):
    P = np.zeros((5, draws, len(ctx.test)))
    for k, mem in enumerate(ctx.members):
        m = mem["model"]
        e, mc, d = ctx.tab(ctx.test, k)
        for dr in range(draws):
            s = ctx.smiles_tokens(ctx.test, k, dr)
            with torch.no_grad():
                t = torch.stack([s, *tab_tokens(m, e, mc, d)], 1)
                P[k, dr] = torch.sigmoid(m.fuse(t)).cpu().numpy()
    ens = P.mean((0, 1))
    y = ctx.y_test
    pred = (ens > NB_THRESHOLD).astype(int)
    # showcase: 3 most confident TP / TN, 3 most confident FP / FN (hydroxamates preferred for TP)
    from rdkit import Chem
    zbg = Chem.MolFromSmarts(ZBG_SMARTS)
    has_zbg = np.array([Chem.MolFromSmiles(ctx.feats["smiles"][i]).HasSubstructMatch(zbg) for i in ctx.test])
    pick = []
    for name, sel, order in [("TP", (y == 1) & (pred == 1) & has_zbg, -ens), ("TN", (y == 0) & (pred == 0), ens),
                             ("FP", (y == 0) & (pred == 1), -ens), ("FN", (y == 1) & (pred == 0), ens)]:
        cand = np.where(sel)[0]
        cand = cand[np.argsort(order[cand])][:3]
        pick += [(int(c), name) for c in cand]
    save("predict", member_draw_probs=P, ens=ens, y=y, has_zbg=has_zbg,
         showcase_pos=np.array([p for p, _ in pick]), showcase_kind=np.array([k for _, k in pick]))
    from sklearn.metrics import roc_auc_score
    log(f"predict: ensemble test AUC {roc_auc_score(y, ens):.4f} | draw-to-draw sd of AUC "
        f"{np.std([roc_auc_score(y, P[:, d].mean(0)) for d in range(draws)]):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage: modality (exact Shapley over 4 players)
# ─────────────────────────────────────────────────────────────────────────────
def stage_modality(ctx, draws=4, n_bg=1000):
    from itertools import combinations
    from sklearn.metrics import roc_auc_score
    n = 4
    subsets = [S for r in range(n + 1) for S in combinations(range(n), r)]
    phi = np.zeros((len(ctx.test), n))
    base_val = np.zeros(len(ctx.test))
    gates = np.zeros((len(ctx.test), n))
    occl = np.zeros((n, len(ctx.test)))       # P(active) with modality i replaced by baseline
    rng = np.random.default_rng(0)
    for k, mem in enumerate(ctx.members):
        m = mem["model"]
        bg_idx = rng.choice(mem["train_idx"], size=min(n_bg, len(mem["train_idx"])), replace=False)
        e, mc, d = ctx.tab(ctx.test, k)
        eb, mb, db = ctx.tab(bg_idx, k)
        for dr in range(draws):
            GUARD.check()
            with torch.no_grad():
                T = torch.stack([ctx.smiles_tokens(ctx.test, k, dr), *tab_tokens(m, e, mc, d)], 1)
                Bm = torch.stack([ctx.smiles_tokens(bg_idx, k, dr), *tab_tokens(m, eb, mb, db)], 1).mean(0)
                v = {}
                for S in subsets:
                    mask = torch.zeros(n, 1, device=DEV)
                    mask[list(S)] = 1
                    v[S] = m.fuse(T * mask + Bm * (1 - mask)).float().cpu().numpy()
                # gate activations after cross-modal attention
                tt = T + m.modal_type_embed(torch.arange(4, device=DEV)).unsqueeze(0)
                gates += m.gate(m.cross_modal(tt)).mean(-1).float().cpu().numpy() / (5 * draws)
            for i in range(n):
                for S in subsets:
                    if i in S:
                        continue
                    w = math.factorial(len(S)) * math.factorial(n - len(S) - 1) / math.factorial(n)
                    phi[:, i] += w * (v[tuple(sorted(S + (i,)))] - v[S]) / (5 * draws)
                occl[i] += 1 / (1 + np.exp(-v[tuple(j for j in range(n) if j != i)])) / (5 * draws)
            base_val += v[()] / (5 * draws)
    full_logit = base_val + phi.sum(1)
    ens = load("predict")["ens"]
    occl_auc = [roc_auc_score(ctx.y_test, occl[i]) for i in range(n)]
    save("modality", phi=phi, base_val=base_val, gates=gates, occl_probs=occl, occl_auc=np.array(occl_auc))
    log("modality Shapley (mean |phi|, logit units): " +
        ", ".join(f"{MOD_NAMES[i]} {np.abs(phi[:, i]).mean():.3f}" for i in range(n)))
    log(f"  efficiency check: corr(sum phi + base, ensemble logit) = "
        f"{np.corrcoef(full_logit, np.log(ens / (1 - ens)))[0, 1]:.4f}")
    log("  occlusion test AUC (modality removed): " + ", ".join(f"{MOD_NAMES[i]} {a:.4f}" for i, a in enumerate(occl_auc)))


# ─────────────────────────────────────────────────────────────────────────────
# Stage: SHAP GradientExplainer on tabular features (+ SMILES token)
# ─────────────────────────────────────────────────────────────────────────────
class HeadFromInputs(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, s, e, mc, d):
        return self.m.fuse(torch.stack([s, *tab_tokens(self.m, e, mc, d)], 1)).unsqueeze(-1)


def stage_shap(ctx, draws=2, n_bg=200, nsamples=200):
    import shap
    N = len(ctx.test)
    out = {"smiles": np.zeros(N), "ecfp": np.zeros((N, data.ECFP_BITS)),
           "maccs": np.zeros((N, data.MACCS_BITS)), "desc": np.zeros((N, data.N_DESC))}
    rng = np.random.default_rng(1)
    for k, mem in enumerate(ctx.members):
        m = mem["model"]
        bg_idx = rng.choice(mem["train_idx"], size=n_bg, replace=False)
        e, mc, d = ctx.tab(ctx.test, k)
        eb, mb, db = ctx.tab(bg_idx, k)
        for dr in range(draws):
            s, sb = ctx.smiles_tokens(ctx.test, k, dr), ctx.smiles_tokens(bg_idx, k, dr)
            expl = shap.GradientExplainer(HeadFromInputs(m), [sb, eb, mb, db], batch_size=100)
            vals = []
            for i in range(0, N, 55):
                GUARD.check()
                sv = expl.shap_values([s[i:i + 55], e[i:i + 55], mc[i:i + 55], d[i:i + 55]],
                                      nsamples=nsamples, rseed=1000 * k + dr)
                vals.append([np.asarray(a).reshape(a.shape[0], -1) for a in sv])
            sv = [np.concatenate([v[j] for v in vals]) for j in range(4)]
            out["smiles"] += sv[0].sum(1) / (5 * draws)
            out["ecfp"] += sv[1] / (5 * draws)
            out["maccs"] += sv[2] / (5 * draws)
            out["desc"] += sv[3] / (5 * draws)
            log(f"shap: member {k+1} draw {dr} done")
    save("shap", **out)
    tot = {k: np.abs(v if v.ndim == 1 else v.sum(1)).mean() for k, v in out.items()}
    log("SHAP mean |group total| (logit): " + ", ".join(f"{k} {v:.3f}" for k, v in tot.items()))


# ─────────────────────────────────────────────────────────────────────────────
# Stage: Integrated Gradients on MoLFormer token embeddings
# ─────────────────────────────────────────────────────────────────────────────
def ig_molecule(ctx, k, pos, draw, steps=32):
    m = ctx.members[k]["model"]
    i = ctx.test[pos]
    ids, mask = ctx.ids([ctx.feats["smiles"][i]])
    emb = m.bert.embeddings.word_embeddings
    x = emb(ids)[0]                                         # [L, h]
    base_ids = ids.clone()
    base_ids[0, 1:-1] = ctx.tok.pad_token_id                # keep <bos>/<eos>
    x0 = emb(base_ids)[0]
    alphas = torch.linspace(0, 1, steps + 1, device=DEV)[1:].view(-1, 1, 1)   # right Riemann sum
    path = (x0 + alphas * (x - x0)).detach().requires_grad_(True)
    e, mc, d = ctx.tab(np.array([i]), k)
    rep = lambda t: t.expand(steps, *t.shape[1:])
    torch.manual_seed(100_000 * draw + pos)
    tokens = m.modality_tokens(None, rep(mask), rep(e), rep(mc), rep(d), inputs_embeds=path)
    logits = m.fuse(tokens)
    grads, = torch.autograd.grad(logits.sum(), path)
    ig = ((x - x0) * grads.mean(0)).sum(-1).detach().float().cpu().numpy()      # [L]
    with torch.no_grad():
        torch.manual_seed(100_000 * draw + pos)
        f1 = m.fuse(m.modality_tokens(None, mask, e, mc, d, inputs_embeds=x[None])).item()
        torch.manual_seed(100_000 * draw + pos)
        f0 = m.fuse(m.modality_tokens(None, mask, e, mc, d, inputs_embeds=x0[None])).item()
    return ig, f1 - f0


def stage_ig(ctx, draws=2, steps=32):
    toks_all, ig_all, gap_all, sum_all = [], [], [], []
    for pos, i in enumerate(ctx.test):
        GUARD.check()
        s = ctx.feats["smiles"][i]
        toks = ctx.tok.convert_ids_to_tokens(ctx.tok(s)["input_ids"])
        acc = np.zeros(len(toks))
        gap = tot = 0.0
        for k in range(5):
            for dr in range(draws):
                ig, g = ig_molecule(ctx, k, pos, dr, steps)
                acc += ig / (5 * draws)
                gap += g / (5 * draws)
                tot += ig.sum() / (5 * draws)
        toks_all.append(np.array(toks, dtype=object))
        ig_all.append(acc)
        gap_all.append(gap)
        sum_all.append(tot)
        if pos % 100 == 0:
            log(f"ig: {pos}/{len(ctx.test)}")
    atom_ig = [a[[bool(ATOM_TOKEN.match(t)) for t in tk]] for a, tk in zip(ig_all, toks_all)]
    save("ig", tokens=np.array(toks_all, dtype=object), token_ig=np.array(ig_all, dtype=object),
         atom_ig=np.array(atom_ig, dtype=object), fx_minus_fbase=np.array(gap_all), ig_sum=np.array(sum_all))
    gap, tot = np.array(gap_all), np.array(sum_all)
    share_atoms = np.mean([np.abs(a).sum() / max(np.abs(t).sum(), 1e-9) for a, t in zip(atom_ig, ig_all)])
    log(f"IG completeness: median |sum IG - (f(x)-f(x0))| / |f(x)-f(x0)| = "
        f"{np.median(np.abs(tot - gap) / np.maximum(np.abs(gap), 1e-6)):.3f} | "
        f"share of |IG| on atom tokens {share_atoms:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# Stage: atoms (project ECFP / MACCS SHAP onto atoms, combine, ZBG analysis)
# ─────────────────────────────────────────────────────────────────────────────
def ecfp_bit_atoms(mol):
    """{bit: [atom sets of each environment]} for the notebook's ECFP4-1024."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=data.ECFP_RADIUS, fpSize=data.ECFP_BITS)
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    gen.GetFingerprint(mol, additionalOutput=ao)
    out = {}
    for bit, envs in ao.GetBitInfoMap().items():
        sets = []
        for center, rad in envs:
            if rad == 0:
                sets.append({center})
            else:
                bonds = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, center)
                atoms = {center}
                for b in bonds:
                    bd = mol.GetBondWithIdx(b)
                    atoms |= {bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()}
                sets.append(atoms)
        out[bit] = sets
    return out


def maccs_key_atoms(mol):
    from rdkit import Chem
    from rdkit.Chem.MACCSkeys import smartsPatts
    out = {}
    for key, (sma, _) in smartsPatts.items():
        if sma == "?":
            continue
        patt = Chem.MolFromSmarts(sma)
        if patt is None:
            continue
        atoms = {a for match in mol.GetSubstructMatches(patt) for a in match}
        if atoms:
            out[key] = atoms
    return out


def project(values_by_feature, feature_atoms, n_atoms):
    w = np.zeros(n_atoms)
    for f, sets in feature_atoms.items():
        if isinstance(sets, set):
            sets = [sets]
        v = values_by_feature[f]
        if v == 0:
            continue
        for s in sets:
            for a in s:
                w[a] += v / len(sets) / len(s)
    return w


def zbg_regions(mol):
    """Atom index sets (zbg, linker, cap) for a hydroxamate, else None."""
    from rdkit import Chem
    m = mol.GetSubstructMatches(Chem.MolFromSmarts(ZBG_SMARTS))
    if not m:
        return None
    zbg = set(m[0])
    c = m[0][0]
    ring_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()]
    if not ring_atoms:
        return zbg, set(), set(range(mol.GetNumAtoms())) - zbg
    paths = [Chem.GetShortestPath(mol, c, r) for r in ring_atoms]
    paths = [p for p in paths if p]
    best = min(paths, key=len) if paths else (c,)
    linker = set(best[1:-1]) - zbg
    cap = set(range(mol.GetNumAtoms())) - zbg - linker
    return zbg, linker, cap


def stage_atoms(ctx):
    from rdkit import Chem
    ig = load("ig")
    sh = load("shap")
    atom_ig, atom_ecfp, atom_maccs, regions = [], [], [], []
    for pos, i in enumerate(ctx.test):
        mol = Chem.MolFromSmiles(ctx.feats["smiles"][i])
        n = mol.GetNumAtoms()
        a_ig = np.asarray(ig["atom_ig"][pos], dtype=float)
        assert len(a_ig) == n
        a_e = project(sh["ecfp"][pos], ecfp_bit_atoms(mol), n)
        a_m = project(sh["maccs"][pos], {k: [s] for k, s in maccs_key_atoms(mol).items()}, n)
        atom_ig.append(a_ig)
        atom_ecfp.append(a_e)
        atom_maccs.append(a_m)
        regions.append(zbg_regions(mol))
    combined = [a + b + c for a, b, c in zip(atom_ig, atom_ecfp, atom_maccs)]
    # region analysis on hydroxamates: attribution density (mean per atom) per region
    rows = []
    for pos, (reg, comb) in enumerate(zip(regions, combined)):
        if reg is None:
            continue
        zbg, linker, cap = reg
        dens = {name: comb[list(s)].mean() if s else np.nan for name, s in (("zbg", zbg), ("linker", linker), ("cap", cap))}
        top = int(np.argmax(comb))
        rows.append(dict(pos=pos, y=int(ctx.y_test[pos]), **{f"dens_{k}": v for k, v in dens.items()},
                         top_in_zbg=top in zbg, top_in_linker=top in linker, top_in_cap=top in cap,
                         share_pos_zbg=np.clip(comb[list(zbg)], 0, None).sum() / max(np.clip(comb, 0, None).sum(), 1e-9)))
    save("atoms", atom_ig=np.array(atom_ig, dtype=object), atom_ecfp=np.array(atom_ecfp, dtype=object),
         atom_maccs=np.array(atom_maccs, dtype=object), combined=np.array(combined, dtype=object),
         regions=np.array([json.dumps(r) for r in rows]))
    import pandas as pd
    r = pd.DataFrame(rows)
    log(f"hydroxamates in test: {len(r)} | top atom in ZBG: {r.top_in_zbg.mean():.2f}, linker {r.top_in_linker.mean():.2f}, "
        f"cap {r.top_in_cap.mean():.2f}")
    log("  attribution density (logit/atom) actives vs inactives: " + ", ".join(
        f"{k} {r[r.y == 1][f'dens_{k}'].mean():+.3f} / {r[r.y == 0][f'dens_{k}'].mean():+.3f}" for k in ("zbg", "linker", "cap")))


# ─────────────────────────────────────────────────────────────────────────────
# Stage: tree (TreeSHAP on XGBoost-ECFP, agreement with V3)
# ─────────────────────────────────────────────────────────────────────────────
def stage_tree(ctx):
    from rdkit import Chem
    from scipy.stats import spearmanr
    from xgboost import XGBClassifier
    X, y = ctx.feats["ecfp"], ctx.feats["labels"].astype(int)
    dev = ctx.split["dev"]
    clf = XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5,
                        n_jobs=6, random_state=42, tree_method="hist", eval_metric="logloss").fit(X[dev], y[dev])
    import xgboost as xgb
    contrib = clf.get_booster().predict(xgb.DMatrix(X[ctx.test]), pred_contribs=True)[:, :-1]   # TreeSHAP
    sh = load("shap")
    v3_imp, xgb_imp = np.abs(sh["ecfp"]).mean(0), np.abs(contrib).mean(0)
    rho = spearmanr(v3_imp, xgb_imp).correlation
    top_v3, top_x = set(np.argsort(-v3_imp)[:20]), set(np.argsort(-xgb_imp)[:20])
    at = load("atoms")
    per_mol = []
    for pos, i in enumerate(ctx.test):
        mol = Chem.MolFromSmiles(ctx.feats["smiles"][i])
        a_x = project(contrib[pos], ecfp_bit_atoms(mol), mol.GetNumAtoms())
        a_v = np.asarray(at["combined"][pos], dtype=float)
        if a_x.std() > 0 and a_v.std() > 0:
            per_mol.append(spearmanr(a_x, a_v).correlation)
    save("tree", xgb_ecfp_shap=contrib, v3_bit_importance=v3_imp, xgb_bit_importance=xgb_imp,
         per_mol_atom_rho=np.array(per_mol), bit_rho=np.array(rho), top20_overlap=np.array(len(top_v3 & top_x)))
    log(f"TreeSHAP vs V3: bit-importance Spearman {rho:.3f} | top-20 bit overlap {len(top_v3 & top_x)}/20 | "
        f"median per-molecule atom-map Spearman {np.median(per_mol):.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Perturbation helpers (LIME + deletion): mask atoms as dummy atoms '*'
# ─────────────────────────────────────────────────────────────────────────────
def masked_smiles(mol, masked):
    from rdkit import Chem
    rw = Chem.RWMol(mol)
    Chem.Kekulize(rw, clearAromaticFlags=True)
    for a in masked:
        at = rw.GetAtomWithIdx(int(a))
        at.SetAtomicNum(0)
        at.SetFormalCharge(0)
        at.SetNoImplicit(True)
        at.SetNumExplicitHs(0)
    return Chem.MolToSmiles(rw)


def stage_lime(ctx, n_mols=100, n_samples=400, stability_mols=25, stability_runs=3):
    from lime.lime_base import LimeBase
    from rdkit import Chem
    from scipy.stats import spearmanr
    pr = load("predict")
    rng = np.random.default_rng(3)
    y = ctx.y_test
    pool = np.concatenate([rng.choice(np.where(y == 1)[0], n_mols // 2, replace=False),
                           rng.choice(np.where(y == 0)[0], n_mols // 2, replace=False)])
    pool = np.unique(np.concatenate([pool, pr["showcase_pos"]]))
    kernel = lambda d: np.sqrt(np.exp(-(d ** 2) / 0.25 ** 2))       # LIME's default kernel, width 0.25
    weights, stab = {}, []

    def explain(pos, seed):
        mol = Chem.MolFromSmiles(ctx.feats["smiles"][ctx.test[pos]])
        n = mol.GetNumAtoms()
        r = np.random.default_rng(seed)
        Z = (r.random((n_samples, n)) > 0.5).astype(int)
        Z[0] = 1
        smi = [masked_smiles(mol, np.where(z == 0)[0]) for z in Z]
        p = ctx.predict_smiles(smi, draw=0)
        dist = 1 - (Z @ Z[0]) / (np.sqrt(Z.sum(1)) * np.sqrt(n) + 1e-9)       # cosine distance
        lb = LimeBase(kernel, random_state=seed)
        _, exp, score, _ = lb.explain_instance_with_data(Z, np.stack([1 - p, p], 1), dist, 1, n,
                                                         feature_selection="none")
        w = np.zeros(n)
        for f, v in exp:
            w[f] = v
        return w, score

    for c, pos in enumerate(pool):
        GUARD.check()
        weights[int(pos)], _ = explain(pos, 0)
        if c < stability_mols:
            ws = [weights[int(pos)]] + [explain(pos, s)[0] for s in range(1, stability_runs)]
            rhos = [spearmanr(ws[a], ws[b]).correlation for a in range(len(ws)) for b in range(a + 1, len(ws))]
            stab.append(np.nanmean(rhos))
        if c % 20 == 0:
            log(f"lime: {c}/{len(pool)}")
    at = load("atoms")
    agree = [spearmanr(weights[p], np.asarray(at["combined"][p], dtype=float)).correlation for p in weights]
    save("lime", pos=np.array(list(weights)), weights=np.array([weights[p] for p in weights], dtype=object),
         stability=np.array(stab), agreement_with_combined=np.array(agree))
    log(f"LIME: {len(weights)} molecules | run-to-run stability (Spearman) {np.nanmean(stab):.3f} | "
        f"agreement with IG+SHAP atom map {np.nanmedian(agree):.3f}")


def stage_deletion(ctx, fractions=(0, 0.1, 0.2, 0.3, 0.4, 0.5), n_random=3):
    from rdkit import Chem
    pr, at, li = load("predict"), load("atoms"), load("lime")
    lime_w = {int(p): np.asarray(w, dtype=float) for p, w in zip(li["pos"], li["weights"])}
    # molecules predicted active (deleting evidence for "active" should lower P(active))
    cand = [p for p in lime_w if pr["ens"][p] > NB_THRESHOLD]
    methods = {"IG+SHAP (combined)": lambda p: np.asarray(at["combined"][p], dtype=float),
               "IG (SMILES only)": lambda p: np.asarray(at["atom_ig"][p], dtype=float),
               "SHAP ECFP->atoms": lambda p: np.asarray(at["atom_ecfp"][p], dtype=float),
               "LIME": lambda p: lime_w[p]}
    rng = np.random.default_rng(5)
    curves = {k: [] for k in list(methods) + ["random"]}
    for c, p in enumerate(cand):
        GUARD.check()
        mol = Chem.MolFromSmiles(ctx.feats["smiles"][ctx.test[p]])
        n = mol.GetNumAtoms()
        batch, keys = [], []
        for name, fn in methods.items():
            order = np.argsort(-fn(p))
            for fr in fractions:
                batch.append(masked_smiles(mol, order[:int(round(fr * n))]))
                keys.append((name, fr))
        for r in range(n_random):
            order = rng.permutation(n)
            for fr in fractions:
                batch.append(masked_smiles(mol, order[:int(round(fr * n))]))
                keys.append(("random", fr))
        probs = ctx.predict_smiles(batch, draw=0)
        for name in curves:
            vals = [probs[i] for i, (k, fr) in enumerate(keys) if k == name]
            curves[name].append(np.mean(np.reshape(vals, (-1, len(fractions))), 0))
    curves = {k: np.array(v) for k, v in curves.items()}
    save("deletion", fractions=np.array(fractions), names=np.array(list(curves)),
         curves=np.stack([curves[k] for k in curves]))
    log(f"deletion ({len(cand)} predicted-active molecules), area under P(active) curve (lower = more faithful): " +
        ", ".join(f"{k} {np.trapz(v.mean(0), fractions):.3f}" for k, v in curves.items()))


# ─────────────────────────────────────────────────────────────────────────────
# Stage: domain (applicability domain + calibration)
# ─────────────────────────────────────────────────────────────────────────────
def stage_domain(ctx):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from sklearn.metrics import brier_score_loss, roc_auc_score
    pr = load("predict")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = {i: gen.GetFingerprint(Chem.MolFromSmiles(ctx.feats["smiles"][i]))
           for i in np.concatenate([ctx.split["dev"], ctx.test])}
    dev_fps = [fps[i] for i in ctx.split["dev"]]
    maxsim = np.array([max(DataStructs.BulkTanimotoSimilarity(fps[i], dev_fps)) for i in ctx.test])
    y, p = ctx.y_test, pr["ens"]
    err = ((p > NB_THRESHOLD).astype(int) != y)
    bins = [0, 0.5, 0.7, 0.9, 1.01]
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (maxsim >= lo) & (maxsim < hi)
        rows.append(dict(bin=f"[{lo:.1f},{min(hi, 1):.1f}{')' if hi < 1 else ']'}", n=int(sel.sum()),
                         error_rate=float(err[sel].mean()) if sel.any() else np.nan,
                         auc=float(roc_auc_score(y[sel], p[sel])) if sel.any() and len(set(y[sel])) == 2 else np.nan))
    frac_pos, mean_pred = [], []
    edges = np.linspace(0, 1, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if sel.sum() >= 5:
            frac_pos.append(y[sel].mean())
            mean_pred.append(p[sel].mean())
    save("domain", maxsim=maxsim, bins=np.array([json.dumps(r) for r in rows]),
         calib_pred=np.array(mean_pred), calib_frac=np.array(frac_pos), brier=np.array(brier_score_loss(y, p)))
    log("applicability domain: " + " | ".join(f"{r['bin']} n={r['n']} err={r['error_rate']:.3f} AUC={r['auc']:.3f}" for r in rows))
    log(f"Brier {brier_score_loss(y, p):.4f}")


STAGES = ["predict", "modality", "shap", "ig", "atoms", "tree", "lime", "deletion", "domain"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=STAGES + ["figures"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(0.2)
    ctx = None
    for st in args.stages:
        if st == "figures":
            import explain_figures
            explain_figures.main()
            continue
        if exists(st) and not args.force:
            log(f"{st}: cached")
            continue
        if ctx is None:
            ctx = Ctx()
        t0 = time.time()
        log(f"== stage {st}")
        globals()[f"stage_{st}"](ctx)
        log(f"== stage {st} done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
