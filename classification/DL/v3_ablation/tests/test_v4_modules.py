"""
Unit checks for the V4 building blocks:
  1. feature cache: shapes, log1p-count fingerprint, chirality actually distinguishes stereoisomers
  2. D-MPNN: output invariant to atom renumbering (graph isomorphism), correct batching
     (a molecule's embedding is the same alone or inside a batch), reverse-edge pairing
  3. every V4 variant builds, runs forward/backward on a real batch
Run:  python tests/test_v4_modules.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backbones
import core
import data
import v4_modules as V


def main():
    torch.manual_seed(0)
    df = data.load_df()
    feats = data.get_features(df)
    v4 = V.get_v4_features(feats["smiles"])
    fp, graphs = v4["ecfp2048c"], v4["graphs"]
    print(f"fingerprint {fp.shape}, mean non-zero bits {np.mean((fp > 0).sum(1)):.1f}, max log1p count {fp.max():.2f}")
    assert fp.shape == (len(df), 2048) and len(graphs) == len(df)

    from rdkit import Chem
    a = V.featurize_v4("C[C@H](N)C(=O)O")[0]
    b = V.featurize_v4("C[C@@H](N)C(=O)O")[0]
    print("chirality distinguishes L/D-alanine:", not np.array_equal(a, b))
    assert not np.array_equal(a, b)

    # D-MPNN invariances
    enc = V.DMPNN(hidden=64, depth=3, dropout=0.0).double().eval()
    smi = df["canonical_smiles"][0]
    mol = Chem.MolFromSmiles(smi)
    perm = np.random.default_rng(0).permutation(mol.GetNumAtoms()).tolist()
    smi_perm = Chem.MolToSmiles(Chem.RenumberAtoms(mol, perm), canonical=False)

    def emb(glist):
        gb = V.collate_graphs(glist)
        gb = {k: (v.double() if v.is_floating_point() else v) for k, v in gb.items()}
        with torch.no_grad():
            return enc(**gb, n_graphs=len(glist))

    g0 = V.featurize_v4(smi)[1:]
    gp = V.featurize_v4(smi_perm)[1:]
    d_perm = (emb([g0]) - emb([gp])).abs().max().item()
    batch = [graphs[i] for i in range(1, 6)] + [g0]
    d_batch = (emb(batch)[-1] - emb([g0])[0]).abs().max().item()
    ei = g0[1]
    rev_ok = all(ei[0, e] == ei[1, e ^ 1] and ei[1, e] == ei[0, e ^ 1] for e in range(ei.shape[1]))
    print(f"D-MPNN atom-renumbering invariance: max |diff| = {d_perm:.2e}")
    print(f"D-MPNN alone vs in a batch:        max |diff| = {d_batch:.2e}")
    print(f"reverse-edge pairing (e ^ 1) correct: {rev_ok}")
    assert d_perm < 1e-9 and d_batch < 1e-9 and rev_ok

    # every V4 variant: build, forward, backward on a real batch
    tok = backbones.load_tokenizer("chemberta_mlm")
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(feats["desc_raw"])
    f2 = dict(feats, **v4)
    idx = np.arange(16)
    for v in ["fp_upgrade", "token_fusion", "graph_branch"]:
        cfg = core.make_cfg(v, backbone="chemberta_mlm")
        ds = core.V3Dataset(f2, idx, tok, sc, fp_key="ecfp2048c", graphs=cfg.graph)
        col = core.make_collate(tok.pad_token_id, fp_tokens=cfg.fusion == "token", graphs=cfg.graph)
        b = col([ds[i] for i in range(len(ds))])
        m = core.build_model(cfg).train()
        out = m(**{k: val for k, val in b.items() if k != "labels"})
        out.sum().backward()
        n_new = sum(p.numel() for n, p in m.named_parameters() if n.startswith(core.NEW_PREFIXES))
        print(f"{v:<13} forward {tuple(out.shape)} ok | trainable {sum(p.numel() for p in m.parameters() if p.requires_grad)/1e6:.1f}M"
              f" | new-module params {n_new/1e6:.2f}M | cfg: {cfg.to_dict()['fp_kind']}, graph={cfg.graph}, fusion={cfg.fusion}")
    print("all V4 checks passed")


if __name__ == "__main__":
    main()
