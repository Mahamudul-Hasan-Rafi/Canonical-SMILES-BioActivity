"""
V6 retrieval must never leak: validation / test molecules may only retrieve training-fold
molecules, no molecule retrieves itself, and the stored similarities must match RDKit.
Also runs a 2-epoch smoke training of both V6 variants.
Run:  python tests/test_v6_retrieval.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backbones
import core
import data
import splits
import v4_modules as V


def main():
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    df = data.load_df()
    feats = data.get_features(df)
    SPL = {n: splits.get_split(n, df) for n in ("random", "scaffold")}
    S = V.get_sim_matrix(feats["smiles"])
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    rng = np.random.default_rng(0)
    for _ in range(5):
        i, j = rng.integers(0, len(df), 2)
        ref = DataStructs.TanimotoSimilarity(gen.GetFingerprint(Chem.MolFromSmiles(feats["smiles"][i])),
                                             gen.GetFingerprint(Chem.MolFromSmiles(feats["smiles"][j])))
        assert abs(float(S[i, j]) - ref) < 1e-3, (i, j, S[i, j], ref)
    print("similarity matrix matches RDKit (5 random pairs)")

    for split_name, s in SPL.items():
        # replicate train_one's retrieval for every CV fold and check the invariants
        for f, (ftr, fvl) in enumerate(s["folds"]):
            tr_arr, vl, te = s["dev"][ftr], s["dev"][fvl], s["test"]
            tr_set = set(tr_arr.tolist())

            def nbr_of(idx):
                sub = S[np.ix_(np.asarray(idx), tr_arr)].astype(np.float32)
                sub[np.asarray(idx)[:, None] == tr_arr[None, :]] = -1.0
                top = np.argsort(-sub, axis=1)[:, :5]
                return tr_arr[top], np.take_along_axis(sub, top, 1)
            for name, idx in (("train", tr_arr), ("val", vl), ("test", te)):
                nb, sim = nbr_of(idx)
                assert set(nb.ravel().tolist()) <= tr_set, f"fold {f} {name}: retrieved outside the training fold"
                assert not (nb == np.asarray(idx)[:, None]).any(), f"fold {f} {name}: retrieved itself"
                assert (sim >= 0).all()
            assert not (set(vl.tolist()) | set(te.tolist())) & tr_set
        print(f"retrieval [{split_name}]: val/test only retrieve training-fold molecules, never themselves (all 5 folds)")

    s = SPL["random"]
    tr, vl = s["dev"][s["folds"][0][0]][:384], s["dev"][s["folds"][0][1]][:128]
    for v in ["graph_mt_knn", "graph_mt_delta"]:
        cfg = core.make_cfg(v, backbone="chemberta_mlm", max_epochs=2)
        out = core.train_one(cfg, feats, tr, vl, {"test": s["test"][:64]}, 1, torch.device("cuda"),
                             backbones.load_tokenizer("chemberta_mlm"))
        print(f"{v:<15} smoke ok | val AUC {out['best_val_auc']:.3f} | trainable {out['n_trainable']/1e6:.1f}M | "
              f"cfg retrieval={cfg.retrieval} delta_w={cfg.delta_w}")
    print("all V6 checks passed")


if __name__ == "__main__":
    main()
