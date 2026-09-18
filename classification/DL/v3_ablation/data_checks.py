"""
CPU-only data audit of the notebook's split (no training).

  * duplicate SMILES and their labels, and which split each copy landed in
  * test-vs-train similarity: max Tanimoto (ECFP4) to any training molecule,
    and whether the test molecule's Murcko scaffold also occurs in training
Writes results/data_checks.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v3_core as C


def main():
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold

    df = C.load_df()
    tr, va, te = C.notebook_splits(df)
    split = np.empty(len(df), dtype=object)
    split[tr], split[va], split[te] = "train", "val", "test"
    smi = df["canonical_smiles"].tolist()
    y = df["bioactivity"].values
    rep = {}

    # duplicates
    groups = df.groupby("canonical_smiles").indices
    dups = {s: ix for s, ix in groups.items() if len(ix) > 1}
    rows = [{"smiles": s, "labels": [int(y[i]) for i in ix], "splits": [split[i] for i in ix]} for s, ix in dups.items()]
    rep["n_duplicate_smiles"] = len(dups)
    rep["n_duplicates_conflicting_labels"] = sum(len(set(r["labels"])) > 1 for r in rows)
    rep["n_duplicates_touching_test"] = sum("test" in r["splits"] for r in rows)
    rep["duplicates"] = rows

    # similarity
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    mols = [Chem.MolFromSmiles(s) for s in smi]
    fps = [gen.GetFingerprint(m) for m in mols]
    scaf = [MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols]
    dev = np.concatenate([tr, va])
    dev_fps = [fps[i] for i in dev]
    dev_scaf = set(scaf[i] for i in dev)
    maxsim = np.array([max(DataStructs.BulkTanimotoSimilarity(fps[i], dev_fps)) for i in te])
    rep["test_max_tanimoto_to_trainval"] = {
        "mean": float(maxsim.mean()), "median": float(np.median(maxsim)),
        "frac_ge_0.9": float((maxsim >= 0.9).mean()), "frac_ge_0.7": float((maxsim >= 0.7).mean()),
        "frac_eq_1.0": float((maxsim >= 0.999).mean()), "frac_lt_0.4": float((maxsim < 0.4).mean())}
    rep["test_scaffold_seen_in_trainval"] = float(np.mean([scaf[i] in dev_scaf for i in te]))
    rep["n_unique_scaffolds"] = len(set(scaf))
    np.save(os.path.join(C.HERE, "results", "test_maxsim.npy"), maxsim)

    out = os.path.join(C.HERE, "results", "data_checks.json")
    with open(out, "w") as fh:
        json.dump(rep, fh, indent=2)
    print(json.dumps({k: v for k, v in rep.items() if k != "duplicates"}, indent=2))
    print("duplicate pairs by split:", sorted({tuple(sorted(r["splits"])) for r in rows}))


if __name__ == "__main__":
    main()
