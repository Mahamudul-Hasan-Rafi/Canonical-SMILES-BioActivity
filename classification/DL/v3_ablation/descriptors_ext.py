"""
Extended descriptor blocks, cached next to the other features.

Our models have always used six Lipinski descriptors. CheMeleon's advantage on potency comes
from being pretrained on Mordred descriptors, so the like-for-like response is to give our own
regressors a real descriptor block rather than six columns.

  rdkit   ~210 RDKit descriptors (Descriptors.descList), the standard cheap set
  mordred ~1600 Mordred 2D descriptors, the set CheMeleon was pretrained on

Non-finite values are replaced by the training-set median at fit time by the caller; here they
are written as NaN and constant/all-NaN columns are dropped once, deterministically.

  python descriptors_ext.py            # builds both caches
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data

CACHE = os.path.join(data.HERE, "results", "cache")


def _clean(X, names):
    """Drop columns that are constant or more than 10 % non-finite; keep the order stable."""
    X = np.asarray(X, dtype=np.float64)
    finite = np.isfinite(X)
    keep = (finite.mean(0) > 0.9)
    with np.errstate(invalid="ignore"):
        Xm = np.where(finite, X, np.nan)
        spread = np.nanmax(Xm, 0) - np.nanmin(Xm, 0)
    keep &= np.isfinite(spread) & (spread > 0)
    return X[:, keep], [n for n, k in zip(names, keep) if k]


def get_rdkit(smiles):
    path = os.path.join(CACHE, "rdkit_desc.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        if len(z["X"]) == len(smiles):
            return z["X"], list(z["names"])
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    RDLogger.DisableLog("rdApp.*")
    names = [n for n, _ in Descriptors.descList]
    fns = [f for _, f in Descriptors.descList]
    rows = []
    for smi in smiles:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            rows.append([np.nan] * len(fns))
            continue
        row = []
        for f in fns:
            try:
                row.append(float(f(m)))
            except Exception:
                row.append(np.nan)
        rows.append(row)
    X, names = _clean(rows, names)
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, X=X, names=np.array(names, dtype=object))
    return X, names


def get_mordred(smiles):
    path = os.path.join(CACHE, "mordred_desc.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        if len(z["X"]) == len(smiles):
            return z["X"], list(z["names"])
    from mordred import Calculator, descriptors
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    calc = Calculator(descriptors, ignore_3D=True)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    ok = [m if m is not None else Chem.MolFromSmiles("C") for m in mols]
    df = calc.pandas(ok, nproc=6, quiet=True)
    names = list(df.columns)
    X = df.apply(lambda c: c.map(lambda v: float(v) if isinstance(v, (int, float)) else np.nan)).values
    X[[i for i, m in enumerate(mols) if m is None], :] = np.nan
    X, names = _clean(X, [str(n) for n in names])
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, X=X, names=np.array(names, dtype=object))
    return X, names


def main():
    df = data.load_df()
    smiles = list(data.get_features(df)["smiles"])
    X, n = get_rdkit(smiles)
    print(f"rdkit   : {X.shape[0]} molecules x {X.shape[1]} descriptors")
    try:
        Xm, nm = get_mordred(smiles)
        print(f"mordred : {Xm.shape[0]} molecules x {Xm.shape[1]} descriptors")
    except Exception as exc:
        print(f"mordred : unavailable ({type(exc).__name__}: {exc})")


if __name__ == "__main__":
    main()
