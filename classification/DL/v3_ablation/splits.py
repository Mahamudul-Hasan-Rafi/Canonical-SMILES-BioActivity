"""
Data splits, computed once and frozen to splits/<name>.npz (with a hash of the
dataset), so every model in every experiment sees exactly the same indices.

  random   : the notebook's split. test = 15% (train_test_split, seed 42);
             CV = StratifiedKFold(5, seed 42) on train+val; single = train / val.
  scaffold : Bemis-Murcko scaffold split. Molecules are grouped by scaffold;
             StratifiedGroupKFold(7, seed 42) fold 0 (735 mols, 13%) is the test set,
             StratifiedGroupKFold(5, seed 42) on the rest gives the CV folds.
             No scaffold is shared between test and dev, or between CV folds.

All indices are positions into the dataframe returned by data.load_df().
"""
import hashlib
import os

import numpy as np

import data

SPLIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits")


def _dataset_hash(df):
    h = hashlib.sha1()
    h.update("\n".join(df["canonical_smiles"]).encode())
    h.update(df["bioactivity"].values.astype(np.int8).tobytes())
    return h.hexdigest()[:16]


def _random(df):
    from sklearn.model_selection import StratifiedKFold, train_test_split
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["bioactivity"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    tr, va, te = train_df.index.values, val_df.index.values, test_df.index.values
    dev = np.concatenate([tr, va])
    y = df["bioactivity"].values[dev].astype(int)
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    return dict(train=tr, val=va, test=te, dev=dev, folds=folds)


def scaffolds(df):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    return np.array([MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s)) for s in df["canonical_smiles"]])


def _scaffold(df):
    from sklearn.model_selection import StratifiedGroupKFold
    y = df["bioactivity"].values.astype(int)
    sc = scaffolds(df)
    outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)
    dev, te = next(iter(outer.split(np.zeros(len(y)), y, groups=sc)))
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(inner.split(np.zeros(len(dev)), y[dev], groups=sc[dev]))
    # single-model protocol on scaffold split: fold 0 of the CV acts as validation
    return dict(train=dev[folds[0][0]], val=dev[folds[0][1]], test=te, dev=dev, folds=folds)


_BUILDERS = {"random": _random, "scaffold": _scaffold}


def get_split(name, df=None):
    """dict(train, val, test, dev, folds=[(tr_pos, vl_pos) positions into dev])."""
    if df is None:
        df = data.load_df()
    os.makedirs(SPLIT_DIR, exist_ok=True)
    path = os.path.join(SPLIT_DIR, f"{name}.npz")
    dh = _dataset_hash(df)
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        if str(z["dataset_hash"]) == dh:
            folds = [(z[f"fold{i}_tr"], z[f"fold{i}_vl"]) for i in range(int(z["n_folds"]))]
            return dict(train=z["train"], val=z["val"], test=z["test"], dev=z["dev"], folds=folds)
        raise RuntimeError(f"splits/{name}.npz was built from a different dataset - delete it deliberately to rebuild")
    s = _BUILDERS[name](df)
    extra = {f"fold{i}_tr": tr for i, (tr, _) in enumerate(s["folds"])}
    extra.update({f"fold{i}_vl": vl for i, (_, vl) in enumerate(s["folds"])})
    np.savez(path, train=s["train"], val=s["val"], test=s["test"], dev=s["dev"],
             n_folds=len(s["folds"]), dataset_hash=dh, **extra)
    return s


def check(name):
    df = data.load_df()
    s = get_split(name, df)
    y = df["bioactivity"].values
    print(f"[{name}] dev {len(s['dev'])} ({y[s['dev']].mean():.3f} active) | test {len(s['test'])} "
          f"({y[s['test']].mean():.3f} active)")
    if name == "scaffold":
        sc = scaffolds(df)
        assert not set(sc[s["test"]]) & set(sc[s["dev"]]), "scaffold leak test/dev"
        for i, (tr, vl) in enumerate(s["folds"]):
            assert not set(sc[s["dev"][tr]]) & set(sc[s["dev"][vl]]), f"scaffold leak fold {i}"
        print("  no scaffold shared between test/dev or between any CV train/val pair")
    for i, (tr, vl) in enumerate(s["folds"]):
        print(f"  fold {i}: train {len(tr)}  val {len(vl)} ({y[s['dev'][vl]].mean():.3f} active)")


if __name__ == "__main__":
    for n in _BUILDERS:
        check(n)
