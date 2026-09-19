"""Dataset loading and the cached ECFP / MACCS / descriptor features."""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_XLSX = r"E:\ML\BioActivity\Dataset\bioactivity_dataset_cleaned_new_v3.xlsx"
CACHE_NPZ = os.path.join(HERE, "cache", "features_v3.npz")
ECFP_BITS, ECFP_RADIUS, MACCS_BITS, N_DESC = 1024, 2, 167, 6
DESC_NAMES = ["MolWt", "MolLogP", "NumHDonors", "NumHAcceptors", "TPSA", "NumRotatableBonds"]


def load_df():
    return pd.read_excel(DATA_XLSX, sheet_name="Sheet1")


def featurize_one(smi):
    """ECFP4 (1024), MACCS (167), 6 RO5 descriptors - identical to notebook cells 5/9."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
    mol = Chem.MolFromSmiles(smi)
    ecfp = np.zeros(ECFP_BITS, np.float32)
    maccs = np.zeros(MACCS_BITS, np.float32)
    desc = np.zeros(N_DESC, np.float32)
    if mol is not None:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=ECFP_RADIUS, fpSize=ECFP_BITS)
        DataStructs.ConvertToNumpyArray(gen.GetFingerprint(mol), ecfp)
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), maccs)
        try:
            desc = np.array([
                Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
                Descriptors.TPSA(mol), Descriptors.NumRotatableBonds(mol),
            ], np.float32)
        except Exception:
            pass
    return ecfp, maccs, desc


def get_features(df=None):
    """dict(smiles, labels, ecfp, maccs, desc_raw) for every row of df (cached)."""
    if df is None:
        df = load_df()
    smiles = df["canonical_smiles"].tolist()
    if os.path.exists(CACHE_NPZ):
        z = np.load(CACHE_NPZ, allow_pickle=True)
        if list(z["smiles"]) == smiles:
            return {k: z[k] for k in z.files}
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(featurize_one, smiles, chunksize=64))
    feats = {
        "smiles": np.array(smiles, dtype=object),
        "labels": df["bioactivity"].values.astype(np.float32),
        "ecfp": np.stack([r[0] for r in res]),
        "maccs": np.stack([r[1] for r in res]),
        "desc_raw": np.stack([r[2] for r in res]),
    }
    os.makedirs(os.path.dirname(CACHE_NPZ), exist_ok=True)
    np.savez_compressed(CACHE_NPZ, **feats)
    return feats


RAW_XLSX = r"E:\ML\BioActivity\Dataset\bioactivity_dataset.xlsx"


def get_pic50(smiles):
    """Median measured pIC50 per molecule from the raw ChEMBL export (for training targets only)."""
    cache = os.path.join(HERE, "cache", "pic50.npy")
    if os.path.exists(cache):
        arr = np.load(cache)
        if len(arr) == len(smiles):
            return arr
    raw = pd.read_excel(RAW_XLSX)
    raw = raw[np.isfinite(raw["pIC50"])]
    med = raw.groupby("canonical_smiles")["pIC50"].median()
    arr = pd.Series(list(smiles)).map(med).values.astype(np.float32)
    assert not np.isnan(arr).any(), "molecule without a raw pIC50"
    np.save(cache, arr)
    return arr
