"""
V4 building blocks (used by core.py only when a V4 option is switched on; the V3 code
path is untouched and stays bit-identical, see tests/test_core_parity.py).

  * upgraded fingerprint : count-based Morgan radius 2, 2048 bits, chirality on,
                           fed as log1p(count)            -> feats["ecfp2048c"]
  * substructure tokens  : the non-zero bits of that fingerprint as a token set
                           (bit id + log1p count), for token-level fusion
  * molecular graph      : Chemprop-style atom / bond features for a D-MPNN
  * DMPNN                : directed message passing neural network (Yang et al., JCIM 2019)
"""
import os

import numpy as np
import torch
import torch.nn as nn

import data

V4_CACHE = os.path.join(data.HERE, "cache", "features_v4.npz")
FP2_BITS = 2048

_ELEMS = [6, 7, 8, 16, 9, 17, 35, 53, 15, 5, 14, 34]          # C N O S F Cl Br I P B Si Se (+ other)
_HYB = ["SP", "SP2", "SP3", "SP3D", "SP3D2"]
_BOND = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
_STEREO = ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE", "STEREOCIS", "STEREOTRANS"]


def _onehot(v, choices):
    out = [0.0] * (len(choices) + 1)
    out[choices.index(v) if v in choices else -1] = 1.0
    return out


def atom_features(a):
    return (_onehot(a.GetAtomicNum(), _ELEMS) + _onehot(a.GetTotalDegree(), [0, 1, 2, 3, 4, 5])
            + _onehot(a.GetFormalCharge(), [-1, 0, 1]) + _onehot(int(a.GetChiralTag()), [0, 1, 2, 3])
            + _onehot(a.GetTotalNumHs(), [0, 1, 2, 3, 4]) + _onehot(str(a.GetHybridization()), _HYB)
            + [float(a.GetIsAromatic()), float(a.IsInRing()), a.GetMass() * 0.01])


def bond_features(b):
    return (_onehot(str(b.GetBondType()), _BOND) + [float(b.GetIsConjugated()), float(b.IsInRing())]
            + _onehot(str(b.GetStereo()), _STEREO))


ATOM_DIM = len(_ELEMS) + 1 + 7 + 4 + 5 + 6 + 6 + 3      # 44
BOND_DIM = len(_BOND) + 1 + 2 + len(_STEREO) + 1         # 14


def featurize_v4(smi):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    mol = Chem.MolFromSmiles(smi)
    fp = np.zeros(FP2_BITS, np.float32)
    if mol is None:
        return fp, np.zeros((1, ATOM_DIM), np.float32), np.zeros((2, 0), np.int64), np.zeros((0, BOND_DIM), np.float32)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP2_BITS, includeChirality=True)
    fp = np.log1p(gen.GetCountFingerprintAsNumPy(mol).astype(np.float32))
    x = np.array([atom_features(a) for a in mol.GetAtoms()], np.float32)
    src, dst, ea = [], [], []
    for b in mol.GetBonds():                       # directed edges in (u->v, v->u) pairs
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = bond_features(b)
        src += [u, v]
        dst += [v, u]
        ea += [f, f]
    ei = np.array([src, dst], np.int64) if src else np.zeros((2, 0), np.int64)
    ea = np.array(ea, np.float32) if ea else np.zeros((0, BOND_DIM), np.float32)
    return fp, x, ei, ea


def get_v4_features(smiles):
    """dict(ecfp2048c [N, 2048] float32, graphs: list of (x, edge_index, edge_attr)), cached."""
    if os.path.exists(V4_CACHE):
        z = np.load(V4_CACHE, allow_pickle=True)
        if list(z["smiles"]) == list(smiles):
            return {"ecfp2048c": z["ecfp2048c"], "graphs": list(z["graphs"])}
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(featurize_v4, list(smiles), chunksize=64))
    fp = np.stack([r[0] for r in res])
    graphs = np.empty(len(res), dtype=object)
    for i, r in enumerate(res):
        graphs[i] = (r[1], r[2], r[3])
    np.savez_compressed(V4_CACHE, smiles=np.array(list(smiles), dtype=object), ecfp2048c=fp, graphs=graphs)
    return {"ecfp2048c": fp, "graphs": list(graphs)}


# ── batching ─────────────────────────────────────────────────────────────────
def collate_fp_tokens(fps):
    """fps [B, 2048] (log1p counts) -> padded token ids / counts / mask of the non-zero bits."""
    nz = [np.nonzero(f)[0] for f in fps]
    T = max(1, max(len(n) for n in nz))
    ids = torch.zeros(len(fps), T, dtype=torch.long)
    cnt = torch.zeros(len(fps), T)
    mask = torch.zeros(len(fps), T, dtype=torch.bool)
    for i, (f, n) in enumerate(zip(fps, nz)):
        ids[i, :len(n)] = torch.from_numpy(n)
        cnt[i, :len(n)] = torch.from_numpy(f[n])
        mask[i, :len(n)] = True
    return {"fp_ids": ids, "fp_cnt": cnt, "fp_mask": mask}


def collate_graphs(graphs):
    xs, srcs, dsts, eas, batch = [], [], [], [], []
    off = 0
    for g, (x, ei, ea) in enumerate(graphs):
        xs.append(x)
        srcs.append(ei[0] + off)
        dsts.append(ei[1] + off)
        eas.append(ea)
        batch.append(np.full(len(x), g))
        off += len(x)
    return {"g_x": torch.from_numpy(np.concatenate(xs)), "g_src": torch.from_numpy(np.concatenate(srcs)),
            "g_dst": torch.from_numpy(np.concatenate(dsts)), "g_ea": torch.from_numpy(np.concatenate(eas)),
            "g_batch": torch.from_numpy(np.concatenate(batch))}


# ── D-MPNN ───────────────────────────────────────────────────────────────────
class DMPNN(nn.Module):
    """Directed MPNN (Chemprop). Edge (u->v) hidden states; message into (u->v) is the sum of
    hidden states of edges entering u, minus the reverse edge (v->u). Mean readout over atoms."""

    def __init__(self, hidden=300, depth=3, dropout=0.1):
        super().__init__()
        self.depth = depth
        self.W_i = nn.Linear(ATOM_DIM + BOND_DIM, hidden, bias=False)
        self.W_h = nn.Linear(hidden, hidden, bias=False)
        self.W_o = nn.Linear(ATOM_DIM + hidden, hidden)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.hidden = hidden

    def forward(self, g_x, g_src, g_dst, g_ea, g_batch, n_graphs):
        n_atoms, n_edges = g_x.size(0), g_src.size(0)
        if n_edges:
            rev = torch.arange(n_edges, device=g_x.device) ^ 1          # edges are stored in (u->v, v->u) pairs
            h0 = self.act(self.W_i(torch.cat([g_x[g_src], g_ea], 1)))
            h = h0
            for _ in range(self.depth - 1):
                inc = torch.zeros(n_atoms, self.hidden, device=g_x.device, dtype=h.dtype).index_add_(0, g_dst, h)
                m = inc[g_src] - h[rev]
                h = self.drop(self.act(h0 + self.W_h(m)))
            a_msg = torch.zeros(n_atoms, self.hidden, device=g_x.device, dtype=h.dtype).index_add_(0, g_dst, h)
        else:
            a_msg = torch.zeros(n_atoms, self.hidden, device=g_x.device, dtype=g_x.dtype)
        a = self.drop(self.act(self.W_o(torch.cat([g_x.to(a_msg.dtype), a_msg], 1))))
        out = torch.zeros(n_graphs, self.hidden, device=g_x.device, dtype=a.dtype).index_add_(0, g_batch, a)
        cnt = torch.zeros(n_graphs, device=g_x.device).index_add_(0, g_batch, torch.ones_like(g_batch, dtype=torch.float))
        return out / cnt.clamp(min=1).unsqueeze(1).to(out.dtype)
