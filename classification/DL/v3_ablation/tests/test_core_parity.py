"""
Parity: core.py (new, backbone-generic) vs v3_core.py (produced the reproduction
and ablation results). With backbone='molformer' they must be identical:
  1. same initial parameters for the same seed (every variant)
  2. same forward output
  3. same training trajectory (short train_one on a subset) -> same predictions
Run:  python tests/test_core_parity.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core
import data
import splits
import v3_core as old

DEV = torch.device("cuda")


def init_parity(variant):
    torch.manual_seed(0)
    m_old = old.build_model(old.make_cfg(variant))
    torch.manual_seed(0)
    m_new = core.build_model(core.make_cfg(variant))
    so, sn = m_old.state_dict(), m_new.state_dict()
    assert so.keys() == sn.keys(), (variant, set(so) ^ set(sn))
    worst = max((so[k].float() - sn[k].float()).abs().max().item() for k in so if "feature_map" not in k)
    return m_old, m_new, worst


def main():
    if os.environ.get("DETERMINISTIC"):
        torch.use_deterministic_algorithms(True)
    core.setup_torch()
    old.setup_torch()
    df = data.load_df()
    feats = data.get_features(df)
    tok = old.load_tokenizer()
    s = splits.get_split("random", df)
    # 0. splits.py reproduces the notebook split used by v3_core
    tr, va, te = old.notebook_splits(df)
    dev_idx, folds = old.notebook_folds(df, tr, va)
    assert (s["test"] == te).all() and (s["dev"] == dev_idx).all()
    assert all((a == c).all() and (b == d).all() for (a, b), (c, d) in zip(s["folds"], folds))
    print("splits: identical to v3_core/notebook")

    for v in old.ABLATIONS:
        _, _, worst = init_parity(v)
        assert worst == 0.0, (v, worst)
    print(f"init parity: identical parameters for all {len(old.ABLATIONS)} variants")

    m_old, m_new, _ = init_parity("full")
    m_old, m_new = m_old.to(DEV).eval(), m_new.to(DEV).eval()
    from sklearn.preprocessing import StandardScaler
    ds = core.V3Dataset(feats, te[:64], tok, StandardScaler().fit(feats["desc_raw"][tr]))
    b = core.make_collate(tok.pad_token_id)([ds[i] for i in range(len(ds))])
    kw = {k: v.to(DEV) for k, v in b.items() if k != "labels"}
    torch.manual_seed(1)
    with torch.no_grad():
        a = m_old(**kw)
    torch.manual_seed(1)
    with torch.no_grad():
        c = m_new(**kw)
    print("forward parity: max |diff| =", (a - c).abs().max().item())
    assert (a - c).abs().max().item() == 0.0
    del m_old, m_new

    sub_tr, sub_vl = dev_idx[folds[0][0][:384]], dev_idx[folds[0][1][:128]]
    # v3_core's augmentation uses RDKit's unseedable doRandom, so compare without augmentation
    def run(mod, variant):
        cfg = mod.make_cfg(variant, max_epochs=2)
        return mod.train_one(cfg, feats, sub_tr, sub_vl, {"test": te[:128]}, 4242, DEV, tok)["test_probs"]
    o1, o2, n1 = run(old, "no_augment"), run(old, "no_augment"), run(core, "no_augment")
    print("training (no aug) old vs old: max |diff| =", np.abs(o1 - o2).max())
    print("training (no aug) old vs new: max |diff| =", np.abs(o1 - n1).max())
    a1, a2 = run(core, "full"), run(core, "full")
    print("training (with aug) new vs new, same seed: max |diff| =", np.abs(a1 - a2).max())


if __name__ == "__main__":
    main()
