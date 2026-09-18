"""
Copies results produced by the legacy runner (run_experiments.py + v3_core.py,
results/jobs/) into the new store (results/store/) under the new job ids, with the
resolved config + hash, so experiments.py/report.py treat them like any other job.

Safe to re-run (idempotent); only complete legacy jobs (json present) are copied.
tests/test_core_parity.py shows the two code paths are bit-identical apart from
the (then unseedable) RDKit augmentation, which is recorded in the metadata.
"""
import glob
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import experiments as E
import splits

HERE = os.path.dirname(os.path.abspath(__file__))
LEGACY = os.path.join(HERE, "results", "jobs")
STORE = os.path.join(HERE, "results", "store")


def main():
    os.makedirs(STORE, exist_ok=True)
    test_idx = splits.get_split("random")["test"]
    n_new = 0
    for path in sorted(glob.glob(os.path.join(LEGACY, "*.json"))):
        meta = json.load(open(path))
        lj = meta["job"]
        j = E.job(lj["kind"], variant=lj["variant"], seed=lj["seed"],
                  fold=lj.get("fold") if lj["kind"] == "cv" else None)
        jid = E.job_id(j)
        dst = os.path.join(STORE, jid + ".json")
        if os.path.exists(dst):
            continue
        cfg, h = E.resolve(j)
        z = np.load(path.replace(".json", ".npz"))
        assert len(z["test_probs"]) == len(test_idx)
        np.savez(os.path.join(STORE, jid + ".npz"), val_probs=z["val_probs"], val_labels=z["val_labels"],
                 test_probs=z["test_probs"], test_labels=z["test_labels"], val_idx=z["val_idx"],
                 test_idx=test_idx)
        new = {k: meta[k] for k in ("best_val_auc", "best_epoch", "epochs_run", "train_seconds",
                                    "history", "n_trainable", "val_idx")}
        new.update(id=jid, job=j, cfg=cfg.to_dict(), cfg_hash=h, test_idx=[int(x) for x in test_idx],
                   code="v3_core.py (legacy runner; RDKit augmentation not seedable)", legacy_id=meta["id"])
        with open(dst, "w") as fh:
            json.dump(new, fh)
        n_new += 1
    print(f"migrated {n_new} new legacy jobs -> results/store ({len(glob.glob(os.path.join(STORE, '*.json')))} total)")


if __name__ == "__main__":
    main()
