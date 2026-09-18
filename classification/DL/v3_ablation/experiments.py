"""
Single source of truth for every training job.

A job is fully described by (protocol, split, backbone, hp_set, variant, seed, fold).
Jobs are stored once in results/store/<job_id>.{json,npz}; an experiment is just a
named list of jobs, so a job shared by two experiments (e.g. the seed-42 full model
is both the reproduction and the ablation reference) is trained only once.

Protocols (identical to the notebook):
  cv     : cell 17 - train on 4/5 of dev, early-stop on the 5th (50 ep, patience 12);
           each fold model also predicts the test set -> 5-model ensemble (cell 18)
  single : cells 15-16 - train on train, early-stop on val (60 ep, patience 12)
"""
import hashlib
import json

import core

REPRO_SEEDS = [42, 123, 2024]
ABLATION_ORDER = ["no_smiles", "smiles_only", "concat_fusion", "frozen_bert", "no_ecfp", "no_maccs",
                  "no_desc", "no_type_emb", "no_gate", "bce_loss", "no_augment", "no_sampler",
                  "last_layer_pool", "cls_pool", "no_llrd"]
NEW_BACKBONES = ["chemberta_mlm", "chemberta_mtr", "chemberta_zinc"]
MAX_EPOCHS = {"cv": 50, "single": 60}


def job(protocol, split="random", backbone="molformer", hp_set="tuned", variant="full", seed=42, fold=None):
    return dict(protocol=protocol, split=split, backbone=backbone, hp_set=hp_set,
                variant=variant, seed=seed, fold=fold)


def cv_jobs(seeds, **kw):
    return [job("cv", seed=s, fold=f, **kw) for s in seeds for f in range(5)]


def job_id(j):
    base = f"{j['protocol']}__{j['split']}__{j['backbone']}__{j['hp_set']}__{j['variant']}__s{j['seed']}"
    return base + (f"__f{j['fold']}" if j["protocol"] == "cv" else "")


def resolve(j):
    """RunCfg for a job + a hash of everything that determines its result."""
    cfg = core.make_cfg(j["variant"], backbone=j["backbone"], hp_set=j["hp_set"],
                        max_epochs=MAX_EPOCHS[j["protocol"]])
    blob = json.dumps({"job": j, "cfg": cfg.to_dict()}, sort_keys=True)
    return cfg, hashlib.sha1(blob.encode()).hexdigest()[:12]


def train_seed(j):
    return j["seed"] * 100 + (j["fold"] or 0)


EXPERIMENTS = {
    # ── already run with v3_core.py (migrated into the store) ────────────────
    "repro": cv_jobs(REPRO_SEEDS) + [job("single", seed=s) for s in REPRO_SEEDS],
    "ablation": cv_jobs([42]) + [j for v in ABLATION_ORDER for j in cv_jobs([42], variant=v)],
    # ── new ──────────────────────────────────────────────────────────────────
    "scaffold": (cv_jobs(REPRO_SEEDS, split="scaffold")
                 + cv_jobs(REPRO_SEEDS, split="scaffold", variant="no_smiles")
                 + cv_jobs([42], split="scaffold", variant="smiles_only")),
    "unbalanced": (cv_jobs(REPRO_SEEDS, variant="no_sampler")
                   + cv_jobs(REPRO_SEEDS, variant="vanilla")),
    "untuned": cv_jobs(REPRO_SEEDS, hp_set="untuned"),
    "backbones": [j for b in NEW_BACKBONES for j in cv_jobs(REPRO_SEEDS, backbone=b)],
}

# order in which the queue runs the new experiments
QUEUE = ["scaffold", "unbalanced", "untuned", "backbones"]


def all_jobs(names):
    seen, out = set(), []
    for n in names:
        for j in EXPERIMENTS[n]:
            jid = job_id(j)
            if jid not in seen:
                seen.add(jid)
                out.append(j)
    return out


if __name__ == "__main__":
    for n, js in EXPERIMENTS.items():
        print(f"{n:<11} {len(js):>3} jobs")
    print("queue (unique):", len(all_jobs(QUEUE)))
