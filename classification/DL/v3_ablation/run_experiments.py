"""
Parallel, resumable, thermally-guarded runner for the V3 reproduction + ablation.

Jobs
  single/<variant>/seed : notebook cell 15 protocol  (train_df -> early stop on val_df, 60 ep, pat 12)
  cv/<variant>/seed/fold: notebook cell 17 protocol  (5-fold on train+val, 50 ep, pat 12), each fold
                          model also scores the held-out test set -> ensemble in summarize.py

Several worker processes share the GPU (the model is small, so one process leaves
most of the 3090 idle). Every worker pauses when the GPU reaches --hot C and
resumes at --cool C; a monitor thread logs temperature / power / utilisation.

Usage
  python run_experiments.py --plan repro            # reproduction only
  python run_experiments.py --plan all --workers 4  # reproduction + ablation
  python run_experiments.py --plan bench            # 2-epoch timing run
Results land in results/jobs/*.json|npz; finished jobs are skipped on restart.
"""
import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
JOBS_DIR = os.path.join(HERE, "results", "jobs")
LOG_DIR = os.path.join(HERE, "logs")

REPRO_SEEDS = [42, 123, 2024]
ABL_SEEDS = [42]
# most informative ablations first, so partial results are already useful
ABL_ORDER = ["no_smiles", "smiles_only", "concat_fusion", "frozen_bert", "no_ecfp", "no_maccs", "no_desc",
             "no_type_emb", "no_gate", "bce_loss", "no_augment", "no_sampler", "last_layer_pool",
             "cls_pool", "no_llrd"]


def job_id(j):
    if j["kind"] == "single":
        return f"single__{j['variant']}__s{j['seed']}"
    return f"cv__{j['variant']}__s{j['seed']}__f{j['fold']}"


def build_plan(plan, bench_n=1):
    import v3_core as C
    jobs = []
    if plan == "bench":
        return [dict(kind="cv", variant="full", seed=i, fold=0, max_epochs=2, bench=True) for i in range(bench_n)]
    # 1) reproduction: full model, 3 seeds, both protocols
    for s in REPRO_SEEDS:
        for f in range(5):
            jobs.append(dict(kind="cv", variant="full", seed=s, fold=f))
    for s in REPRO_SEEDS:
        jobs.append(dict(kind="single", variant="full", seed=s))
    if plan == "all":
        # 2) ablations: 5-fold CV; the seed-42 'full' run is shared with the reproduction,
        #    the 3 'full' seeds give the seed-to-seed noise floor
        assert set(ABL_ORDER) == set(C.ABLATIONS) - {"full"}
        for s in ABL_SEEDS:
            for v in ABL_ORDER:
                for f in range(5):
                    jobs.append(dict(kind="cv", variant=v, seed=s, fold=f))
    return jobs


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────
_W = {}


def _worker_init(threads, hot, cool, log_path, mem_frac):
    import torch
    import v3_core as C
    torch.set_num_threads(threads)
    # Hard VRAM cap per worker: on Windows (WDDM) exceeding 24 GB silently spills
    # into system RAM and everything crawls, so never let the workers get there.
    torch.cuda.set_per_process_memory_fraction(mem_frac)
    C.setup_torch()
    df = C.load_df()
    _W["C"] = C
    _W["feats"] = C.get_features(df)
    _W["tok"] = C.load_tokenizer()
    tr, va, te = C.notebook_splits(df)
    dev_idx, folds = C.notebook_folds(df, tr, va)
    _W.update(tr=tr, va=va, te=te, dev_idx=dev_idx, folds=folds)
    _W["device"] = torch.device("cuda")
    _W["log_path"] = log_path

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} [pid {os.getpid()}] {msg}"
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    _W["log"] = log
    _W["guard"] = C.ThermalGuard(hot_c=hot, cool_c=cool, every_s=20, log=log)


def _run_job(j):
    import numpy as np
    C = _W["C"]
    jid = job_id(j)
    log = _W["log"]
    try:
        extra = {"max_epochs": j["max_epochs"]} if "max_epochs" in j else {}
        if j["kind"] == "single":
            extra.setdefault("max_epochs", 60)
            cfg = C.make_cfg(j["variant"], **extra)
            tr_idx, vl_idx = _W["tr"], _W["va"]
        else:
            extra.setdefault("max_epochs", 50)
            cfg = C.make_cfg(j["variant"], **extra)
            f_tr, f_vl = _W["folds"][j["fold"]]
            tr_idx, vl_idx = _W["dev_idx"][f_tr], _W["dev_idx"][f_vl]
        seed = j["seed"] * 100 + j.get("fold", 0)
        log(f"START {jid}")
        out = C.train_one(cfg, _W["feats"], tr_idx, vl_idx, {"test": _W["te"]}, seed,
                          _W["device"], _W["tok"], log=log, guard=_W["guard"])
        meta = {k: out[k] for k in ("best_val_auc", "best_epoch", "epochs_run", "train_seconds",
                                    "history", "n_trainable")}
        meta.update(job=j, id=jid, val_idx=[int(x) for x in vl_idx])
        if j.get("bench"):
            log(f"BENCH {jid}: {out['train_seconds']:.1f}s for {out['epochs_run']} epochs")
            return jid, meta
        np.savez(os.path.join(JOBS_DIR, jid + ".npz"), val_probs=out["val_probs"], val_labels=out["val_labels"],
                 test_probs=out["test_probs"], test_labels=out["test_labels"], val_idx=vl_idx)
        with open(os.path.join(JOBS_DIR, jid + ".json"), "w") as fh:
            json.dump(meta, fh)
        log(f"DONE  {jid}  valAUC={out['best_val_auc']:.4f} ep={out['best_epoch']}/{out['epochs_run']} "
            f"{out['train_seconds']/60:.1f}min")
        return jid, meta
    except Exception:
        log(f"FAIL  {jid}\n{traceback.format_exc()}")
        return jid, None


# ─────────────────────────────────────────────────────────────────────────────
# Monitor
# ─────────────────────────────────────────────────────────────────────────────
def monitor(stop, path, every=60):
    while not stop.is_set():
        try:
            q = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used,fan.speed",
                                "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
            with open(path, "a") as fh:
                fh.write(f"{time.strftime('%H:%M:%S')}, {q}\n")
        except Exception:
            pass
        stop.wait(every)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="all", choices=["bench", "repro", "all"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--bench-n", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2, help="CPU threads per worker")
    ap.add_argument("--hot", type=int, default=78, help="pause training at this GPU temp (C)")
    ap.add_argument("--cool", type=int, default=70, help="resume at this GPU temp (C)")
    args = ap.parse_args()

    os.makedirs(JOBS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    import v3_core as C
    C.get_features()  # build the feature cache once, before workers start

    jobs = build_plan(args.plan, args.bench_n)
    todo = [j for j in jobs if j.get("bench") or not os.path.exists(os.path.join(JOBS_DIR, job_id(j) + ".json"))]
    log_path = os.path.join(LOG_DIR, "run.log")
    print(f"{len(jobs)} jobs in plan, {len(todo)} to run, {args.workers} workers, "
          f"thermal pause at {args.hot}C / resume {args.cool}C", flush=True)

    stop = threading.Event()
    th = threading.Thread(target=monitor, args=(stop, os.path.join(LOG_DIR, "gpu.csv")), daemon=True)
    th.start()
    t0 = time.time()
    done = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_worker_init,
                  initargs=(args.threads, args.hot, args.cool, log_path, min(0.25, 0.85 / args.workers)), maxtasksperchild=8) as pool:
        for jid, meta in pool.imap_unordered(_run_job, todo):
            done += 1
            el = (time.time() - t0) / 60
            status = "ok" if meta else "FAILED"
            extra = f" valAUC={meta['best_val_auc']:.4f} ({meta['train_seconds']/60:.1f} min)" if meta else ""
            print(f"[{done}/{len(todo)}] {jid} {status}{extra} | elapsed {el:.1f} min, "
                  f"ETA {el / done * (len(todo) - done):.0f} min", flush=True)
    stop.set()
    print(f"All done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
