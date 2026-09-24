"""
Parallel, resumable, thermally guarded job runner.

  python runner.py --exp scaffold unbalanced untuned backbones   # run in this order
  python runner.py --queue                                      # experiments.QUEUE
  python runner.py --smoke                                      # 1 short job per distinct config
  python runner.py --status                                     # what is done / missing

A job is skipped when results/store/<job_id>.json exists with the same config hash.
If the hash differs (the job's definition changed) the runner stops and says so,
instead of silently mixing results from different configurations.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
STORE = os.path.join(HERE, "results", "store")
LOGS = os.path.join(HERE, "logs")
_W = {}


def _worker_init(threads, hot, cool, log_path, mem_frac):
    import torch
    torch.set_num_threads(threads)
    torch.cuda.set_per_process_memory_fraction(mem_frac)   # never spill past 24 GB (WDDM)
    import core
    import data
    import gpu
    core.setup_torch()
    df = data.load_df()
    _W.update(core=core, feats=data.get_features(df), df=df, tok={}, splits={}, log_path=log_path,
              device=torch.device("cuda"))

    def log(msg):
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [pid {os.getpid()}] {msg}\n")
    _W["log"] = log
    _W["guard"] = gpu.ThermalGuard(hot_c=hot, cool_c=cool, every_s=20, log=log)


def _get(kind, name):
    import backbones
    import splits
    cache = _W[kind]
    if name not in cache:
        cache[name] = backbones.load_tokenizer(name) if kind == "tok" else splits.get_split(name, _W["df"])
    return cache[name]


def _run_job(args):
    import numpy as np
    import experiments as E
    j, smoke = args
    jid = E.job_id(j)
    log = _W["log"]
    try:
        cfg, h = E.resolve(j)
        s = _get("splits", j["split"])
        if j["protocol"] == "cv":
            f_tr, f_vl = s["folds"][j["fold"]]
            tr_idx, vl_idx = s["dev"][f_tr], s["dev"][f_vl]
        else:
            tr_idx, vl_idx = s["train"], s["val"]
        te_idx = s["test"]
        if smoke:
            cfg.max_epochs = 1
            tr_idx, vl_idx, te_idx = tr_idx[:512], vl_idx[:256], te_idx[:128]
        log(f"START {jid}{' (smoke)' if smoke else ''}")
        out = _W["core"].train_one(cfg, _W["feats"], tr_idx, vl_idx, {"test": te_idx}, E.train_seed(j),
                                   _W["device"], _get("tok", j["backbone"]), log=log, guard=_W["guard"],
                                   return_pic50=True)
        meta = {k: out[k] for k in ("best_val_auc", "best_epoch", "epochs_run", "train_seconds",
                                    "history", "n_trainable")}
        meta.update(id=jid, job=j, cfg=cfg.to_dict(), cfg_hash=h, code="core.py",
                    val_idx=[int(x) for x in vl_idx], test_idx=[int(x) for x in te_idx],
                    finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        if smoke:
            log(f"SMOKE {jid} ok  valAUC={out['best_val_auc']:.3f}  {out['train_seconds']:.0f}s  "
                f"trainable={out['n_trainable']/1e6:.1f}M")
            return jid, meta
        arrays = dict(val_probs=out["val_probs"], val_labels=out["val_labels"],
                      test_probs=out["test_probs"], test_labels=out["test_labels"],
                      val_idx=vl_idx, test_idx=te_idx)
        for k in ("val_pic50", "test_pic50"):     # present whenever the model has a potency head
            if k in out:
                arrays[k] = out[k]
        np.savez(os.path.join(STORE, jid + ".npz"), **arrays)
        tmp = os.path.join(STORE, jid + ".json.tmp")
        with open(tmp, "w") as fh:
            json.dump(meta, fh)
        os.replace(tmp, os.path.join(STORE, jid + ".json"))   # json last = job complete
        log(f"DONE  {jid}  valAUC={out['best_val_auc']:.4f} ep={out['best_epoch']}/{out['epochs_run']} "
            f"{out['train_seconds']/60:.1f}min")
        return jid, meta
    except Exception:
        log(f"FAIL  {jid}\n{traceback.format_exc()}")
        return jid, None


def status(jobs):
    import experiments as E
    done, stale, todo = [], [], []
    for j in jobs:
        path = os.path.join(STORE, E.job_id(j) + ".json")
        if os.path.exists(path):
            h = json.load(open(path)).get("cfg_hash")
            (done if h == E.resolve(j)[1] else stale).append(j)
        else:
            todo.append(j)
    return done, stale, todo


def main():
    import experiments as E
    import gpu
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--exp", nargs="+", choices=list(E.EXPERIMENTS))
    g.add_argument("--queue", action="store_true")
    g.add_argument("--smoke", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--hot", type=int, default=78)
    ap.add_argument("--cool", type=int, default=70)
    args = ap.parse_args()

    os.makedirs(STORE, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    import data
    data.get_features()
    import splits
    for n in ("random", "scaffold"):
        splits.get_split(n)

    if args.smoke:
        seen, jobs = set(), []
        for j in E.all_jobs(E.QUEUE):
            key = (j["split"], j["backbone"], j["hp_set"], j["variant"], j["protocol"])
            if key not in seen:
                seen.add(key)
                jobs.append(j)
        todo = [(j, True) for j in jobs]
    else:
        names = E.QUEUE if args.queue else args.exp
        jobs = E.all_jobs(names)
        done, stale, todo_j = status(jobs)
        print(f"{names}: {len(jobs)} jobs | done {len(done)} | stale {len(stale)} | to run {len(todo_j)}")
        if stale:
            print("STALE results (job definition changed since they were produced):")
            for j in stale:
                print("   ", E.job_id(j))
            print("Delete them deliberately or revert the change; refusing to mix configurations.")
            sys.exit(1)
        if args.status:
            for j in todo_j:
                print("  todo", E.job_id(j))
            return
        todo = [(j, False) for j in todo_j]
    if not todo:
        return

    log_path = os.path.join(LOGS, "smoke.log" if args.smoke else "runner.log")
    stop = threading.Event()
    threading.Thread(target=gpu.monitor, args=(stop, os.path.join(LOGS, "gpu.csv")), daemon=True).start()
    t0, n_done, n_fail = time.time(), 0, 0
    frac = min(0.25, 0.85 / args.workers)
    with mp.get_context("spawn").Pool(args.workers, initializer=_worker_init,
                                      initargs=(args.threads, args.hot, args.cool, log_path, frac),
                                      maxtasksperchild=8) as pool:
        for jid, meta in pool.imap_unordered(_run_job, todo):
            n_done += 1
            n_fail += meta is None
            el = (time.time() - t0) / 60
            print(f"[{n_done}/{len(todo)}] {jid} {'ok' if meta else 'FAILED'}"
                  + (f" valAUC={meta['best_val_auc']:.4f}" if meta else "")
                  + f" | {el:.0f} min elapsed, ETA {el / n_done * (len(todo) - n_done):.0f} min", flush=True)
    stop.set()
    print(f"finished in {(time.time() - t0) / 60:.1f} min, {n_fail} failed", flush=True)


if __name__ == "__main__":
    main()
