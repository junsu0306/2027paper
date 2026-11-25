"""Claim-based multi-GPU job runner.

An experiment module exposes `jobs() -> list[dict]` (each with a unique "id" and an optional "priority", lower first)
and `run(job, device) -> list[dict]` (result rows). Workers claim jobs with O_EXCL files and write
<run>/results/<id>.jsonl atomically, so a job is done iff its result file exists; errors go to <id>.error.json.

  python -m qcbobs.runner work   --exp experiments/E01_three_places --run runs/E01 --need-mib 3000
  python -m qcbobs.runner status --exp ... --run ...
  python -m qcbobs.runner unclaim-dead --run ...     (drop claims whose worker process is gone)

GPU courtesy (MVD AGENTS.md): before every claim the worker waits until the GPU has need-mib + 1,024 MiB free.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_exp(path: str):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    spec = importlib.util.spec_from_file_location(f"exp_{p.name}", p / "exp.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(p))
    spec.loader.exec_module(mod)
    return mod


def _safe(i: str) -> str:
    return i.replace("/", "_").replace(":", "_")


def free_mib() -> int:
    import torch
    f, _ = torch.cuda.mem_get_info()
    return f // (1 << 20)


def work(args):
    import torch
    sys.path.insert(0, str(ROOT / "src"))
    from qcbobs import env
    env.setup()
    env.backend()
    exp = load_exp(args.exp)
    run = Path(args.run)
    (run / "results").mkdir(parents=True, exist_ok=True)
    (run / "claims").mkdir(parents=True, exist_ok=True)
    jobs = sorted(exp.jobs(), key=lambda j: (j.get("priority", 0), j["id"]))
    (run / "jobs.json").write_text(json.dumps(jobs, indent=0)) if not (run / "jobs.json").exists() else None
    dev = torch.device("cuda")
    prov = env.provenance()
    while True:
        if (run / "STOP").exists():
            print("STOP file found", flush=True)
            return
        todo = [j for j in jobs if not (run / "results" / f"{_safe(j['id'])}.jsonl").exists()
                and not (run / "claims" / _safe(j["id"])).exists()]
        if not todo:
            print("no unclaimed jobs left", flush=True)
            return
        while free_mib() < args.need_mib + 1024:
            print(f"waiting for memory: {free_mib()} MiB free", flush=True)
            time.sleep(60)
        claimed = None
        for j in todo:
            try:
                fd = os.open(run / "claims" / _safe(j["id"]), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            os.write(fd, json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "t": time.time(),
                                     "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}).encode())
            os.close(fd)
            claimed = j
            break
        if claimed is None:
            continue
        t0 = time.time()
        sid = _safe(claimed["id"])
        try:
            rows = exp.run(claimed, dev)
            for r in rows:
                r.setdefault("job", claimed["id"])
                r["prov"] = prov
            tmp = run / "results" / f".{sid}.tmp"
            with tmp.open("w") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            tmp.rename(run / "results" / f"{sid}.jsonl")
            print(f"done {claimed['id']} rows {len(rows)} {time.time() - t0:.0f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            (run / "results" / f"{sid}.error.json").write_text(json.dumps(
                {"job": claimed, "error": repr(e)[:500], "tb": traceback.format_exc()[-3000:], "sec": time.time() - t0}))
            print(f"ERROR {claimed['id']}: {e!r}", flush=True)
        finally:
            torch.cuda.empty_cache()


def status(args):
    run = Path(args.run)
    exp = load_exp(args.exp)
    jobs = exp.jobs()
    res = {p.name[:-6] for p in (run / "results").glob("*.jsonl")} if (run / "results").exists() else set()
    err = {p.name[:-11] for p in (run / "results").glob("*.error.json")} if (run / "results").exists() else set()
    cl = {p.name for p in (run / "claims").glob("*")} if (run / "claims").exists() else set()
    ids = [_safe(j["id"]) for j in jobs]
    done = sum(i in res for i in ids)
    print(f"jobs {len(ids)} done {done} errors {len(err)} running {len([i for i in ids if i in cl and i not in res and i not in err])}"
          f" waiting {len([i for i in ids if i not in cl])}")
    for e in sorted(err)[:10]:
        print(" error:", e, json.loads((run / 'results' / f'{e}.error.json').read_text())["error"][:200])


def unclaim_dead(args):
    run = Path(args.run)
    n = 0
    for c in (run / "claims").glob("*"):
        if (run / "results" / f"{c.name}.jsonl").exists() or (run / "results" / f"{c.name}.error.json").exists():
            continue
        pid = json.loads(c.read_text()).get("pid")
        if pid and not Path(f"/proc/{pid}").exists():
            c.unlink()
            n += 1
    print("released", n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["work", "status", "unclaim-dead"])
    ap.add_argument("--exp")
    ap.add_argument("--run", required=True)
    ap.add_argument("--need-mib", type=int, default=3000)
    a = ap.parse_args()
    {"work": work, "status": status, "unclaim-dead": unclaim_dead}[a.cmd](a)


if __name__ == "__main__":
    main()
