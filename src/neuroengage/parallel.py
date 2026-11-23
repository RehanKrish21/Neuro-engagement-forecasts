"""Run TRIBE extraction on several GPUs of one machine (e.g. Kaggle's T4 x2).

One ``neuroengage extract`` process per GPU, each pinned with
``CUDA_VISIBLE_DEVICES`` and given every k-th video of the manifest. All
processes write to the same ``preds/`` folder (one file per video, so they
never collide) but use separate TRIBE feature caches, because the cache's
shared array files are not safe for concurrent writers. Finished videos are
skipped, so re-running resumes.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from .tribe_runner import merge_shards

logger = logging.getLogger(__name__)


def _done(preds_dir: Path, ids: set[str]) -> int:
    return sum(1 for p in preds_dir.glob("*.npz") if p.stem in ids)


def _tail(path: Path, n: int = 25) -> str:
    try:
        return "".join(path.read_text(errors="replace").splitlines(keepends=True)[-n:])
    except OSError:
        return ""


def run_extract_on_gpus(
    manifest_path: str | Path,
    out_dir: str | Path,
    gpus: list[int] | range,
    cache_folder: str | Path,
    precision: str = "fp32",
    max_hours: float | None = None,
    poll_seconds: float = 60.0,
    command: list[str] | None = None,
) -> Path:
    """Extract all videos in ``manifest_path`` using one process per GPU.

    Prints progress (videos done, rate, ETA) every ``poll_seconds``; shard logs
    go to ``<out_dir>/logs/shard<k>.log``. Returns the path of the merged
    ``network_timeseries`` table. ``command`` replaces the
    ``python -m neuroengage.cli extract`` prefix (used in tests).
    """
    out_dir = Path(out_dir)
    preds_dir = out_dir / "preds"
    logs = out_dir / "logs"
    preds_dir.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    ids = set(pd.read_csv(manifest_path, dtype={"video_id": str})["video_id"])
    gpus = list(gpus)
    n = len(gpus)
    if n == 0:
        raise ValueError("No GPUs given")
    prefix = command or [sys.executable, "-m", "neuroengage.cli", "extract"]

    procs = []
    for k, gpu in enumerate(gpus):
        argv = [*prefix, "--manifest", str(manifest_path), "--out", str(out_dir),
                "--cache", str(Path(cache_folder) / f"gpu{gpu}"), "--precision", precision,
                "--shard-index", str(k), "--num-shards", str(n)]
        if max_hours:
            argv += ["--max-hours", f"{max_hours:.3f}"]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"}
        log = open(logs / f"shard{k}.log", "w")
        procs.append((subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env), log))
    print(f"Started {n} extraction process(es) on GPU(s) {gpus}; logs in {logs}", flush=True)

    start_done, t0 = _done(preds_dir, ids), time.time()
    try:
        while any(p.poll() is None for p, _ in procs):
            time.sleep(poll_seconds)
            done = _done(preds_dir, ids)
            new, elapsed = done - start_done, time.time() - t0
            msg = f"[{elapsed / 60:5.0f} min] {done}/{len(ids)} videos done"
            if new > 0:
                eta_h = (len(ids) - done) * elapsed / new / 3600
                msg += f" ({elapsed / new:.0f} s/video overall, ~{eta_h:.1f} h left)"
            print(msg, flush=True)
    finally:
        for p, log in procs:
            if p.poll() is None:
                p.terminate()
            log.close()

    for k, (p, _) in enumerate(procs):
        if p.returncode != 0:
            print(f"⚠️  shard {k} exited with code {p.returncode}. Last log lines:\n"
                  f"{_tail(logs / f'shard{k}.log')}", flush=True)
    print(f"✅ {_done(preds_dir, ids)}/{len(ids)} videos have predictions", flush=True)

    if n == 1:
        path = next((p for p in (out_dir / "network_timeseries.parquet",
                                 out_dir / "network_timeseries.csv") if p.exists()), None)
        if path is None:
            raise RuntimeError(f"Extraction produced no table; see {logs / 'shard0.log'}")
        return path
    return merge_shards(out_dir)
