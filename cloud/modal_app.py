"""Run the SnapUGC pipeline on Modal GPUs.

One-time setup (on your machine):

    pip install modal
    modal setup                                   # log in
    modal secret create huggingface HF_TOKEN=hf_xxx   # token with Llama 3.2 access
    modal volume create neuroengage-data

Get the data into the volume (prints the label CSV and video folder it finds):

    modal run cloud/modal_app.py::download --split train

Pilot (50 videos, 1 GPU), then the real run, with the paths printed above:

    modal run cloud/modal_app.py --labels /data/snapugc/train/<labels>.csv \\
        --videos-dir /data/snapugc/train/<videos> --n 50 --shards 1
    modal run cloud/modal_app.py --labels ... --videos-dir ... --n 3000 --shards 8

Results land in the volume under /data/runs/<run>/; fetch them with

    modal volume get neuroengage-data /runs/<run>/results ./results

Everything is resumable: re-running the same command skips videos whose
predictions are already cached in the volume.
"""

from __future__ import annotations

import modal

GPU = "L40S"  # 48 GB. "A100-40GB" / "A100-80GB" / "H100" also work; "T4" will likely run out of memory.
REPO = "https://github.com/RehanKrish21/Neuro-engagement-forecasts.git"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system 'tribev2[plotting] @ git+https://github.com/facebookresearch/tribev2.git'",
        f"uv pip install --system 'neuroengage @ git+{REPO}' pyarrow gdown",
        # TRIBE transcribes speech by shelling out to `uvx whisperx`; pre-install it into the image.
        "uvx whisperx --help > /dev/null || true",
    )
    .env({"HF_HOME": "/data/hf_cache"})  # model weights downloaded once, kept in the volume
)

app = modal.App("neuroengage", image=image)
volume = modal.Volume.from_name("neuroengage-data", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")


SNAPUGC_FILE_IDS = {"train": "1Mv5Esq5gGuxRTayabRUb5NmHwEN3JdbD", "val": "1iY7qUfqivYzUqUID501dKQDpI3NQil7y"}


@app.function(volumes={"/data": volume}, timeout=6 * 60 * 60, ephemeral_disk=800 * 1024)
def download(split: str = "train") -> None:
    """Fetch a SnapUGC split from Google Drive straight into the volume and unpack it."""
    import glob
    import os
    import shutil
    from collections import Counter
    from pathlib import Path

    import gdown

    dest = Path(f"/data/snapugc/{split}")
    dest.mkdir(parents=True, exist_ok=True)
    if not any(dest.rglob("*.mp4")):
        tmp = Path("/tmp/dl")
        tmp.mkdir(exist_ok=True)
        archive = gdown.download(id=SNAPUGC_FILE_IDS[split], output=f"{tmp}/", quiet=False)
        shutil.unpack_archive(archive, str(dest))
        volume.commit()
    csvs = sorted(glob.glob(f"{dest}/**/*.csv", recursive=True))
    dirs = Counter(os.path.dirname(p) for p in glob.glob(f"{dest}/**/*.mp4", recursive=True))
    print("label files:", csvs)
    print("video folders:", dirs.most_common(3))
    print("Pass these as --labels and --videos-dir to `modal run cloud/modal_app.py`.")


@app.function(volumes={"/data": volume}, timeout=60 * 60)
def prepare(labels: str, videos_dir: str, n: int, run_dir: str, seed: int = 0) -> int:
    from neuroengage.snapugc import build_manifest, load_labels
    from pathlib import Path

    Path(run_dir).mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(load_labels(labels), videos_dir, n=n or None, seed=seed)
    manifest.to_csv(f"{run_dir}/manifest.csv", index=False)
    volume.commit()
    return len(manifest)


@app.function(gpu=GPU, volumes={"/data": volume}, secrets=[hf_secret], timeout=24 * 60 * 60)
def extract_shard(run_dir: str, shard_index: int, num_shards: int) -> str:
    from neuroengage.cli import main

    volume.reload()
    main([
        "extract", "--manifest", f"{run_dir}/manifest.csv", "--out", f"{run_dir}/work",
        "--cache", "/tmp/tribe_cache",  # bulky per-video features stay off the volume
        "--shard-index", str(shard_index), "--num-shards", str(num_shards),
    ])
    volume.commit()
    return f"shard {shard_index} done"


@app.function(volumes={"/data": volume}, timeout=2 * 60 * 60)
def train(run_dir: str, num_shards: int) -> str:
    from pathlib import Path

    from neuroengage.cli import main

    volume.reload()
    if num_shards > 1:
        main(["merge-shards", "--out", f"{run_dir}/work"])
    main([
        "train-scores", "--manifest", f"{run_dir}/manifest.csv",
        "--timeseries", f"{run_dir}/work/network_timeseries",
        "--targets", "nawp,ecr",
        "--baseline-cols", "duration_s,title_n_words,has_description",
        "--title", "SnapUGC: TRIBE v2 brain features vs baselines",
        "--out", f"{run_dir}/results",
    ])
    volume.commit()
    return Path(f"{run_dir}/results/report.md").read_text()


@app.local_entrypoint()
def main(
    labels: str = "",
    videos_dir: str = "",
    n: int = 50,
    shards: int = 1,
    run: str = "",
    seed: int = 0,
):
    if not labels or not videos_dir:
        raise SystemExit("Pass --labels and --videos-dir (run `modal run cloud/modal_app.py::download` first)")
    run_dir = f"/data/runs/{run or f'snapugc_n{n}_seed{seed}'}"
    count = prepare.remote(labels, videos_dir, n, run_dir, seed)
    print(f"Manifest: {count} videos -> {run_dir}/manifest.csv")
    for msg in extract_shard.starmap([(run_dir, k, shards) for k in range(shards)]):
        print(msg)
    print(train.remote(run_dir, shards))
