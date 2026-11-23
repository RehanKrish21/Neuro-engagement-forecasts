"""Run the SnapUGC pipeline on Modal GPUs.

One-time setup (on your machine):

    pip install modal
    modal setup                                   # log in
    modal secret create huggingface HF_TOKEN=hf_xxx   # token with Llama 3.2 access
    modal volume create neuroengage-data

Get the labels and a stratified sample of videos into the volume (prints the
--labels / --videos-dir to use next):

    modal run cloud/modal_app.py::download --split train --n 3000

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
        f"uv pip install --system 'neuroengage @ git+{REPO}' pyarrow gdown accelerate torchaudio==2.6.0",
        # TRIBE transcribes speech by shelling out to `uvx whisperx`; pre-install it into the image.
        "uvx whisperx --help > /dev/null || true",
    )
    .env({"HF_HOME": "/data/hf_cache"})  # model weights downloaded once, kept in the volume
)

app = modal.App("neuroengage", image=image)
volume = modal.Volume.from_name("neuroengage-data", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")


SNAPUGC_FILE_IDS = {"train": "1Mv5Esq5gGuxRTayabRUb5NmHwEN3JdbD", "val": "1iY7qUfqivYzUqUID501dKQDpI3NQil7y"}


@app.function(volumes={"/data": volume}, timeout=6 * 60 * 60)
def download(split: str = "train", n: int = 300, seed: int = 0, oversample: float = 1.3) -> None:
    """Fetch a SnapUGC label CSV and a stratified sample of its videos into the volume.

    SnapUGC's Drive links are label files (Id, Title, Description, Download_link,
    ECR); videos come from each row's Download_link. Re-running resumes.
    """
    from pathlib import Path

    import gdown
    from neuroengage.snapugc import available_targets, download_videos, load_labels, sample_candidates

    dest = Path(f"/data/snapugc/{split}")
    dest.mkdir(parents=True, exist_ok=True)
    found = sorted(dest.glob("*.csv"))
    labels_csv = found[0] if found else Path(
        gdown.download(id=SNAPUGC_FILE_IDS[split], output=f"{dest}/", quiet=False)
    )
    labels = load_labels(labels_csv)
    print(f"{len(labels):,} labelled videos; targets: {available_targets(labels)}")
    log = download_videos(sample_candidates(labels, int(n * oversample), seed=seed), dest / "videos")
    volume.commit()
    print(log["status"].value_counts().to_string())
    print(f"--labels {labels_csv} --videos-dir {dest / 'videos'}")


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
