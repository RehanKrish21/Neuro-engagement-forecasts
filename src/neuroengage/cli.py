"""Command line: ``neuroengage {demo,extract,audio-extract,merge-shards,train,train-scores,snapugc-download,snapugc-prepare,score}``."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import joblib
import pandas as pd

from .features import second_features, video_features
from .retention import resample_to_seconds
from .tribe_runner import load_table


SYNTHETIC_NOTE = (
    "> **Synthetic data.** These numbers show the pipeline recovers a signal "
    "planted by `neuroengage.synthetic`. They are not evidence that TRIBE v2 "
    "predicts real engagement."
)


def _demo(args) -> None:
    from .pipeline import run_score_training, run_training
    from .synthetic import make_dataset, make_score_dataset

    kwargs = {"beta_engagement": 0.0, "beta_aversion": 0.0} if args.null else {}
    kind = "NULL synthetic data (no planted signal)" if args.null else "synthetic data"
    if args.snapugc:
        manifest, ts = make_score_dataset(seed=args.seed, **kwargs)
        run_score_training(
            manifest, ts, ["nawp", "ecr"], args.out, baseline_cols=["duration_s"],
            hook_seconds=5, seed=args.seed,
            title=f"SnapUGC-style demo on {kind}", note=SYNTHETIC_NOTE,
        )
    else:
        manifest, ts, ret = make_dataset(seed=args.seed, **kwargs)
        run_training(
            manifest, ts, ret, args.out,
            baseline_cols=["duration_s", "log_followers", "post_hour"],
            seed=args.seed, title=f"Pipeline demo on {kind}", note=SYNTHETIC_NOTE,
        )
    print(f"Wrote {args.out}/report.md")


def _snapugc_prepare(args) -> None:
    from .snapugc import build_manifest, done_ids, load_labels

    overrides = {k: v for k, v in (("nawp", args.nawp_col), ("ecr", args.ecr_col)) if v}
    labels = load_labels(args.labels, id_col=args.id_col or None, target_cols=overrides)
    manifest = build_manifest(labels, args.videos_dir, n=args.n, seed=args.seed,
                              include_ids=done_ids(args.done_dir))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out, index=False)
    print(f"Wrote {len(manifest)} videos to {args.out}")


def _snapugc_download(args) -> None:
    from .snapugc import done_ids, download_videos, load_labels, sample_candidates

    Path(args.videos_dir).mkdir(parents=True, exist_ok=True)
    labels = load_labels(args.labels, id_col=args.id_col or None)
    done = done_ids(args.done_dir) & set(labels["video_id"])
    need = max(args.n - len(done), 0)
    if done:
        print(f"{len(done)} videos already have predictions; downloading candidates for {need} more")
    if args.redownload_done and done:
        # videos processed in earlier runs: fetch the files again (e.g. for audio features)
        redo = labels[labels["video_id"].isin(done)]
        fb = None
        if args.mirror:
            from .snapugc import hf_mirror_index

            fb = hf_mirror_index(args.mirror, cache_path=Path(args.videos_dir).parent / f"{args.mirror}_index.json")
        rlog = download_videos(redo, args.videos_dir, workers=args.workers, fallback_urls=fb)
        print("Re-downloaded earlier videos:", rlog["status"].value_counts().to_dict())
    if need == 0:
        print("Nothing to download")
        return
    fallback = None
    if args.mirror:
        from .snapugc import hf_mirror_index

        fallback = hf_mirror_index(args.mirror, cache_path=Path(args.videos_dir).parent / f"{args.mirror}_index.json")
        in_mirror = labels["video_id"].isin(fallback)
        print(f"Mirror {args.mirror}: {len(fallback):,} videos indexed; covers {int(in_mirror.sum()):,} of "
              f"{len(labels):,} labelled videos. Dead official links fall back to it.")
        labels = labels[in_mirror | labels.get("url", pd.Series(index=labels.index)).notna()]
    candidates = sample_candidates(labels, int(need * args.oversample), seed=args.seed, exclude_ids=done,
                                   keep_without_url=bool(fallback))
    log = download_videos(candidates, args.videos_dir, workers=args.workers, fallback_urls=fallback)
    print(log["status"].value_counts().to_string())
    ok = log["status"].isin(["ok", "ok_mirror", "exists"]).sum()
    print(f"{ok}/{len(log)} videos available in {args.videos_dir}")


def _exploratory(spec: str) -> set[str] | None:
    if not spec:
        return None
    if spec == "snapugc-run1":
        from .pipeline import run1_exploratory_ids

        return run1_exploratory_ids()
    lines = Path(spec).read_text().splitlines()
    return {ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")}


def _raw(preds_dir: str, manifest: pd.DataFrame):
    if not preds_dir:
        return None
    from .tribe_runner import raw_feature_table

    table = raw_feature_table(preds_dir, ids=set(manifest["video_id"].astype(str)))
    print(f"Raw features found for {len(table)} videos")
    return table


def _audio(audio_dir: str, manifest: pd.DataFrame):
    if not audio_dir:
        return None
    from .audio import audio_feature_table

    table = audio_feature_table(audio_dir, ids=set(manifest["video_id"].astype(str)))
    print(f"Audio features found for {len(table)} videos")
    return table


def _audio_extract(args) -> None:
    from .audio import AudioTagger, extract_audio_features

    manifest = pd.read_csv(args.manifest, dtype={"video_id": str})
    tagger = None if args.no_tagger else AudioTagger(device=args.device)
    log = extract_audio_features(manifest, args.out, tagger=tagger, overwrite=args.overwrite)
    print(log["status"].value_counts().to_string())


def _train_scores(args) -> None:
    from .pipeline import run_score_training

    manifest = pd.read_csv(args.manifest, dtype={"video_id": str})
    ts = load_table(args.timeseries)
    baseline = args.baseline_cols.split(",") if args.baseline_cols else None
    run_score_training(
        manifest, ts, args.targets.split(","), args.out,
        baseline_cols=baseline,
        group_col=args.group_col or None,
        normalize_within_group=args.normalize_within_group,
        hook_seconds=args.hook_seconds,
        n_splits=args.folds, seed=args.seed, title=args.title, n_boot=args.n_boot,
        exploratory_ids=_exploratory(args.exploratory_ids),
        raw_features=_raw(args.raw_from, manifest),
        audio_features=_audio(args.audio_from, manifest),
    )
    print(f"Wrote {args.out}/report.md")


def _extract(args) -> None:
    from .tribe_runner import TribeExtractor

    manifest = pd.read_csv(args.manifest, dtype={"video_id": str})
    table = "network_timeseries"
    if args.num_shards > 1:
        if not 0 <= args.shard_index < args.num_shards:
            raise SystemExit("--shard-index must be in [0, --num-shards)")
        manifest = manifest.iloc[args.shard_index :: args.num_shards]
        table = f"network_timeseries_shard{args.shard_index:03d}"
    if args.limit:
        manifest = manifest.head(args.limit)
    cache = Path(args.cache) if args.cache else Path(args.out) / "cache"
    deadline = time.time() + args.max_hours * 3600 if args.max_hours else None
    TribeExtractor(cache_folder=cache, device=args.device, precision=args.precision).run(
        manifest, args.out, overwrite=args.overwrite, table_name=table, deadline=deadline
    )
    print(f"Wrote {table} to {args.out}")


def _merge(args) -> None:
    from .tribe_runner import merge_shards

    print(f"Wrote {merge_shards(args.out)}")


def _train(args) -> None:
    from .pipeline import run_training

    manifest = pd.read_csv(args.manifest, dtype={"video_id": str})
    ts = load_table(args.timeseries)
    curves = pd.read_csv(args.retention, dtype={"video_id": str})
    if "duration_s" in manifest:
        durations = manifest.set_index("video_id")["duration_s"]
    else:
        durations = ts.groupby("video_id")["t"].max() + 1
    per_second = resample_to_seconds(curves, durations)
    baseline = args.baseline_cols.split(",") if args.baseline_cols else None
    run_training(manifest, ts, per_second, args.out, baseline_cols=baseline,
                 n_splits=args.folds, seed=args.seed)
    print(f"Wrote {args.out}/report.md")


def _score(args) -> None:
    models = joblib.load(args.models)
    ts = load_table(args.timeseries)
    vf = video_features(ts)
    manifest = pd.read_csv(args.manifest, dtype={"video_id": str}) if args.manifest else vf[["video_id"]]
    video = manifest.merge(vf.drop(columns=["duration_s"]) if "duration_s" in manifest else vf,
                           on="video_id")
    m = models["avg_retention"]
    video["pred_avg_retention_vs_creator_z"] = m["model"].predict(video[m["cols"]].to_numpy(float))
    sf = second_features(ts)
    d = models["dropoff"]
    sf["pred_dropoff_hazard"] = d["model"].predict(sf[d["cols"]].to_numpy(float))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    video[["video_id", "pred_avg_retention_vs_creator_z"]].sort_values(
        "pred_avg_retention_vs_creator_z", ascending=False
    ).to_csv(out / "video_scores.csv", index=False)
    sf[["video_id", "t", "pred_dropoff_hazard"]].to_csv(out / "dropoff_risk.csv", index=False)
    print(f"Wrote {out}/video_scores.csv and {out}/dropoff_risk.csv")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="neuroengage", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="Run the full pipeline on synthetic data")
    d.add_argument("--out", default="results/demo")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--null", action="store_true", help="Synthetic data with no brain signal")
    d.add_argument("--snapugc", action="store_true", help="Per-video scores (NAWP/ECR) instead of curves")
    d.set_defaults(func=_demo)

    sp = sub.add_parser("snapugc-prepare", help="Build a manifest from SnapUGC labels + videos")
    sp.add_argument("--labels", required=True, help="SnapUGC label CSV (train or val)")
    sp.add_argument("--videos-dir", required=True, help="Folder containing <Id>.mp4 files")
    sp.add_argument("--n", type=int, default=None, help="Stratified subsample size")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--id-col", default="", help="Override the video id column name")
    sp.add_argument("--nawp-col", default="", help="Override the NAWP column name")
    sp.add_argument("--ecr-col", default="", help="Override the ECR column name")
    sp.add_argument("--out", default="work/snapugc_manifest.csv")
    sp.add_argument("--done-dir", default="", help="preds/ folder of earlier runs: always include them")
    sp.set_defaults(func=_snapugc_prepare)

    sd = sub.add_parser("snapugc-download", help="Download a stratified sample of SnapUGC videos")
    sd.add_argument("--labels", required=True, help="SnapUGC label CSV with a Download_link column")
    sd.add_argument("--videos-dir", required=True)
    sd.add_argument("--n", type=int, required=True, help="Videos you want in the final run")
    sd.add_argument("--oversample", type=float, default=1.3, help="Extra candidates for dead links")
    sd.add_argument("--workers", type=int, default=8)
    sd.add_argument("--seed", type=int, default=0)
    sd.add_argument("--id-col", default="")
    sd.add_argument("--done-dir", default="", help="preds/ folder of earlier runs: skip those videos")
    sd.add_argument("--redownload-done", action="store_true",
                    help="Also fetch video files for videos processed in earlier runs (for audio features)")
    sd.add_argument("--mirror", default="", choices=["", "tcm03"],
                    help="Fall back to an unofficial Hugging Face copy of the SnapUGC videos for dead links")
    sd.set_defaults(func=_snapugc_download)

    ts_ = sub.add_parser("train-scores", help="Evaluate brain features vs baselines on per-video scores")
    ts_.add_argument("--manifest", required=True, help="CSV with video_id, targets, baseline cols")
    ts_.add_argument("--timeseries", default="work/network_timeseries")
    ts_.add_argument("--targets", default="nawp,ecr",
                     help="Comma-separated target columns (missing ones are skipped)")
    ts_.add_argument("--baseline-cols", default="duration_s", help="Comma-separated columns")
    ts_.add_argument("--group-col", default="", help="e.g. creator_id; empty = shuffled K-fold")
    ts_.add_argument("--normalize-within-group", action="store_true")
    ts_.add_argument("--hook-seconds", type=float, default=5.0,
                     help="Length of the opening window (SnapUGC's ECR uses 5 s)")
    ts_.add_argument("--folds", type=int, default=5)
    ts_.add_argument("--seed", type=int, default=0)
    ts_.add_argument("--title", default="Results")
    ts_.add_argument("--n-boot", type=int, default=2000, help="Bootstrap resamples for 95%% CIs")
    ts_.add_argument("--audio-from", default="",
                     help="audio features folder (from audio-extract): adds the audio comparison")
    ts_.add_argument("--raw-from", default="",
                     help="preds/ folder: add the raw-feature baseline (TRIBE's own inputs)")
    ts_.add_argument("--exploratory-ids", default="",
                     help="'snapugc-run1' or a file of IDs used to choose the analysis; "
                          "adds a confirmatory test on the other videos")
    ts_.add_argument("--out", default="results/scores_run")
    ts_.set_defaults(func=_train_scores)

    e = sub.add_parser("extract", help="Run TRIBE v2 over videos (needs GPU + tribev2)")
    e.add_argument("--manifest", required=True, help="CSV with video_id,path")
    e.add_argument("--out", default="work")
    e.add_argument("--device", default="auto")
    e.add_argument("--overwrite", action="store_true")
    e.add_argument("--cache", default="", help="Feature cache dir (default <out>/cache)")
    e.add_argument("--limit", type=int, default=0, help="Only the first N videos (pilot runs)")
    e.add_argument("--shard-index", type=int, default=0)
    e.add_argument("--num-shards", type=int, default=1,
                   help="Split the manifest across machines; merge with merge-shards")
    e.add_argument("--precision", choices=["fp32", "fp16"], default="fp32",
                   help="fp16 is much faster on T4-class GPUs; check agreement with fp32 first")
    e.add_argument("--max-hours", type=float, default=0,
                   help="Stop starting new videos after this many hours (0 = no limit)")
    e.set_defaults(func=_extract)

    au = sub.add_parser("audio-extract", help="Audio descriptors + AudioSet tags/embeddings (AST) per video")
    au.add_argument("--manifest", required=True, help="CSV with video_id,path")
    au.add_argument("--out", required=True, help="Folder for <video_id>.npz audio features")
    au.add_argument("--no-tagger", action="store_true", help="Descriptors only (no AST model)")
    au.add_argument("--device", default="auto")
    au.add_argument("--overwrite", action="store_true")
    au.set_defaults(func=_audio_extract)

    mg = sub.add_parser("merge-shards", help="Combine sharded extract outputs")
    mg.add_argument("--out", default="work")
    mg.set_defaults(func=_merge)

    t = sub.add_parser("train", help="Evaluate brain features vs baselines and fit models")
    t.add_argument("--manifest", required=True, help="CSV with video_id,creator_id[,duration_s,...]")
    t.add_argument("--timeseries", default="work/network_timeseries")
    t.add_argument("--retention", required=True, help="CSV with video_id,t,retention")
    t.add_argument("--baseline-cols", default="", help="Comma-separated manifest columns")
    t.add_argument("--folds", type=int, default=5)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", default="results/run")
    t.set_defaults(func=_train)

    s = sub.add_parser("score", help="Score new videos with trained models")
    s.add_argument("--models", required=True)
    s.add_argument("--timeseries", required=True)
    s.add_argument("--manifest", default="", help="Needed if baseline cols came from it")
    s.add_argument("--out", default="results/scores")
    s.set_defaults(func=_score)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
