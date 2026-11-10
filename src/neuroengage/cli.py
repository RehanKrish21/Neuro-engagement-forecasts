"""Command line: ``neuroengage {demo,extract,train,train-scores,snapugc-prepare,score}``."""

from __future__ import annotations

import argparse
import logging
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
    from .snapugc import build_manifest, load_labels

    overrides = {k: v for k, v in (("nawp", args.nawp_col), ("ecr", args.ecr_col)) if v}
    labels = load_labels(args.labels, id_col=args.id_col or None, target_cols=overrides)
    manifest = build_manifest(labels, args.videos_dir, n=args.n, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out, index=False)
    print(f"Wrote {len(manifest)} videos to {args.out}")


def _train_scores(args) -> None:
    from .pipeline import run_score_training

    manifest = pd.read_csv(args.manifest)
    ts = load_table(args.timeseries)
    baseline = args.baseline_cols.split(",") if args.baseline_cols else None
    run_score_training(
        manifest, ts, args.targets.split(","), args.out,
        baseline_cols=baseline,
        group_col=args.group_col or None,
        normalize_within_group=args.normalize_within_group,
        hook_seconds=args.hook_seconds,
        n_splits=args.folds, seed=args.seed, title=args.title,
    )
    print(f"Wrote {args.out}/report.md")


def _extract(args) -> None:
    from .tribe_runner import TribeExtractor

    manifest = pd.read_csv(args.manifest)
    TribeExtractor(cache_folder=Path(args.out) / "cache", device=args.device).run(
        manifest, args.out, overwrite=args.overwrite
    )
    print(f"Wrote network time series to {args.out}")


def _train(args) -> None:
    from .pipeline import run_training

    manifest = pd.read_csv(args.manifest)
    ts = load_table(args.timeseries)
    curves = pd.read_csv(args.retention)
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
    manifest = pd.read_csv(args.manifest) if args.manifest else vf[["video_id"]]
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
    sp.set_defaults(func=_snapugc_prepare)

    ts_ = sub.add_parser("train-scores", help="Evaluate brain features vs baselines on per-video scores")
    ts_.add_argument("--manifest", required=True, help="CSV with video_id, targets, baseline cols")
    ts_.add_argument("--timeseries", default="work/network_timeseries")
    ts_.add_argument("--targets", default="nawp,ecr", help="Comma-separated target columns")
    ts_.add_argument("--baseline-cols", default="duration_s", help="Comma-separated columns")
    ts_.add_argument("--group-col", default="", help="e.g. creator_id; empty = shuffled K-fold")
    ts_.add_argument("--normalize-within-group", action="store_true")
    ts_.add_argument("--hook-seconds", type=float, default=5.0,
                     help="Length of the opening window (SnapUGC's ECR uses 5 s)")
    ts_.add_argument("--folds", type=int, default=5)
    ts_.add_argument("--seed", type=int, default=0)
    ts_.add_argument("--title", default="Results")
    ts_.add_argument("--out", default="results/scores_run")
    ts_.set_defaults(func=_train_scores)

    e = sub.add_parser("extract", help="Run TRIBE v2 over videos (needs GPU + tribev2)")
    e.add_argument("--manifest", required=True, help="CSV with video_id,path")
    e.add_argument("--out", default="work")
    e.add_argument("--device", default="auto")
    e.add_argument("--overwrite", action="store_true")
    e.set_defaults(func=_extract)

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
