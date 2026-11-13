"""End-to-end training/evaluation: network time series + retention -> report."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import second_features, video_features
from .modeling import (
    creator_normalize,
    evaluate_dropoff_model,
    evaluate_video_model,
    fit_final,
)
from .retention import dropoff_targets, video_targets

VIDEO_TARGETS = ("avg_retention", "hook_retention", "completion")
POSITION_COLS = ["t_frac", "log_t"]


def _str_ids(*frames: pd.DataFrame) -> list[pd.DataFrame]:
    """IDs like 0012 or 12345 must not turn into integers when read from CSV."""
    return [f.assign(video_id=f["video_id"].astype(str)) for f in frames]


def build_tables(
    manifest: pd.DataFrame,
    long_ts: pd.DataFrame,
    retention_per_second: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Join features and targets.

    Returns ``(video_table, second_table, video_brain_cols, second_brain_cols)``.
    """
    manifest, long_ts, retention_per_second = _str_ids(manifest, long_ts, retention_per_second)
    vf = video_features(long_ts)
    brain_cols = [c for c in vf.columns if "__" in c]
    vt = video_targets(retention_per_second)
    video = manifest.merge(vf.drop(columns=["duration_s"], errors="ignore"), on="video_id")
    video = video.merge(vt, on="video_id")
    if "duration_s" not in video:
        video = video.merge(vf[["video_id", "duration_s"]], on="video_id")
    for target in VIDEO_TARGETS:
        video[f"{target}__creator_z"] = creator_normalize(video, target).to_numpy()

    sf = second_features(long_ts)
    sec_brain_cols = [c for c in sf.columns if c not in {"video_id", "t", *POSITION_COLS}]
    hz = dropoff_targets(retention_per_second)[["video_id", "t", "hazard"]]
    second = sf.merge(hz, on=["video_id", "t"], how="inner")
    second = second.merge(manifest[["video_id", "creator_id"]], on="video_id")
    return video, second, brain_cols, sec_brain_cols


def run_training(
    manifest: pd.DataFrame,
    long_ts: pd.DataFrame,
    retention_per_second: pd.DataFrame,
    out_dir: str | Path,
    baseline_cols: list[str] | None = None,
    n_splits: int = 5,
    seed: int = 0,
    title: str = "Results",
    note: str = "",
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video, second, brain_cols, sec_brain_cols = build_tables(
        manifest, long_ts, retention_per_second
    )
    baseline_cols = [c for c in (baseline_cols or ["duration_s"]) if c in video.columns]
    if not baseline_cols:
        raise ValueError("No baseline columns found in manifest")

    video_sets = {
        "baseline": baseline_cols,
        "brain": brain_cols,
        "baseline+brain": baseline_cols + brain_cols,
    }
    video_summaries = []
    for target in VIDEO_TARGETS:
        res = evaluate_video_model(
            video, f"{target}__creator_z", video_sets, n_splits=n_splits, seed=seed
        )
        video_summaries.append(res.summary.assign(target=target))
        res.oof_predictions.to_csv(out_dir / f"oof_{target}.csv", index=False)
    video_summary = pd.concat(video_summaries, ignore_index=True)
    video_summary.to_csv(out_dir / "video_level_metrics.csv", index=False)

    drop_sets = {"position": POSITION_COLS, "position+brain": POSITION_COLS + sec_brain_cols}
    drop = evaluate_dropoff_model(second, "hazard", drop_sets, n_splits=n_splits, seed=seed)
    drop.summary.to_csv(out_dir / "dropoff_metrics.csv", index=False)
    drop.oof_predictions.to_csv(out_dir / "oof_dropoff.csv", index=False)

    models = {
        "avg_retention": {
            "cols": video_sets["baseline+brain"],
            "model": fit_final(video, "avg_retention__creator_z", video_sets["baseline+brain"], "ridge", seed),
        },
        "dropoff": {
            "cols": drop_sets["position+brain"],
            "model": fit_final(second, "hazard", drop_sets["position+brain"], "hgb", seed),
        },
    }
    joblib.dump(models, out_dir / "models.joblib")

    top = top_coefficients(models["avg_retention"]["model"], models["avg_retention"]["cols"])
    top.to_csv(out_dir / "avg_retention_top_features.csv", index=False)

    fig_path = None
    try:
        from .plots import plot_summary

        fig_path = plot_summary(video_summary, drop.summary, out_dir / "summary.png", title=title)
    except ImportError:
        pass

    info = {
        "n_videos": int(video["video_id"].nunique()),
        "n_creators": int(video["creator_id"].nunique()),
        "n_video_seconds": int(len(second)),
        "baseline_cols": baseline_cols,
        "n_brain_features_video": len(brain_cols),
        "n_brain_features_second": len(sec_brain_cols),
    }
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2))
    write_report(out_dir, info, video_summary, drop.summary, top, fig_path, title, note)
    return {"video": video_summary, "dropoff": drop.summary, "info": info}


def top_coefficients(ridge_pipeline, cols: list[str], k: int = 12) -> pd.DataFrame:
    """Largest standardized ridge coefficients (direction and size)."""
    coef = ridge_pipeline[-1].coef_
    df = pd.DataFrame({"feature": cols, "std_coef": coef})
    return df.reindex(df["std_coef"].abs().sort_values(ascending=False).index).head(k)


def _fmt(df: pd.DataFrame) -> str:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda x: f"{x:.3f}")
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def write_report(out_dir, info, video_summary, drop_summary, top, fig_path, title, note) -> None:
    v = video_summary[["target", "feature_set", "spearman_mean", "spearman_std", "r2_mean",
                       "spearman_gain_vs_baseline", "folds_better_than_baseline"]]
    d = drop_summary[["feature_set", "within_video_spearman_mean", "within_video_spearman_std",
                      "r2_mean", "within_video_spearman_gain_vs_position"]]
    parts = [f"# {title}", ""]
    if note:
        parts += [note, ""]
    parts += [
        f"{info['n_videos']} videos from {info['n_creators']} creators, "
        f"{info['n_video_seconds']} video-seconds. Grouped cross-validation: "
        "video-level folds hold out whole creators; drop-off folds hold out whole videos.",
        "",
    ]
    if fig_path:
        parts += [f"![summary]({Path(fig_path).name})", ""]
    parts += [
        "## Video-level: within-creator performance",
        "",
        "Targets are z-scored within creator, so the question is \"is this video better "
        "than this creator's usual?\".",
        "",
        _fmt(v),
        "",
        "## Per-second drop-off",
        "",
        "`within_video_spearman`: rank agreement between predicted and actual drop-off "
        "across each held-out video's seconds (first 3 s excluded).",
        "",
        _fmt(d),
        "",
        "## Largest standardized coefficients (avg_retention, baseline+brain ridge)",
        "",
        _fmt(top),
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(parts))


def run_score_training(
    manifest: pd.DataFrame,
    long_ts: pd.DataFrame,
    targets: list[str],
    out_dir: str | Path,
    baseline_cols: list[str] | None = None,
    group_col: str | None = None,
    normalize_within_group: bool = False,
    hook_seconds: float = 3.0,
    n_splits: int = 5,
    seed: int = 0,
    title: str = "Results",
    note: str = "",
) -> dict:
    """Video-level evaluation against per-video scores (no retention curves).

    For datasets that report one number per video, such as SnapUGC's NAWP and
    ECR. ``manifest`` holds ``video_id``, the target columns and any baseline
    columns. ``group_col`` (e.g. creator) makes folds hold out whole groups and,
    with ``normalize_within_group``, z-scores targets within group.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest, long_ts = _str_ids(manifest, long_ts)
    vf = video_features(long_ts, hook_seconds=hook_seconds)
    brain_cols = [c for c in vf.columns if "__" in c]
    dup = [c for c in vf.columns if c in manifest.columns and c != "video_id"]
    video = manifest.merge(vf.drop(columns=dup), on="video_id")
    if video.empty:
        raise ValueError("No overlap between manifest video_ids and the time series")
    missing_t = [t for t in targets if t not in video]
    if missing_t:
        raise KeyError(f"Targets not in manifest: {missing_t}")
    baseline_cols = [c for c in (baseline_cols or ["duration_s"]) if c in video.columns]
    if not baseline_cols:
        raise ValueError("No baseline columns found")

    sets = {
        "baseline": baseline_cols,
        "brain": brain_cols,
        "baseline+brain": baseline_cols + brain_cols,
    }
    summaries, fitted = [], {}
    for target in targets:
        col = target
        if group_col and normalize_within_group:
            col = f"{target}__group_z"
            video[col] = creator_normalize(video, target, creator_col=group_col).to_numpy()
        res = evaluate_video_model(video, col, sets, group_col=group_col, n_splits=n_splits, seed=seed)
        summaries.append(res.summary.assign(target=target))
        res.oof_predictions.to_csv(out_dir / f"oof_{target}.csv", index=False)
        rows = video[video[col].notna()]
        fitted[target] = {
            "cols": sets["baseline+brain"],
            "model": fit_final(rows, col, sets["baseline+brain"], "ridge", seed),
        }
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(out_dir / "video_level_metrics.csv", index=False)
    joblib.dump(fitted, out_dir / "models.joblib")

    tops = []
    for target, m in fitted.items():
        tops.append(top_coefficients(m["model"], m["cols"]).assign(target=target))
    top = pd.concat(tops, ignore_index=True)
    top.to_csv(out_dir / "top_features.csv", index=False)

    fig_path = None
    try:
        from .plots import plot_summary

        fig_path = plot_summary(summary, None, out_dir / "summary.png", title=title)
    except ImportError:
        pass

    info = {
        "n_videos": int(len(video)),
        "targets": targets,
        "baseline_cols": baseline_cols,
        "group_col": group_col,
        "hook_seconds": hook_seconds,
        "n_brain_features": len(brain_cols),
    }
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2))

    v = summary[["target", "feature_set", "spearman_mean", "spearman_std", "r2_mean",
                 "spearman_gain_vs_baseline", "folds_better_than_baseline"]]
    folds = (f"folds hold out whole `{group_col}` groups" if group_col
             else "shuffled K-fold over videos (no creator IDs)")
    parts = [f"# {title}", ""]
    if note:
        parts += [note, ""]
    parts += [
        f"{info['n_videos']} videos, {n_splits}-fold cross-validation, {folds}. "
        f"Baseline: {', '.join(f'`{c}`' for c in baseline_cols)}.",
        "",
    ]
    if fig_path:
        parts += [f"![summary]({Path(fig_path).name})", ""]
    parts += ["## Held-out performance", "", _fmt(v), ""]
    for target in targets:
        parts += [f"## Largest standardized coefficients: {target} (baseline+brain ridge)", "",
                  _fmt(top[top.target == target].drop(columns="target")), ""]
    (out_dir / "report.md").write_text("\n".join(parts))
    return {"video": summary, "info": info}
