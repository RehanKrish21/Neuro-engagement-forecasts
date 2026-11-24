"""End-to-end training/evaluation: network time series + retention -> report."""

from __future__ import annotations

import json
import logging
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
    pooled_metrics,
)

# Pre-registered (docs/preregistration.md): the primary comparison is
# PRIMARY_SET vs "baseline" on pooled out-of-fold Spearman for each target.
PRIMARY_SET = "avg(baseline,brain)"
COMPACT_STATS = ("__mean", "__hook")
MIN_CONFIRMATORY = 30


def run1_exploratory_ids() -> set[str]:
    """SnapUGC video IDs analysed in run 1, where the primary comparison was chosen."""
    path = Path(__file__).parent / "data" / "snapugc_run1_ids.txt"
    return {ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")}
from .retention import dropoff_targets, video_targets

logger = logging.getLogger(__name__)

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
    n_boot: int = 2000,
    exploratory_ids: set[str] | None = None,
    raw_features: pd.DataFrame | None = None,
) -> dict:
    """Video-level evaluation against per-video scores (no retention curves).

    ``exploratory_ids``: videos already used to choose the analysis (run 1).
    Models still train on them, but the pooled metrics are additionally
    reported on the remaining videos only ("confirmatory"), which is the
    fair test of a comparison chosen after seeing the exploratory data.

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
    missing_t = [t for t in targets if t not in video or video[t].isna().all()]
    if missing_t:
        logger.warning("Skipping targets not in the data: %s", missing_t)
        targets = [t for t in targets if t not in missing_t]
    if not targets:
        raise KeyError(f"None of the targets are in the manifest; columns: {list(manifest.columns)}")
    baseline_cols = [c for c in (baseline_cols or ["duration_s"]) if c in video.columns]
    if not baseline_cols:
        raise ValueError("No baseline columns found")

    compact_cols = [c for c in brain_cols if c.endswith(COMPACT_STATS)]
    sets = {
        "baseline": baseline_cols,
        "brain": brain_cols,
        "brain_compact": compact_cols,
        "baseline+brain": baseline_cols + brain_cols,
        "baseline+brain_compact": baseline_cols + compact_cols,
    }
    ensembles = {
        "avg(baseline,brain)": ("baseline", "brain"),
        "avg(baseline,brain_compact)": ("baseline", "brain_compact"),
    }
    summaries, pooled, fitted = [], [], {}
    for target in targets:
        col = target
        if group_col and normalize_within_group:
            col = f"{target}__group_z"
            video[col] = creator_normalize(video, target, creator_col=group_col).to_numpy()
        res = evaluate_video_model(video, col, sets, group_col=group_col, n_splits=n_splits,
                                   seed=seed, ensembles=ensembles)
        summaries.append(res.summary.assign(target=target))
        oof = res.oof_predictions
        pooled.append(pooled_metrics(oof, n_boot=n_boot, seed=seed).assign(target=target, subset="all videos"))
        if exploratory_ids:
            new_oof = oof[~oof["video_id"].astype(str).isin(exploratory_ids)]
            if len(new_oof) >= MIN_CONFIRMATORY:
                pooled.append(pooled_metrics(new_oof.reset_index(drop=True), n_boot=n_boot, seed=seed)
                              .assign(target=target, subset=f"confirmatory: {len(new_oof)} new videos"))
            else:
                logger.warning("Only %d videos outside the exploratory set; confirmatory test needs %d",
                               len(new_oof), MIN_CONFIRMATORY)
        res.oof_predictions.to_csv(out_dir / f"oof_{target}.csv", index=False)
        rows = video[video[col].notna()]
        fitted[target] = {
            "cols": sets["baseline+brain"],
            "model": fit_final(rows, col, sets["baseline+brain"], "ridge", seed),
        }
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(out_dir / "video_level_metrics.csv", index=False)
    pooled_df = pd.concat(pooled, ignore_index=True)
    pooled_df.to_csv(out_dir / "pooled_metrics.csv", index=False)
    joblib.dump(fitted, out_dir / "models.joblib")

    tops = []
    for target, m in fitted.items():
        tops.append(top_coefficients(m["model"], m["cols"]).assign(target=target))
    top = pd.concat(tops, ignore_index=True)
    top.to_csv(out_dir / "top_features.csv", index=False)

    fig_path = None
    try:
        from .plots import plot_pooled

        conf = pooled_df[pooled_df["subset"].str.startswith("confirmatory")]
        shown = conf if len(conf) else pooled_df[pooled_df["subset"] == "all videos"]
        sub = shown["subset"].iloc[0]
        fig_path = plot_pooled(shown, out_dir / "summary.png", title=f"{title} ({sub})")
    except ImportError:
        pass

    info = {
        "n_videos": int(len(video)),
        "targets": targets,
        "baseline_cols": baseline_cols,
        "group_col": group_col,
        "hook_seconds": hook_seconds,
        "n_brain_features": len(brain_cols),
        "n_brain_compact_features": len(compact_cols),
        "primary_comparison": f"{PRIMARY_SET} vs baseline",
        "n_exploratory_excluded": int(len(set(video["video_id"]) & set(exploratory_ids or ()))),
    }
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2))

    def ci(lo, hi):
        return [f"[{a:.2f}, {b:.2f}]" for a, b in zip(lo, hi)]

    raw_section: list[str] = []
    if raw_features is not None and len(raw_features):
        raw_section = _raw_comparison(video, raw_features, targets, sets, group_col,
                                      normalize_within_group, n_splits, seed, n_boot, out_dir, ci)
        info["n_videos_with_raw"] = int(video["video_id"].isin(set(raw_features["video_id"])).sum())
        (out_dir / "run_info.json").write_text(json.dumps(info, indent=2))

    pv = pd.DataFrame({
        "subset": pooled_df["subset"],
        "target": pooled_df["target"],
        "feature_set": [f"**{f}** (primary)" if f == PRIMARY_SET else f for f in pooled_df["feature_set"]],
        "spearman": pooled_df["spearman"].round(3),
        "95% CI": ci(pooled_df["spearman_lo"], pooled_df["spearman_hi"]),
        "pairwise_acc": pooled_df["pairwise_acc"].round(3),
        "pairwise 95% CI": ci(pooled_df["pairwise_lo"], pooled_df["pairwise_hi"]),
        "gain_vs_baseline": pooled_df["gain_vs_baseline"].round(3),
        "gain 95% CI": ci(pooled_df["gain_lo"], pooled_df["gain_hi"]),
        "P(gain>0)": pooled_df["p_gain_positive"].round(2),
    })
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
    parts += [
        "## Held-out performance (pooled out-of-fold, 95% bootstrap CIs)",
        "",
        f"Pre-registered primary comparison: **{PRIMARY_SET} vs baseline** "
        "(docs/preregistration.md). `pairwise_acc` = share of video pairs ranked in the right "
        "order (0.5 = chance). `brain_compact` = only the `mean` and `hook` statistics of each "
        "brain signal; `avg(a,b)` = fixed 50/50 average of the two models' ranks (nothing fitted). "
        "Gains are paired (same bootstrap resamples). When present, the **confirmatory** rows "
        "exclude the videos used to choose this analysis (run 1) and are the ones that test it.",
        "",
        _fmt(pv),
        "",
        "### Per-fold detail",
        "",
        _fmt(v),
        "",
    ]
    parts += raw_section
    for target in targets:
        parts += [f"## Largest standardized coefficients: {target} (baseline+brain ridge)", "",
                  _fmt(top[top.target == target].drop(columns="target")), ""]
    (out_dir / "report.md").write_text("\n".join(parts))
    return {"video": summary, "info": info, "pooled": pooled_df}


MIN_RAW_VIDEOS = 50


def _raw_comparison(video, raw_features, targets, sets, group_col, normalize_within_group,
                    n_splits, seed, n_boot, out_dir, ci) -> list[str]:
    """Exploratory: TRIBE's input features vs its predicted brain responses."""
    raw_features = raw_features.assign(video_id=raw_features["video_id"].astype(str))
    data = video.merge(raw_features, on="video_id")
    if len(data) < MIN_RAW_VIDEOS:
        logger.warning("Raw-feature comparison skipped: %d videos have raw features (need %d)",
                       len(data), MIN_RAW_VIDEOS)
        return ["## Raw-feature baseline (exploratory)", "",
                f"Skipped: only {len(data)} videos have saved raw features (need {MIN_RAW_VIDEOS}). "
                "Videos predicted before version 0.8.0 have none.", ""]
    raw_cols = [c for c in raw_features.columns if c.startswith("raw:")]
    by_mod = {}
    for c in raw_cols:
        by_mod.setdefault(c.split(":")[1], []).append(c)
    rsets = {"raw": raw_cols, **{f"raw_{m}": cols for m, cols in sorted(by_mod.items())},
             "brain": sets["brain"], "baseline": sets["baseline"]}
    set_models = {k: "ridge_wide" for k in rsets if k.startswith("raw")}
    ensembles = {"avg(raw,brain)": ("raw", "brain")}
    tables = []
    for target in targets:
        col = target
        if group_col and normalize_within_group:
            col = f"{target}__group_z"
            data[col] = creator_normalize(data, target, creator_col=group_col).to_numpy()
        res = evaluate_video_model(data, col, rsets, group_col=group_col, n_splits=n_splits, seed=seed,
                                   ensembles=ensembles, set_models=set_models)
        res.oof_predictions.to_csv(out_dir / f"oof_raw_{target}.csv", index=False)
        tables.append(pooled_metrics(res.oof_predictions, baseline="raw", n_boot=n_boot, seed=seed)
                      .assign(target=target))
    t = pd.concat(tables, ignore_index=True)
    t.to_csv(out_dir / "raw_comparison_metrics.csv", index=False)
    try:
        from .plots import plot_pooled

        plot_pooled(t.assign(subset="raw"), out_dir / "raw_comparison.png",
                    title=f"Brain features vs TRIBE's raw inputs ({len(data)} videos)")
    except ImportError:
        pass
    tv = pd.DataFrame({
        "target": t["target"], "feature_set": t["feature_set"],
        "spearman": t["spearman"].round(3), "95% CI": ci(t["spearman_lo"], t["spearman_hi"]),
        "pairwise_acc": t["pairwise_acc"].round(3), "pairwise 95% CI": ci(t["pairwise_lo"], t["pairwise_hi"]),
        "gain_vs_raw": t["gain_vs_baseline"].round(3), "gain 95% CI": ci(t["gain_lo"], t["gain_hi"]),
        "P(gain>0)": t["p_gain_positive"].round(2),
    })
    dims = ", ".join(f"{m}: {len(c)}" for m, c in sorted(by_mod.items()))
    return [
        "## Raw-feature baseline (exploratory, added 2026-11-24)",
        "",
        f"{len(data)} videos with saved raw features. `raw` = the features TRIBE itself is built on "
        f"(V-JEPA 2 video, Wav2Vec-BERT audio, Llama 3.2 text: whole-video and first-5-s averages; "
        f"dimensions {dims}), fed to a ridge model with strong regularisation. Gains are relative to "
        "`raw`: does TRIBE's predicted brain activity add anything beyond its own inputs? Not part of "
        "the pre-registered primary test.",
        "",
        "![raw comparison](raw_comparison.png)",
        "",
        _fmt(tv),
        "",
    ]
