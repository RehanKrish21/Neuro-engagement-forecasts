"""SnapUGC (Li et al., ECCV 2024) loader.

SnapUGC is ~90-120k Snapchat Spotlight short videos (5-60 s, each with
more than 2,000 views), labelled with two engagement metrics:

* **NAWP**, normalized average watch percentage: how much of the video people
  watch, normalized so different lengths are comparable.
* **ECR**, engagement continuation rate: probability a viewer watches at
  least 5 seconds. Essentially a hook score.

Download: https://github.com/dasongli1/SnapUGC_Engagement (dataset/readme.md
links the train and val sets). The label CSVs key videos by ``Id`` with
the video at ``<videos_dir>/<Id>.mp4`` and also carry ``Title`` and
``Description``. Label column names are auto-detected case-insensitively
and can be overridden.

Check the dataset's terms before use. SnapUGC has no creator IDs, so
evaluation uses plain shuffled K-fold rather than creator-grouped folds.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ID_ALIASES = ("id", "video_id", "videoid", "vid")
TARGET_ALIASES = {
    "nawp": ("nawp", "normalized_average_watch_percentage", "avg_watch_percentage"),
    "ecr": ("ecr", "engagement_continuation_rate", "continuation_rate"),
}


def _find(columns, aliases) -> str | None:
    lower = {c.lower().strip(): c for c in columns}
    for a in aliases:
        if a in lower:
            return lower[a]
    return None


def load_labels(
    labels_csv: str | Path,
    id_col: str | None = None,
    target_cols: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Read a SnapUGC label CSV into ``video_id, nawp, ecr[, title, description]``.

    Targets missing from the file are skipped with a warning (for example a
    test split without labels), but at least one must be present.
    """
    raw = pd.read_csv(labels_csv)
    id_col = id_col or _find(raw.columns, ID_ALIASES)
    if id_col is None:
        raise KeyError(f"No video id column in {labels_csv}; columns: {list(raw.columns)}")
    out = pd.DataFrame({"video_id": raw[id_col].astype(str)})
    target_cols = target_cols or {}
    for name, aliases in TARGET_ALIASES.items():
        col = target_cols.get(name) or _find(raw.columns, aliases)
        if col is None:
            logger.warning("SnapUGC: no %s column in %s", name.upper(), labels_csv)
            continue
        out[name] = pd.to_numeric(raw[col], errors="coerce")
    targets = [c for c in TARGET_ALIASES if c in out]
    if not targets:
        raise KeyError(
            f"No NAWP/ECR column in {labels_csv}; columns: {list(raw.columns)}. "
            "Pass target_cols={'nawp': ..., 'ecr': ...}."
        )
    for text_col in ("title", "description"):
        col = _find(raw.columns, (text_col,))
        if col is not None:
            out[text_col] = raw[col].fillna("").astype(str)
    return out.dropna(subset=targets, how="all").drop_duplicates("video_id")


def text_baselines(labels: pd.DataFrame) -> pd.DataFrame:
    """Cheap metadata baselines from title/description (if present)."""
    out = pd.DataFrame({"video_id": labels["video_id"]})
    for col in ("title", "description"):
        if col in labels:
            text = labels[col]
            out[f"{col}_n_words"] = text.str.split().str.len().fillna(0)
            out[f"has_{col}"] = (text.str.strip() != "").astype(float)
    return out


def build_manifest(
    labels: pd.DataFrame,
    videos_dir: str | Path,
    n: int | None = None,
    seed: int = 0,
    stratify_on: str | None = "nawp",
    n_bins: int = 10,
) -> pd.DataFrame:
    """Manifest of labelled videos present on disk, optionally subsampled.

    TRIBE v2 costs real GPU time per video, so you will usually start with a
    few thousand videos. Subsampling is stratified across ``n_bins``
    quantile bins of ``stratify_on`` so the subset keeps the full range of
    engagement rather than piling up around the median.
    """
    videos_dir = Path(videos_dir)
    df = labels.copy()
    df["path"] = [str(videos_dir / f"{vid}.mp4") for vid in df["video_id"]]
    exists = np.array([Path(p).is_file() for p in df["path"]])
    if not exists.any():
        raise FileNotFoundError(f"None of the labelled videos found under {videos_dir}")
    if (~exists).any():
        logger.warning("SnapUGC: %d labelled videos missing on disk", int((~exists).sum()))
    df = df[exists]
    if n is not None and n < len(df):
        if stratify_on and stratify_on in df and df[stratify_on].notna().all():
            bins = pd.qcut(df[stratify_on].rank(method="first"), n_bins, labels=False)
            df = (
                df.groupby(bins, group_keys=False)
                .apply(lambda g: g.sample(min(len(g), int(np.ceil(n / n_bins))), random_state=seed))
                .sample(frac=1.0, random_state=seed)
                .head(n)
            )
        else:
            df = df.sample(n=n, random_state=seed)
    cols = ["video_id", "path", *[c for c in ("nawp", "ecr") if c in df]]
    manifest = df[cols].merge(text_baselines(df), on="video_id")
    return manifest.reset_index(drop=True)
