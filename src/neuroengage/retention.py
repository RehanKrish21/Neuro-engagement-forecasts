"""Audience-retention curves -> per-second and per-video targets.

Canonical format is long: ``video_id, t, retention`` where ``t`` is seconds
from the start and ``retention`` is the fraction of starting viewers still
watching (1.0 at t=0, non-increasing in a clean curve).

YouTube's Analytics API reports retention against *elapsed video ratio*
(``elapsedVideoTimeRatio``, 0.01..1.0) with ``audienceWatchRatio``; use
:func:`from_youtube` to convert that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def from_youtube(report: pd.DataFrame, durations: pd.Series | dict) -> pd.DataFrame:
    """Convert a YouTube audience-retention report to canonical long format.

    ``report`` columns: ``video_id``, ``elapsedVideoTimeRatio``,
    ``audienceWatchRatio``. ``durations`` maps video_id -> length in seconds.
    """
    durations = pd.Series(durations, dtype=float)
    df = report.copy()
    df["t"] = df["elapsedVideoTimeRatio"] * df["video_id"].map(durations)
    df["retention"] = df["audienceWatchRatio"]
    return df[["video_id", "t", "retention"]]


def resample_to_seconds(curves: pd.DataFrame, durations: pd.Series | dict) -> pd.DataFrame:
    """Interpolate each curve onto integer seconds ``0..ceil(duration)-1``.

    Applies a running minimum so retention never increases (re-watches can
    push raw curves above earlier values) and anchors t=0 at 1.0 when the
    curve does not start there.
    """
    durations = pd.Series(durations, dtype=float)
    out = []
    for vid, g in curves.groupby("video_id", sort=False):
        if vid not in durations.index:
            raise KeyError(f"No duration for video {vid!r}")
        g = g.sort_values("t")
        t, r = g["t"].to_numpy(float), g["retention"].to_numpy(float)
        if t[0] > 0:
            t, r = np.r_[0.0, t], np.r_[1.0, r]
        grid = np.arange(int(np.ceil(durations[vid])))
        ret = np.interp(grid, t, r)
        ret = np.minimum.accumulate(np.clip(ret, 0.0, 1.0))
        out.append(pd.DataFrame({"video_id": vid, "t": grid, "retention": ret}))
    return pd.concat(out, ignore_index=True)


def dropoff_targets(per_second: pd.DataFrame) -> pd.DataFrame:
    """Add ``hazard``: share of viewers still present at t-1 who leave by t.

    The hazard (rather than the raw drop) is the right per-second target
    because it does not shrink just because few viewers remain late in a video.
    """
    df = per_second.sort_values(["video_id", "t"]).copy()
    prev = df.groupby("video_id")["retention"].shift(1).fillna(1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        hazard = np.where(prev > 0, (prev - df["retention"]) / prev, 0.0)
    df["hazard"] = np.clip(hazard, 0.0, 1.0)
    return df


def video_targets(per_second: pd.DataFrame, hook_seconds: int = 3) -> pd.DataFrame:
    """Per-video summary targets from per-second retention.

    * ``avg_retention``: mean of the curve, i.e. average fraction watched.
    * ``hook_retention``: retention at ``hook_seconds`` (did the opening work?).
    * ``completion``: retention at the final second.
    """
    rows = []
    for vid, g in per_second.groupby("video_id", sort=False):
        r = g.sort_values("t")["retention"].to_numpy(float)
        rows.append(
            {
                "video_id": vid,
                "avg_retention": r.mean(),
                "hook_retention": r[min(hook_seconds, len(r) - 1)],
                "completion": r[-1],
            }
        )
    return pd.DataFrame(rows)
