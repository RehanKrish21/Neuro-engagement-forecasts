"""Turn network time series into model features.

Two granularities:

* :func:`video_features` - one row per video (for predicting watch time,
  completion rate, shares, ...).
* :func:`second_features` - one row per video-second (for predicting *where*
  viewers drop off, aligned with per-second audience-retention curves).

Two composite signals come from the neuroforecasting literature:

* ``affect_balance`` = valuation (mPFC/OFC) - anterior insula. Tong et al.
  (2020, PNAS) found nucleus-accumbens activity (positive arousal) and
  anterior-insula activity (negative arousal) forecast YouTube engagement in
  opposite directions. TRIBE v2's released weights are cortical only, so
  mPFC/OFC stands in for the accumbens here.
* ``attention_index`` = dorsal attention - default mode: focused external
  attention versus mind-wandering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .rois import NETWORK_NAMES

COMPOSITES = {
    "affect_balance": ("valuation_mpfc", "salience_insula"),
    "attention_index": ("dorsal_attention", "default_mode"),
}
EXTRA_SIGNALS = ("global", "pattern_shift")


def add_composites(ts: pd.DataFrame) -> pd.DataFrame:
    ts = ts.copy()
    for name, (pos, neg) in COMPOSITES.items():
        if pos in ts and neg in ts:
            ts[name] = ts[pos] - ts[neg]
    return ts


def signal_columns(ts: pd.DataFrame) -> list[str]:
    wanted = (*NETWORK_NAMES, *EXTRA_SIGNALS, *COMPOSITES)
    return [c for c in wanted if c in ts.columns]


def _slope(t: np.ndarray, y: np.ndarray) -> float:
    if len(t) < 2 or np.ptp(t) == 0:
        return 0.0
    return float(np.polyfit(t, y, 1)[0])


def video_features(long_ts: pd.DataFrame, hook_seconds: float = 3.0) -> pd.DataFrame:
    """One feature row per video from long-format network time series.

    ``long_ts`` needs ``video_id`` and ``t`` columns plus signal columns, as
    produced by :func:`neuroengage.tribe_runner.to_long`.
    """
    long_ts = add_composites(long_ts)
    cols = signal_columns(long_ts)
    rows = []
    for vid, g in long_ts.groupby("video_id", sort=False):
        g = g.sort_values("t")
        t = g["t"].to_numpy(float)
        duration = float(t[-1] - t[0] + 1) if len(t) else 0.0
        hook = t < t[0] + hook_seconds
        third = max(1, len(t) // 3)
        row: dict[str, float | str] = {"video_id": vid, "duration_s": duration}
        for c in cols:
            y = g[c].to_numpy(float)
            rest = y[~hook] if (~hook).any() else y
            row[f"{c}__mean"] = y.mean()
            row[f"{c}__std"] = y.std()
            row[f"{c}__hook"] = y[hook].mean()
            row[f"{c}__hook_vs_rest"] = y[hook].mean() - rest.mean()
            row[f"{c}__slope"] = _slope(t, y)
            row[f"{c}__late_vs_early"] = y[-third:].mean() - y[:third].mean()
            row[f"{c}__peak_time_frac"] = float(np.argmax(y)) / max(1, len(y) - 1)
        rows.append(row)
    return pd.DataFrame(rows)


def second_features(long_ts: pd.DataFrame, lags: tuple[int, ...] = (1, 2)) -> pd.DataFrame:
    """One feature row per (video, second).

    For each signal: raw value, value z-scored within the video, change from
    the previous second, a trailing 3-second mean, and lagged values. Also
    includes position in the video (seconds and fraction), since drop-off is
    always heaviest at the start.
    """
    long_ts = add_composites(long_ts)
    cols = signal_columns(long_ts)
    out = []
    for vid, g in long_ts.groupby("video_id", sort=False):
        g = g.sort_values("t").reset_index(drop=True)
        feats = pd.DataFrame({"video_id": vid, "t": g["t"]})
        duration = g["t"].iloc[-1] - g["t"].iloc[0] + 1
        feats["t_frac"] = (g["t"] - g["t"].iloc[0]) / duration
        feats["log_t"] = np.log1p(g["t"] - g["t"].iloc[0])
        new = {}
        for c in cols:
            y = g[c]
            sd = y.std(ddof=0)
            new[c] = y
            new[f"{c}__z"] = (y - y.mean()) / (sd if sd > 0 else 1.0)
            new[f"{c}__delta"] = y.diff().fillna(0.0)
            new[f"{c}__roll3"] = y.rolling(3, min_periods=1).mean()
            for lag in lags:
                new[f"{c}__lag{lag}"] = y.shift(lag).bfill()
        out.append(pd.concat([feats, pd.DataFrame(new)], axis=1))
    return pd.concat(out, ignore_index=True)
