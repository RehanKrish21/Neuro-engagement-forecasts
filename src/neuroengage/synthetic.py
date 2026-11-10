"""Synthetic data with a *known, planted* brain-engagement relationship.

This exists so the pipeline can be run end to end and tested without a GPU,
TRIBE v2 weights, or private analytics. Results on it show the pipeline can
recover a signal that is there by construction. They say nothing about
whether TRIBE v2 features predict real-world engagement; only real data can.

Generative story per video:

* a latent ``engagement`` process e(t) and ``aversion`` process a(t) (AR(1));
* network time series are noisy mixtures: valuation and dorsal attention
  track +e, default mode tracks -e, anterior insula tracks +a, the rest is
  noise;
* per-second drop-off hazard = baseline decay x creator effect x
  exp(-beta_e * e(t) + beta_a * a(t)).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .rois import NETWORK_NAMES

LOADINGS = {
    "valuation_mpfc": (0.8, 0.0),
    "dorsal_attention": (0.6, 0.0),
    "default_mode": (-0.6, 0.0),
    "salience_insula": (0.0, 0.8),
    "ventral_attention": (0.2, 0.3),
}


def _ar1(rng: np.random.Generator, n: int, phi: float = 0.85) -> np.ndarray:
    x = np.zeros(n)
    eps = rng.normal(0, np.sqrt(1 - phi**2), n)
    x[0] = rng.normal()
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def make_dataset(
    n_creators: int = 20,
    videos_per_creator: int = 12,
    min_seconds: int = 15,
    max_seconds: int = 60,
    beta_engagement: float = 0.3,
    beta_aversion: float = 0.25,
    noise: float = 1.5,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ``(manifest, network_timeseries, retention_per_second)``.

    ``manifest`` has ``video_id, creator_id, duration_s, log_followers,
    post_hour``. Set ``beta_engagement = beta_aversion = 0`` for a null
    dataset where brain features carry no signal.
    """
    rng = np.random.default_rng(seed)
    manifest, ts_frames, ret_frames = [], [], []
    for c in range(n_creators):
        creator = f"creator_{c:02d}"
        creator_effect = np.exp(rng.normal(0, 0.3))
        log_followers = rng.normal(10, 1.5)
        for v in range(videos_per_creator):
            vid = f"{creator}_v{v:02d}"
            n = int(rng.integers(min_seconds, max_seconds + 1))
            t = np.arange(n)
            e, a = _ar1(rng, n), _ar1(rng, n, phi=0.7)
            cols = {}
            for name in NETWORK_NAMES:
                le, la = LOADINGS.get(name, (0.0, 0.0))
                cols[name] = le * e + la * a + noise * rng.normal(0, 0.6, n)
            cols["global"] = np.mean(list(cols.values()), axis=0)
            cols["pattern_shift"] = rng.gamma(2.0, 0.05, n) + 0.1 * np.clip(a, 0, None)
            cols["pattern_shift"][0] = 0.0
            ts_frames.append(pd.DataFrame({"video_id": vid, "t": t, **cols}))

            base = 0.12 * np.exp(-t / 2.5) + 0.012
            hazard = base * creator_effect * np.exp(-beta_engagement * e + beta_aversion * a)
            hazard = np.clip(hazard * np.exp(rng.normal(0, 0.15, n)), 0, 0.95)
            retention = np.cumprod(1 - hazard)
            ret_frames.append(pd.DataFrame({"video_id": vid, "t": t, "retention": retention}))
            manifest.append(
                {
                    "video_id": vid,
                    "creator_id": creator,
                    "duration_s": float(n),
                    "log_followers": log_followers,
                    "post_hour": int(rng.integers(0, 24)),
                }
            )
    return (
        pd.DataFrame(manifest),
        pd.concat(ts_frames, ignore_index=True),
        pd.concat(ret_frames, ignore_index=True),
    )


def make_vertex_predictions(
    vertex_labels: np.ndarray, network_ts: pd.DataFrame, noise: float = 0.5, seed: int = 0
) -> np.ndarray:
    """Expand network-level time series to fake vertex-level predictions.

    Every vertex in a network gets that network's signal plus noise; useful
    for testing :mod:`neuroengage.rois` round trips.
    """
    from .rois import NETWORKS

    rng = np.random.default_rng(seed)
    n_t, n_v = len(network_ts), len(vertex_labels)
    preds = rng.normal(0, noise, (n_t, n_v))
    labels = np.asarray(vertex_labels, dtype=object)
    for net in NETWORKS:
        mask = np.isin(labels, net.areas)
        preds[:, mask] += network_ts[net.name].to_numpy()[:, None]
    return preds
