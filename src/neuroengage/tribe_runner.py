"""Thin wrapper around Meta's TRIBE v2 for batch feature extraction.

TRIBE v2 (``facebookresearch/tribev2``) is an optional dependency: it needs a
GPU, gated access to Llama 3.2-3B on Hugging Face, and is released under
CC BY-NC 4.0. Everything downstream of this module works on the cached
``.npz`` files it writes, so modeling can run on a laptop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .rois import build_network_masks, load_hcp_vertex_labels, network_timeseries

logger = logging.getLogger(__name__)


@dataclass
class TribePrediction:
    """Predicted cortical response for one video.

    ``preds`` is ``(n_timesteps, n_vertices)``; ``times`` holds the start
    time in seconds of each row. TRIBE v2 already shifts predictions by 5 s
    to undo hemodynamic lag, so row ``t`` lines up with stimulus second ``t``.
    """

    video_id: str
    preds: np.ndarray
    times: np.ndarray

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, video_id=self.video_id, preds=self.preds, times=self.times)

    @classmethod
    def load(cls, path: str | Path) -> "TribePrediction":
        with np.load(path, allow_pickle=False) as data:
            return cls(str(data["video_id"]), data["preds"], data["times"])


class TribeExtractor:
    """Runs TRIBE v2 over videos and caches predictions and network time series."""

    def __init__(
        self,
        checkpoint: str = "facebook/tribev2",
        cache_folder: str | Path = "./cache/tribe",
        device: str = "auto",
    ) -> None:
        try:
            from tribev2 import TribeModel
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "TRIBE v2 is not installed. See the README section "
                "'Installing TRIBE v2'."
            ) from exc
        self.model = TribeModel.from_pretrained(
            checkpoint, cache_folder=str(cache_folder), device=device
        )
        # Keep empty (silent/blank) seconds so rows stay on a regular grid.
        self.model.remove_empty_segments = False
        self._masks: dict[str, np.ndarray] | None = None

    @property
    def masks(self) -> dict[str, np.ndarray]:
        if self._masks is None:
            self._masks = build_network_masks(load_hcp_vertex_labels("fsaverage5"))
        return self._masks

    def predict(self, video_id: str, video_path: str | Path) -> TribePrediction:
        events = self.model.get_events_dataframe(video_path=str(video_path))
        preds, segments = self.model.predict(events=events, verbose=False)
        times = np.array([float(s.start) for s in segments])
        order = np.argsort(times)
        times = times[order] - times.min()
        return TribePrediction(video_id, preds[order].astype(np.float32), times)

    def run(
        self,
        manifest: pd.DataFrame,
        out_dir: str | Path,
        overwrite: bool = False,
        skip_errors: bool = True,
        table_name: str = "network_timeseries",
    ) -> pd.DataFrame:
        """Extract every video in ``manifest`` (columns ``video_id``, ``path``).

        Re-running resumes: videos with a cached ``.npz`` are not recomputed.
        Videos that fail are logged to ``failures.csv`` and skipped.

        Writes ``<out_dir>/preds/<video_id>.npz`` (full vertex predictions) and
        returns the long-format network time series, also saved as
        ``<out_dir>/network_timeseries.parquet`` (or ``.csv`` without pyarrow).
        """
        out_dir = Path(out_dir)
        (out_dir / "preds").mkdir(parents=True, exist_ok=True)
        frames, failures = [], []
        total = len(manifest)
        for i, row in enumerate(manifest.itertuples(index=False), 1):
            vid = str(row.video_id)
            cache = out_dir / "preds" / f"{vid}.npz"
            try:
                if cache.exists() and not overwrite:
                    pred = TribePrediction.load(cache)
                else:
                    logger.info("TRIBE v2 [%d/%d]: %s", i, total, vid)
                    pred = self.predict(vid, row.path)
                    pred.save(cache)
            except Exception as exc:  # corrupt/silent/too-short videos
                if not skip_errors:
                    raise
                logger.warning("Skipping %s: %s", vid, exc)
                failures.append({"video_id": vid, "error": repr(exc)})
                continue
            frames.append(to_long(pred, self.masks))
        if failures:
            name = "failures.csv" if table_name == "network_timeseries" else f"failures_{table_name}.csv"
            pd.DataFrame(failures).to_csv(out_dir / name, index=False)
        if not frames:
            raise RuntimeError("TRIBE v2 failed on every video; see failures.csv")
        long = pd.concat(frames, ignore_index=True)
        save_table(long, out_dir / table_name)
        return long


def to_long(pred: TribePrediction, masks: dict[str, np.ndarray]) -> pd.DataFrame:
    """Network time series for one prediction, with ``video_id`` and ``t`` columns."""
    ts = network_timeseries(pred.preds, masks)
    ts.insert(0, "t", pred.times)
    ts.insert(0, "video_id", pred.video_id)
    return ts


def save_table(df: pd.DataFrame, stem: str | Path) -> Path:
    stem = Path(stem)
    try:
        path = stem.with_suffix(".parquet")
        df.to_parquet(path, index=False)
    except ImportError:
        path = stem.with_suffix(".csv")
        df.to_csv(path, index=False)
    return path


def load_table(stem: str | Path) -> pd.DataFrame:
    stem = Path(stem)
    if stem.suffix in {".parquet", ".csv"}:
        candidates = [stem]
    else:
        candidates = [stem.with_suffix(".parquet"), stem.with_suffix(".csv")]
    for path in candidates:
        if path.exists():
            if path.suffix == ".parquet":
                return pd.read_parquet(path)
            return pd.read_csv(path, dtype={"video_id": str})
    raise FileNotFoundError(f"No table at {stem} (.parquet or .csv)")


def merge_shards(out_dir: str | Path, pattern: str = "network_timeseries_shard*") -> Path:
    """Concatenate per-shard time series tables into ``network_timeseries``."""
    out_dir = Path(out_dir)
    paths = sorted(p for p in out_dir.glob(pattern) if p.suffix in {".parquet", ".csv"})
    if not paths:
        raise FileNotFoundError(f"No {pattern} tables in {out_dir}")
    frames = [load_table(p) for p in paths]
    merged = pd.concat(frames, ignore_index=True).drop_duplicates(["video_id", "t"])
    return save_table(merged, out_dir / "network_timeseries")
