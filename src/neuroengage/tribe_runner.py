"""Thin wrapper around Meta's TRIBE v2 for batch feature extraction.

TRIBE v2 (``facebookresearch/tribev2``) is an optional dependency: it needs a
GPU, gated access to Llama 3.2-3B on Hugging Face, and is released under
CC BY-NC 4.0. Everything downstream of this module works on the cached
``.npz`` files it writes, so modeling can run on a laptop.
"""

from __future__ import annotations

import logging
import os
import time
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
    duration: float | None = None  # stimulus length in seconds, if known

    def trimmed(self) -> "TribePrediction":
        """Drop rows that start at or after the end of the stimulus.

        TRIBE predicts in fixed-length windows (100 s by default); with
        empty segments kept, the window's padding after the video ends comes
        back as extra rows that describe no stimulus at all.
        """
        if self.duration is None:
            return self
        keep = self.times < self.duration - 1e-6
        return TribePrediction(self.video_id, self.preds[keep], self.times[keep], self.duration)

    def save(self, path: str | Path) -> None:
        extra = {} if self.duration is None else {"duration": self.duration}
        np.savez_compressed(path, video_id=self.video_id, preds=self.preds, times=self.times, **extra)

    @classmethod
    def load(cls, path: str | Path) -> "TribePrediction":
        with np.load(path, allow_pickle=False) as data:
            duration = float(data["duration"]) if "duration" in data.files else None
            return cls(str(data["video_id"]), data["preds"], data["times"], duration).trimmed()


class TribeExtractor:
    """Runs TRIBE v2 over videos and caches predictions and network time series."""

    def __init__(
        self,
        checkpoint: str = "facebook/tribev2",
        cache_folder: str | Path = "./cache/tribe",
        device: str = "auto",
        low_memory: bool | None = None,
        precision: str = "fp32",
        num_workers: int | None = None,
    ) -> None:
        """Load TRIBE v2.

        ``low_memory`` loads the Llama 3.2-3B text model in float16 straight onto
        the GPU (TRIBE's ``device="accelerate"`` mode, needs the ``accelerate``
        package) instead of float32 via CPU RAM: ~6.4 GB instead of ~13 GB, which
        is what lets it run on a free Colab T4 (15 GB GPU, ~12 GB RAM). Text
        features then differ slightly from TRIBE's float32 setup. ``None``
        enables it automatically on GPUs or machines with less than 30 GB.

        ``precision="fp16"`` runs feature extraction and TRIBE under CUDA
        autocast (float16). On T4-class GPUs this is much faster for the V-JEPA 2
        video model, which dominates run time, but features differ slightly from
        TRIBE's float32 setup and large models can overflow in float16: a video
        whose predictions come out non-finite fails instead of being saved. Use a
        separate ``cache_folder`` per precision, since TRIBE's feature cache does
        not record it. Check agreement with float32 on a few videos first.

        ``num_workers`` sets TRIBE's data-loader processes (default: up to 4).
        """
        if precision not in ("fp32", "fp16"):
            raise ValueError("precision must be 'fp32' or 'fp16'")
        self.precision = precision
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        try:
            from tribev2 import TribeModel
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "TRIBE v2 is not installed. See the README section "
                "'Installing TRIBE v2'."
            ) from exc
        if low_memory is None:
            low_memory = needs_low_memory()
        self.low_memory = low_memory
        config_update: dict = {"data.num_workers": num_workers or min(4, os.cpu_count() or 1)}
        if low_memory:
            try:
                import accelerate  # noqa: F401
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "low_memory mode needs the 'accelerate' package: pip install accelerate"
                ) from exc
            config_update["data.text_feature.device"] = "accelerate"
            logger.info("TRIBE v2: low-memory mode (Llama 3.2 in float16 on GPU)")
        self.model = TribeModel.from_pretrained(
            checkpoint, cache_folder=str(cache_folder), device=device,
            config_update=config_update,
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
        import contextlib

        precision = getattr(self, "precision", "fp32")
        autocast = contextlib.nullcontext()
        if precision == "fp16":
            import torch

            if torch.cuda.is_available():
                autocast = torch.autocast("cuda", dtype=torch.float16)
        with autocast:
            events = self.model.get_events_dataframe(video_path=str(video_path))
            preds, segments = self.model.predict(events=events, verbose=False)
        preds = np.asarray(preds, dtype=np.float32)
        if not np.isfinite(preds).all():
            raise FloatingPointError(
                f"non-finite predictions ({precision}); use precision='fp32' for this video"
            )
        times = np.array([float(s.start) for s in segments])
        order = np.argsort(times)
        t0 = times.min()
        duration = stimulus_duration(events, t0)
        pred = TribePrediction(video_id, preds[order], times[order] - t0, duration)
        return pred.trimmed()

    def run(
        self,
        manifest: pd.DataFrame,
        out_dir: str | Path,
        overwrite: bool = False,
        skip_errors: bool = True,
        table_name: str = "network_timeseries",
        deadline: float | None = None,
    ) -> pd.DataFrame:
        """Extract every video in ``manifest`` (columns ``video_id``, ``path``).

        Re-running resumes: videos with a cached ``.npz`` are not recomputed.
        Videos that fail are logged to ``failures.csv`` and skipped. With a
        ``deadline`` (``time.time()`` value), no new video is started after it;
        already-cached videos are still included.

        Writes ``<out_dir>/preds/<video_id>.npz`` (full vertex predictions) and
        returns the long-format network time series, also saved as
        ``<out_dir>/network_timeseries.parquet`` (or ``.csv`` without pyarrow).
        """
        out_dir = Path(out_dir)
        (out_dir / "preds").mkdir(parents=True, exist_ok=True)
        frames, failures = [], []
        last_exc: Exception | None = None
        total = len(manifest)
        deferred = 0
        for i, row in enumerate(manifest.itertuples(index=False), 1):
            vid = str(row.video_id)
            cache = out_dir / "preds" / f"{vid}.npz"
            try:
                if cache.exists() and not overwrite:
                    pred = TribePrediction.load(cache)
                elif deadline is not None and time.time() > deadline:
                    deferred += 1
                    continue
                else:
                    logger.info("TRIBE v2 [%d/%d]: %s", i, total, vid)
                    pred = self.predict(vid, row.path)
                    pred.save(cache)
            except Exception as exc:  # corrupt/silent/too-short videos
                if not skip_errors:
                    raise
                chain = describe_exception(exc)
                logger.warning("Skipping %s: %s", vid, chain)
                failures.append({"video_id": vid, "error": chain})
                last_exc = exc
                continue
            frames.append(to_long(pred, self.masks))
        if deferred:
            logger.warning("Time budget reached: %d videos not started (re-run to continue)", deferred)
        if failures:
            name = "failures.csv" if table_name == "network_timeseries" else f"failures_{table_name}.csv"
            pd.DataFrame(failures).to_csv(out_dir / name, index=False)
        if not frames:
            # Chain the last error so the notebook shows the original traceback.
            raise RuntimeError(
                f"TRIBE v2 failed on every video. Last error: "
                f"{failures[-1]['error'] if failures else 'unknown'}"
            ) from last_exc
        long = pd.concat(frames, ignore_index=True)
        save_table(long, out_dir / table_name)
        return long


def stimulus_duration(events: pd.DataFrame, t0: float = 0.0) -> float | None:
    """End of the last event (video/audio/word) relative to ``t0``, in seconds."""
    if not {"start", "duration"} <= set(events.columns):
        return None
    end = pd.to_numeric(events["start"], errors="coerce") + pd.to_numeric(
        events["duration"], errors="coerce"
    )
    end = end.max()
    return None if pd.isna(end) else float(end) - t0


def needs_low_memory(threshold_gb: float = 30.0) -> bool:
    """True when the GPU or system RAM is too small for TRIBE's float32 Llama."""
    try:
        ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        ram_gb = float("inf")
    gpu_gb = float("inf")
    try:
        import torch

        if torch.cuda.is_available():
            gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:  # pragma: no cover - torch missing or broken
        pass
    return min(ram_gb, gpu_gb) < threshold_gb


def describe_exception(exc: BaseException) -> str:
    """``Outer -> ... -> root cause`` for wrapped errors.

    TRIBE's feature extractors re-raise loading failures as a generic
    "Model loading went wrong"; the useful message is the root cause.
    """
    parts, seen = [], set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        parts.append(f"{type(cur).__name__}: {cur}".strip())
        cur = cur.__cause__ or cur.__context__
    return " -> ".join(parts)


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
