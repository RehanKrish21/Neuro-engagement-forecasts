"""SnapUGC (Li et al., ECCV 2024) loader.

SnapUGC is ~90-120k Snapchat Spotlight short videos (5-60 s, each with
more than 2,000 views), labelled with two engagement metrics:

* **NAWP**, normalized average watch percentage: how much of the video people
  watch, normalized so different lengths are comparable.
* **ECR**, engagement continuation rate: probability a viewer watches at
  least 5 seconds. Essentially a hook score.

Download: https://github.com/dasongli1/SnapUGC_Engagement (dataset/readme.md
links the train and val label files). Those are CSVs, not video archives:
``train_data.csv`` has columns ``Id, Title, Description, Download_link, ECR``
(106,192 rows; no NAWP), one Snapchat CDN link per video. Use
:func:`sample_candidates` + :func:`download_videos` to fetch only the videos
you need as ``<videos_dir>/<Id>.mp4``. Label column names are auto-detected
case-insensitively and can be overridden.

Check the dataset's terms before use. SnapUGC has no creator IDs, so
evaluation uses plain shuffled K-fold rather than creator-grouped folds.
"""

from __future__ import annotations

import logging
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ID_ALIASES = ("id", "video_id", "videoid", "vid")
URL_ALIASES = ("download_link", "url", "video_url", "link", "download_url")
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
    url_col = _find(raw.columns, URL_ALIASES)
    if url_col is not None:
        out["url"] = raw[url_col]
    return out.dropna(subset=targets, how="all").drop_duplicates("video_id")


def available_targets(df: pd.DataFrame) -> list[str]:
    """Engagement targets present in ``df`` (e.g. only ``ecr`` for the train CSV)."""
    return [t for t in TARGET_ALIASES if t in df.columns and df[t].notna().any()]


def _stratified(df: pd.DataFrame, n: int, seed: int, stratify_on: str | None, n_bins: int) -> pd.DataFrame:
    if n >= len(df):
        return df
    stratify_on = stratify_on or next(iter(available_targets(df)), None)
    if stratify_on and stratify_on in df and df[stratify_on].notna().all():
        bins = pd.qcut(df[stratify_on].rank(method="first"), n_bins, labels=False)
        per_bin = int(np.ceil(n / n_bins))
        return (
            df.groupby(bins, group_keys=False)
            .apply(lambda g: g.sample(min(len(g), per_bin), random_state=seed))
            .sample(frac=1.0, random_state=seed)
            .head(n)
        )
    return df.sample(n=n, random_state=seed)


def done_ids(preds_dir: str | Path | None) -> set[str]:
    """IDs that already have TRIBE predictions (``<preds_dir>/<id>.npz``)."""
    if not preds_dir or not Path(preds_dir).is_dir():
        return set()
    return {p.stem for p in Path(preds_dir).glob("*.npz")}


def sample_candidates(
    labels: pd.DataFrame,
    n: int,
    seed: int = 0,
    stratify_on: str | None = None,
    n_bins: int = 10,
    exclude_ids: set[str] | None = None,
    keep_without_url: bool = False,
) -> pd.DataFrame:
    """Stratified sample of ``n`` labelled videos to download.

    Ask for more than you need (e.g. 1.3x): some CDN links will be dead, and
    :func:`build_manifest` then picks the final set from what downloaded.
    ``exclude_ids`` (e.g. videos already processed in an earlier run) are
    never sampled.
    """
    df = labels
    if exclude_ids:
        df = df[~df["video_id"].isin(exclude_ids)]
    if "url" in df and not keep_without_url:
        df = df[df["url"].notna() & df["url"].astype(str).str.startswith("http")]
    return _stratified(df, n, seed, stratify_on, n_bins).reset_index(drop=True)


def _is_mp4(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(12)
    except OSError:
        return False
    return len(head) >= 8 and head[4:8] in (b"ftyp", b"moov", b"mdat", b"free", b"wide")


def _fetch(url: str, dest: Path, timeout: float, retries: int) -> tuple[str, int]:
    tmp = dest.with_suffix(".part")
    last = "error"
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as out:
                while chunk := resp.read(1 << 16):
                    out.write(chunk)
            if not _is_mp4(tmp):
                tmp.unlink(missing_ok=True)
                return "not_mp4", 0
            size = tmp.stat().st_size
            tmp.replace(dest)
            return "ok", size
        except urllib.error.HTTPError as exc:
            tmp.unlink(missing_ok=True)
            if exc.code in (403, 404, 410):  # expired/removed: retrying will not help
                return f"http_{exc.code}", 0
            last = f"http_{exc.code}"
        except Exception as exc:  # timeouts, resets
            tmp.unlink(missing_ok=True)
            last = type(exc).__name__
        time.sleep(1.5 * (attempt + 1))
    return last, 0


HF_MIRROR_REPOS = {
    # Unofficial Hugging Face re-upload of SnapUGC video files (user tcm03), named
    # <SnapUGC Id>.mp4. Checked 2026-11-24: 121,506 unique IDs, full-length videos,
    # all 135 run-1 IDs present. Its own labels are 3 coarse classes, so labels
    # always come from the official SnapUGC CSV; only the video files are used.
    "tcm03": ("SnapUGC_0", "SnapUGC_1", "SnapUGC_2", "SnapUGC_3", "SnapUGC"),
}


def hf_mirror_index(
    mirror: str = "tcm03",
    cache_path: str | Path | None = None,
    fetch=None,
) -> dict[str, str]:
    """Map SnapUGC video ID -> download URL in a Hugging Face mirror.

    Built from the Hub API's file listings (a few MB of JSON, no video
    downloads) and optionally cached to ``cache_path``. ``fetch(url) -> dict``
    can be injected for tests.
    """
    import json

    if cache_path and Path(cache_path).exists():
        return json.loads(Path(cache_path).read_text())
    if fetch is None:
        def fetch(url):
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
    index: dict[str, str] = {}
    for repo in HF_MIRROR_REPOS[mirror]:
        info = fetch(f"https://huggingface.co/api/datasets/{mirror}/{repo}")
        for sib in info.get("siblings", []):
            name = sib["rfilename"]
            if name.endswith(".mp4"):
                vid = name.rsplit("/", 1)[-1][:-4]
                index.setdefault(vid, f"https://huggingface.co/datasets/{mirror}/{repo}/resolve/main/{name}")
    if cache_path:
        Path(cache_path).write_text(json.dumps(index))
    logger.info("HF mirror %s: %d videos indexed", mirror, len(index))
    return index


def download_videos(
    candidates: pd.DataFrame,
    videos_dir: str | Path,
    workers: int = 8,
    timeout: float = 60.0,
    retries: int = 2,
    fallback_urls: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Download ``<videos_dir>/<video_id>.mp4`` for each row with a ``url``.

    Resumable: valid files already on disk are skipped. If the official link
    fails and ``fallback_urls`` (e.g. :func:`hf_mirror_index`) has the video,
    it is fetched from there instead (status ``ok_mirror``). Returns one row
    per video with ``status`` (``ok``, ``ok_mirror``, ``exists``, ``http_403``,
    ``not_mp4``, ...) and appends it to ``<videos_dir>/_download_log.csv``.
    """
    if "url" not in candidates:
        raise KeyError("candidates need a 'url' column (load_labels keeps Download_link as url)")
    videos_dir = Path(videos_dir)
    videos_dir.mkdir(parents=True, exist_ok=True)

    def one(row) -> dict:
        dest = videos_dir / f"{row.video_id}.mp4"
        if dest.exists() and _is_mp4(dest):
            return {"video_id": row.video_id, "status": "exists", "bytes": dest.stat().st_size}
        status, size = ("no_url", 0)
        if isinstance(row.url, str) and row.url.startswith("http"):
            status, size = _fetch(str(row.url), dest, timeout, retries)
        if status != "ok" and fallback_urls and row.video_id in fallback_urls:
            m_status, m_size = _fetch(fallback_urls[row.video_id], dest, timeout, retries)
            if m_status == "ok":
                return {"video_id": row.video_id, "status": "ok_mirror", "bytes": m_size}
            status = f"{status}|mirror_{m_status}"
        return {"video_id": row.video_id, "status": status, "bytes": size}

    rows = list(candidates[["video_id", "url"]].itertuples(index=False))
    results, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, res in enumerate(pool.map(one, rows), 1):
            results.append(res)
            if i % 50 == 0 or i == len(rows):
                ok = sum(r["status"] in ("ok", "ok_mirror", "exists") for r in results)
                logger.info("SnapUGC download %d/%d (%d ok, %.0f s)", i, len(rows), ok, time.time() - t0)
    log = pd.DataFrame(results)
    log_path = videos_dir / "_download_log.csv"
    log.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
    return log


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
    stratify_on: str | None = None,
    n_bins: int = 10,
    include_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Manifest of labelled videos present on disk, optionally subsampled.

    ``include_ids`` are videos that already have predictions (from an earlier
    run): they are always kept, even if their video file is no longer on
    disk, and new videos fill the manifest up to ``n``. That way runs
    accumulate instead of starting over.

    TRIBE v2 costs real GPU time per video, so you will usually start with a
    few thousand videos. Subsampling is stratified across ``n_bins``
    quantile bins of ``stratify_on`` so the subset keeps the full range of
    engagement rather than piling up around the median.
    """
    videos_dir = Path(videos_dir)
    df = labels.copy()
    df["path"] = [str(videos_dir / f"{vid}.mp4") for vid in df["video_id"]]
    include_ids = set(include_ids or ())
    keep = df["video_id"].isin(include_ids).to_numpy()
    exists = np.array([Path(p).is_file() for p in df["path"]])
    if not (exists | keep).any():
        raise FileNotFoundError(f"None of the labelled videos found under {videos_dir}")
    logger.info("SnapUGC: %d of %d labelled videos are on disk, %d already processed",
                int(exists.sum()), len(df), int(keep.sum()))
    kept, rest = df[keep], df[exists & ~keep]
    if n is not None:
        rest = _stratified(rest, max(n - len(kept), 0), seed, stratify_on, n_bins)
    df = pd.concat([kept, rest])
    cols = ["video_id", "path", *[c for c in ("nawp", "ecr") if c in df]]
    manifest = df[cols].merge(text_baselines(df), on="video_id")
    return manifest.reset_index(drop=True)
