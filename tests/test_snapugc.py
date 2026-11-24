import numpy as np
import pandas as pd
import pytest

from neuroengage.cli import main
from neuroengage.pipeline import run_score_training
from neuroengage.snapugc import build_manifest, load_labels
from neuroengage.synthetic import make_score_dataset


def _labels_csv(tmp_path, n=40, cols=("Id", "Title", "Description", "NAWP", "ECR")):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            cols[0]: [f"vid{i:03d}" for i in range(n)],
            cols[1]: ["a short title"] * n,
            cols[2]: [""] * (n // 2) + ["some description here"] * (n - n // 2),
            cols[3]: rng.uniform(size=n),
            cols[4]: rng.uniform(size=n),
        }
    )
    path = tmp_path / "labels.csv"
    df.to_csv(path, index=False)
    return path, df


def test_load_labels_autodetects_columns(tmp_path):
    path, _ = _labels_csv(tmp_path)
    labels = load_labels(path)
    assert {"video_id", "nawp", "ecr", "title", "description"} <= set(labels.columns)
    assert labels["video_id"].iloc[0] == "vid000"


def test_load_labels_overrides_and_missing(tmp_path):
    path, _ = _labels_csv(tmp_path, cols=("key", "Title", "Description", "watch", "cont"))
    with pytest.raises(KeyError):
        load_labels(path)
    labels = load_labels(path, id_col="key", target_cols={"nawp": "watch", "ecr": "cont"})
    assert labels[["nawp", "ecr"]].notna().all().all()


def test_build_manifest_filters_missing_and_stratifies(tmp_path):
    path, df = _labels_csv(tmp_path, n=100)
    vids = tmp_path / "videos"
    vids.mkdir()
    for vid in df["Id"][:80]:  # 20 videos missing on disk
        (vids / f"{vid}.mp4").write_bytes(b"")
    m = build_manifest(load_labels(path), vids, n=40, seed=1)
    assert len(m) == 40
    assert m["video_id"].isin(df["Id"][:80]).all()
    assert {"path", "nawp", "ecr", "title_n_words", "has_description"} <= set(m.columns)
    # stratified: subsample spans the target range
    assert m["nawp"].min() < 0.2 and m["nawp"].max() > 0.8


def test_score_pipeline_recovers_signal_and_null(tmp_path):
    manifest, ts = make_score_dataset(seed=4, n_creators=10, videos_per_creator=12)
    res = run_score_training(manifest, ts, ["nawp", "ecr"], tmp_path / "sig", hook_seconds=5, n_boot=100)
    v = res["video"].set_index(["target", "feature_set"])
    assert v.loc[("ecr", "baseline+brain"), "spearman_gain_vs_baseline"] > 0.2
    assert (tmp_path / "sig" / "summary.png").exists()

    manifest, ts = make_score_dataset(
        seed=4, n_creators=10, videos_per_creator=12, beta_engagement=0.0, beta_aversion=0.0
    )
    res = run_score_training(manifest, ts, ["nawp", "ecr"], tmp_path / "null", hook_seconds=5, n_boot=100)
    v = res["video"].set_index(["target", "feature_set"])
    for target in ["nawp", "ecr"]:
        assert v.loc[(target, "baseline+brain"), "spearman_gain_vs_baseline"] < 0.1


def test_cli_prepare_and_train_scores(tmp_path):
    manifest, ts = make_score_dataset(seed=6, n_creators=6, videos_per_creator=10)
    vids = tmp_path / "videos"
    vids.mkdir()
    for vid in manifest["video_id"]:
        (vids / f"{vid}.mp4").write_bytes(b"")
    manifest.rename(columns={"video_id": "Id", "nawp": "NAWP", "ecr": "ECR"}).to_csv(
        tmp_path / "labels.csv", index=False
    )
    ts.to_csv(tmp_path / "ts.csv", index=False)
    main(["snapugc-prepare", "--labels", str(tmp_path / "labels.csv"),
          "--videos-dir", str(vids), "--out", str(tmp_path / "m.csv")])
    m = pd.read_csv(tmp_path / "m.csv")
    assert len(m) == len(manifest)
    main(["train-scores", "--manifest", str(tmp_path / "m.csv"), "--timeseries",
          str(tmp_path / "ts.csv"), "--folds", "3", "--n-boot", "100", "--out", str(tmp_path / "run")])
    metrics = pd.read_csv(tmp_path / "run" / "video_level_metrics.csv")
    assert set(metrics["target"]) == {"nawp", "ecr"}


def test_numeric_looking_ids_survive_csv_roundtrip(tmp_path):
    manifest, ts = make_score_dataset(seed=7, n_creators=6, videos_per_creator=10)
    ids = {v: f"{i:05d}" for i, v in enumerate(manifest["video_id"])}  # e.g. "00042"
    manifest["video_id"] = manifest["video_id"].map(ids)
    ts["video_id"] = ts["video_id"].map(ids)
    manifest.to_csv(tmp_path / "m.csv", index=False)
    ts.to_csv(tmp_path / "ts.csv", index=False)
    main(["train-scores", "--manifest", str(tmp_path / "m.csv"), "--timeseries",
          str(tmp_path / "ts.csv"), "--folds", "3", "--n-boot", "100", "--out", str(tmp_path / "run")])
    info = pd.read_json(tmp_path / "run" / "run_info.json", typ="series")
    assert info["n_videos"] == len(manifest)


def _serve(directory, monkeypatch):
    import functools
    import http.server
    import threading

    for var in ("no_proxy", "NO_PROXY"):
        monkeypatch.setenv(var, "127.0.0.1,localhost")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    handler.log_message = lambda *a, **k: None
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def test_download_videos_validates_resumes_and_logs(tmp_path, monkeypatch):
    from neuroengage.snapugc import download_videos, sample_candidates

    src = tmp_path / "cdn"
    src.mkdir()
    mp4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 2000
    for name in ("a", "b", "c"):
        (src / f"{name}.bin").write_bytes(mp4)
    (src / "html.bin").write_bytes(b"<html>expired</html>")
    server, base = _serve(src, monkeypatch)
    try:
        labels = pd.DataFrame({
            "video_id": ["a", "b", "c", "gone", "page", "nourl"],
            "ecr": [0.1, 0.5, 0.9, 0.3, 0.7, 0.2],
            "url": [f"{base}/a.bin", f"{base}/b.bin", f"{base}/c.bin", f"{base}/missing.bin",
                    f"{base}/html.bin", None],
        })
        cand = sample_candidates(labels, n=10)
        assert "nourl" not in set(cand["video_id"])
        out = tmp_path / "videos"
        log = download_videos(cand, out, workers=3, retries=0).set_index("video_id")["status"]
        assert log[["a", "b", "c"]].tolist() == ["ok", "ok", "ok"]
        assert log["gone"] == "http_404" and log["page"] == "not_mp4"
        assert sorted(p.name for p in out.glob("*.mp4")) == ["a.mp4", "b.mp4", "c.mp4"]
        assert not list(out.glob("*.part"))
        again = download_videos(cand, out, workers=3, retries=0).set_index("video_id")["status"]
        assert again[["a", "b", "c"]].tolist() == ["exists"] * 3
        assert len(pd.read_csv(out / "_download_log.csv")) == 10
    finally:
        server.shutdown()


def test_ecr_only_labels_flow(tmp_path):
    """The real train_data.csv has ECR but no NAWP."""
    from neuroengage.snapugc import available_targets

    manifest, ts = make_score_dataset(seed=8, n_creators=8, videos_per_creator=10)
    csv = tmp_path / "train_data.csv"
    manifest.drop(columns=["nawp"]).rename(columns={"video_id": "Id", "ecr": "ECR"}).assign(
        Title="t", Description="", Download_link="https://example.invalid/x"
    ).to_csv(csv, index=False)
    labels = load_labels(csv)
    assert available_targets(labels) == ["ecr"] and "url" in labels
    vids = tmp_path / "v"
    vids.mkdir()
    for v in labels["video_id"]:
        (vids / f"{v}.mp4").write_bytes(b"")
    m = build_manifest(labels, vids, n=60)
    assert len(m) == 60
    res = run_score_training(m, ts, ["nawp", "ecr"], tmp_path / "run", hook_seconds=5, n_boot=100)
    assert set(res["video"]["target"]) == {"ecr"}


def test_runs_accumulate_with_done_ids(tmp_path):
    from neuroengage.snapugc import done_ids, sample_candidates

    path, df = _labels_csv(tmp_path, n=100)
    labels = load_labels(path)
    labels["url"] = "https://example.invalid/x"
    vids = tmp_path / "videos"
    vids.mkdir()
    preds = tmp_path / "preds"
    preds.mkdir()
    first = set(df["Id"][:30])                       # processed in run 1; videos since deleted
    for v in first:
        (preds / f"{v}.npz").write_bytes(b"")
    for v in df["Id"][30:]:
        (vids / f"{v}.mp4").write_bytes(b"")
    assert done_ids(preds) == first and done_ids(tmp_path / "missing") == set()
    cand = sample_candidates(labels, 50, exclude_ids=first)
    assert not set(cand["video_id"]) & first
    m = build_manifest(labels, vids, n=60, include_ids=first)
    assert len(m) == 60 and first <= set(m["video_id"])        # all earlier work kept
    m_small = build_manifest(labels, vids, n=10, include_ids=first)
    assert set(m_small["video_id"]) == first                     # never drops finished work
