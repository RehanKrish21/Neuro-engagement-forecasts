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
    res = run_score_training(manifest, ts, ["nawp", "ecr"], tmp_path / "sig", hook_seconds=5)
    v = res["video"].set_index(["target", "feature_set"])
    assert v.loc[("ecr", "baseline+brain"), "spearman_gain_vs_baseline"] > 0.2
    assert (tmp_path / "sig" / "summary.png").exists()

    manifest, ts = make_score_dataset(
        seed=4, n_creators=10, videos_per_creator=12, beta_engagement=0.0, beta_aversion=0.0
    )
    res = run_score_training(manifest, ts, ["nawp", "ecr"], tmp_path / "null", hook_seconds=5)
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
          str(tmp_path / "ts.csv"), "--folds", "3", "--out", str(tmp_path / "run")])
    metrics = pd.read_csv(tmp_path / "run" / "video_level_metrics.csv")
    assert set(metrics["target"]) == {"nawp", "ecr"}
