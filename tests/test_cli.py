import pandas as pd

from neuroengage.cli import main
from neuroengage.synthetic import make_dataset


def test_train_and_score_roundtrip(tmp_path):
    manifest, ts, ret = make_dataset(n_creators=6, videos_per_creator=6, seed=5)
    manifest.to_csv(tmp_path / "videos.csv", index=False)
    ts.to_csv(tmp_path / "ts.csv", index=False)
    ret.to_csv(tmp_path / "retention.csv", index=False)
    main([
        "train", "--manifest", str(tmp_path / "videos.csv"), "--timeseries", str(tmp_path / "ts.csv"),
        "--retention", str(tmp_path / "retention.csv"), "--baseline-cols", "duration_s",
        "--folds", "3", "--out", str(tmp_path / "run"),
    ])
    main([
        "score", "--models", str(tmp_path / "run" / "models.joblib"),
        "--timeseries", str(tmp_path / "ts.csv"), "--manifest", str(tmp_path / "videos.csv"),
        "--out", str(tmp_path / "scores"),
    ])
    scores = pd.read_csv(tmp_path / "scores" / "video_scores.csv")
    assert len(scores) == len(manifest)
    risk = pd.read_csv(tmp_path / "scores" / "dropoff_risk.csv")
    assert len(risk) == len(ts)
