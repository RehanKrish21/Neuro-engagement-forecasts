import numpy as np
import pandas as pd

from neuroengage.modeling import creator_normalize
from neuroengage.pipeline import run_training
from neuroengage.synthetic import make_dataset


def test_creator_normalize():
    df = pd.DataFrame({"creator_id": ["a", "a", "b", "c", "c"], "y": [1, 3, 5, 10, 10]})
    z = creator_normalize(df, "y").to_numpy()
    np.testing.assert_allclose(z, [-1, 1, 0, 0, 0])


def _run(tmp_path, **kw):
    manifest, ts, ret = make_dataset(n_creators=10, videos_per_creator=8, seed=3, **kw)
    return run_training(
        manifest, ts, ret, tmp_path, baseline_cols=["duration_s", "log_followers"], n_splits=5
    )


def test_pipeline_recovers_planted_signal(tmp_path):
    res = _run(tmp_path)
    v = res["video"].set_index(["target", "feature_set"])
    assert v.loc[("avg_retention", "baseline+brain"), "spearman_gain_vs_baseline"] > 0.2
    d = res["dropoff"].set_index("feature_set")
    assert d.loc["position+brain", "within_video_spearman_gain_vs_position"] > 0.1
    for f in ["report.md", "models.joblib", "video_level_metrics.csv", "summary.png"]:
        assert (tmp_path / f).exists()


def test_pipeline_finds_nothing_in_null_data(tmp_path):
    res = _run(tmp_path, beta_engagement=0.0, beta_aversion=0.0)
    v = res["video"].set_index(["target", "feature_set"])
    assert v.loc[("avg_retention", "baseline+brain"), "spearman_gain_vs_baseline"] < 0.1
    d = res["dropoff"].set_index("feature_set")
    assert d.loc["position+brain", "within_video_spearman_gain_vs_position"] < 0.05
