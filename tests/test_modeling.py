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


def test_pooled_metrics_ci_and_pairwise():
    from neuroengage.modeling import pairwise_accuracy, pooled_metrics

    rng = np.random.default_rng(0)
    y = rng.normal(size=300)
    oof = pd.DataFrame({"video_id": [str(i) for i in range(300)], "y_true": y,
                        "pred__baseline": y + rng.normal(0, 2, 300),
                        "pred__good": y + rng.normal(0, 0.3, 300),
                        "pred__noise": rng.normal(size=300)})
    m = pooled_metrics(oof, n_boot=200).set_index("feature_set")
    assert m.loc["good", "spearman_lo"] > m.loc["baseline", "spearman_hi"] - 0.2
    assert m.loc["good", "gain_lo"] > 0 and m.loc["good", "p_gain_positive"] > 0.99
    assert m.loc["noise", "pairwise_lo"] < 0.5 < m.loc["noise", "pairwise_hi"]
    for s in m.index:
        assert m.loc[s, "spearman_lo"] <= m.loc[s, "spearman"] <= m.loc[s, "spearman_hi"]
    assert pairwise_accuracy(np.array([1, 2, 3]), np.array([1, 2, 3])) == 1.0
    assert pairwise_accuracy(np.array([1, 2, 3]), np.array([3, 2, 1])) == 0.0


def test_ensemble_and_compact_sets_reported(tmp_path):
    from neuroengage.pipeline import PRIMARY_SET, run_score_training
    from neuroengage.synthetic import make_score_dataset

    manifest, ts = make_score_dataset(seed=9, n_creators=6, videos_per_creator=10)
    res = run_score_training(manifest, ts, ["ecr"], tmp_path, hook_seconds=5, n_boot=50)
    pooled = pd.read_csv(tmp_path / "pooled_metrics.csv")
    assert {"brain_compact", PRIMARY_SET, "avg(baseline,brain_compact)"} <= set(pooled["feature_set"])
    report = (tmp_path / "report.md").read_text()
    assert "(primary)" in report and "pairwise" in report
    assert res["info"]["n_brain_compact_features"] < res["info"]["n_brain_features"]


def test_confirmatory_subset_excludes_exploratory_ids(tmp_path):
    from neuroengage.pipeline import run1_exploratory_ids, run_score_training
    from neuroengage.synthetic import make_score_dataset

    assert len(run1_exploratory_ids()) == 135
    manifest, ts = make_score_dataset(seed=10, n_creators=8, videos_per_creator=10)
    explo = set(manifest["video_id"][:30])
    run_score_training(manifest, ts, ["ecr"], tmp_path, hook_seconds=5, n_boot=50, exploratory_ids=explo)
    pooled = pd.read_csv(tmp_path / "pooled_metrics.csv")
    assert set(pooled["subset"]) == {"all videos", "confirmatory: 50 new videos"}
    assert "confirmatory" in (tmp_path / "report.md").read_text()
