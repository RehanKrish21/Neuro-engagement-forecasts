import numpy as np
import pandas as pd

from neuroengage.tribe_runner import TribeExtractor, TribePrediction, load_table


def test_run_skips_failures_and_caches(tmp_path):
    ext = TribeExtractor.__new__(TribeExtractor)  # bypass loading the real model
    n_v = 12
    ext._masks = {"a": np.arange(n_v) < 6, "b": np.arange(n_v) >= 6}
    calls = []

    def fake_predict(vid, path):
        calls.append(vid)
        if vid == "bad":
            raise RuntimeError("no audio stream")
        return TribePrediction(vid, np.random.default_rng(0).normal(size=(5, n_v)), np.arange(5.0))

    ext.predict = fake_predict
    manifest = pd.DataFrame({"video_id": ["ok1", "bad", "ok2"], "path": ["x", "y", "z"]})
    long = ext.run(manifest, tmp_path)
    assert set(long["video_id"]) == {"ok1", "ok2"}
    assert pd.read_csv(tmp_path / "failures.csv")["video_id"].tolist() == ["bad"]
    assert len(load_table(tmp_path / "network_timeseries")) == 10

    calls.clear()
    ext.run(manifest, tmp_path)  # second run: cached videos are not recomputed
    assert calls == ["bad"]


def test_merge_shards(tmp_path):
    from neuroengage.tribe_runner import merge_shards, save_table

    for k in range(3):
        df = pd.DataFrame({"video_id": f"v{k}", "t": range(4), "a": np.arange(4.0)})
        save_table(df, tmp_path / f"network_timeseries_shard{k:03d}")
    merged = load_table(merge_shards(tmp_path))
    assert sorted(merged["video_id"].unique()) == ["v0", "v1", "v2"]
    assert len(merged) == 12
