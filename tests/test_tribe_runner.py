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


def test_failure_reports_root_cause(tmp_path):
    import pytest

    ext = TribeExtractor.__new__(TribeExtractor)
    ext._masks = {"a": np.ones(4, bool)}

    def fake_predict(vid, path):
        try:
            raise OSError("You are trying to access a gated repo")
        except OSError as inner:
            raise RuntimeError("Model loading went wrong") from inner

    ext.predict = fake_predict
    with pytest.raises(RuntimeError, match="gated repo") as info:
        ext.run(pd.DataFrame({"video_id": ["v"], "path": ["x"]}), tmp_path)
    assert isinstance(info.value.__cause__, RuntimeError)
    err = pd.read_csv(tmp_path / "failures.csv")["error"].iloc[0]
    assert "Model loading went wrong -> OSError: You are trying to access a gated repo" in err


def test_needs_low_memory_threshold():
    from neuroengage.tribe_runner import needs_low_memory

    assert needs_low_memory(threshold_gb=1e9) is True
    assert needs_low_memory(threshold_gb=0.0) is False


class _Seg:
    def __init__(self, start):
        self.start = start


class _FakeTribeModel:
    """Mimics TRIBE: predicts a fixed 100 s window regardless of video length."""

    def get_events_dataframe(self, video_path):
        return pd.DataFrame({"type": ["Video", "Word"], "start": [0.0, 3.0], "duration": [52.3, 0.4]})

    def predict(self, events, verbose=False):
        return np.zeros((100, 8)), [_Seg(float(t)) for t in range(100)]


def test_predict_trims_window_padding(tmp_path):
    ext = TribeExtractor.__new__(TribeExtractor)
    ext.model = _FakeTribeModel()
    pred = ext.predict("v", "clip.mp4")
    assert pred.preds.shape[0] == 53  # seconds 0..52 of a 52.3 s video
    assert pred.times[-1] == 52.0 and pred.duration == 52.3
    pred.save(tmp_path / "v.npz")
    loaded = TribePrediction.load(tmp_path / "v.npz")
    assert loaded.preds.shape[0] == 53 and loaded.duration == 52.3


def test_load_trims_untrimmed_cache_with_duration(tmp_path):
    raw = TribePrediction("v", np.zeros((100, 4)), np.arange(100.0), duration=10.0)
    np.savez_compressed(tmp_path / "v.npz", video_id="v", preds=raw.preds, times=raw.times, duration=10.0)
    assert TribePrediction.load(tmp_path / "v.npz").preds.shape[0] == 10


def test_nonfinite_predictions_fail_and_deadline_defers(tmp_path):
    class NaNModel(_FakeTribeModel):
        def predict(self, events, verbose=False):
            out = np.zeros((100, 8))
            out[3, 2] = np.nan
            return out, [_Seg(float(t)) for t in range(100)]

    import pytest

    ext = TribeExtractor.__new__(TribeExtractor)
    ext.model = NaNModel()
    with pytest.raises(FloatingPointError):
        ext.predict("v", "clip.mp4")

    ext = TribeExtractor.__new__(TribeExtractor)
    ext._masks = {"a": np.ones(8, bool)}
    ext.model = _FakeTribeModel()
    manifest = pd.DataFrame({"video_id": ["a", "b", "c"], "path": ["x", "y", "z"]})
    ext.run(manifest.head(1), tmp_path)                       # "a" gets cached
    long = ext.run(manifest, tmp_path, deadline=0.0)          # past deadline: only cached "a"
    assert set(long["video_id"]) == {"a"}
    assert not (tmp_path / "failures.csv").exists()           # deferred is not a failure
