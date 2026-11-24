import numpy as np
import pandas as pd

from neuroengage.cli import main
from neuroengage.pipeline import run_score_training
from neuroengage.synthetic import make_score_dataset
from neuroengage.tribe_runner import (
    TribeExtractor, TribePrediction, raw_feature_table, summarize_raw,
)


class Seg:
    def __init__(self, start, duration):
        self.start, self.duration = start, duration


def test_summarize_raw_trims_padding_and_hook_window():
    t = np.arange(20, dtype=np.float32)                         # 20 samples over a 10 s window (2 Hz)
    video = np.broadcast_to(t, (1, 2, 3, 20)).copy()             # B, L, H, T: value = sample index
    audio = np.broadcast_to(t, (1, 4, 20)).copy()                # no layer axis
    out = summarize_raw([({"video": video, "audio": audio}, [Seg(100.0, 10.0)])], t0=100.0, duration=6.0)
    assert out["video__mean"].shape == (6,) and out["audio__mean"].shape == (4,)
    assert np.allclose(out["video__mean"], np.arange(12).mean())  # t < 6 s -> samples 0..11
    assert np.allclose(out["video__hook"], np.arange(10).mean())  # t < 5 s -> samples 0..9
    assert out["video__present"][0] == 1.0


class FakeEncoder:
    feature_dims = {"text": (2, 3), "video": (2, 3)}

    def __init__(self):
        self.hooks = []

    def register_forward_pre_hook(self, fn):
        self.hooks.append(fn)
        enc = self

        class Handle:
            def remove(self):
                enc.hooks.remove(fn)
        return Handle()


class Batch:
    def __init__(self, data, segments):
        self.data, self.segments = data, segments


class FakeTribe:
    def __init__(self):
        self._model = FakeEncoder()

    def get_events_dataframe(self, video_path):
        return pd.DataFrame({"type": ["Video"], "start": [0.0], "duration": [8.0]})

    def predict(self, events, verbose=False):
        batch = Batch({"video": np.ones((1, 2, 3, 200), np.float32)}, [Seg(0.0, 100.0)])  # no speech
        for fn in list(self._model.hooks):
            fn(self._model, (batch,))
        return np.zeros((100, 4)), [Seg(float(t), 1.0) for t in range(100)]


def test_predict_captures_raw_and_roundtrip(tmp_path):
    ext = TribeExtractor.__new__(TribeExtractor)
    ext.model = FakeTribe()
    pred = ext.predict("v1", "x.mp4")
    assert pred.preds.shape[0] == 8 and set(pred.raw) == {"video__mean", "video__hook", "video__present"}
    assert not ext.model._model.hooks                            # hook removed afterwards
    pred.save(tmp_path / "v1.npz")
    again = TribePrediction.load(tmp_path / "v1.npz")
    assert np.allclose(again.raw["video__mean"], 1.0)


def test_raw_feature_table_fills_missing_modality(tmp_path):
    a = {"video__mean": np.ones(6, np.float32), "video__hook": np.ones(6, np.float32),
         "video__present": np.ones(1, np.float32), "text__mean": np.full(4, 2.0, np.float32),
         "text__hook": np.full(4, 2.0, np.float32), "text__present": np.ones(1, np.float32)}
    b = {k: v for k, v in a.items() if k.startswith("video")}   # nobody speaks in b
    TribePrediction("a", np.zeros((3, 2)), np.arange(3.0), 3.0, a).save(tmp_path / "a.npz")
    TribePrediction("b", np.zeros((3, 2)), np.arange(3.0), 3.0, b).save(tmp_path / "b.npz")
    TribePrediction("old", np.zeros((3, 2)), np.arange(3.0), 3.0).save(tmp_path / "old.npz")  # pre-0.8
    t = raw_feature_table(tmp_path).set_index("video_id")
    assert list(t.index) == ["a", "b"]
    assert t.loc["b", "raw:text:present"] == 0 and t.loc["a", "raw:text:present"] == 1
    assert t.loc["b", [c for c in t.columns if c.startswith("raw:text:mean")]].sum() == 0
    assert len([c for c in t.columns if c.startswith("raw:video:")]) == 6 + 6 + 1


def _synthetic_raw(manifest, ts, seed=0, dims=200):
    """Raw features that carry the same planted signal as the brain features, plus noise."""
    rng = np.random.default_rng(seed)
    sig = ts.groupby("video_id")[["valuation_mpfc", "salience_insula", "dorsal_attention"]].mean()
    w = rng.normal(size=(3, dims))
    x = sig.to_numpy() @ w + rng.normal(0, 3, (len(sig), dims))
    cols = [f"raw:video:mean:{i}" for i in range(dims)]
    out = pd.DataFrame(x, columns=cols)
    out.insert(0, "video_id", sig.index)
    out["raw:video:present"] = 1.0
    return out


def test_raw_section_in_report(tmp_path):
    manifest, ts = make_score_dataset(seed=12, n_creators=8, videos_per_creator=10)
    raw = _synthetic_raw(manifest, ts)
    run_score_training(manifest, ts, ["ecr"], tmp_path, hook_seconds=5, n_boot=50, raw_features=raw)
    m = pd.read_csv(tmp_path / "raw_comparison_metrics.csv")
    assert {"raw", "raw_video", "brain", "baseline", "avg(raw,brain)"} <= set(m["feature_set"])
    assert m.set_index("feature_set").loc["raw", "gain_vs_baseline"] == 0
    report = (tmp_path / "report.md").read_text()
    assert "Raw-feature baseline (exploratory" in report and (tmp_path / "raw_comparison.png").exists()

    run_score_training(manifest, ts, ["ecr"], tmp_path / "few", hook_seconds=5, n_boot=50,
                       raw_features=raw.head(10))
    assert "Skipped: only 10 videos" in (tmp_path / "few" / "report.md").read_text()


def test_cli_raw_from(tmp_path):
    manifest, ts = make_score_dataset(seed=13, n_creators=6, videos_per_creator=10)
    raw = _synthetic_raw(manifest, ts, dims=20)
    preds = tmp_path / "preds"
    preds.mkdir()
    for _, r in raw.iterrows():
        vec = r[[c for c in raw.columns if c.startswith("raw:video:mean")]].to_numpy(np.float32)
        TribePrediction(r.video_id, np.zeros((2, 2)), np.arange(2.0), 2.0,
                        {"video__mean": vec, "video__hook": vec, "video__present": np.ones(1, np.float32)}
                        ).save(preds / f"{r.video_id}.npz")
    manifest.to_csv(tmp_path / "m.csv", index=False)
    ts.to_csv(tmp_path / "ts.csv", index=False)
    main(["train-scores", "--manifest", str(tmp_path / "m.csv"), "--timeseries", str(tmp_path / "ts.csv"),
          "--targets", "ecr", "--n-boot", "50", "--raw-from", str(preds), "--out", str(tmp_path / "run")])
    assert (tmp_path / "run" / "raw_comparison_metrics.csv").exists()
