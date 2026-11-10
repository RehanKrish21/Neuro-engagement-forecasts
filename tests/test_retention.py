import numpy as np
import pandas as pd
import pytest

from neuroengage.retention import dropoff_targets, from_youtube, resample_to_seconds, video_targets


def test_resample_monotone_and_anchored():
    curves = pd.DataFrame(
        {"video_id": "a", "t": [0.5, 2.0, 4.0, 9.5], "retention": [0.9, 0.7, 0.72, 0.4]}
    )
    out = resample_to_seconds(curves, {"a": 10})
    r = out["retention"].to_numpy()
    assert len(r) == 10 and r[0] == 1.0
    assert np.all(np.diff(r) <= 1e-12)


def test_missing_duration_raises():
    with pytest.raises(KeyError):
        resample_to_seconds(pd.DataFrame({"video_id": ["x"], "t": [0], "retention": [1]}), {})


def test_hazard_matches_definition():
    per = pd.DataFrame({"video_id": "a", "t": [0, 1, 2, 3], "retention": [0.8, 0.6, 0.6, 0.3]})
    hz = dropoff_targets(per)["hazard"].to_numpy()
    np.testing.assert_allclose(hz, [0.2, 0.25, 0.0, 0.5])


def test_video_targets():
    per = pd.DataFrame({"video_id": "a", "t": range(5), "retention": [1, 0.8, 0.7, 0.6, 0.5]})
    vt = video_targets(per, hook_seconds=3).iloc[0]
    assert vt.avg_retention == pytest.approx(0.72)
    assert vt.hook_retention == pytest.approx(0.6)
    assert vt.completion == pytest.approx(0.5)


def test_from_youtube():
    rep = pd.DataFrame(
        {"video_id": ["a", "a"], "elapsedVideoTimeRatio": [0.5, 1.0], "audienceWatchRatio": [0.6, 0.3]}
    )
    out = from_youtube(rep, {"a": 20})
    assert out["t"].tolist() == [10.0, 20.0]
