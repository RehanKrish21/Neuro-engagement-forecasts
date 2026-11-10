import numpy as np
import pandas as pd

from neuroengage.features import second_features, video_features
from neuroengage.rois import NETWORK_NAMES


def _ts(n=10, vid="v"):
    df = pd.DataFrame({n_: np.arange(n, dtype=float) for n_ in NETWORK_NAMES})
    df["salience_insula"] = 0.0
    df["global"] = 0.0
    df["pattern_shift"] = 0.0
    df.insert(0, "t", np.arange(n))
    df.insert(0, "video_id", vid)
    return df


def test_video_features_values():
    vf = video_features(pd.concat([_ts(10, "a"), _ts(20, "b")]), hook_seconds=3)
    assert list(vf.video_id) == ["a", "b"]
    a = vf.iloc[0]
    assert a.duration_s == 10
    assert a["valuation_mpfc__hook"] == 1.0  # mean of 0,1,2
    assert a["valuation_mpfc__slope"] == 1.0
    # affect_balance = valuation - insula = valuation here
    assert a["affect_balance__mean"] == a["valuation_mpfc__mean"]
    assert a["valuation_mpfc__peak_time_frac"] == 1.0


def test_second_features_shape_and_no_leakage_across_videos():
    sf = second_features(pd.concat([_ts(5, "a"), _ts(5, "b")]))
    assert len(sf) == 10
    first_b = sf[sf.video_id == "b"].iloc[0]
    assert first_b["valuation_mpfc__delta"] == 0.0
    assert first_b["valuation_mpfc__lag1"] == 0.0  # backfilled, not from video a
    assert not sf.isna().any().any()
