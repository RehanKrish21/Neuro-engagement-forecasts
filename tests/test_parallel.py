import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from neuroengage.parallel import run_extract_on_gpus
from neuroengage.tribe_runner import load_table

FAKE = textwrap.dedent("""
    import argparse, os, sys, time
    import numpy as np, pandas as pd
    from neuroengage.tribe_runner import TribePrediction, save_table, to_long
    p = argparse.ArgumentParser()
    for a in ("--manifest", "--out", "--cache", "--precision", "--max-hours"):
        p.add_argument(a)
    p.add_argument("--shard-index", type=int); p.add_argument("--num-shards", type=int)
    a = p.parse_args()
    if os.environ.get("FAIL_GPU") == os.environ["CUDA_VISIBLE_DEVICES"]:
        print("boom: simulated CUDA error"); sys.exit(3)
    m = pd.read_csv(a.manifest, dtype={"video_id": str}).iloc[a.shard_index::a.num_shards]
    masks = {"net": np.ones(4, bool)}
    frames = []
    for v in m.video_id:
        pred = TribePrediction(v, np.random.rand(5, 4).astype("float32"), np.arange(5.0), 5.0)
        pred.save(os.path.join(a.out, "preds", v + ".npz")); time.sleep(0.05)
        frames.append(to_long(pred, masks))
    name = "network_timeseries" if a.num_shards == 1 else f"network_timeseries_shard{a.shard_index:03d}"
    save_table(pd.concat(frames), os.path.join(a.out, name))
    print("gpu", os.environ["CUDA_VISIBLE_DEVICES"], "precision", a.precision, "cache", a.cache)
""")


@pytest.fixture
def setup(tmp_path):
    script = tmp_path / "fake_extract.py"
    script.write_text(FAKE)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame({"video_id": [f"{i:04d}" for i in range(10)], "path": "x"}).to_csv(manifest, index=False)
    return script, manifest, tmp_path / "work"


def test_two_gpus_split_and_merge(setup, tmp_path):
    script, manifest, out = setup
    path = run_extract_on_gpus(manifest, out, [0, 1], tmp_path / "cache", precision="fp16",
                               poll_seconds=0.1, command=[sys.executable, str(script)])
    ts = load_table(path)
    assert sorted(ts["video_id"].unique()) == [f"{i:04d}" for i in range(10)]  # IDs stay strings
    log0 = (out / "logs" / "shard0.log").read_text()
    assert "gpu 0 precision fp16" in log0 and "gpu0" in log0
    assert "gpu 1" in (out / "logs" / "shard1.log").read_text()


def test_failed_shard_reported_and_rest_merged(setup, tmp_path, monkeypatch, capsys):
    script, manifest, out = setup
    monkeypatch.setenv("FAIL_GPU", "1")
    path = run_extract_on_gpus(manifest, out, [0, 1], tmp_path / "cache",
                               poll_seconds=0.1, command=[sys.executable, str(script)])
    printed = capsys.readouterr().out
    assert "shard 1 exited with code 3" in printed and "simulated CUDA error" in printed
    assert load_table(path)["video_id"].nunique() == 5


def test_single_gpu_returns_table(setup, tmp_path):
    script, manifest, out = setup
    path = run_extract_on_gpus(manifest, out, [0], tmp_path / "cache",
                               poll_seconds=0.1, command=[sys.executable, str(script)])
    assert path.name.startswith("network_timeseries.") and load_table(path)["video_id"].nunique() == 10
