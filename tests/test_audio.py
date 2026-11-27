import numpy as np
import pandas as pd

from neuroengage.audio import SR, audio_feature_table, descriptors, extract_audio_features, load_audio


def _clicks(bpm, seconds=8.0, start=0.0, amp=0.5):
    wav = np.zeros(int(seconds * SR), np.float32)
    rng = np.random.default_rng(0)
    for t in np.arange(start, seconds, 60.0 / bpm):
        i = int(t * SR)
        wav[i:i + 400] += amp * rng.standard_normal(min(400, len(wav) - i)).astype(np.float32)
    return wav


def test_silence_and_empty():
    assert descriptors(np.zeros(0, np.float32))["has_audio"] == 0
    assert descriptors(np.zeros(SR * 3, np.float32))["has_audio"] == 0


def test_beat_tempo_and_onsets():
    d = descriptors(_clicks(120))
    assert d["has_audio"] == 1 and abs(d["tempo_bpm"] - 120) < 8 and d["beat_strength"] > 0.3
    assert 1.5 < d["onset_rate_hook"] < 2.6                                  # ~2 clicks per second
    noise = np.random.default_rng(1).standard_normal(SR * 8).astype(np.float32) * 0.1
    assert descriptors(noise)["beat_strength"] < d["beat_strength"]


def test_quiet_start_and_loudness_rise():
    wav = np.concatenate([np.zeros(SR * 2, np.float32), 0.3 * np.random.default_rng(2).standard_normal(SR * 4).astype(np.float32)])
    d = descriptors(wav)
    assert d["silence_frac_first2s"] > 0.9 and d["loud_hook_vs_rest"] < 0
    loud = descriptors(0.3 * np.random.default_rng(3).standard_normal(SR * 4).astype(np.float32))
    assert loud["loud_first1s"] > -15 and loud["silence_frac_first2s"] == 0


def test_extract_and_table_without_tagger(tmp_path):
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        import pytest
        pytest.skip("ffmpeg not installed")
    wav_path = tmp_path / "a.wav"
    import wave
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((_clicks(100) * 20000).astype(np.int16).tobytes())
    mp4 = tmp_path / "v1.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=8", "-i", str(wav_path),
                    "-shortest", "-c:v", "libx264", "-c:a", "aac", str(mp4)], check=True)
    assert abs(len(load_audio(mp4)) / SR - 8) < 0.2
    man = pd.DataFrame({"video_id": ["v1", "missing"], "path": [str(mp4), str(tmp_path / "nope.mp4")]})
    log = extract_audio_features(man, tmp_path / "audio").set_index("video_id")["status"]
    assert log["v1"] == "ok" and log["missing"] == "no_video"
    t = audio_feature_table(tmp_path / "audio").set_index("video_id")
    assert abs(t.loc["v1", "aud:tempo_bpm"] - 100) < 8 and t.loc["v1", "aud:has_audio"] == 1


def _synthetic_audio_table(manifest, ts, seed=0, emb_dims=40):
    """Audio features where 'music in the hook' carries part of the planted signal."""
    rng = np.random.default_rng(seed)
    sig = ts.groupby("video_id")["valuation_mpfc"].mean()
    rows = pd.DataFrame({"video_id": sig.index})
    rows["aud:has_audio"] = 1.0
    rows["aud:loud_first1s"] = rng.normal(-20, 5, len(rows))
    rows["aud:beat_strength"] = rng.uniform(0, 1, len(rows))
    rows["aud:tag_music_hook"] = 1 / (1 + np.exp(-(sig.to_numpy() * 2 + rng.normal(0, 1, len(rows)))))
    rows["aud:tag_speech_hook"] = rng.uniform(0, 1, len(rows))
    rows["aud:time_to_speech_s"] = rng.integers(0, 6, len(rows)).astype(float)
    emb = rng.normal(size=(len(rows), emb_dims)) + sig.to_numpy()[:, None] * 0.3
    for j in range(emb_dims):
        rows[f"audemb:hook:{j}"] = emb[:, j]
    return rows


def test_audio_section_in_report(tmp_path):
    from neuroengage.pipeline import run_score_training
    from neuroengage.synthetic import make_score_dataset

    manifest, ts = make_score_dataset(seed=21, n_creators=8, videos_per_creator=10)
    audio = _synthetic_audio_table(manifest, ts)
    run_score_training(manifest, ts, ["ecr"], tmp_path, hook_seconds=5, n_boot=50, audio_features=audio)
    m = pd.read_csv(tmp_path / "audio_comparison_metrics.csv")
    expected = {"baseline", "brain", "audio_descriptors", "audio_tags", "audio_interpretable", "audio_ast",
                "brain+audio_interpretable", "baseline+audio_interpretable", "avg(brain,audio_ast)"}
    assert expected <= set(m["feature_set"])
    assert "## Audio features (exploratory" in (tmp_path / "report.md").read_text()
    assert (tmp_path / "audio_comparison.png").exists()


def test_audio_section_uses_raw_overlap(tmp_path):
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from test_raw_features import _synthetic_raw

    from neuroengage.pipeline import run_score_training
    from neuroengage.synthetic import make_score_dataset

    manifest, ts = make_score_dataset(seed=22, n_creators=8, videos_per_creator=10)
    raw = _synthetic_raw(manifest, ts, dims=30)
    audio = _synthetic_audio_table(manifest, ts)
    run_score_training(manifest, ts, ["ecr"], tmp_path, hook_seconds=5, n_boot=50,
                       raw_features=raw, audio_features=audio)
    m = set(pd.read_csv(tmp_path / "audio_comparison_metrics.csv")["feature_set"])
    assert {"raw_video", "raw_video+audio_ast", "avg(raw_video,audio_ast)"} <= m


def test_cli_audio_from(tmp_path):
    from neuroengage.cli import main
    from neuroengage.synthetic import make_score_dataset

    manifest, ts = make_score_dataset(seed=23, n_creators=6, videos_per_creator=10)
    audio = _synthetic_audio_table(manifest, ts, emb_dims=8)
    adir = tmp_path / "audio"
    adir.mkdir()
    for _, r in audio.iterrows():
        data = {f"desc__{c[4:]}": np.float32(r[c]) for c in audio.columns
                if c.startswith("aud:") and not c.startswith("aud:tag_") and c != "aud:time_to_speech_s"}
        data.update({f"tag__{c[4:]}": np.float32(r[c]) for c in audio.columns
                     if c.startswith("aud:tag_") or c == "aud:time_to_speech_s"})
        data["emb__hook"] = r[[c for c in audio.columns if c.startswith("audemb:")]].to_numpy(np.float32)
        np.savez_compressed(adir / f"{r.video_id}.npz", **data)
    t = audio_feature_table(adir)
    assert len(t) == len(audio) and "aud:tag_music_hook" in t and "audemb:hook:7" in t
    manifest.to_csv(tmp_path / "m.csv", index=False)
    ts.to_csv(tmp_path / "ts.csv", index=False)
    main(["train-scores", "--manifest", str(tmp_path / "m.csv"), "--timeseries", str(tmp_path / "ts.csv"),
          "--targets", "ecr", "--n-boot", "50", "--audio-from", str(adir), "--out", str(tmp_path / "run")])
    assert (tmp_path / "run" / "audio_comparison_metrics.csv").exists()
