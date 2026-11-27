"""Audio features beyond TRIBE's speech-oriented inputs.

TRIBE hears each video through Wav2Vec-BERT (a *speech* model) and a speech
transcript. Short-form hooks often live in music and sound: a beat drop, a
trending song, laughter, silence. Two complementary feature sets:

* :func:`descriptors` - interpretable signal-processing measures of the
  opening (loudness, silence, sound onsets, beat strength, brightness), in
  plain numpy.
* :class:`AudioTagger` - AST (Audio Spectrogram Transformer) fine-tuned on
  AudioSet (2M clips, 527 sound classes): second-by-second probabilities of
  speech, music, singing, laughter, ... for the first 5 s, video-level class
  probabilities, and a 768-number general-audio embedding.

Audio is decoded with ffmpeg, so this step is independent of TRIBE and can run
on any video file (including videos processed by TRIBE in earlier runs).
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SR = 16000
HOOK_SECONDS = 5.0
AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
TAGS = {  # feature name -> AudioSet class name
    "speech": "Speech",
    "music": "Music",
    "singing": "Singing",
    "rapping": "Rapping",
    "laughter": "Laughter",
    "sound_effect": "Sound effect",
    "animal": "Animal",
    "silence": "Silence",
}
PER_SECOND_TAGS = ("speech", "music")


def load_audio(path: str | Path, sr: int = SR, timeout: float = 180.0) -> np.ndarray:
    """Mono float32 waveform at ``sr`` Hz via ffmpeg; empty if the file has no audio."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
           "-ar", str(sr), "-f", "f32le", "-"]
    try:
        res = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        logger.warning("ffmpeg failed on %s: %s", path, exc)
        return np.zeros(0, np.float32)
    if res.returncode != 0 or not res.stdout:
        return np.zeros(0, np.float32)
    return np.frombuffer(res.stdout, dtype=np.float32).copy()


def _frames(wav: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if len(wav) < frame:
        wav = np.pad(wav, (0, frame - len(wav)))
    n = 1 + (len(wav) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return wav[idx]


def descriptors(wav: np.ndarray, sr: int = SR, hook: float = HOOK_SECONDS) -> dict[str, float]:
    """Interpretable audio measures, emphasising the opening ``hook`` seconds.

    Loudness is in dBFS (0 = full scale). Onsets are peaks of spectral flux
    (sudden increases in sound energy: hits, cuts, syllables). Beat strength
    is the peak autocorrelation of the onset envelope at tempi of 60-200 BPM
    (near 0 = no regular beat, near 1 = strong steady beat).
    """
    keys = ["has_audio", "loud_first1s", "loud_hook", "loud_mean", "loud_hook_vs_rest",
            "loud_rise_first1s", "silence_frac_first2s", "onset_rate_hook", "onset_rate_mean",
            "beat_strength", "tempo_bpm", "centroid_hook_hz", "centroid_mean_hz"]
    out = dict.fromkeys(keys, 0.0)
    if len(wav) == 0 or not np.isfinite(wav).all():
        return out
    frame, hop = 1024, 512
    fr = _frames(wav.astype(np.float64), frame, hop)
    t = (np.arange(len(fr)) * hop + frame / 2) / sr
    power = (fr ** 2).mean(axis=1)
    frame_db = np.maximum(10 * np.log10(power + 1e-12), -90.0)
    if frame_db.max() < -60:  # digital silence
        return out
    out["has_audio"] = 1.0
    hk, first1, first2 = t < hook, t < 1.0, t < 2.0
    rest = ~hk if (~hk).any() else hk

    def m(x, mask):
        return float(x[mask].mean()) if mask.any() else float(x.mean())

    def loud(mask):  # energy-average, then dB (averaging dB values over-weights near-silence)
        sel = power[mask] if mask.any() else power
        return float(max(10 * np.log10(sel.mean() + 1e-12), -90.0))

    out["loud_first1s"] = loud(first1)
    out["loud_hook"] = loud(hk)
    out["loud_mean"] = loud(np.ones_like(hk))
    out["loud_hook_vs_rest"] = out["loud_hook"] - loud(rest)
    out["loud_rise_first1s"] = loud((t >= 0.5) & first1) - loud(t < 0.5)
    out["silence_frac_first2s"] = float((frame_db[first2] < -45).mean()) if first2.any() else 0.0

    spec = np.abs(np.fft.rfft(fr * np.hanning(frame), axis=1))
    freqs = np.fft.rfftfreq(frame, 1 / sr)
    centroid = (spec * freqs).sum(axis=1) / (spec.sum(axis=1) + 1e-10)
    out["centroid_hook_hz"] = m(centroid, hk)
    out["centroid_mean_hz"] = float(centroid.mean())

    flux = np.r_[0.0, np.maximum(np.diff(np.log1p(spec), axis=0), 0).sum(axis=1)]
    thresh = flux.mean() + flux.std()
    peak = (flux[1:-1] > flux[:-2]) & (flux[1:-1] >= flux[2:]) & (flux[1:-1] > thresh)
    onset_t = t[1:-1][peak]
    dur = max(len(wav) / sr, 1e-3)
    out["onset_rate_hook"] = float((onset_t < hook).sum() / min(hook, dur))
    out["onset_rate_mean"] = float(len(onset_t) / dur)

    # light smoothing so beat periods that fall between frames still form one clear peak
    env = np.convolve(flux, [0.1, 0.2, 0.4, 0.2, 0.1], mode="same")
    env = env - env.mean()
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]
    if ac[0] > 0:
        ac = ac / ac[0]
        fps = sr / hop
        lo, hi = int(0.3 * fps), min(int(1.0 * fps), len(ac) - 2)
        # only genuine local peaks count: a steady beat repeats; smooth envelopes don't
        peaks = [k for k in range(max(lo, 1), hi + 1) if ac[k] > ac[k - 1] and ac[k] >= ac[k + 1] and ac[k] > 0]
        if peaks:
            best = max(ac[j] for j in peaks)
            # a beat every T also "repeats" at 2T: take the shortest lag nearly as strong (avoids half-tempo)
            k = min(j for j in peaks if ac[j] >= 0.85 * best)
            out["beat_strength"] = float(best)
            out["tempo_bpm"] = float(60 * fps / k)
    return out


class AudioTagger:
    """AudioSet sound-event tagging and embeddings with AST."""

    def __init__(self, model_name: str = AST_MODEL, device: str = "auto") -> None:
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification

        self.torch = torch
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.fe = ASTFeatureExtractor.from_pretrained(model_name)
        self.model = ASTForAudioClassification.from_pretrained(model_name).to(self.device).eval()
        name_to_id = {v: int(k) for k, v in self.model.config.id2label.items()}
        missing = [c for c in TAGS.values() if c not in name_to_id]
        if missing:
            raise KeyError(f"AudioSet classes not found in {model_name}: {missing}")
        self.tag_ids = {k: name_to_id[c] for k, c in TAGS.items()}

    def _run(self, chunks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        inputs = self.fe([c.astype(np.float32) for c in chunks], sampling_rate=SR, return_tensors="pt")
        x = inputs["input_values"].to(self.device)
        with self.torch.no_grad():
            pooled = self.model.audio_spectrogram_transformer(x).pooler_output
            probs = self.torch.sigmoid(self.model.classifier(pooled))
        return probs.float().cpu().numpy(), pooled.float().cpu().numpy()

    def tag(self, wav: np.ndarray, hook: float = HOOK_SECONDS) -> tuple[dict[str, float], dict[str, np.ndarray]]:
        """Scalar tag features and embeddings (``hook``: first 5 s; ``mean``: 10-s windows)."""
        n_sec = int(hook)
        per_sec = [wav[s * SR:(s + 1) * SR] for s in range(n_sec)]
        hook_chunk = wav[: int(hook * SR)]
        win = 10 * SR
        full = [wav[i:i + win] for i in range(0, max(len(wav), 1), win)][:6]
        chunks = [c if len(c) else np.zeros(SR // 10, np.float32) for c in per_sec + [hook_chunk] + full]
        probs, emb = self._run(chunks)
        p_sec, p_hook, p_full = probs[:n_sec], probs[n_sec], probs[n_sec + 1:]
        feats: dict[str, float] = {}
        for name, cid in self.tag_ids.items():
            feats[f"tag_{name}_hook"] = float(p_hook[cid])
            feats[f"tag_{name}_mean"] = float(p_full[:, cid].mean())
        for name in PER_SECOND_TAGS:
            for s in range(n_sec):
                feats[f"tag_{name}_s{s}"] = float(p_sec[s, self.tag_ids[name]])
        speech = p_sec[:, self.tag_ids["speech"]]
        above = np.flatnonzero(speech > 0.5)
        feats["time_to_speech_s"] = float(above[0]) if len(above) else float(n_sec)
        return feats, {"hook": emb[n_sec], "mean": emb[n_sec + 1:].mean(axis=0)}


def extract_audio_features(
    manifest: pd.DataFrame,
    out_dir: str | Path,
    tagger: AudioTagger | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Compute and cache ``<out_dir>/<video_id>.npz`` for every manifest video on disk."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    for i, row in enumerate(manifest.itertuples(index=False), 1):
        vid, path = str(row.video_id), Path(row.path)
        dest = out_dir / f"{vid}.npz"
        if dest.exists() and not overwrite:
            rows.append({"video_id": vid, "status": "exists"})
            continue
        if not path.is_file():
            rows.append({"video_id": vid, "status": "no_video"})
            continue
        try:
            wav = load_audio(path)
            data = {f"desc__{k}": np.float32(v) for k, v in descriptors(wav).items()}
            if tagger is not None:
                feats, emb = tagger.tag(wav)
                data.update({f"tag__{k}": np.float32(v) for k, v in feats.items()})
                data.update({f"emb__{k}": v.astype(np.float32) for k, v in emb.items()})
            np.savez_compressed(dest, **data)
            rows.append({"video_id": vid, "status": "ok"})
        except Exception as exc:  # corrupt file etc.
            logger.warning("Audio features failed for %s: %s", vid, exc)
            rows.append({"video_id": vid, "status": f"error: {type(exc).__name__}"})
        if i % 50 == 0:
            logger.info("Audio features %d/%d (%.0f s)", i, len(manifest), time.time() - t0)
    return pd.DataFrame(rows)


def audio_feature_table(audio_dir: str | Path, ids: set[str] | None = None) -> pd.DataFrame:
    """One row per video: ``aud:<name>`` scalars and ``audemb:<hook|mean>:<i>`` embeddings."""
    rows = []
    for path in sorted(Path(audio_dir).glob("*.npz")):
        if ids is not None and path.stem not in ids:
            continue
        with np.load(path) as d:
            row: dict[str, float | str] = {"video_id": path.stem}
            for k in d.files:
                kind, name = k.split("__", 1)
                if kind in ("desc", "tag"):
                    row[f"aud:{name}"] = float(d[k])
                elif kind == "emb":
                    row.update({f"audemb:{name}:{j}": float(x) for j, x in enumerate(d[k])})
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["video_id"])
    return pd.DataFrame(rows).fillna(0.0)
