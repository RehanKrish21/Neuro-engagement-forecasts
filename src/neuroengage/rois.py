"""Group TRIBE v2's cortical predictions into functional networks.

TRIBE v2 predicts activity on the fsaverage5 mesh (10,242 vertices per
hemisphere, 20,484 total, left hemisphere first). Individual vertices are
noisy and not interpretable on their own, so we average them into a small
set of networks that the neuroforecasting literature links to attention,
valuation and aversion.

Networks are defined as unions of HCP-MMP1 (Glasser et al., 2016) areas, the
same parcellation TRIBE v2 ships helpers for (``tribev2.utils.get_hcp_labels``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FSAVERAGE5_VERTICES_PER_HEMI = 10242
FSAVERAGE5_VERTICES = 2 * FSAVERAGE5_VERTICES_PER_HEMI


@dataclass(frozen=True)
class Network:
    name: str
    areas: tuple[str, ...]
    rationale: str


NETWORKS: tuple[Network, ...] = (
    Network(
        "visual_early",
        ("V1", "V2", "V3", "V4"),
        "Low-level visual drive: motion, contrast, cuts.",
    ),
    Network(
        "visual_motion",
        ("MT", "MST", "FST", "V4t", "LO1", "LO2", "LO3"),
        "Motion and object processing.",
    ),
    Network(
        "face_body",
        ("FFC", "PIT", "VVC", "PHA1", "TPOJ2", "TPOJ3"),
        "Faces, bodies and scenes: people on screen.",
    ),
    Network(
        "auditory",
        ("A1", "LBelt", "MBelt", "PBelt", "RI", "A4", "A5"),
        "Sound and music.",
    ),
    Network(
        "language",
        ("44", "45", "47l", "55b", "STSda", "STSdp", "STSva", "STSvp", "PSL", "TPOJ1"),
        "Speech comprehension and narrative.",
    ),
    Network(
        "dorsal_attention",
        ("FEF", "PEF", "LIP", "VIP", "MIP", "AIP", "IPS1", "IP0", "IP1", "7AL", "7PC"),
        "Goal-directed, sustained attention (where the eyes go).",
    ),
    Network(
        "ventral_attention",
        ("PGi", "PFm", "PF", "STV"),
        "Stimulus-driven reorienting: surprise and salient events.",
    ),
    Network(
        "salience_insula",
        ("AAIC", "AVI", "MI", "FOP4", "FOP5", "a24pr", "p32pr", "a32pr"),
        "Anterior insula/cingulate: arousal; anterior-insula activity forecasts "
        "aversion and skipping (Tong et al., 2020).",
    ),
    Network(
        "valuation_mpfc",
        ("10r", "10v", "10pp", "OFC", "pOFC", "11l", "13l", "s32", "25", "a24", "p32"),
        "Medial prefrontal / orbitofrontal valuation. Cortical stand-in for the "
        "ventral striatum, which TRIBE v2's released weights do not cover.",
    ),
    Network(
        "default_mode",
        ("7m", "31pv", "31pd", "v23ab", "d23ab", "PCV", "RSC", "POS1"),
        "Default-mode network: rises when attention drifts.",
    ),
)

NETWORK_NAMES: tuple[str, ...] = tuple(n.name for n in NETWORKS)


def build_network_masks(
    vertex_labels: np.ndarray | list[str],
    networks: tuple[Network, ...] = NETWORKS,
    strict: bool = False,
) -> dict[str, np.ndarray]:
    """Return ``{network_name: boolean vertex mask}``.

    Parameters
    ----------
    vertex_labels:
        One HCP-MMP1 area name per vertex (``""`` for unlabeled vertices),
        e.g. from :func:`load_hcp_vertex_labels`.
    strict:
        If True, raise when an area listed in a network is missing from
        ``vertex_labels``. Otherwise missing areas are ignored, but a network
        that ends up with no vertices still raises.
    """
    labels = np.asarray(vertex_labels, dtype=object)
    present = set(labels.tolist())
    masks: dict[str, np.ndarray] = {}
    for net in networks:
        missing = [a for a in net.areas if a not in present]
        if strict and missing:
            raise ValueError(f"{net.name}: areas not in vertex labels: {missing}")
        mask = np.isin(labels, net.areas)
        if not mask.any():
            raise ValueError(f"Network {net.name!r} matched no vertices")
        masks[net.name] = mask
    return masks


def network_timeseries(preds: np.ndarray, masks: dict[str, np.ndarray]) -> pd.DataFrame:
    """Average vertex predictions within each network.

    ``preds`` has shape ``(n_timesteps, n_vertices)``, as returned by
    ``TribeModel.predict``. Returns a ``(n_timesteps, n_networks + 2)`` frame
    with one column per network plus two whole-cortex signals:

    * ``global``: mean predicted activity across cortex.
    * ``pattern_shift``: 1 - correlation between consecutive cortical
      patterns. High values mark moments where the predicted brain state
      changes sharply (event boundaries, cuts, topic shifts), a cheap proxy
      for novelty-driven attention capture. The first timestep is 0.
    """
    preds = np.asarray(preds, dtype=np.float64)
    if preds.ndim != 2:
        raise ValueError(f"preds must be 2-D (time, vertices), got {preds.shape}")
    for name, mask in masks.items():
        if mask.shape[0] != preds.shape[1]:
            raise ValueError(
                f"mask {name!r} has {mask.shape[0]} vertices, preds have {preds.shape[1]}"
            )
    out = {name: preds[:, mask].mean(axis=1) for name, mask in masks.items()}
    out["global"] = preds.mean(axis=1)
    out["pattern_shift"] = _pattern_shift(preds)
    return pd.DataFrame(out)


def _pattern_shift(preds: np.ndarray) -> np.ndarray:
    shift = np.zeros(preds.shape[0])
    if preds.shape[0] < 2:
        return shift
    centered = preds - preds.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    norms[norms == 0] = 1.0
    unit = centered / norms[:, None]
    corr = np.einsum("ij,ij->i", unit[1:], unit[:-1])
    shift[1:] = 1.0 - corr
    return shift


def load_hcp_vertex_labels(mesh: str = "fsaverage5") -> np.ndarray:
    """Per-vertex HCP-MMP1 area names for ``mesh`` (left then right hemisphere).

    Uses TRIBE v2's own helper, so the vertex order matches its predictions.
    Needs ``tribev2`` installed with its plotting extras (``mne``); the
    parcellation is downloaded on first use.
    """
    try:
        from tribev2.utils import get_hcp_vertex_labels
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "load_hcp_vertex_labels needs the tribev2 package. Install it from "
            "https://github.com/facebookresearch/tribev2 (see README)."
        ) from exc
    return np.asarray(get_hcp_vertex_labels(mesh=mesh, combine=False), dtype=object)
