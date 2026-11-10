import numpy as np
import pandas as pd
import pytest

from neuroengage.rois import (
    FSAVERAGE5_VERTICES,
    NETWORK_NAMES,
    NETWORKS,
    build_network_masks,
    network_timeseries,
)
from neuroengage.synthetic import make_vertex_predictions


@pytest.fixture
def vertex_labels():
    """Fake fsaverage5 labelling: each network's areas get a block of vertices."""
    labels = np.full(FSAVERAGE5_VERTICES, "", dtype=object)
    areas = sorted({a for n in NETWORKS for a in n.areas})
    rng = np.random.default_rng(0)
    idx = rng.permutation(FSAVERAGE5_VERTICES)[: len(areas) * 50]
    for i, area in enumerate(areas):
        labels[idx[i * 50 : (i + 1) * 50]] = area
    return labels


def test_networks_have_unique_names_and_disjoint_areas():
    assert len(set(NETWORK_NAMES)) == len(NETWORK_NAMES)
    seen = {}
    for net in NETWORKS:
        for area in net.areas:
            assert area not in seen, f"{area} in {seen.get(area)} and {net.name}"
            seen[area] = net.name


def test_masks_cover_expected_vertices(vertex_labels):
    masks = build_network_masks(vertex_labels, strict=True)
    assert set(masks) == set(NETWORK_NAMES)
    for net in NETWORKS:
        assert masks[net.name].sum() == 50 * len(net.areas)


def test_missing_network_raises():
    with pytest.raises(ValueError, match="matched no vertices"):
        build_network_masks(np.array(["V1", "V2"], dtype=object))


def test_network_timeseries_recovers_planted_signal(vertex_labels):
    rng = np.random.default_rng(1)
    planted = pd.DataFrame({n: rng.normal(size=30) for n in NETWORK_NAMES})
    preds = make_vertex_predictions(vertex_labels, planted, noise=0.3)
    ts = network_timeseries(preds, build_network_masks(vertex_labels))
    assert ts.shape == (30, len(NETWORK_NAMES) + 2)
    for name in NETWORK_NAMES:
        assert np.corrcoef(ts[name], planted[name])[0, 1] > 0.95


def test_pattern_shift_detects_state_change():
    preds = np.zeros((6, 100))
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=100), rng.normal(size=100)
    preds[:3] = a
    preds[3:] = b
    shift = network_timeseries(preds, {"all": np.ones(100, bool)})["pattern_shift"]
    assert shift[0] == 0
    assert shift[3] > 0.5
    assert np.allclose(shift[[1, 2, 4, 5]], 0, atol=1e-9)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="vertices"):
        network_timeseries(np.zeros((4, 10)), {"x": np.ones(11, bool)})
