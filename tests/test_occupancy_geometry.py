"""Voxel-index conversion and label volumes — the geometry the occupancy scan rests on."""

import numpy as np
import pytest

from cosolvkit.analysis.core.models import Hotspot
from cosolvkit.analysis.sites.occupancy import (
    assert_inside_grid,
    build_label_volume,
    positions_to_voxel_indices,
)


def _hotspot(site_id, blob, shape=(8, 8, 8)):
    mask = np.zeros(shape, dtype=bool)
    mask[blob] = True
    h = Hotspot(rank=site_id, site_id=site_id, cosolvent="FMD",
                n_voxels=int(mask.sum()), voxel_mask=mask)
    h.grid_origin = np.zeros(3)
    h.grid_delta = np.full(3, 0.5)
    return h


def test_voxel_centres_map_to_their_own_index():
    origin, delta = np.zeros(3), np.full(3, 0.5)
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 1.5, 2.0]])
    idx = positions_to_voxel_indices(pos, origin, delta)
    assert idx.tolist() == [[0, 0, 0], [2, 3, 4]]


def test_rounds_to_nearest_centre_not_down():
    """Voxel centres sit AT origin + i*delta, so a point 0.6 delta along belongs to voxel 1.

    np.floor would put it in voxel 0 and shift every blob by half a voxel.
    """
    origin, delta = np.zeros(3), np.full(3, 0.5)
    idx = positions_to_voxel_indices(np.array([[0.30, 0.30, 0.30]]), origin, delta)
    assert idx.tolist() == [[1, 1, 1]]


def test_label_volume_numbers_hotspots_from_one():
    a = _hotspot(1, np.s_[1:3, 1:3, 1:3])
    b = _hotspot(2, np.s_[5:7, 5:7, 5:7])
    vol, ordered = build_label_volume([a, b])
    assert vol[1, 1, 1] == 1
    assert vol[5, 5, 5] == 2
    assert vol[0, 0, 0] == 0
    assert [h.site_id for h in ordered] == [1, 2]


def test_label_volume_rejects_mismatched_grids():
    a = _hotspot(1, np.s_[1:3, 1:3, 1:3])
    b = _hotspot(2, np.s_[5:7, 5:7, 5:7])
    b.grid_delta = np.full(3, 0.8)
    with pytest.raises(ValueError, match="grid"):
        build_label_volume([a, b])


def test_grid_guard_accepts_a_contained_protein():
    inside = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    assert_inside_grid(inside, np.zeros(3), np.full(3, 0.5), (8, 8, 8))


def test_grid_guard_rejects_an_unaligned_trajectory():
    """A protein translated off the grid means the trajectory was never superposed."""
    outside = np.array([[500.0, 500.0, 500.0], [501.0, 501.0, 501.0]])
    with pytest.raises(ValueError, match="align"):
        assert_inside_grid(outside, np.zeros(3), np.full(3, 0.5), (8, 8, 8))
