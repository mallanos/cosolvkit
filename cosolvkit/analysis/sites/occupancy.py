#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Which probe molecule sat in which hotspot, in which source file and frames.
#

import logging

import numpy as np

logger = logging.getLogger(__name__)


def positions_to_voxel_indices(positions, origin, delta):
    """Map Cartesian positions to grid indices.

    Voxel centres sit at ``origin + i * delta`` — the same convention
    ``find_pocket_residues`` uses — so each cell spans +/- delta/2 about its centre and
    the conversion rounds to the nearest centre. ``np.floor`` would bias every lookup by
    half a voxel.

    :param positions: shape ``(n, 3)`` Cartesian coordinates in Angstroms.
    :param origin: shape ``(3,)`` grid origin.
    :param delta: shape ``(3,)`` voxel spacing.
    :return: shape ``(n, 3)`` integer indices, unclipped — callers must mask out-of-range.
    """
    return np.rint((np.asarray(positions, dtype=float) - origin) / delta).astype(int)


def build_label_volume(hotspots):
    """Paint each hotspot's mask into one integer volume, numbered from 1.

    One volume per cosolvent, never one shared across cosolvents: watershed masks are
    disjoint within a cosolvent but may overlap between them, and a shared volume would
    let one label silently overwrite another.

    :param hotspots: hotspots sharing a grid, all with ``voxel_mask``/``grid_origin``/
        ``grid_delta`` set.
    :return: ``(label_volume, ordered_hotspots)`` where ``label_volume == i + 1`` marks
        ``ordered_hotspots[i]``.
    :raises ValueError: if the hotspots do not share one grid.
    """
    if not hotspots:
        raise ValueError("build_label_volume requires at least one hotspot.")

    first = hotspots[0]
    shape = first.voxel_mask.shape
    for h in hotspots[1:]:
        if (h.voxel_mask.shape != shape
                or not np.allclose(h.grid_origin, first.grid_origin)
                or not np.allclose(h.grid_delta, first.grid_delta)):
            raise ValueError(
                f"Hotspot {h.site_id} is on a different grid from hotspot "
                f"{first.site_id}; masks cannot share a label volume."
            )

    vol = np.zeros(shape, dtype=np.int32)
    for i, h in enumerate(hotspots, start=1):
        vol[h.voxel_mask] = i
    return vol, list(hotspots)


def assert_inside_grid(protein_positions, origin, delta, shape, min_fraction=0.5):
    """Fail unless the protein actually sits inside the density grid.

    The grids live in the aligned, superposed frame. Handed an unaligned trajectory every
    voxel lookup falls outside the grid and the scan returns a silent zero, so this is
    checked once up front rather than debugged later.

    :raises ValueError: if fewer than *min_fraction* of atoms fall inside the grid.
    """
    idx = positions_to_voxel_indices(protein_positions, origin, delta)
    inside = np.all((idx >= 0) & (idx < np.asarray(shape)), axis=1)
    fraction = float(inside.mean()) if len(inside) else 0.0
    if fraction < min_fraction:
        raise ValueError(
            f"Only {fraction:.1%} of protein atoms fall inside the density grid "
            f"(need >= {min_fraction:.0%}). The trajectory is almost certainly not "
            f"aligned onto the frame the maps were built in — align it first."
        )
