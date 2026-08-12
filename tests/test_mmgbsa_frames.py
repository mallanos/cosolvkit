"""Frame selection, Amber masks and the hand-built MMGBSA trajectory."""

import json

import numpy as np
import pytest

try:
    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader
    HAS_MDA = True
except ImportError:
    HAS_MDA = False

from cosolvkit.analysis.core.models import ProbeOccupancy
from cosolvkit.analysis.sites.mmgbsa import (
    amber_exclude_mask,
    check_single_topology,
    select_frames,
    write_frame_trajectory,
)

pytestmark = pytest.mark.skipif(not HAS_MDA, reason="MDAnalysis not available")


def _occ(label="r0", frames=range(50), resid=279):
    return ProbeOccupancy(
        source_label=label, topology=f"/tmp/{label}.prmtop",
        trajectory=f"/tmp/{label}.dcd", probe_resname="FMD", probe_resid=resid,
        probe_resindex=resid - 1, frames=list(frames), n_frames_scanned=100, stride=1)


# --- Amber masks -----------------------------------------------------------

def test_exclude_mask_drops_the_kept_resid_from_a_contiguous_range():
    assert amber_exclude_mask(range(279, 316), keep=279) == ":280-315"


def test_exclude_mask_splits_when_the_kept_resid_is_mid_range():
    assert amber_exclude_mask(range(279, 316), keep=300) == ":279-299,301-315"


def test_exclude_mask_handles_the_last_resid_and_a_lone_survivor():
    assert amber_exclude_mask(range(279, 316), keep=315) == ":279-314"
    assert amber_exclude_mask([279, 280], keep=279) == ":280"


def test_exclude_mask_is_empty_when_nothing_survives():
    assert amber_exclude_mask([279], keep=279) == ""


# --- frame selection -------------------------------------------------------

def test_selection_is_reproducible_under_a_fixed_seed():
    a = select_frames(_occ(), n_frames=10, seed=7)
    b = select_frames(_occ(), n_frames=10, seed=7)
    assert a == b
    assert len(a) == 10


def test_selection_draws_only_from_occupied_frames():
    occ = _occ(frames=[3, 9, 27])
    assert set(select_frames(occ, n_frames=2, seed=0)) <= {3, 9, 27}


def test_asking_for_more_frames_than_exist_returns_all_of_them():
    occ = _occ(frames=[1, 2, 3])
    assert select_frames(occ, n_frames=99, seed=0) == [1, 2, 3]


def test_clustering_is_reserved_not_silently_random():
    with pytest.raises(NotImplementedError):
        select_frames(_occ(), n_frames=5, strategy="cluster")


# --- topology identity -----------------------------------------------------

def _topology_universe(n_atoms=6, charge_offset=0.0):
    u = mda.Universe.empty(n_atoms, n_residues=2, n_segments=1,
                           atom_resindex=[0, 0, 0, 1, 1, 1],
                           residue_segindex=[0, 0], trajectory=True)
    u.add_TopologyAttr("name", ["CA", "CB", "CG"] * 2)
    u.add_TopologyAttr("type", ["C", "C", "C"] * 2)
    u.add_TopologyAttr("resname", ["ALA", "FMD"])
    u.add_TopologyAttr("resid", [1, 279])
    u.add_TopologyAttr("charge", [0.1 + charge_offset] * n_atoms)
    u.load_new(np.zeros((1, n_atoms, 3), dtype=np.float32), order="fac",
               format=MemoryReader,
               dimensions=np.array([[40.0] * 3 + [90.0] * 3]))
    return u


def test_replicas_differing_only_in_box_share_a_topology():
    """Regression: real replicas differ in BOX_DIMENSIONS and %VERSION only.

    A byte or md5 comparison calls these different and is wrong.
    """
    a, b = _topology_universe(), _topology_universe()
    b.dimensions = np.array([41.0, 41.0, 41.0, 90.0, 90.0, 90.0])
    got = check_single_topology([_occ("r0"), _occ("r1")],
                                open_universe=lambda t: {"/tmp/r0.prmtop": a,
                                                         "/tmp/r1.prmtop": b}[t])
    assert got == "/tmp/r0.prmtop"


def test_empty_occupancy_list_raises_clearly():
    with pytest.raises(ValueError):
        check_single_topology([])


def test_genuinely_different_systems_raise():
    a, b = _topology_universe(), _topology_universe(charge_offset=0.5)
    with pytest.raises(ValueError, match="topolog"):
        check_single_topology([_occ("r0"), _occ("r1")],
                              open_universe=lambda t: {"/tmp/r0.prmtop": a,
                                                       "/tmp/r1.prmtop": b}[t])


# --- trajectory writing ----------------------------------------------------

def _frame_universe(n_frames=10, tag=0.0):
    u = mda.Universe.empty(3, n_residues=1, n_segments=1, atom_resindex=[0, 0, 0],
                           residue_segindex=[0], trajectory=True)
    u.add_TopologyAttr("name", ["CA", "CB", "CG"])
    u.add_TopologyAttr("resname", ["ALA"])
    u.add_TopologyAttr("resid", [1])
    coords = np.zeros((n_frames, 3, 3), dtype=np.float32)
    for f in range(n_frames):
        coords[f, :, 0] = f + tag          # frame index is recoverable from x
    u.load_new(coords, order="fac", format=MemoryReader,
               dimensions=np.tile(np.array([40.0] * 3 + [90.0] * 3), (n_frames, 1)))
    return u


def test_written_trajectory_has_the_selected_frames_in_order(tmp_path):
    u = _frame_universe()
    out = tmp_path / "frames.dcd"
    manifest = write_frame_trajectory([(_occ(), [7, 2, 5])], str(out),
                                      open_universe=lambda t, x: u)

    # Re-read through a PDB of the same topology; the x coordinate encodes the
    # original frame index, so the written order is directly checkable.
    topology_pdb = tmp_path / "top.pdb"
    _frame_universe().atoms.write(str(topology_pdb))
    written = mda.Universe(str(topology_pdb), str(out))

    assert len(written.trajectory) == 3
    xs = [round(float(ts.positions[0, 0])) for ts in written.trajectory]
    assert xs == [2, 5, 7], "frames must be written in ascending order"
    assert manifest["n_frames"] == 3


def test_manifest_maps_new_index_back_to_source_and_frame(tmp_path):
    u = _frame_universe()
    out = tmp_path / "frames.dcd"
    manifest = write_frame_trajectory([(_occ("r0"), [2, 5])], str(out),
                                      open_universe=lambda t, x: u)

    assert manifest["frames"] == [
        {"index": 0, "source_label": "r0", "original_frame": 2},
        {"index": 1, "source_label": "r0", "original_frame": 5},
    ]
    on_disk = json.loads((tmp_path / "frames.json").read_text())
    assert on_disk == manifest


def test_frames_pool_across_two_replica_sources(tmp_path):
    """Replicas share a topology, so their frames belong in one trajectory."""
    u0, u1 = _frame_universe(tag=0.0), _frame_universe(tag=100.0)
    out = tmp_path / "frames.dcd"
    manifest = write_frame_trajectory(
        [(_occ("r0"), [1]), (_occ("r1"), [2])], str(out),
        open_universe=lambda t, x: {"/tmp/r0.dcd": u0, "/tmp/r1.dcd": u1}[x])

    assert manifest["n_frames"] == 2
    assert [f["source_label"] for f in manifest["frames"]] == ["r0", "r1"]
