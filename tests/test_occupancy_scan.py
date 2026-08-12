"""The scan: a probe inside the blob is recorded, one outside is not, per source file."""

import numpy as np
import pytest

try:
    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader
    HAS_MDA = True
except ImportError:
    HAS_MDA = False

from cosolvkit.analysis.config import SimulationEntry
from cosolvkit.analysis.core.models import Hotspot

pytestmark = pytest.mark.skipif(not HAS_MDA, reason="MDAnalysis not available")

BOX = np.array([40.0, 40.0, 40.0, 90.0, 90.0, 90.0])


def _universe(probe_track, n_protein=4):
    """One 2-atom FMD molecule whose position per frame is given by probe_track.

    Protein atoms sit at fixed positions inside the grid so the alignment guard passes.
    """
    n_probe = 2
    n_atoms = n_protein + n_probe
    atom_resindex = [0] * n_protein + [1] * n_probe
    u = mda.Universe.empty(n_atoms, n_residues=2, n_segments=1,
                           atom_resindex=atom_resindex, residue_segindex=[0, 0],
                           trajectory=True)
    u.add_TopologyAttr("name", ["CA", "CB", "CG", "CD", "C1", "N1"])
    u.add_TopologyAttr("type", ["C", "C", "C", "C", "C", "N"])
    u.add_TopologyAttr("resname", ["ALA", "FMD"])
    u.add_TopologyAttr("resid", [1, 279])
    u.add_TopologyAttr("segid", ["A"])

    n_frames = len(probe_track)
    coords = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
    protein_xyz = np.array([[1.0, 1.0, 1.0], [1.5, 1.0, 1.0],
                            [1.0, 1.5, 1.0], [1.0, 1.0, 1.5]])
    for f, probe_xyz in enumerate(probe_track):
        coords[f, :n_protein] = protein_xyz
        coords[f, n_protein:] = probe_xyz
    u.load_new(coords, order="fac", format=MemoryReader,
               dimensions=np.tile(BOX, (n_frames, 1)))
    return u


def _hotspot():
    """A blob covering voxels 2..3 in each axis, i.e. 1.0-1.5 A with delta 0.5."""
    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[2:4, 2:4, 2:4] = True
    h = Hotspot(rank=1, site_id=1, cosolvent="FMD", n_voxels=int(mask.sum()),
                voxel_mask=mask)
    h.grid_origin = np.zeros(3)
    h.grid_delta = np.full(3, 0.5)
    return h


def _annotator(monkeypatch, universes, **kwargs):
    """Build an annotator whose Universe construction is stubbed to the given objects.

    ``universes`` maps a simulation label to a Universe; trajectories are named
    ``/tmp/<label>.dcd`` so the stub can recover the label from the path.
    """
    import os

    from cosolvkit.analysis.sites import occupancy as occ_mod

    sims = [SimulationEntry(trajectory=f"/tmp/{lbl}.dcd", topology=f"/tmp/{lbl}.prmtop",
                            cosolvents=["FMD"], label=lbl) for lbl in universes]
    monkeypatch.setattr(
        occ_mod, "_open_universe",
        lambda topology, trajectory: universes[
            os.path.splitext(os.path.basename(trajectory))[0]],
    )
    return occ_mod.OccupancyAnnotator(sims, **kwargs), sims


def test_probe_inside_the_blob_is_recorded(monkeypatch):
    # frames 0 and 1 inside the blob (1.25 A), frame 2 far outside
    u = _universe([[1.25, 1.25, 1.25], [1.25, 1.25, 1.25], [15.0, 15.0, 15.0]])
    ann, _ = _annotator(monkeypatch, {"r0": u})
    h = _hotspot()
    ann.annotate({"FMD": [h]})

    assert len(h.probe_occupancy) == 1
    occ = h.probe_occupancy[0]
    assert occ.frames == [0, 1]
    assert occ.probe_resid == 279
    assert occ.source_label == "r0"
    assert occ.n_frames_scanned == 3


def test_probe_that_never_enters_leaves_no_record(monkeypatch):
    u = _universe([[15.0, 15.0, 15.0], [16.0, 16.0, 16.0]])
    ann, _ = _annotator(monkeypatch, {"r0": u})
    h = _hotspot()
    ann.annotate({"FMD": [h]})
    assert h.probe_occupancy == []


def test_two_sources_produce_two_records_not_one(monkeypatch):
    """Same resid in both replicas — provenance must keep them apart."""
    u0 = _universe([[1.25, 1.25, 1.25], [15.0, 15.0, 15.0]])
    u1 = _universe([[15.0, 15.0, 15.0], [1.25, 1.25, 1.25]])
    ann, _ = _annotator(monkeypatch, {"r0": u0, "r1": u1})
    h = _hotspot()
    ann.annotate({"FMD": [h]})

    assert len(h.probe_occupancy) == 2
    by_source = {o.source_label: o.frames for o in h.probe_occupancy}
    assert by_source == {"r0": [0], "r1": [1]}


def test_stride_is_recorded_and_limits_scanned_frames(monkeypatch):
    u = _universe([[1.25, 1.25, 1.25]] * 6)
    ann, _ = _annotator(monkeypatch, {"r0": u}, stride=2)
    h = _hotspot()
    ann.annotate({"FMD": [h]})
    occ = h.probe_occupancy[0]
    assert occ.stride == 2
    assert occ.frames == [0, 2, 4]
    assert occ.n_frames_scanned == 3


def test_unaligned_trajectory_fails_loudly(monkeypatch):
    u = _universe([[1.25, 1.25, 1.25]])
    u.atoms.positions = u.atoms.positions + 500.0
    ann, _ = _annotator(monkeypatch, {"r0": u})
    with pytest.raises(ValueError, match="align"):
        ann.annotate({"FMD": [_hotspot()]})


def test_pocket_residues_get_source_keyed_contacts(monkeypatch):
    u = _universe([[1.25, 1.25, 1.25]])
    ann, _ = _annotator(monkeypatch, {"r0": u}, contact_cutoff=6.0)
    h = _hotspot()
    ann.annotate({"FMD": [h]})

    assert h.pocket_residues, "the annotator should have found the lining residue"
    pr = h.pocket_residues[0]
    assert pr.resname == "ALA"
    assert pr.contact_frames("FMD", source="r0") == [0]
