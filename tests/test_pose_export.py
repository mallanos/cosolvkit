"""A pose is protein plus exactly one probe molecule, with a self-consistent manifest."""

import json

import numpy as np
import pytest

try:
    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader
    HAS_MDA = True
except ImportError:
    HAS_MDA = False

from cosolvkit.analysis.core.models import PoseRef

pytestmark = pytest.mark.skipif(not HAS_MDA, reason="MDAnalysis not available")


def _universe_two_probes():
    """4 protein atoms, two 2-atom FMD molecules (resid 279 near, 280 far)."""
    atom_resindex = [0, 0, 0, 0, 1, 1, 2, 2]
    u = mda.Universe.empty(8, n_residues=3, n_segments=1,
                           atom_resindex=atom_resindex, residue_segindex=[0, 0, 0],
                           trajectory=True)
    u.add_TopologyAttr("name", ["CA", "CB", "CG", "CD", "C1", "N1", "C1", "N1"])
    u.add_TopologyAttr("type", ["C", "C", "C", "C", "C", "N", "C", "N"])
    u.add_TopologyAttr("resname", ["ALA", "FMD", "FMD"])
    u.add_TopologyAttr("resid", [1, 279, 280])
    u.add_TopologyAttr("segid", ["A"])
    coords = np.zeros((2, 8, 3), dtype=np.float32)
    coords[:, :4] = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0],
                              [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]])
    coords[:, 4:6] = np.array([[3.0, 1.0, 1.0], [3.5, 1.0, 1.0]])
    coords[:, 6:8] = np.array([[30.0, 30.0, 30.0], [30.5, 30.0, 30.0]])
    u.load_new(coords, order="fac", format=MemoryReader,
               dimensions=np.tile(np.array([40.0] * 3 + [90.0] * 3), (2, 1)))
    return u


def _pose_ref():
    return PoseRef(source_label="r0", topology="/tmp/s.prmtop", trajectory="/tmp/t.dcd",
                   frame=1, probe_resname="FMD", probe_resid=279, probe_resindex=1,
                   episode=(0, 2))


def test_pose_holds_protein_and_exactly_one_probe(tmp_path):
    from cosolvkit.analysis.sites.poses import write_pose

    out = tmp_path / "pose.pdb"
    u = _universe_two_probes()
    write_pose(_pose_ref(), str(out), open_universe=lambda t, x: u)

    written = mda.Universe(str(out))
    probes = written.select_atoms("resname FMD")
    assert len(np.unique(probes.resids)) == 1
    assert set(np.unique(probes.resids)) == {279}
    assert len(written.select_atoms("protein")) == 4


def test_manifest_is_self_consistent(tmp_path):
    from cosolvkit.analysis.sites.poses import write_pose

    out = tmp_path / "pose.pdb"
    u = _universe_two_probes()
    manifest = write_pose(_pose_ref(), str(out), pocket_cutoff=5.0,
                          open_universe=lambda t, x: u)

    assert manifest["ligand_selection"] == "resname FMD"
    assert manifest["provenance"]["frame"] == 1
    assert manifest["provenance"]["source_label"] == "r0"
    assert manifest["provenance"]["probe_resid"] == 279
    assert len(manifest["pocket_selection"]) == len(manifest["pocket_resnames"])
    assert manifest["pocket_selection"], "the probe sits 1 A from the protein"


def test_pocket_selection_indexes_the_written_file_not_the_source(tmp_path):
    """Indices must address pose.pdb, whose numbering differs from the source topology."""
    from cosolvkit.analysis.sites.poses import write_pose

    out = tmp_path / "pose.pdb"
    u = _universe_two_probes()
    manifest = write_pose(_pose_ref(), str(out), pocket_cutoff=5.0,
                          open_universe=lambda t, x: u)

    written = mda.Universe(str(out))
    n_res = len(written.residues)
    assert all(1 <= i <= n_res for i in manifest["pocket_selection"])
    for i, name in zip(manifest["pocket_selection"], manifest["pocket_resnames"]):
        assert written.residues[i - 1].resname == name


def _universe_duplicate_resid_same_resname():
    """4 protein atoms, two 2-atom FMD molecules that both carry resid 279.

    Only ``resindex`` distinguishes them: this must not silently pull in both.
    """
    atom_resindex = [0, 0, 0, 0, 1, 1, 2, 2]
    u = mda.Universe.empty(8, n_residues=3, n_segments=1,
                           atom_resindex=atom_resindex, residue_segindex=[0, 0, 0],
                           trajectory=True)
    u.add_TopologyAttr("name", ["CA", "CB", "CG", "CD", "C1", "N1", "C1", "N1"])
    u.add_TopologyAttr("type", ["C", "C", "C", "C", "C", "N", "C", "N"])
    u.add_TopologyAttr("resname", ["ALA", "FMD", "FMD"])
    u.add_TopologyAttr("resid", [1, 279, 279])
    u.add_TopologyAttr("segid", ["A"])
    coords = np.zeros((2, 8, 3), dtype=np.float32)
    coords[:, :4] = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0],
                              [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]])
    coords[:, 4:6] = np.array([[3.0, 1.0, 1.0], [3.5, 1.0, 1.0]])
    coords[:, 6:8] = np.array([[30.0, 30.0, 30.0], [30.5, 30.0, 30.0]])
    u.load_new(coords, order="fac", format=MemoryReader,
               dimensions=np.tile(np.array([40.0] * 3 + [90.0] * 3), (2, 1)))
    return u


def test_duplicate_resid_does_not_pull_in_a_second_probe(tmp_path):
    """resindex, not resid, must select the probe: a repeated resid is not ambiguous."""
    from cosolvkit.analysis.sites.poses import write_pose

    out = tmp_path / "pose.pdb"
    u = _universe_duplicate_resid_same_resname()
    write_pose(_pose_ref(), str(out), open_universe=lambda t, x: u)

    written = mda.Universe(str(out))
    probes = written.select_atoms("resname FMD")
    assert len(probes) == 2, "exactly one 2-atom probe molecule, not both"
    assert len(np.unique(probes.resids)) == 1


def test_pocket_selection_refuses_a_pose_whose_resids_are_not_its_positions(tmp_path):
    """AutoPath reads a pocket_selection list as RESIDS and maps them through
    build_residue_mapping; write_pose computes 1-based POSITIONS. The two coincide for a
    prmtop-derived pose only by accident, so a pose numbered otherwise must fail loudly
    rather than silently restrain the wrong residues."""
    from cosolvkit.analysis.sites.poses import write_pose

    u = _universe_two_probes()
    # Protein residue keeps position 1 in the written pose but carries resid 7.
    u.residues.resids = [7, 279, 280]

    with pytest.raises(ValueError, match="pocket_selection is ambiguous"):
        write_pose(_pose_ref(), str(tmp_path / "pose.pdb"), pocket_cutoff=5.0,
                   open_universe=lambda t, x: u)


def test_manifest_is_written_beside_the_pose(tmp_path):
    from cosolvkit.analysis.sites.poses import write_pose

    out = tmp_path / "pose.pdb"
    u = _universe_two_probes()
    manifest = write_pose(_pose_ref(), str(out), open_universe=lambda t, x: u)

    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert on_disk == manifest
