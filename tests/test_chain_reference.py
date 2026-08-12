"""A prmtop has no chains; a reference PDB restores them by residue order."""

import numpy as np
import pytest

try:
    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader
    HAS_MDA = True
except ImportError:
    HAS_MDA = False

from cosolvkit.analysis.core.models import Hotspot, PocketResidue

pytestmark = pytest.mark.skipif(not HAS_MDA, reason="MDAnalysis not available")


def _protein(segids, resnames=("ALA", "GLY")):
    u = mda.Universe.empty(len(segids), n_residues=len(segids),
                           n_segments=len(set(segids)),
                           atom_resindex=list(range(len(segids))),
                           residue_segindex=[sorted(set(segids)).index(s) for s in segids],
                           trajectory=True)
    u.add_TopologyAttr("name", ["CA"] * len(segids))
    u.add_TopologyAttr("resname", list(resnames)[:len(segids)])
    u.add_TopologyAttr("resid", list(range(1, len(segids) + 1)))
    u.add_TopologyAttr("segid", sorted(set(segids)))
    coords = np.zeros((1, len(segids), 3), dtype=np.float32)
    u.load_new(coords, order="fac", format=MemoryReader,
               dimensions=np.array([[40.0] * 3 + [90.0] * 3]))
    return u


def _hotspot_with_residues():
    h = Hotspot(rank=1, site_id=1, cosolvent="FMD")
    h.pocket_residues = [
        PocketResidue(resid=1, resindex=0, resname="ALA", chain="SYSTEM",
                      n_contact_voxels=1, min_dist_ang=3.0, contact_fraction=0.1),
        PocketResidue(resid=2, resindex=1, resname="GLY", chain="SYSTEM",
                      n_contact_voxels=1, min_dist_ang=3.0, contact_fraction=0.1),
    ]
    return h


def test_chains_are_restored_and_the_source_is_recorded():
    from cosolvkit.analysis.sites.occupancy import apply_chain_reference

    topology = _protein(["SYSTEM", "SYSTEM"])
    reference = _protein(["A", "B"])
    h = _hotspot_with_residues()
    apply_chain_reference([h], reference, topology)

    assert [pr.chain for pr in h.pocket_residues] == ["A", "B"]
    assert all(pr.chain_source == "chain_reference" for pr in h.pocket_residues)


def test_residue_count_mismatch_raises():
    from cosolvkit.analysis.sites.occupancy import apply_chain_reference

    topology = _protein(["SYSTEM", "SYSTEM"])
    reference = _protein(["A"], resnames=("ALA",))
    with pytest.raises(ValueError, match="residue count"):
        apply_chain_reference([_hotspot_with_residues()], reference, topology)


def test_resname_mismatch_raises_rather_than_swapping_chains():
    from cosolvkit.analysis.sites.occupancy import apply_chain_reference

    topology = _protein(["SYSTEM", "SYSTEM"], resnames=("ALA", "GLY"))
    reference = _protein(["A", "B"], resnames=("TRP", "GLY"))
    with pytest.raises(ValueError, match="resname"):
        apply_chain_reference([_hotspot_with_residues()], reference, topology)


def test_config_accepts_and_resolves_chain_reference(tmp_path):
    from cosolvkit.analysis.config import AnalysisConfig

    (tmp_path / "cfg.yaml").write_text(
        "out_path: out\n"
        "simulations:\n"
        "  - trajectory: t.dcd\n"
        "    topology: system.prmtop\n"
        "    cosolvents: ['FMD']\n"
        "    label: r0\n"
        "    chain_reference: system.pdb\n"
    )
    cfg = AnalysisConfig.from_yaml(str(tmp_path / "cfg.yaml"))
    assert cfg.simulations[0].chain_reference == str(tmp_path / "system.pdb")


def test_chain_reference_is_optional(tmp_path):
    from cosolvkit.analysis.config import AnalysisConfig

    (tmp_path / "cfg.yaml").write_text(
        "out_path: out\n"
        "simulations:\n"
        "  - trajectory: t.dcd\n"
        "    topology: system.prmtop\n"
        "    cosolvents: ['FMD']\n"
    )
    cfg = AnalysisConfig.from_yaml(str(tmp_path / "cfg.yaml"))
    assert cfg.simulations[0].chain_reference is None
