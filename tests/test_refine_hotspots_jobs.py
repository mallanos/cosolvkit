"""Job generation: directory tags, molecule choice, and the rendered driver script."""

import pytest

from cosolvkit.analysis.core.models import Hotspot, ProbeOccupancy
from cosolvkit.cli.refine_hotspots_jobs import (
    render_autopath_script,
    select_mmgbsa_jobs,
    target_tag,
)


class _Args:
    mmgbsa_n_frames = 5
    mmgbsa_n_molecules = 1
    mmgbsa_source = None
    seed = 0


def _occ(label, resid, frames):
    return ProbeOccupancy(source_label=label, topology="/tmp/a.prmtop",
                          trajectory=f"/tmp/{label}.dcd", probe_resname="FMD",
                          probe_resid=resid, probe_resindex=resid - 1,
                          frames=list(frames), n_frames_scanned=100, stride=1)


def _hotspot(*occs):
    h = Hotspot(rank=1, site_id=12, cosolvent="FMD")
    h.probe_occupancy = list(occs)
    return h


def test_tags_disambiguate_hotspots_across_cosolvents():
    """Hotspot site_id is unique only within a cosolvent."""
    h = Hotspot(rank=1, site_id=3, cosolvent="FMD")
    assert target_tag(h, is_binding_site=False) == "hs_FMD_3"
    assert target_tag(h, is_binding_site=True) == "bs_3"


def test_one_job_per_molecule_best_first():
    h = _hotspot(_occ("r0", 279, range(10)), _occ("r0", 280, range(40, 100)))
    jobs = select_mmgbsa_jobs(h, _Args())
    assert len(jobs) == 1
    assert jobs[0][0].probe_resid == 280, "the molecule with more bound frames wins"
    assert len(jobs[0][1]) == 5


def test_more_molecules_can_be_requested():
    class Args(_Args):
        mmgbsa_n_molecules = 2

    h = _hotspot(_occ("r0", 279, range(10)), _occ("r0", 280, range(40, 100)))
    assert len(select_mmgbsa_jobs(h, Args())) == 2


def test_frames_of_one_molecule_pool_across_replicas():
    """Same molecule, two replicas sharing a topology — both contribute frames."""
    h = _hotspot(_occ("r0", 279, range(10)), _occ("r1", 279, range(20, 30)))
    jobs = select_mmgbsa_jobs(h, _Args())

    assert len(jobs) == 1, "one molecule means one job, not one per replica"
    merged, frames = jobs[0]
    assert merged.frames == list(range(10)) + list(range(20, 30))
    assert len(frames) == 5
    assert set(frames) <= set(merged.frames)


def test_source_filter_restricts_selection():
    class Args(_Args):
        mmgbsa_source = "r1"

    h = _hotspot(_occ("r0", 279, range(10)), _occ("r1", 280, range(20, 30)))
    jobs = select_mmgbsa_jobs(h, Args())
    assert all(o.source_label == "r1" for o, _ in jobs)


def test_hotspot_with_no_occupancy_yields_no_jobs():
    assert select_mmgbsa_jobs(Hotspot(rank=1, site_id=1, cosolvent="FMD"), _Args()) == []


def test_script_imports_autopath_from_the_module_not_the_package():
    """`from autopath import AutoPath` fails — AutoPath is not in the lazy-import table."""
    manifest = {"pose_pdb": "/tmp/bs_12/pose.pdb", "ligand_selection": "resname FMD",
                "pocket_selection": [4, 5], "pocket_resnames": ["TYR", "ALA"]}
    script = render_autopath_script(manifest, mmgbsa=None, mode="smd")
    assert "from autopath.autopath_core import AutoPath" in script
    assert "from autopath import AutoPath" not in script
    assert "resname FMD" in script


def test_script_validates_the_pocket_selection_after_preparation():
    manifest = {"pose_pdb": "/tmp/bs_12/pose.pdb", "ligand_selection": "resname FMD",
                "pocket_selection": [4, 5], "pocket_resnames": ["TYR", "ALA"]}
    script = render_autopath_script(manifest, mmgbsa=None, mode="smd")
    assert "EXPECTED_POCKET_RESNAMES" in script
    assert "raise SystemExit" in script


def test_mmgbsa_only_script_does_not_build_a_system():
    manifest = {"pose_pdb": "/tmp/bs_12/pose.pdb", "ligand_selection": "resname FMD",
                "pocket_selection": [4], "pocket_resnames": ["TYR"]}
    mmgbsa = {"sysname": "bs_12_FMD279_r0", "prmtop": "/tmp/a.prmtop",
              "trajectory": "/tmp/bs_12/mmgbsa/frames.dcd",
              "ligand_amber_selection": ":279",
              "strip_amber_selection": ":HOH:WAT:Na+:Cl-:280-315",
              "output_folder": "/tmp/bs_12/mmgbsa"}
    script = render_autopath_script(manifest, mmgbsa=mmgbsa, mode="mmgbsa")
    assert "prepare_mmgbsa_batch" in script
    assert "AutoPath(" not in script
    assert ":280-315" in script
