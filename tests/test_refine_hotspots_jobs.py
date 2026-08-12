"""Job generation: directory tags, molecule choice, and the rendered driver script."""

import pytest

from cosolvkit.analysis.core.models import Hotspot, ProbeOccupancy
from cosolvkit.cli.refine_hotspots_jobs import (
    SOLVENT_STRIP,
    _slurm_script,
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
    assert sum(len(f) for _, f in jobs[0][1]) == 5


def test_more_molecules_can_be_requested():
    class Args(_Args):
        mmgbsa_n_molecules = 2

    h = _hotspot(_occ("r0", 279, range(10)), _occ("r0", 280, range(40, 100)))
    jobs = select_mmgbsa_jobs(h, Args())
    assert len(jobs) == 2
    assert [j[0].probe_resid for j in jobs] == [280, 279], "best-occupied first"


def test_frames_of_one_molecule_pool_across_replicas():
    """Same molecule, two replicas sharing a topology — both contribute frames, but
    frame indices are trajectory-local, so each chosen frame must stay paired with the
    record (and trajectory) it actually came from."""
    h = _hotspot(_occ("r0", 279, range(10)), _occ("r1", 279, range(20, 30)))
    jobs = select_mmgbsa_jobs(h, _Args())

    assert len(jobs) == 1, "one molecule means one job, not one per replica"
    representative, selections = jobs[0]
    assert {o.source_label for o, _ in selections} == {"r0", "r1"}
    assert sum(len(f) for _, f in selections) == 5
    for occ, frames in selections:
        assert set(frames) <= set(occ.frames), (
            "a chosen frame must be present in the record it is paired with"
        )


def test_source_filter_restricts_selection():
    """The filtered-out record is the one that would otherwise win, and more molecules
    are requested than survive the filter — so a no-op filter shows up as an extra job
    from the wrong source rather than as a silently identical result."""
    class Args(_Args):
        mmgbsa_source = "r1"
        mmgbsa_n_molecules = 3

    h = _hotspot(_occ("r0", 279, range(100)),   # most-occupied, but wrong source
                 _occ("r0", 281, range(50)),
                 _occ("r1", 280, range(20, 30)))
    jobs = select_mmgbsa_jobs(h, Args())

    assert len(jobs) == 1, "only the r1 molecule is eligible"
    assert [o.probe_resid for o, _ in jobs] == [280]
    assert all(o.source_label == "r1" for o, _ in jobs)
    for _, selections in jobs:
        assert all(occ.source_label == "r1" for occ, _ in selections), (
            "no frame may come from a record the filter excluded"
        )
        assert all(set(frames) <= set(range(20, 30)) for _, frames in selections)


def test_frame_strategy_reaches_select_frames():
    """The reserved clustering hook must be reachable from the CLI path, which means
    selection has to go through mmgbsa.select_frames rather than a private RNG."""
    class Args(_Args):
        mmgbsa_frame_strategy = "cluster"

    h = _hotspot(_occ("r0", 279, range(10)))
    with pytest.raises(NotImplementedError, match="cluster"):
        select_mmgbsa_jobs(h, Args())


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
    assert "check_pocket(POSE_PDB)" in script, "the check must actually be invoked"


def test_mmgbsa_only_script_does_not_build_a_system():
    manifest = {"pose_pdb": "/tmp/bs_12/pose.pdb", "ligand_selection": "resname FMD",
                "pocket_selection": [4], "pocket_resnames": ["TYR"]}
    mmgbsa = {"sysname": "bs_12_FMD279_r0", "prmtop": "/tmp/a.prmtop",
              "trajectory": "/tmp/bs_12/mmgbsa/frames.dcd",
              "ligand_amber_selection": ":279",
              "strip_amber_selection": ":POP:HOH:WAT:NA:CL:K:MG:280-315",
              "mmpbsa_in": "/tmp/bs_12/mmgbsa/mmgbsa.in",
              "output_folder": "/tmp/bs_12/mmgbsa", "radii": "mbondi2"}
    script = render_autopath_script(manifest, mmgbsa=mmgbsa, mode="mmgbsa")
    assert "prepare_mmgbsa_batch" in script
    assert "AutoPath(" not in script
    assert ":280-315" in script


def test_script_passes_an_mmpbsa_input_file():
    """prepare_mmgbsa_batch bare-open()s mmpbsa_in and defaults to a 'mmgbsa.in' that
    ships nowhere, so an unset mmpbsa_in kills the job with FileNotFoundError."""
    manifest = {"pose_pdb": "/tmp/bs_12/pose.pdb", "ligand_selection": "resname FMD",
                "pocket_selection": [4], "pocket_resnames": ["TYR"]}
    mmgbsa = {"sysname": "s", "prmtop": "/tmp/a.prmtop", "trajectory": "/tmp/f.dcd",
              "ligand_amber_selection": ":279", "strip_amber_selection": ":NA:CL",
              "mmpbsa_in": "/tmp/bs_12/mmgbsa/mmgbsa.in",
              "output_folder": "/tmp/bs_12/mmgbsa", "radii": "mbondi2"}
    script = render_autopath_script(manifest, mmgbsa=mmgbsa, mode="mmgbsa")
    assert "mmpbsa_in='/tmp/bs_12/mmgbsa/mmgbsa.in'" in script


def test_script_documents_the_two_stage_mmgbsa_handoff():
    manifest = {"pose_pdb": "/tmp/bs_12/pose.pdb", "ligand_selection": "resname FMD",
                "pocket_selection": [4], "pocket_resnames": ["TYR"]}
    mmgbsa = {"sysname": "s", "prmtop": "/tmp/a.prmtop", "trajectory": "/tmp/f.dcd",
              "ligand_amber_selection": ":279", "strip_amber_selection": ":NA:CL",
              "mmpbsa_in": "/tmp/m/mmgbsa.in", "output_folder": "/tmp/m", "radii": "mbondi2"}
    script = render_autopath_script(manifest, mmgbsa=mmgbsa, mode="mmgbsa")
    assert "run_mmgbsa_batch.sh" in script, (
        "the driver must say that a second submission stage is required"
    )


def test_script_renders_one_call_per_molecule():
    manifest = {"pose_pdb": "/tmp/bs_12/pose.pdb", "ligand_selection": "resname FMD",
                "pocket_selection": [4], "pocket_resnames": ["TYR"]}
    specs = [
        {"sysname": "a", "prmtop": "/t/a.prmtop", "trajectory": "/t/a/frames.dcd",
         "ligand_amber_selection": ":279", "strip_amber_selection": ":NA",
         "mmpbsa_in": "/t/a/mmgbsa.in", "output_folder": "/t/a", "radii": "mbondi2"},
        {"sysname": "b", "prmtop": "/t/a.prmtop", "trajectory": "/t/b/frames.dcd",
         "ligand_amber_selection": ":280", "strip_amber_selection": ":NA",
         "mmpbsa_in": "/t/b/mmgbsa.in", "output_folder": "/t/b", "radii": "mbondi2"},
    ]
    script = render_autopath_script(manifest, mmgbsa=specs, mode="mmgbsa")
    assert script.count("prepare_mmgbsa_batch(") == 2, (
        "MMPBSA takes one ligand mask, so each molecule needs its own call"
    )
    assert ":279" in script and ":280" in script


# ---------------------------------------------------------------------------
# The SLURM script
# ---------------------------------------------------------------------------

def test_slurm_script_cds_into_the_target_directory(tmp_path):
    """AutoPath names its system after the PDB basename and creates it relative to the
    working directory; every target's pose is called pose.pdb, so without the cd all
    jobs would write ./pose_fixed/ into the shared submit directory."""
    tag_dir = tmp_path / "bs_12"
    tag_dir.mkdir()
    script = _slurm_script("bs_12", str(tag_dir / "run_autopath.py"),
                           workdir=str(tag_dir), mode="both")
    assert f"cd {tag_dir}" in script
    assert script.index("cd ") < script.index("python "), "the cd must precede python"


def test_slurm_script_requests_a_gpu_for_the_smd_leg(tmp_path):
    smd = _slurm_script("bs_12", str(tmp_path / "d.py"), workdir=str(tmp_path),
                        mode="smd")
    both = _slurm_script("bs_12", str(tmp_path / "d.py"), workdir=str(tmp_path),
                         mode="both")
    assert "#SBATCH --gres=gpu:1" in smd
    assert "#SBATCH --gres=gpu:1" in both


def test_slurm_script_asks_for_no_gpu_when_only_preparing_mmgbsa(tmp_path):
    only = _slurm_script("bs_12", str(tmp_path / "d.py"), workdir=str(tmp_path),
                         mode="mmgbsa")
    assert "#SBATCH --gres" not in only
    assert "GPU" in only, "the absence of a GPU request should be explained"


def test_slurm_template_receives_the_working_directory(tmp_path):
    template = tmp_path / "tpl.q"
    template.write_text("#!/bin/bash\n# {{NAME}}\ncd {{WORKDIR}}\npython {{SCRIPT}}\n")
    tag_dir = tmp_path / "bs_1"
    script = _slurm_script("bs_1", str(tag_dir / "run.py"),
                           template_path=str(template), workdir=str(tag_dir))
    assert f"cd {tag_dir}" in script
    assert "{{WORKDIR}}" not in script
