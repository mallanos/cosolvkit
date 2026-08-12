"""End-to-end job generation: what actually lands on disk for a target.

``generate_jobs``, ``_build_mmgbsa_inputs``, ``_rank_targets`` and ``_slurm_script`` are
the layer with no other integration coverage, which is why every defect the final review
found lived here. These tests drive them over a small real system on disk.
"""

import os
import types

import numpy as np
import pytest

try:
    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader
    HAS_MDA = True
except ImportError:
    HAS_MDA = False

from cosolvkit.analysis.config import BindingSitesConfig, SimulationEntry
from cosolvkit.analysis.core.models import Hotspot, ProbeOccupancy

pytestmark = pytest.mark.skipif(not HAS_MDA, reason="MDAnalysis not available")

N_FRAMES = 4


def _system(tmp_path, label="r0", cosolvents=("FMD", "BEN")):
    """Write a tiny PDB + DCD: 4 ALA residues, 3 FMD molecules and 2 BEN molecules.

    Residues are numbered 1..9, i.e. resid == position, which is what a prmtop gives and
    what ``write_pose`` insists on. Everything sits within a few Angstroms so the pocket
    search finds the protein.
    """
    layout = [("ALA", ["N", "CA", "C", "O"])] * 4
    layout += [("FMD", ["C1", "N1"])] * 3
    layout += [("BEN", ["C1", "C2"])] * 2

    names, resnames, resids, atom_resindex = [], [], [], []
    for r, (resname, atoms) in enumerate(layout):
        names += atoms
        resnames.append(resname)
        resids.append(r + 1)
        atom_resindex += [r] * len(atoms)

    n = len(names)
    u = mda.Universe.empty(n, n_residues=len(layout), n_segments=1,
                           atom_resindex=atom_resindex,
                           residue_segindex=[0] * len(layout), trajectory=True)
    u.add_TopologyAttr("name", names)
    u.add_TopologyAttr("type", [x[0] for x in names])
    u.add_TopologyAttr("resname", resnames)
    u.add_TopologyAttr("resid", resids)
    u.add_TopologyAttr("segid", ["A"])

    base = np.random.RandomState(0).rand(n, 3).astype(np.float32) * 5.0
    coords = np.stack([base + f * 0.05 for f in range(N_FRAMES)])
    u.load_new(coords, order="fac", format=MemoryReader,
               dimensions=np.tile([40.0, 40.0, 40.0, 90.0, 90.0, 90.0], (N_FRAMES, 1)))

    pdb = str(tmp_path / f"{label}.pdb")
    dcd = str(tmp_path / f"{label}.dcd")
    u.atoms.write(pdb)
    with mda.Writer(dcd, n_atoms=n) as w:
        for _ in u.trajectory:
            w.write(u.atoms)
    return SimulationEntry(trajectory=dcd, topology=pdb,
                           cosolvents=list(cosolvents), label=label)


def _occ(sim, resid, resindex, resname="FMD", frames=range(N_FRAMES)):
    return ProbeOccupancy(source_label=sim.label, topology=sim.topology,
                          trajectory=sim.trajectory, probe_resname=resname,
                          probe_resid=resid, probe_resindex=resindex,
                          frames=list(frames), n_frames_scanned=N_FRAMES, stride=1)


def _hotspot(*occs):
    h = Hotspot(rank=1, site_id=12, cosolvent="FMD")
    h.probe_occupancy = list(occs)
    return h


def _config(sims):
    return types.SimpleNamespace(simulations=list(sims), out_path="/nonexistent",
                                 binding_sites=BindingSitesConfig())


def _args(**overrides):
    base = dict(target="hotspots", top_n=5, gap_tolerance=0, mode="both",
                slurm_template=None, submit=False, checkpoint=None,
                mmgbsa_n_frames=3, mmgbsa_n_molecules=1, mmgbsa_source=None,
                mmgbsa_frame_strategy="random", seed=0)
    base.update(overrides)
    return types.SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# generate_jobs
# ---------------------------------------------------------------------------

def test_generated_job_cds_into_its_own_directory(tmp_path):
    """Without the cd, AutoPath's ./pose_fixed/ resolves under the submit directory and
    every target clobbers every other target's system.pdb, equilibration/ and sMD/."""
    from cosolvkit.cli.refine_hotspots_jobs import generate_jobs

    sim = _system(tmp_path)
    out = tmp_path / "refine"
    out.mkdir()
    results = {"FMD": [_hotspot(_occ(sim, 5, 4))]}

    qfiles = generate_jobs(_config([sim]), results, str(out), _args())

    assert len(qfiles) == 1
    tag_dir = os.path.join(str(out), "hs_FMD_12")
    body = open(qfiles[0]).read()
    assert f"cd {tag_dir}" in body
    assert body.index("cd ") < body.index("python ")


def test_two_targets_get_two_separate_directories(tmp_path):
    from cosolvkit.cli.refine_hotspots_jobs import generate_jobs

    sim = _system(tmp_path)
    out = tmp_path / "refine"
    out.mkdir()
    a = _hotspot(_occ(sim, 5, 4))
    b = Hotspot(rank=2, site_id=13, cosolvent="FMD")
    b.probe_occupancy = [_occ(sim, 6, 5)]

    qfiles = generate_jobs(_config([sim]), {"FMD": [a, b]}, str(out), _args())

    assert len(qfiles) == 2
    workdirs = set()
    for q in qfiles:
        line = next(ln for ln in open(q).read().splitlines() if ln.startswith("cd "))
        workdirs.add(line)
    assert len(workdirs) == 2, "the two jobs must not share a working directory"


def test_generated_job_writes_a_pose_and_a_driver(tmp_path):
    from cosolvkit.cli.refine_hotspots_jobs import generate_jobs

    sim = _system(tmp_path)
    out = tmp_path / "refine"
    out.mkdir()
    generate_jobs(_config([sim]), {"FMD": [_hotspot(_occ(sim, 5, 4))]}, str(out),
                  _args())

    tag_dir = out / "hs_FMD_12"
    assert (tag_dir / "pose.pdb").is_file()
    assert (tag_dir / "manifest.json").is_file()
    assert (tag_dir / "run_autopath.py").is_file()
    assert (out / "submit_all.sh").is_file()


def test_target_without_occupancy_is_skipped(tmp_path):
    from cosolvkit.cli.refine_hotspots_jobs import generate_jobs

    sim = _system(tmp_path)
    out = tmp_path / "refine"
    out.mkdir()
    empty = Hotspot(rank=1, site_id=99, cosolvent="FMD")
    qfiles = generate_jobs(_config([sim]), {"FMD": [empty]}, str(out), _args())
    assert qfiles == []


# ---------------------------------------------------------------------------
# _build_mmgbsa_inputs
# ---------------------------------------------------------------------------

def test_mmgbsa_input_file_is_written_and_passed(tmp_path):
    """prepare_mmgbsa_batch bare-open()s mmpbsa_in, whose default 'mmgbsa.in' ships
    nowhere — an unset mmpbsa_in kills the job before it computes anything."""
    from cosolvkit.cli.refine_hotspots_jobs import _build_mmgbsa_inputs

    sim = _system(tmp_path)
    tag_dir = tmp_path / "hs_FMD_12"
    tag_dir.mkdir()
    specs = _build_mmgbsa_inputs(_config([sim]), _hotspot(_occ(sim, 5, 4)),
                                 "hs_FMD_12", str(tag_dir), _args())

    assert len(specs) == 1
    mmpbsa_in = specs[0]["mmpbsa_in"]
    assert os.path.isfile(mmpbsa_in)
    text = open(mmpbsa_in).read()
    assert "&general" in text and "&gb" in text
    # prepare_mmgbsa_batch rewrites these by prefix match; they must survive verbatim.
    for prefix in ("#startframe", "#endframe", "#interval", "strip_mask"):
        assert any(ln.startswith(prefix) for ln in text.splitlines()), prefix


def test_strip_mask_names_the_ions_as_amber_names_them(tmp_path):
    """A real prmtop calls them NA and CL; ':Na+'/':Cl-' match nothing and leave every
    counter-ion inside the MMGBSA receptor."""
    from cosolvkit.cli.refine_hotspots_jobs import SOLVENT_STRIP, _build_mmgbsa_inputs

    assert ":NA" in SOLVENT_STRIP and ":CL" in SOLVENT_STRIP
    assert "Na+" not in SOLVENT_STRIP and "Cl-" not in SOLVENT_STRIP

    sim = _system(tmp_path)
    tag_dir = tmp_path / "hs_FMD_12"
    tag_dir.mkdir()
    specs = _build_mmgbsa_inputs(_config([sim]), _hotspot(_occ(sim, 5, 4)),
                                 "hs_FMD_12", str(tag_dir), _args())
    strip = specs[0]["strip_amber_selection"]
    assert ":NA" in strip and ":CL" in strip
    assert "Na+" not in strip and "Cl-" not in strip
    assert open(specs[0]["mmpbsa_in"]).read().count(strip) == 1


def test_every_cosolvent_species_is_stripped_from_the_receptor(tmp_path):
    """A SimulationEntry may list several cosolvents; only the one ligand molecule stays.

    FMD is 5,6,7 and BEN is 8,9. Keeping FMD 5 must strip 6,7,8,9.
    """
    from cosolvkit.cli.refine_hotspots_jobs import _build_mmgbsa_inputs

    sim = _system(tmp_path, cosolvents=("FMD", "BEN"))
    tag_dir = tmp_path / "hs_FMD_12"
    tag_dir.mkdir()
    specs = _build_mmgbsa_inputs(_config([sim]), _hotspot(_occ(sim, 5, 4)),
                                 "hs_FMD_12", str(tag_dir), _args())

    strip = specs[0]["strip_amber_selection"]
    assert strip.endswith(":6-9"), strip
    assert specs[0]["ligand_amber_selection"] == ":5"


def test_second_species_stays_when_the_simulation_lists_only_one(tmp_path):
    """The mask follows the config, not a guess: a run declaring only FMD strips only
    the other FMD copies."""
    from cosolvkit.cli.refine_hotspots_jobs import _build_mmgbsa_inputs

    sim = _system(tmp_path, cosolvents=("FMD",))
    tag_dir = tmp_path / "hs_FMD_12"
    tag_dir.mkdir()
    specs = _build_mmgbsa_inputs(_config([sim]), _hotspot(_occ(sim, 5, 4)),
                                 "hs_FMD_12", str(tag_dir), _args())
    assert specs[0]["strip_amber_selection"].endswith(":6-7")


def test_each_molecule_gets_its_own_job_and_directory(tmp_path):
    """--mmgbsa-n-molecules > 1 must produce N jobs, not silently one: MMPBSA takes a
    single ligand mask, so one molecule is one job."""
    from cosolvkit.cli.refine_hotspots_jobs import _build_mmgbsa_inputs

    sim = _system(tmp_path)
    tag_dir = tmp_path / "hs_FMD_12"
    tag_dir.mkdir()
    hotspot = _hotspot(_occ(sim, 5, 4, frames=range(N_FRAMES)),
                       _occ(sim, 6, 5, frames=range(N_FRAMES)))
    specs = _build_mmgbsa_inputs(_config([sim]), hotspot, "hs_FMD_12", str(tag_dir),
                                 _args(mmgbsa_n_molecules=2))

    assert len(specs) == 2
    # Masks are indices into each job's OWN stripped complex, not the original resids.
    # The two can legitimately coincide: when molecule 5 is the ligand, 6 is stripped,
    # and vice versa, so each ends up at the same position in its own complex. What
    # must NOT happen is a raw resid, which ante-MMPBSA would resolve to nothing.
    import MDAnalysis as mda

    from cosolvkit.cli.refine_hotspots_jobs import (
        SOLVENT_STRIP, post_strip_ligand_index,
    )
    strip_names = {n for n in SOLVENT_STRIP.strip(":").split(":") if n}
    u = mda.Universe(specs[0]["prmtop"])
    probe_resids = {int(r) for r in u.select_atoms("resname FMD").resids}
    for s, resid in zip(sorted(specs, key=lambda x: x["sysname"]), (5, 6)):
        expected = post_strip_ligand_index(
            u, strip_resnames=strip_names,
            excluded_resids=probe_resids - {resid}, keep_resid=resid)
        assert s["ligand_amber_selection"] == f":{expected}"
    folders = {s["output_folder"] for s in specs}
    assert len(folders) == 2, "one molecule's frames must not overwrite another's"
    trajectories = {s["trajectory"] for s in specs}
    assert len(trajectories) == 2
    for s in specs:
        assert os.path.isfile(s["trajectory"])
        assert os.path.isfile(s["mmpbsa_in"])
    assert len({s["sysname"] for s in specs}) == 2


def test_generated_driver_calls_prepare_once_per_molecule(tmp_path):
    from cosolvkit.cli.refine_hotspots_jobs import generate_jobs

    sim = _system(tmp_path)
    out = tmp_path / "refine"
    out.mkdir()
    hotspot = _hotspot(_occ(sim, 5, 4), _occ(sim, 6, 5))
    generate_jobs(_config([sim]), {"FMD": [hotspot]}, str(out),
                  _args(mmgbsa_n_molecules=2))

    driver = (out / "hs_FMD_12" / "run_autopath.py").read_text()
    assert driver.count("prepare_mmgbsa_batch(") == 2
    assert "mmpbsa_in=" in driver
    assert "run_mmgbsa_batch.sh" in driver, "the two-stage handoff must be documented"


def test_frames_stay_paired_with_their_own_trajectory(tmp_path):
    """Frame indices are trajectory-local: pooling two replicas into one record with one
    trajectory path was a previously-fixed Critical."""
    from cosolvkit.cli.refine_hotspots_jobs import select_mmgbsa_jobs

    r0 = _system(tmp_path, label="r0")
    # Replicas share a topology but have their own trajectory — the case where a frame
    # index is only meaningful next to the file it indexes.
    r1 = SimulationEntry(trajectory=_system(tmp_path, label="r1").trajectory,
                         topology=r0.topology, cosolvents=list(r0.cosolvents),
                         label="r1")
    hotspot = _hotspot(_occ(r0, 5, 4), _occ(r1, 5, 4))
    jobs = select_mmgbsa_jobs(hotspot, _args(mmgbsa_n_frames=6))

    assert len(jobs) == 1
    _, selections = jobs[0]
    assert {occ.trajectory for occ, _ in selections} == {r0.trajectory, r1.trajectory}
    for occ, frames in selections:
        assert set(frames) <= set(occ.frames)


# ---------------------------------------------------------------------------
# _rank_targets
# ---------------------------------------------------------------------------

def _write_map(path, cosolvent):
    from gridData import Grid

    g = Grid(np.zeros((4, 4, 4)), origin=np.zeros(3), delta=np.full(3, 0.5))
    g.export(os.path.join(path, f"map_agfe_{cosolvent}.dx"))


def test_ranking_passes_the_merged_field_maps(tmp_path, monkeypatch):
    """Without field_maps, binding-site scoring falls back to a member-count-biased
    best-of-members and refine_hotspots can pick a different top-N than
    binding_sites.csv reports."""
    from cosolvkit.analysis.sites import binding_sites as bs_mod
    from cosolvkit.cli import refine_hotspots_jobs as jobs_mod

    ckpt = tmp_path / "merged"
    ckpt.mkdir()
    _write_map(str(ckpt), "FMD")

    captured = {}

    def fake(results, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(bs_mod, "identify_binding_sites", fake)
    jobs_mod._rank_targets(_config([]), {"FMD": [Hotspot(1, 1, "FMD")]},
                           _args(target="binding_sites", checkpoint=str(ckpt)))

    assert "field_maps" in captured
    assert "FMD" in captured["field_maps"], "the merged map on disk must be loaded"


def test_missing_maps_are_warned_about_by_name(tmp_path, monkeypatch, caplog):
    from cosolvkit.analysis.sites import binding_sites as bs_mod
    from cosolvkit.cli import refine_hotspots_jobs as jobs_mod

    ckpt = tmp_path / "merged"
    ckpt.mkdir()
    monkeypatch.setattr(bs_mod, "identify_binding_sites", lambda results, **kw: [])

    with caplog.at_level("WARNING"):
        jobs_mod._rank_targets(_config([]), {"FMD": [Hotspot(1, 1, "FMD")]},
                               _args(target="binding_sites", checkpoint=str(ckpt)))

    assert "binding_sites.csv" in caplog.text, (
        "the warning must name the consequence, not just the missing file"
    )


# ---------------------------------------------------------------------------
# Interpreter pinning — found by a real SLURM smoke test, where a bare `python`
# on the compute node resolved to a base install with no MDAnalysis.
# ---------------------------------------------------------------------------

def test_slurm_script_pins_an_absolute_interpreter():
    """A bare `python` resolves against the node's PATH and dies on import."""
    from cosolvkit.cli.refine_hotspots_jobs import _slurm_script

    script = _slurm_script("bs_1", "/jobs/bs_1/run_autopath.py",
                           workdir="/jobs/bs_1", mode="mmgbsa",
                           python_exe="/envs/autopath/bin/python")
    run_line = [l for l in script.splitlines()
                if l.strip().endswith("run_autopath.py")][0]
    assert run_line.startswith("/envs/autopath/bin/python "), run_line
    assert not run_line.startswith("python "), "bare python would use the node default"


def test_slurm_script_defaults_to_the_running_interpreter():
    import sys

    from cosolvkit.cli.refine_hotspots_jobs import _slurm_script

    script = _slurm_script("bs_1", "/jobs/bs_1/run_autopath.py",
                           workdir="/jobs/bs_1", mode="mmgbsa")
    assert f"{sys.executable} /jobs/bs_1/run_autopath.py" in script


def test_slurm_template_gets_the_python_placeholder(tmp_path):
    from cosolvkit.cli.refine_hotspots_jobs import _slurm_script

    tpl = tmp_path / "tpl.q"
    tpl.write_text("#!/bin/bash\ncd {{WORKDIR}}\n{{PYTHON}} {{SCRIPT}}\n")
    script = _slurm_script("bs_1", "/jobs/bs_1/run_autopath.py",
                           template_path=str(tpl), workdir="/jobs/bs_1",
                           mode="mmgbsa", python_exe="/envs/autopath/bin/python")
    assert "/envs/autopath/bin/python /jobs/bs_1/run_autopath.py" in script
    assert "{{PYTHON}}" not in script


# ---------------------------------------------------------------------------
# ante-MMPBSA ligand numbering — found by a real MMPBSA submission that died in
# Strip('') because the ligand prmtop came out empty.
# ---------------------------------------------------------------------------

def _stripping_universe():
    """protein 1-2, water 3-4, three probes 5-7 (of which 6 is the ligand)."""
    import MDAnalysis as mda

    resnames = ["ALA", "GLY", "HOH", "HOH", "FMD", "FMD", "FMD"]
    resids = [1, 2, 3, 4, 5, 6, 7]
    u = mda.Universe.empty(len(resnames), n_residues=len(resnames), n_segments=1,
                           atom_resindex=list(range(len(resnames))),
                           residue_segindex=[0] * len(resnames), trajectory=True)
    u.add_TopologyAttr("name", ["CA"] * len(resnames))
    u.add_TopologyAttr("resname", resnames)
    u.add_TopologyAttr("resid", resids)
    return u


def test_ligand_index_is_renumbered_against_the_stripped_complex():
    """ante-MMPBSA strips first, then resolves -n against the renumbered complex."""
    from cosolvkit.cli.refine_hotspots_jobs import post_strip_ligand_index

    # survivors: ALA 1, GLY 2, FMD 6  ->  the ligand is residue 3, NOT 6
    idx = post_strip_ligand_index(_stripping_universe(), strip_resnames={"HOH"},
                                  excluded_resids={5, 7}, keep_resid=6)
    assert idx == 3, "passing the original resid 6 would select nothing after stripping"


def test_ligand_index_raises_when_the_ligand_is_itself_stripped():
    from cosolvkit.cli.refine_hotspots_jobs import post_strip_ligand_index

    with pytest.raises(ValueError, match="does not survive"):
        post_strip_ligand_index(_stripping_universe(), strip_resnames={"HOH", "FMD"},
                                excluded_resids=set(), keep_resid=6)


def test_ligand_index_counts_only_survivors_before_it():
    from cosolvkit.cli.refine_hotspots_jobs import post_strip_ligand_index

    # nothing stripped: the index is just the position, which here equals the resid
    idx = post_strip_ligand_index(_stripping_universe(), strip_resnames=set(),
                                  excluded_resids=set(), keep_resid=6)
    assert idx == 6


# ---------------------------------------------------------------------------
# Metal handling under decomposition — found by a real MMPBSA run where sander
# aborted with "bad atom type: Mn" the moment idecomp was switched on.
# ---------------------------------------------------------------------------

def _metal_universe(metal_xyz, ligand_xyz, metal_resname="MN"):
    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader

    resnames = ["ALA", metal_resname, "FMD"]
    u = mda.Universe.empty(3, n_residues=3, n_segments=1, atom_resindex=[0, 1, 2],
                           residue_segindex=[0, 0, 0], trajectory=True)
    u.add_TopologyAttr("name", ["CA", metal_resname, "C1"])
    u.add_TopologyAttr("resname", resnames)
    u.add_TopologyAttr("resid", [1, 2, 3])
    coords = np.array([[[0.0, 0.0, 0.0], metal_xyz, ligand_xyz]], dtype=np.float32)
    u.load_new(coords, order="fac", format=MemoryReader,
               dimensions=np.array([[80.0] * 3 + [90.0] * 3]))
    return u


def test_distant_metal_is_reported_with_its_distance():
    from cosolvkit.cli.refine_hotspots_jobs import metals_blocking_decomp

    u = _metal_universe([20.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    found = metals_blocking_decomp(u, ligand_resid=3)
    assert len(found) == 1
    name, resid, dist = found[0]
    assert (name, resid) == ("MN", 2)
    assert dist == pytest.approx(20.0)


def test_a_protein_without_metals_reports_nothing():
    from cosolvkit.cli.refine_hotspots_jobs import metals_blocking_decomp

    u = _metal_universe([20.0, 0.0, 0.0], [0.0, 0.0, 0.0], metal_resname="GLY")
    assert metals_blocking_decomp(u, ligand_resid=3) == []


def test_decomp_block_is_present_only_when_requested():
    from cosolvkit.cli.refine_hotspots_jobs import (
        DECOMP_BLOCK, MMPBSA_IN_TEMPLATE,
    )

    on = MMPBSA_IN_TEMPLATE.format(strip_mask=":X", decomp=DECOMP_BLOCK)
    off = MMPBSA_IN_TEMPLATE.format(strip_mask=":X", decomp="")
    assert "idecomp=1" in on and "csv_format=1" in on
    assert "&decomp" not in off


def test_no_decomp_flag_parses():
    from cosolvkit.cli.refine_hotspots import build_parser

    assert build_parser().parse_args(["--config", "a.yaml"]).decomp is True
    assert build_parser().parse_args(["--config", "a.yaml", "--no-decomp"]).decomp is False
