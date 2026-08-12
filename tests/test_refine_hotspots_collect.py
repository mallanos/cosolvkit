"""`refine_hotspots --collect`: attaching finished MMGBSA results to the checkpoint.

End-to-end: writes a real checkpoint, real job-directory files (mmgbsa.in,
FINAL_RESULTS_mmpbsa.dat, FINAL_DECOMP_mmpbsa.dat, frames.json) and a synthetic
topology, then runs the collector against them exactly as it would run against a real
``<out>/*/mmgbsa/*/`` tree.
"""

import json
import textwrap
from types import SimpleNamespace

import pytest

try:
    import MDAnalysis as mda
    HAS_MDA = True
except ImportError:
    HAS_MDA = False

from cosolvkit.analysis.core.models import PocketResidue, ProbeOccupancy
from cosolvkit.analysis.sites.detect import HotspotDetector
from cosolvkit.cli.refine_hotspots import (
    _attach_mmgbsa_result,
    _cosolvent_list,
    _find_target_hotspot,
    _parse_target_tag,
    _representative_occupancy,
    collect_results,
)

pytestmark = pytest.mark.skipif(not HAS_MDA, reason="MDAnalysis not available")


# ---------------------------------------------------------------------------
# _parse_target_tag / _cosolvent_list — pure, no I/O
# ---------------------------------------------------------------------------

def test_parse_target_tag_hotspot():
    assert _parse_target_tag("hs_FMD_12") == (False, "FMD", 12)


def test_parse_target_tag_binding_site():
    assert _parse_target_tag("bs_7") == (True, None, 7)


def test_parse_target_tag_rejects_unknown_shape():
    with pytest.raises(ValueError):
        _parse_target_tag("weird_1")


def test_cosolvent_list_dedupes_in_first_seen_order():
    config = SimpleNamespace(simulations=[
        SimpleNamespace(cosolvents=["FMD", "BEN"]),
        SimpleNamespace(cosolvents=["BEN", "MPD"]),
    ])
    assert _cosolvent_list(config) == ["FMD", "BEN", "MPD"]


# ---------------------------------------------------------------------------
# _find_target_hotspot / _representative_occupancy / _attach_mmgbsa_result
# ---------------------------------------------------------------------------

def _occ(source, resname, resid, n_frames_bound):
    return ProbeOccupancy(
        source_label=source, topology="/tmp/t.prmtop", trajectory="/tmp/t.dcd",
        probe_resname=resname, probe_resid=resid, probe_resindex=resid - 1,
        frames=list(range(n_frames_bound)), n_frames_scanned=100, stride=1,
    )


def test_find_target_hotspot_for_a_hotspot_tag(make_hotspot):
    h1 = make_hotspot(site_id=1, cosolvent="FMD")
    h2 = make_hotspot(site_id=2, cosolvent="FMD")
    results = {"FMD": [h1, h2]}
    assert _find_target_hotspot(results, False, "FMD", 2, "FMD", 289, []) is h2


def test_find_target_hotspot_for_a_binding_site_tag_with_one_candidate(make_hotspot):
    h = make_hotspot(site_id=1, cosolvent="FMD")
    h.probe_occupancy = [_occ("r0", "FMD", 289, 5)]
    results = {"FMD": [h]}
    assert _find_target_hotspot(results, True, None, 99, "FMD", 289, ["r0"]) is h


def test_find_target_hotspot_disambiguates_by_source_label_overlap(make_hotspot):
    """Same molecule occupying two hotspots of the same cosolvent: pick the one whose
    occupancy source labels match the job's frames.json, not an arbitrary one."""
    h1 = make_hotspot(site_id=1, cosolvent="FMD")
    h1.probe_occupancy = [_occ("r0", "FMD", 289, 3)]
    h2 = make_hotspot(site_id=2, cosolvent="FMD")
    h2.probe_occupancy = [_occ("r1", "FMD", 289, 50)]
    results = {"FMD": [h1, h2]}
    picked = _find_target_hotspot(results, True, None, 0, "FMD", 289,
                                  source_labels=["r0"])
    assert picked is h1, "source-label overlap must win over raw frame count"


def test_find_target_hotspot_returns_none_when_nothing_matches(make_hotspot):
    h = make_hotspot(site_id=1, cosolvent="FMD")
    assert _find_target_hotspot({"FMD": [h]}, False, "FMD", 99, "FMD", 289, []) is None


def test_representative_occupancy_matches_select_mmgbsa_jobs_tiebreak(make_hotspot):
    h = make_hotspot(site_id=1, cosolvent="FMD")
    h.probe_occupancy = [_occ("r_b", "FMD", 289, 5), _occ("r_a", "FMD", 289, 5)]
    rep = _representative_occupancy(h, "FMD", 289)
    # Equal n_frames_bound -> tie-break on source_label, same as select_mmgbsa_jobs.
    assert rep.source_label == "r_b"


def test_attach_mmgbsa_result_replaces_rather_than_duplicates(make_hotspot):
    from cosolvkit.analysis.core.models import MmgbsaResult

    h = make_hotspot(site_id=1, cosolvent="FMD")
    first = MmgbsaResult(probe_resname="FMD", probe_resid=289, source_label="r0",
                         delta_total=-3.0, std_dev=1.0, std_err=0.4, n_frames=5,
                         results_path="/tmp/a/FINAL_RESULTS_mmpbsa.dat")
    updated = MmgbsaResult(probe_resname="FMD", probe_resid=289, source_label="r0",
                           delta_total=-9.0, std_dev=1.0, std_err=0.4, n_frames=5,
                           results_path="/tmp/a/FINAL_RESULTS_mmpbsa.dat")
    _attach_mmgbsa_result(h, first)
    _attach_mmgbsa_result(h, updated)
    assert len(h.mmgbsa) == 1
    assert h.mmgbsa[0].delta_total == -9.0


# ---------------------------------------------------------------------------
# End-to-end: real checkpoint + real job directory + synthetic topology
# ---------------------------------------------------------------------------

FINAL_RESULTS_TEXT = textwrap.dedent("""\
    Differences (Complex - Receptor - Ligand):
    Energy Component            Average              Std. Dev.   Std. Err. of Mean
    -------------------------------------------------------------------------------
    VDWAALS                     -6.0928                1.6139              0.7217
    EEL                         -5.5309                3.4263              1.5323

    DELTA TOTAL                 -3.2442                1.1038              0.4936

    """)

# resid 1 = ALA, 2 = MN (stripped), 3 = LEU, 4 = MET, 5 = FMD (the ligand). Stripping
# MN shifts everything from resid 3 on down by one in the DELTAS numbering.
DECOMP_TEXT = textwrap.dedent("""\
    DELTAS:
    Total Energy Decomposition:
    Residue,Location,Internal,,,van der Waals,,,Electrostatic,,,Polar Solvation,,,Non-Polar Solv.,,,TOTAL,,
    ,,Avg.,Std. Dev.,Std. Err. of Mean,Avg.,Std. Dev.,Std. Err. of Mean,Avg.,Std. Dev.,Std. Err. of Mean,Avg.,Std. Dev.,Std. Err. of Mean,Avg.,Std. Dev.,Std. Err. of Mean,Avg.,Std. Dev.,Std. Err. of Mean
    ALA   1,R ALA   1,0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.01,0.001,-0.1,0.01,0.001,0.0,0.0,0.0,0.0,0.001,0.0001
    LEU   2,R LEU   2,0.0,0.0,0.0,0.0,0.0,0.0,0.2,0.01,0.001,-0.2,0.01,0.001,0.0,0.0,0.0,0.0,0.001,0.0001
    MET   3,R MET   3,0.0,0.0,0.0,0.0,0.0,0.0,-5.0,0.5,0.05,3.0,0.3,0.03,0.0,0.0,0.0,-2.0,0.1,0.01
    FMD   4,L FMD   4,0.0,0.0,0.0,-1.0,0.1,0.01,-2.0,0.2,0.02,3.0,0.3,0.03,-0.1,0.01,0.001,-0.1,0.05,0.005

    """)


def _write_synthetic_topology(path):
    u = mda.Universe.empty(5, n_residues=5, n_segments=1, atom_resindex=list(range(5)),
                           residue_segindex=[0] * 5, trajectory=True)
    u.add_TopologyAttr("name", ["CA"] * 5)
    u.add_TopologyAttr("resname", ["ALA", "MN", "LEU", "MET", "FMD"])
    u.add_TopologyAttr("resid", [1, 2, 3, 4, 5])
    u.atoms.write(str(path))


@pytest.fixture
def collected_checkpoint(tmp_path, make_hotspot):
    """A checkpoint + refine/ tree with one finished FMD5 MMGBSA+decomposition job."""
    topo_path = tmp_path / "system.pdb"
    _write_synthetic_topology(topo_path)

    h = make_hotspot(site_id=1, cosolvent="FMD")
    h.pocket_residues = [
        PocketResidue(resid=1, resindex=0, resname="ALA", chain="A",
                     n_contact_voxels=1, min_dist_ang=3.0, contact_fraction=0.1),
        PocketResidue(resid=3, resindex=2, resname="LEU", chain="A",
                     n_contact_voxels=1, min_dist_ang=3.0, contact_fraction=0.1),
        PocketResidue(resid=4, resindex=3, resname="MET", chain="A",
                     n_contact_voxels=1, min_dist_ang=3.0, contact_fraction=0.1),
    ]
    h.probe_occupancy = [ProbeOccupancy(
        source_label="r0", topology=str(topo_path), trajectory="/tmp/t.dcd",
        probe_resname="FMD", probe_resid=5, probe_resindex=4,
        frames=[0, 1, 2], n_frames_scanned=10, stride=1,
    )]

    checkpoint_dir = tmp_path / "merged"
    HotspotDetector.save_checkpoint({"FMD": [h]}, str(checkpoint_dir))

    out_dir = tmp_path / "refine"
    job_dir = out_dir / "hs_FMD_1" / "mmgbsa" / "FMD5"
    job_dir.mkdir(parents=True)
    (job_dir / "mmgbsa.in").write_text('strip_mask= ":MN"\n')
    (job_dir / "FINAL_RESULTS_mmpbsa.dat").write_text(FINAL_RESULTS_TEXT)
    (job_dir / "FINAL_DECOMP_mmpbsa.dat").write_text(DECOMP_TEXT)
    (job_dir / "frames.json").write_text(json.dumps({
        "n_frames": 3,
        "frames": [{"index": i, "source_label": "r0", "original_frame": i}
                  for i in range(3)],
    }))

    config = SimpleNamespace(simulations=[SimpleNamespace(cosolvents=["FMD"])])
    return config, str(checkpoint_dir), str(out_dir)


def test_collect_attaches_mmgbsa_total_to_the_hotspot(collected_checkpoint):
    config, checkpoint_dir, out_dir = collected_checkpoint
    results = collect_results(config, checkpoint_dir, out_dir)

    h = results["FMD"][0]
    assert len(h.mmgbsa) == 1
    assert h.mmgbsa[0].delta_total == pytest.approx(-3.2442)
    assert h.mmgbsa[0].n_frames == 3
    assert h.mmgbsa[0].probe_resid == 5


def test_collect_maps_decomposition_onto_pocket_residues_by_original_resid(
    collected_checkpoint
):
    config, checkpoint_dir, out_dir = collected_checkpoint
    results = collect_results(config, checkpoint_dir, out_dir)

    by_resid = {pr.resid: pr for pr in results["FMD"][0].pocket_residues}
    # resid 1 (before the strip point): stripped index == resid, identity by
    # coincidence.
    assert by_resid[1].mmgbsa_decomposition["Electrostatic"]["average"] == pytest.approx(0.1)
    assert by_resid[1].mmgbsa_location == "R"
    # resid 3 (after MN is stripped): stripped index 2 must map to resid 3, not 2.
    assert by_resid[3].mmgbsa_decomposition["Electrostatic"]["average"] == pytest.approx(0.2)
    # resid 4: stripped index 3.
    assert by_resid[4].mmgbsa_decomposition["Electrostatic"]["average"] == pytest.approx(-5.0)


def test_collect_writes_both_csvs(collected_checkpoint):
    import pandas as pd

    config, checkpoint_dir, out_dir = collected_checkpoint
    collect_results(config, checkpoint_dir, out_dir)

    results_csv = pd.read_csv(f"{out_dir}/mmgbsa_results.csv")
    assert len(results_csv) == 1
    assert results_csv.loc[0, "delta_total"] == pytest.approx(-3.2442)

    decomp_csv = pd.read_csv(f"{out_dir}/mmgbsa_decomposition.csv")
    assert len(decomp_csv) == 4  # 3 receptor residues + 1 ligand row
    # The auditable pair: both numberings survive into the CSV.
    row = decomp_csv[decomp_csv["stripped_resid"] == 2].iloc[0]
    assert row["resid"] == 3
    assert row["resname"] == "LEU"


def test_collect_is_idempotent(collected_checkpoint):
    config, checkpoint_dir, out_dir = collected_checkpoint
    collect_results(config, checkpoint_dir, out_dir)
    results = collect_results(config, checkpoint_dir, out_dir)

    h = results["FMD"][0]
    assert len(h.mmgbsa) == 1, "re-collecting must replace, not duplicate"

    import pandas as pd
    assert len(pd.read_csv(f"{out_dir}/mmgbsa_results.csv")) == 1
    assert len(pd.read_csv(f"{out_dir}/mmgbsa_decomposition.csv")) == 4


def test_collect_survives_a_job_that_has_not_finished_yet(tmp_path, make_hotspot):
    """A job dir with no FINAL_RESULTS_mmpbsa.dat must not crash the collector."""
    h = make_hotspot(site_id=1, cosolvent="FMD")
    h.probe_occupancy = [ProbeOccupancy(
        source_label="r0", topology="/tmp/t.prmtop", trajectory="/tmp/t.dcd",
        probe_resname="FMD", probe_resid=5, probe_resindex=4,
        frames=[0, 1], n_frames_scanned=10, stride=1,
    )]
    checkpoint_dir = tmp_path / "merged"
    HotspotDetector.save_checkpoint({"FMD": [h]}, str(checkpoint_dir))

    out_dir = tmp_path / "refine"
    job_dir = out_dir / "hs_FMD_1" / "mmgbsa" / "FMD5"
    job_dir.mkdir(parents=True)
    (job_dir / "mmgbsa.in").write_text('strip_mask= ""\n')

    config = SimpleNamespace(simulations=[SimpleNamespace(cosolvents=["FMD"])])
    results = collect_results(config, str(checkpoint_dir), str(out_dir))
    assert results["FMD"][0].mmgbsa == []
