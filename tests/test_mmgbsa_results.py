"""Finished-MMGBSA parsing and the stripped-complex -> original-resid mapping.

The mapping is the load-bearing piece: ante-MMPBSA strips solvent/ions/other-probe
copies (and, for decomposition, unsupported metals) and renumbers survivors from 1.
Every test that touches ``stripped_index_to_resid`` therefore includes at least one
stripped residue BEFORE the residue being mapped, so an identity-mapping bug (index
happens to equal resid) cannot pass silently.
"""

import os
import textwrap

import pytest

try:
    import MDAnalysis as mda
    HAS_MDA = True
except ImportError:
    HAS_MDA = False

from cosolvkit.analysis.core.models import MmgbsaResult
from cosolvkit.analysis.sites.mmgbsa_results import (
    collect_decomposition,
    collect_job,
    parse_decomposition,
    parse_probe_dirname,
    parse_results,
    parse_strip_mask,
    stripped_index_to_resid,
)

pytestmark = pytest.mark.skipif(not HAS_MDA, reason="MDAnalysis not available")


# ---------------------------------------------------------------------------
# Fixture file contents — modelled on a real FosAKP FMD289 MMGBSA run.
# ---------------------------------------------------------------------------

FINAL_RESULTS_TEXT = textwrap.dedent("""\
    GENERALIZED BORN:

    Complex:
    Energy Component            Average              Std. Dev.   Std. Err. of Mean
    -------------------------------------------------------------------------------
    VDWAALS                  -2227.5670                9.4423              4.2227
    TOTAL                   -24004.4472               83.3559             37.2779


    Receptor:
    Energy Component            Average              Std. Dev.   Std. Err. of Mean
    -------------------------------------------------------------------------------
    TOTAL                   -24012.1924               83.0000             37.0000


    Ligand:
    Energy Component            Average              Std. Dev.   Std. Err. of Mean
    -------------------------------------------------------------------------------
    TOTAL                      -11.7441                0.1506              0.0674


    Differences (Complex - Receptor - Ligand):
    Energy Component            Average              Std. Dev.   Std. Err. of Mean
    -------------------------------------------------------------------------------
    VDWAALS                     -6.0928                1.6139              0.7217
    EEL                         -5.5309                3.4263              1.5323
    EGB                          9.9886                3.2409              1.4494
    ESURF                       -1.6091                0.0661              0.0296

    DELTA G gas                -11.6237                3.5745              1.5986
    DELTA G solv                 8.3795                3.2435              1.4505

    DELTA TOTAL                 -3.2442                1.1038              0.4936

    """)

# Stripped-complex numbering: original resid 2 (MN) is stripped, so stripped index 2
# maps to original resid 3 (LEU), not original resid 2. Index 1 (ALA, resid 1) sits
# BEFORE the stripped residue and is unaffected by it — the contrast is the point.
DECOMP_TEXT = textwrap.dedent("""\
    | Run on a test machine
    Energy Decomposition Analysis (All units kcal/mol): Generalized Born solvent
    Complex:
    Total Energy Decomposition:
    Residue,Location,Internal,,,van der Waals,,,Electrostatic,,,Polar Solvation,,,Non-Polar Solv.,,,TOTAL,,
    ,,Avg.,Std. Dev.,Std. Err. of Mean,Avg.,Std. Dev.,Std. Err. of Mean,Avg.,Std. Dev.,Std. Err. of Mean,Avg.,Std. Dev.,Std. Err. of Mean,Avg.,Std. Dev.,Std. Err. of Mean,Avg.,Std. Dev.,Std. Err. of Mean
    ALA   1,R ALA   1,0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.01,0.001,-0.1,0.01,0.001,0.0,0.0,0.0,0.0,0.001,0.0001
    LEU   2,R LEU   2,0.0,0.0,0.0,0.0,0.0,0.0,0.2,0.01,0.001,-0.2,0.01,0.001,0.0,0.0,0.0,0.0,0.001,0.0001
    MET   3,R MET   3,0.0,0.0,0.0,0.0,0.0,0.0,-5.0,0.5,0.05,3.0,0.3,0.03,0.0,0.0,0.0,-2.0,0.1,0.01
    FMD   4,L FMD   4,0.0,0.0,0.0,-1.0,0.1,0.01,-2.0,0.2,0.02,3.0,0.3,0.03,-0.1,0.01,0.001,-0.1,0.05,0.005

    Run on a test machine, done.

    DELTAS:
    Total Energy Decomposition:
    Residue,Location,Internal,,,van der Waals,,,Electrostatic,,,Polar Solvation,,,Non-Polar Solv.,,,TOTAL,,
    ,,Avg.,Std. Dev.,Std. Err. of Mean,Avg.,Std. Dev.,Std. Err. of Mean,Avg.,Std. Dev.,Std. Err. of Mean,Avg.,Std. Dev.,Std. Err. of Mean,Avg.,Std. Dev.,Std. Err. of Mean,Avg.,Std. Dev.,Std. Err. of Mean
    ALA   1,R ALA   1,0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.01,0.001,-0.1,0.01,0.001,0.0,0.0,0.0,0.0,0.001,0.0001
    LEU   2,R LEU   2,0.0,0.0,0.0,0.0,0.0,0.0,0.2,0.01,0.001,-0.2,0.01,0.001,0.0,0.0,0.0,0.0,0.001,0.0001
    MET   3,R MET   3,0.0,0.0,0.0,0.0,0.0,0.0,-5.0,0.5,0.05,3.0,0.3,0.03,0.0,0.0,0.0,-2.0,0.1,0.01
    FMD   4,L FMD   4,0.0,0.0,0.0,-1.0,0.1,0.01,-2.0,0.2,0.02,3.0,0.3,0.03,-0.1,0.01,0.001,-0.1,0.05,0.005

    """)


def _synthetic_universe(resnames, resids):
    """A tiny topology-only Universe: one CA atom per residue."""
    n = len(resnames)
    u = mda.Universe.empty(n, n_residues=n, n_segments=1,
                           atom_resindex=list(range(n)),
                           residue_segindex=[0] * n, trajectory=False)
    u.add_TopologyAttr("name", ["CA"] * n)
    u.add_TopologyAttr("resname", resnames)
    u.add_TopologyAttr("resid", resids)
    return u


@pytest.fixture
def synthetic_topology():
    """resid 1=ALA, 2=MN (stripped), 3=LEU, 4=MET, 5=FMD (the ligand)."""
    return _synthetic_universe(["ALA", "MN", "LEU", "MET", "FMD"], [1, 2, 3, 4, 5])


# ---------------------------------------------------------------------------
# parse_strip_mask
# ---------------------------------------------------------------------------

def test_parse_strip_mask_splits_resnames_from_resid_ranges():
    resnames, resids = parse_strip_mask(
        ':POP:HOH:WAT:NA:CL:K:MG:279-288,290-315:MN'
    )
    assert resnames == {"POP", "HOH", "WAT", "NA", "CL", "K", "MG", "MN"}
    assert resids == set(range(279, 289)) | set(range(290, 316))
    assert 289 not in resids


def test_parse_strip_mask_handles_no_excluded_resids():
    resnames, resids = parse_strip_mask(":POP:HOH:WAT")
    assert resnames == {"POP", "HOH", "WAT"}
    assert resids == set()


# ---------------------------------------------------------------------------
# stripped_index_to_resid — the critical correctness surface
# ---------------------------------------------------------------------------

def test_mapping_shifts_after_a_stripped_residue_but_not_before(synthetic_topology):
    mapping = stripped_index_to_resid(synthetic_topology, {"MN"}, set())
    assert mapping == {1: 1, 2: 3, 3: 4, 4: 5}
    # The residue before the strip point is unaffected — the contrast that would let
    # a wrong (identity) mapping slip through unnoticed.
    assert mapping[1] == 1
    # Every index at or after the strip point genuinely differs from its resid.
    assert mapping[2] != 2
    assert mapping[3] != 3
    assert mapping[4] != 4


def test_mapping_also_respects_excluded_resids(synthetic_topology):
    """Other probe copies are excluded by resid, not resname, e.g. a second FMD."""
    u = _synthetic_universe(["ALA", "FMD", "LEU", "MET", "FMD"], [1, 2, 3, 4, 5])
    mapping = stripped_index_to_resid(u, set(), {2})
    assert mapping == {1: 1, 2: 3, 3: 4, 4: 5}


def test_round_trip_with_post_strip_ligand_index(synthetic_topology):
    """The inverse relationship the whole module exists to guarantee."""
    from cosolvkit.cli.refine_hotspots_jobs import post_strip_ligand_index

    strip_resnames = {"MN"}
    excluded_resids = set()
    for keep_resid in (1, 3, 4, 5):
        idx = post_strip_ligand_index(synthetic_topology, strip_resnames,
                                      excluded_resids, keep_resid)
        mapping = stripped_index_to_resid(synthetic_topology, strip_resnames,
                                          excluded_resids)
        assert mapping[idx] == keep_resid


REAL_TOPOLOGY = (
    "/mnt/forli/group/pscio/cosolvkitv2/simulation_outputs/FosAKP/"
    "out_250ns_panel_manu/formamide/r2/system.prmtop"
)


@pytest.mark.skipif(not os.path.isfile(REAL_TOPOLOGY),
                    reason="real FosAKP topology not reachable from this machine")
def test_round_trip_against_the_real_fosakp_topology():
    """FMD 289 is stripped-index 277 in the real hs_FMD_1 run (solvent + FMD
    279-315 + MN 139 + MN 278 stripped) — verified independently against the actual
    FINAL_DECOMP_mmpbsa.dat, whose 'FMD 277' DELTAS row is this same molecule.
    MN 139 sits BEFORE resid 289, so this is not the trivial before-the-strip case.
    """
    from cosolvkit.cli.refine_hotspots_jobs import (SOLVENT_STRIP,
                                                    post_strip_ligand_index)

    u = mda.Universe(REAL_TOPOLOGY)
    strip_resnames = {n for n in SOLVENT_STRIP.strip(":").split(":") if n} | {"MN"}
    all_fmd = {int(r.resid) for r in u.residues if r.resname == "FMD"}
    excluded_resids = all_fmd - {289}

    idx = post_strip_ligand_index(u, strip_resnames, excluded_resids, keep_resid=289)
    assert idx == 277

    mapping = stripped_index_to_resid(u, strip_resnames, excluded_resids)
    assert mapping[277] == 289
    # And the shift-by-one-metal case the smoke test's pocket residues never exercise:
    # MN 139 is stripped, so everything from resid 140 on shifts down by one.
    assert mapping[139] == 140


# ---------------------------------------------------------------------------
# parse_results / parse_decomposition
# ---------------------------------------------------------------------------

def test_parse_results_reads_the_differences_table(tmp_path):
    path = tmp_path / "FINAL_RESULTS_mmpbsa.dat"
    path.write_text(FINAL_RESULTS_TEXT)

    parsed = parse_results(str(path))
    assert parsed["delta_total"] == pytest.approx(-3.2442)
    assert parsed["std_dev"] == pytest.approx(1.1038)
    assert parsed["std_err"] == pytest.approx(0.4936)
    assert parsed["components"]["VDWAALS"] == {
        "average": pytest.approx(-6.0928),
        "std_dev": pytest.approx(1.6139),
        "std_err": pytest.approx(0.7217),
    }
    assert "DELTA TOTAL" in parsed["components"]


def test_parse_decomposition_returns_the_deltas_table(tmp_path):
    path = tmp_path / "FINAL_DECOMP_mmpbsa.dat"
    path.write_text(DECOMP_TEXT)

    df = parse_decomposition(str(path))
    assert len(df) == 4
    assert set(df["resid"]) == {1, 2, 3, 4}
    assert list(df["location"]) == ["R", "R", "R", "L"]


REAL_DECOMP = (
    "/gpfs/group/forli/mllanos/cosolvkitV2/benchmarking/FosAKP/refine_smoketest/"
    "hs_FMD_1/mmgbsa/FMD289/FINAL_DECOMP_mmpbsa.dat"
)
REAL_RESULTS = (
    "/gpfs/group/forli/mllanos/cosolvkitV2/benchmarking/FosAKP/refine_smoketest/"
    "hs_FMD_1/mmgbsa/FMD289/FINAL_RESULTS_mmpbsa.dat"
)


@pytest.mark.skipif(not os.path.isfile(REAL_RESULTS),
                    reason="real FosAKP MMGBSA run not reachable from this machine")
def test_parse_results_against_the_real_run():
    """Check the parse against the file's own contents, not a pinned number.

    This file is a real run that gets regenerated whenever the smoke test is re-run with
    different settings (igb 8 -> 5 moved it from -3.2442 to -3.9864), so hardcoding the
    energy tests the run rather than the parser.
    """
    parsed = parse_results(REAL_RESULTS)

    expected = {}
    in_diff = False
    for line in open(REAL_RESULTS):
        if line.startswith("Differences (Complex - Receptor - Ligand)"):
            in_diff = True
            continue
        if in_diff and line.strip().startswith("DELTA TOTAL"):
            expected["total"] = float(line.split()[2])
        if in_diff and line.strip().startswith("VDWAALS"):
            expected["vdw"] = float(line.split()[1])

    assert parsed["delta_total"] == pytest.approx(expected["total"])
    assert parsed["components"]["VDWAALS"]["average"] == pytest.approx(expected["vdw"])
    for key in ("VDWAALS", "EEL", "EGB", "ESURF"):
        assert key in parsed["components"], key
    assert parsed["std_err"] > 0


@pytest.mark.skipif(not os.path.isfile(REAL_DECOMP),
                    reason="real FosAKP decomposition not reachable from this machine")
def test_parse_decomposition_against_the_real_run():
    df = parse_decomposition(REAL_DECOMP)
    assert len(df) == 277
    ligand_row = df[df["location"] == "L"].iloc[0]
    assert ligand_row["resname"] == "FMD"
    assert int(ligand_row["resid"]) == 277


# ---------------------------------------------------------------------------
# parse_probe_dirname
# ---------------------------------------------------------------------------

def test_parse_probe_dirname_splits_resname_and_resid():
    assert parse_probe_dirname("FMD289") == ("FMD", 289)


def test_parse_probe_dirname_rejects_a_bad_name():
    with pytest.raises(ValueError):
        parse_probe_dirname("not-a-probe-dir")


# ---------------------------------------------------------------------------
# collect_job
# ---------------------------------------------------------------------------

def test_collect_job_returns_none_when_not_finished_yet(tmp_path, caplog):
    with caplog.at_level("INFO"):
        result = collect_job(str(tmp_path), "FMD", 289, "r0")
    assert result is None
    assert "no FINAL_RESULTS_mmpbsa.dat" in caplog.text


def test_collect_job_uses_frames_json_n_frames(tmp_path):
    (tmp_path / "FINAL_RESULTS_mmpbsa.dat").write_text(FINAL_RESULTS_TEXT)
    (tmp_path / "frames.json").write_text(
        '{"n_frames": 5, "frames": [{"source_label": "formamide_r2"}]}'
    )

    result = collect_job(str(tmp_path), "FMD", 289, "formamide_r2")
    assert isinstance(result, MmgbsaResult)
    assert result.n_frames == 5
    assert result.delta_total == pytest.approx(-3.2442)
    assert result.probe_resname == "FMD"
    assert result.probe_resid == 289
    assert result.results_path == str(tmp_path / "FINAL_RESULTS_mmpbsa.dat")


def test_collect_job_falls_back_to_rewritten_mmgbsa_in_endframe(tmp_path):
    """frames.json missing — n_frames must come from the *_mmgbsa.in AutoPath wrote."""
    (tmp_path / "FINAL_RESULTS_mmpbsa.dat").write_text(FINAL_RESULTS_TEXT)
    (tmp_path / "hs_FMD_1_FMD289_formamide_r2_mmgbsa.in").write_text(
        "&general\nstartframe = 1,\nendframe = 5,\ninterval = 1,\n/\n"
    )

    result = collect_job(str(tmp_path), "FMD", 289, "formamide_r2")
    assert result.n_frames == 5


# ---------------------------------------------------------------------------
# collect_decomposition
# ---------------------------------------------------------------------------

def test_collect_decomposition_maps_stripped_indices_to_original_resids(
    tmp_path, synthetic_topology
):
    path = tmp_path / "FINAL_DECOMP_mmpbsa.dat"
    path.write_text(DECOMP_TEXT)

    df = collect_decomposition(str(tmp_path), synthetic_topology, {"MN"}, set())
    assert list(df["stripped_resid"]) == [1, 2, 3, 4]
    # The load-bearing assertion: original resid, not stripped index.
    assert list(df["resid"]) == [1, 3, 4, 5]
    assert list(df["resname"]) == ["ALA", "LEU", "MET", "FMD"]


def test_collect_decomposition_returns_none_when_absent(tmp_path, synthetic_topology):
    assert collect_decomposition(str(tmp_path), synthetic_topology, {"MN"}, set()) is None


def test_collect_decomposition_raises_on_resname_mismatch(tmp_path):
    """A wrong strip mask silently mis-maps residues; this must fail loudly instead."""
    path = tmp_path / "FINAL_DECOMP_mmpbsa.dat"
    path.write_text(DECOMP_TEXT)

    # Nothing stripped: stripped index 2 maps to resid 2, but the decomp table's
    # index 2 is "LEU" while this topology's resid 2 is "MN" — a real mismatch.
    wrong_topology = _synthetic_universe(["ALA", "MN", "LEU", "MET", "FMD"],
                                         [1, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="does not match"):
        collect_decomposition(str(tmp_path), wrong_topology, set(), set())
