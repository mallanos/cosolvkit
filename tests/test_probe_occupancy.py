"""ProbeOccupancy: frame bookkeeping for one probe molecule in one hotspot."""

import pytest

from cosolvkit.analysis.core.models import ProbeOccupancy


def _occ(frames, stride=1, n_scanned=100):
    return ProbeOccupancy(
        source_label="formamide_r0",
        topology="/tmp/system.prmtop",
        trajectory="/tmp/traj.dcd",
        probe_resname="FMD",
        probe_resid=279,
        probe_resindex=278,
        frames=list(frames),
        n_frames_scanned=n_scanned,
        stride=stride,
    )


def test_gap_free_run_is_one_episode():
    assert _occ([4, 5, 6, 7]).episodes() == [(4, 7)]


def test_gap_splits_into_two_episodes():
    assert _occ([1, 2, 10, 11]).episodes() == [(1, 2), (10, 11)]


def test_gap_tolerance_rejoins_a_single_frame_gap():
    # frames 5 and 7 are two apart; stride 1 * (1 + 1) = 2 bridges them
    assert _occ([4, 5, 7, 8]).episodes(gap_tolerance=1) == [(4, 8)]
    assert _occ([4, 5, 7, 8]).episodes(gap_tolerance=0) == [(4, 5), (7, 8)]


def test_stride_makes_spaced_frames_contiguous():
    assert _occ([0, 10, 20], stride=10).episodes() == [(0, 20)]


def test_empty_frames_has_no_episodes():
    occ = _occ([])
    assert occ.episodes() == []
    assert occ.longest_episode() is None
    assert occ.n_frames_bound == 0


def test_longest_episode_picks_the_widest_and_ties_go_to_the_first():
    assert _occ([1, 2, 10, 11, 12]).longest_episode() == (10, 12)
    assert _occ([1, 2, 10, 11]).longest_episode() == (1, 2)


def test_occupancy_fraction_and_amber_mask():
    occ = _occ([1, 2, 3], n_scanned=10)
    assert occ.n_frames_bound == 3
    assert occ.occupancy_fraction == pytest.approx(0.3)
    assert occ.amber_mask == ":279"


def test_round_trip_through_dict():
    occ = _occ([1, 2, 3])
    back = ProbeOccupancy.from_dict(occ.to_dict())
    assert back == occ


from cosolvkit.analysis.core.models import Hotspot, PoseRef


def _hotspot_with(*occs):
    h = Hotspot(rank=1, site_id=1, cosolvent="FMD")
    h.probe_occupancy = list(occs)
    return h


def _named(label, resid, frames, stride=1, n_scanned=100):
    return ProbeOccupancy(
        source_label=label, topology=f"/tmp/{label}.prmtop",
        trajectory=f"/tmp/{label}.dcd", probe_resname="FMD",
        probe_resid=resid, probe_resindex=resid - 1, frames=list(frames),
        n_frames_scanned=n_scanned, stride=stride,
    )


def test_new_hotspot_has_no_occupancy():
    h = Hotspot(rank=1, site_id=1, cosolvent="FMD")
    assert h.probe_occupancy == []
    assert h.n_probe_molecules == 0
    assert h.total_residence_frames == 0
    assert h.best_pose() is None


def test_rollups_count_molecules_and_frames():
    h = _hotspot_with(_named("r0", 279, [1, 2]), _named("r1", 280, [5, 6, 7]))
    assert h.n_probe_molecules == 2
    assert h.total_residence_frames == 5
    assert sorted(h.occupancy_by_probe()) == ["FMD"]
    assert len(h.occupancy_by_probe()["FMD"]) == 2


def test_best_pose_picks_the_longest_episode():
    short = _named("r0", 279, [1, 2])
    long_ = _named("r1", 280, [40, 41, 42, 43])
    pose = _hotspot_with(short, long_).best_pose()
    assert pose.probe_resid == 280
    assert pose.source_label == "r1"
    assert pose.episode == (40, 43)


def test_best_pose_frame_is_a_sampled_frame_near_the_midpoint():
    # midpoint of (0, 30) is 15, which is not sampled; frame 20 is the nearest that is
    pose = _hotspot_with(_named("r0", 279, [0, 20, 30], stride=10)).best_pose()
    assert pose.frame == 20
    assert pose.amber_mask == ":279"


def test_best_pose_tie_breaks_deterministically_on_source_then_resid():
    a = _named("r1", 300, [10, 11])
    b = _named("r0", 279, [50, 51])
    assert _hotspot_with(a, b).best_pose().source_label == "r0"
    assert _hotspot_with(b, a).best_pose().source_label == "r0"
