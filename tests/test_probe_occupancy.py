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
