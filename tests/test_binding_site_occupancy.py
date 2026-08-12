"""A binding site's occupancy is the union of its member hotspots'."""

from cosolvkit.analysis.core.models import BindingSite, Hotspot, ProbeOccupancy


def _member(site_id, resid, frames):
    h = Hotspot(rank=1, site_id=site_id, cosolvent="FMD")
    h.probe_occupancy = [ProbeOccupancy(
        source_label="r0", topology="/tmp/a.prmtop", trajectory="/tmp/a.dcd",
        probe_resname="FMD", probe_resid=resid, probe_resindex=resid - 1,
        frames=list(frames), n_frames_scanned=100, stride=1)]
    return h


def test_occupancy_is_the_union_over_members():
    bs = BindingSite(site_id=3, member_hotspots=[_member(1, 279, [1, 2]),
                                                 _member(2, 280, [5])])
    assert len(bs.probe_occupancy) == 2
    assert bs.n_probe_molecules == 2
    assert bs.total_residence_frames == 3


def test_best_pose_delegates_to_the_best_member():
    bs = BindingSite(site_id=3, member_hotspots=[_member(1, 279, [1, 2]),
                                                 _member(2, 280, [40, 41, 42, 43])])
    pose = bs.best_pose()
    assert pose.probe_resid == 280
    assert pose.episode == (40, 43)


def test_a_site_with_no_occupancy_has_no_pose():
    bs = BindingSite(site_id=3, member_hotspots=[Hotspot(rank=1, site_id=1,
                                                         cosolvent="FMD")])
    assert bs.probe_occupancy == []
    assert bs.best_pose() is None


def test_row_carries_the_occupancy_columns():
    bs = BindingSite(site_id=3, member_hotspots=[_member(1, 279, [1, 2])])
    row = bs.to_row()
    assert row["n_probe_molecules"] == 1
    assert row["total_residence_frames"] == 2
    assert row["best_probe"] == "FMD"
    bad = {k: v for k, v in row.items() if isinstance(v, (list, dict))}
    assert bad == {}
