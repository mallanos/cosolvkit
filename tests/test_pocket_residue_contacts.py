"""Contacts are keyed by source file first: a frame index is meaningless without one."""

from cosolvkit.analysis.core.models import PocketResidue


def _pr():
    pr = PocketResidue(resid=42, resindex=41, resname="TYR", chain="A",
                       n_contact_voxels=5, min_dist_ang=3.1, contact_fraction=0.25)
    pr.cosolvent_contacts = {
        "formamide_r0": {"FMD": {279: [1, 2, 3], 280: [7]}},
        "formamide_r1": {"FMD": {279: [2, 9]}},
    }
    return pr


def test_frames_union_across_sources_by_default():
    assert _pr().contact_frames("FMD") == [1, 2, 3, 7, 9]


def test_frames_can_be_restricted_to_one_source():
    assert _pr().contact_frames("FMD", source="formamide_r1") == [2, 9]


def test_molecules_union_and_per_source():
    pr = _pr()
    assert pr.contact_molecules("FMD") == [279, 280]
    assert pr.contact_molecules("FMD", source="formamide_r1") == [279]


def test_event_count_sums_molecule_frame_pairs():
    assert _pr().n_contact_events("FMD") == 6
    assert _pr().n_contact_events("FMD", source="formamide_r0") == 4


def test_unknown_probe_or_source_is_empty_not_an_error():
    pr = _pr()
    assert pr.contact_frames("BEN") == []
    assert pr.contact_frames("FMD", source="nope") == []


def test_round_trip_preserves_the_three_level_shape():
    pr = _pr()
    back = PocketResidue.from_dict(pr.to_dict())
    assert back.cosolvent_contacts == pr.cosolvent_contacts
    assert back.chain_source == "topology"
