"""The refine_hotspots CLI: argument surface and the occupancy table it writes."""

import pytest

from cosolvkit.analysis.core.models import Hotspot, ProbeOccupancy
from cosolvkit.cli.refine_hotspots import build_parser, occupancy_rows


def _hotspot_with_two_records():
    h = Hotspot(rank=1, site_id=12, cosolvent="FMD")
    h.probe_occupancy = [
        ProbeOccupancy(source_label="r0", topology="/tmp/a.prmtop",
                       trajectory="/tmp/a.dcd", probe_resname="FMD", probe_resid=279,
                       probe_resindex=278, frames=[1, 2, 3], n_frames_scanned=10,
                       stride=1),
        ProbeOccupancy(source_label="r1", topology="/tmp/b.prmtop",
                       trajectory="/tmp/b.dcd", probe_resname="FMD", probe_resid=280,
                       probe_resindex=279, frames=[8], n_frames_scanned=10, stride=1),
    ]
    return h


def test_defaults_match_the_spec():
    args = build_parser().parse_args(["--config", "a.yaml"])
    assert args.target == "binding_sites"
    assert args.top_n == 5
    assert args.mode == "both"
    assert args.stride == 1
    assert args.min_atoms == 1
    assert args.gap_tolerance == 0
    assert args.mmgbsa_n_frames == 20
    assert args.mmgbsa_n_molecules == 1
    assert args.mmgbsa_source is None
    assert args.seed == 0
    assert args.submit is False
    assert args.annotate_only is False


def test_config_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_occupancy_rows_are_one_per_record():
    rows = occupancy_rows({"FMD": [_hotspot_with_two_records()]})
    assert len(rows) == 2
    first = next(r for r in rows if r["probe_resid"] == 279)
    assert first["site_id"] == 12
    assert first["cosolvent"] == "FMD"
    assert first["source_label"] == "r0"
    assert first["n_frames_bound"] == 3
    assert first["occupancy_fraction"] == 0.3
    assert first["n_episodes"] == 1
    assert first["longest_start"] == 1
    assert first["longest_end"] == 3


def test_hotspots_without_occupancy_contribute_no_rows():
    assert occupancy_rows({"FMD": [Hotspot(rank=1, site_id=1, cosolvent="FMD")]}) == []
