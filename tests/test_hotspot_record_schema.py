"""to_record is the canonical form; to_row is a flat projection safe for CSV."""

import numpy as np
import pytest

from cosolvkit.analysis.core.models import (Hotspot, MmgbsaResult, PocketResidue,
                                            ProbeOccupancy)


def _loaded_hotspot():
    mask = np.zeros((6, 6, 6), dtype=bool)
    mask[1:3, 1:3, 1:3] = True
    h = Hotspot(rank=1, site_id=7, cosolvent="FMD", n_voxels=8,
                centroid=np.array([1.0, 1.0, 1.0]), agfe_min=-2.5,
                agfe_mean_top_pct=-1.8, voxel_mask=mask,
                favorable_atomtypes=["HBD"], per_type_agfe={"HBD": -2.5})
    h.grid_origin = np.zeros(3)
    h.grid_delta = np.full(3, 0.5)
    h.add_property("accessible_fraction", 0.42)
    h.pocket_residues = [PocketResidue(resid=42, resindex=41, resname="TYR", chain="A",
                                       n_contact_voxels=5, min_dist_ang=3.1,
                                       contact_fraction=0.25)]
    h.probe_occupancy = [ProbeOccupancy(
        source_label="formamide_r0", topology="/tmp/s.prmtop", trajectory="/tmp/t.dcd",
        probe_resname="FMD", probe_resid=279, probe_resindex=278,
        frames=[10, 11, 12], n_frames_scanned=100, stride=1)]
    return h


def test_to_row_is_flat_enough_for_a_csv():
    row = _loaded_hotspot().to_row()
    bad = {k: v for k, v in row.items() if isinstance(v, (list, dict))}
    assert bad == {}, f"non-scalar columns would corrupt the CSV: {sorted(bad)}"


def test_to_row_carries_occupancy_summaries():
    row = _loaded_hotspot().to_row()
    assert row["n_probe_molecules"] == 1
    assert row["total_residence_frames"] == 3
    assert row["best_probe"] == "FMD"
    assert row["best_source"] == "formamide_r0"
    assert row["best_frame"] == 11


def test_to_row_summaries_are_none_without_occupancy():
    h = Hotspot(rank=1, site_id=1, cosolvent="FMD")
    row = h.to_row()
    assert row["n_probe_molecules"] == 0
    assert row["best_probe"] is None
    assert row["best_frame"] is None


def test_to_dict_is_an_alias_of_to_row():
    h = _loaded_hotspot()
    assert h.to_dict() == h.to_row()


def test_record_round_trip_preserves_nested_data():
    h = _loaded_hotspot()
    rec = h.to_record()
    assert rec["schema"] == 2
    back = Hotspot.from_record(rec, h.voxel_mask, h.grid_origin, h.grid_delta)
    assert back.site_id == 7
    assert back.properties["accessible_fraction"] == 0.42
    assert len(back.pocket_residues) == 1
    assert back.pocket_residues[0].resname == "TYR"
    assert len(back.probe_occupancy) == 1
    assert back.probe_occupancy[0].frames == [10, 11, 12]


def test_schema_1_records_still_load():
    """Checkpoints written before this change carry a flat dict plus _properties."""
    h = _loaded_hotspot()
    legacy = {
        "rank": 1, "site_id": 7, "cosolvent": "FMD", "n_voxels": 8,
        "centroid_x": 1.0, "centroid_y": 1.0, "centroid_z": 1.0,
        "agfe_min": -2.5, "agfe_mean_top_pct": -1.8,
        "favorable_atomtypes": "HBD", "agfe_HBD": -2.5,
        "_properties": {"accessible_fraction": 0.42},
    }
    back = Hotspot.from_record(legacy, h.voxel_mask, h.grid_origin, h.grid_delta)
    assert back.site_id == 7
    assert back.properties["accessible_fraction"] == 0.42
    assert back.probe_occupancy == []


def test_unknown_schema_raises_rather_than_guessing():
    h = _loaded_hotspot()
    rec = h.to_record()
    rec["schema"] = 99
    with pytest.raises(ValueError, match="schema"):
        Hotspot.from_record(rec, h.voxel_mask, h.grid_origin, h.grid_delta)


def _mmgbsa_result(delta_total=-3.24, resid=289, source="formamide_r2"):
    return MmgbsaResult(
        probe_resname="FMD", probe_resid=resid, source_label=source,
        delta_total=delta_total, std_dev=1.1, std_err=0.49, n_frames=5,
        components={"DELTA TOTAL": {"average": delta_total, "std_dev": 1.1,
                                    "std_err": 0.49}},
        results_path="/tmp/FINAL_RESULTS_mmpbsa.dat",
    )


def test_to_row_summarizes_mmgbsa_as_best_and_count():
    h = _loaded_hotspot()
    h.mmgbsa = [_mmgbsa_result(delta_total=-3.24), _mmgbsa_result(delta_total=-9.0)]
    row = h.to_row()
    assert row["mmgbsa_delta_total"] == -9.0
    assert row["mmgbsa_n_results"] == 2


def test_to_row_mmgbsa_summary_is_none_without_results():
    row = _loaded_hotspot().to_row()
    assert row["mmgbsa_delta_total"] is None
    assert row["mmgbsa_n_results"] == 0


def test_record_round_trip_preserves_mmgbsa():
    h = _loaded_hotspot()
    h.mmgbsa = [_mmgbsa_result()]
    h.pocket_residues[0].mmgbsa_decomposition = {
        "van_der_Waals": {"average": -1.2, "std_dev": 0.3, "std_err": 0.1}
    }
    h.pocket_residues[0].mmgbsa_location = "R"

    rec = h.to_record()
    back = Hotspot.from_record(rec, h.voxel_mask, h.grid_origin, h.grid_delta)

    assert len(back.mmgbsa) == 1
    assert back.mmgbsa[0].probe_resid == 289
    assert back.mmgbsa[0].delta_total == pytest.approx(-3.24)
    assert back.pocket_residues[0].mmgbsa_location == "R"
    assert back.pocket_residues[0].mmgbsa_decomposition["van_der_Waals"]["average"] == -1.2


def test_export_writes_no_json_and_a_flat_csv(tmp_path):
    """Regression: nested fields must not reach the CSV, and the JSON export is gone."""
    import logging

    import pandas as pd

    from cosolvkit.analysis.sites.detect import HotspotDetector

    # Built with __new__ so no Universe or grid is needed; export_results only reads
    # out_path, logger and (when label_map=True) _labeled_arrays.
    detector = HotspotDetector.__new__(HotspotDetector)
    detector._out_path = str(tmp_path)
    detector.logger = logging.getLogger("test")
    detector._labeled_arrays = {}

    HotspotDetector.export_results(detector, {"FMD": [_loaded_hotspot()]}, label_map=False)

    assert (tmp_path / "hotspot_sites_FMD.csv").exists()
    assert not (tmp_path / "hotspot_sites_FMD.json").exists()
    df = pd.read_csv(tmp_path / "hotspot_sites_FMD.csv")
    assert "pocket_residues" not in df.columns
    assert df.loc[0, "n_probe_molecules"] == 1
