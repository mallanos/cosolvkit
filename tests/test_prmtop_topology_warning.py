"""A PDB topology fuses solvent residues and collides probe resids with protein resids."""

import logging

from cosolvkit.analysis.config import AnalysisConfig


def _write(tmp_path, topology):
    (tmp_path / "cfg.yaml").write_text(
        "out_path: out\n"
        "simulations:\n"
        "  - trajectory: t.dcd\n"
        f"    topology: {topology}\n"
        "    cosolvents: ['FMD']\n"
    )
    return str(tmp_path / "cfg.yaml")


def test_pdb_topology_warns(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        AnalysisConfig.from_yaml(_write(tmp_path, "system.pdb"))
    assert any("prmtop" in r.message for r in caplog.records)


def test_prmtop_topology_is_silent(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        AnalysisConfig.from_yaml(_write(tmp_path, "system.prmtop"))
    assert not [r for r in caplog.records if "prmtop" in r.message]
