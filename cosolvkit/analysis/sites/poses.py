#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Extract a bound probe pose as a complex PDB for AutoPath.
#

import json
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


def _open_universe(topology, trajectory):
    """Indirection so tests can inject a synthetic Universe."""
    import MDAnalysis as mda
    return mda.Universe(topology, trajectory)


def write_pose(pose_ref, out_pdb, protein_selection="protein", pocket_cutoff=5.0,
               open_universe=None):
    """Write protein plus one bound probe molecule as a complex PDB for AutoPath.

    The probe keeps its original resname — no silent rename to ``UNK``, so provenance
    survives into the AutoPath run — and is written with its hydrogens, which OpenFF
    parametrisation needs.

    ``pocket_selection`` is derived from the WRITTEN file, not from the source topology:
    residue numbering in the pose is its own, and AutoPath's ``pocket_selection`` must
    index the structure it is actually given.

    :param pose_ref: :class:`PoseRef` naming the source, frame and probe molecule.
    :param out_pdb: path to write; ``manifest.json`` is written beside it.
    :param protein_selection: MDAnalysis selection for the receptor.
    :param pocket_cutoff: Angstroms; residues within this of the probe form the pocket.
    :param open_universe: optional ``(topology, trajectory) -> Universe`` override.
    :return: the manifest dict, also written to ``manifest.json``.
    """
    import MDAnalysis as mda
    from MDAnalysis.lib.distances import minimize_vectors

    opener = open_universe or _open_universe
    u = opener(pose_ref.topology, pose_ref.trajectory)
    u.trajectory[pose_ref.frame]

    protein = u.select_atoms(protein_selection)
    probe = u.select_atoms(f"resindex {pose_ref.probe_resindex}")
    if len(probe) == 0:
        raise ValueError(
            f"No atoms for {pose_ref.probe_resname} resid {pose_ref.probe_resid} in "
            f"{pose_ref.topology}."
        )
    found = set(probe.resnames)
    if found != {pose_ref.probe_resname}:
        raise ValueError(
            f"resindex {pose_ref.probe_resindex} holds resname(s) {sorted(found)}, "
            f"not {pose_ref.probe_resname!r} as the PoseRef claims — wrong topology?"
        )

    # Bring the probe to its nearest image of the protein so the written complex is
    # contiguous rather than split across the periodic boundary.
    ref = protein.positions.mean(axis=0)
    probe.positions = minimize_vectors(probe.positions - ref, u.dimensions) + ref

    spread = float(np.ptp(probe.positions, axis=0).max()) if len(probe) > 1 else 0.0

    complex_ag = protein + probe
    complex_ag.write(out_pdb)

    written = mda.Universe(out_pdb)
    written_probe = written.select_atoms(f"resname {pose_ref.probe_resname}")
    pocket = written.select_atoms(
        f"({protein_selection}) and around {pocket_cutoff} resname "
        f"{pose_ref.probe_resname}"
    )
    resindex_to_position = {r.resindex: i + 1 for i, r in enumerate(written.residues)}
    pocket_selection = sorted(resindex_to_position[r.resindex]
                              for r in pocket.residues)
    pocket_resnames = [written.residues[i - 1].resname for i in pocket_selection]

    manifest = {
        "pose_pdb": os.path.abspath(out_pdb),
        "ligand_selection": f"resname {pose_ref.probe_resname}",
        "ligand_amber_mask": pose_ref.amber_mask,
        "pocket_selection": pocket_selection,
        "pocket_resnames": pocket_resnames,
        "probe_max_extent_ang": round(spread, 3),
        "provenance": {
            "source_label": pose_ref.source_label,
            "topology": pose_ref.topology,
            "trajectory": pose_ref.trajectory,
            "frame": int(pose_ref.frame),
            "probe_resname": pose_ref.probe_resname,
            "probe_resid": int(pose_ref.probe_resid),
            "probe_resindex": int(pose_ref.probe_resindex),
            "episode": [int(pose_ref.episode[0]), int(pose_ref.episode[1])],
        },
    }

    manifest_path = os.path.join(os.path.dirname(os.path.abspath(out_pdb)),
                                 "manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    logger.info("Wrote pose %s (%d probe atoms, max extent %.2f A).",
                out_pdb, len(written_probe), spread)
    return manifest
