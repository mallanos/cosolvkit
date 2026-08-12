#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Build MMGBSA inputs from hotspot occupancy: selected frames, not a contiguous slice.
#

import json
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


def _open_universe(topology, trajectory=None):
    """Indirection so tests can inject a synthetic Universe."""
    import MDAnalysis as mda
    return mda.Universe(topology) if trajectory is None else mda.Universe(topology,
                                                                          trajectory)


def amber_exclude_mask(all_resids, keep):
    """Amber mask selecting every resid in *all_resids* except *keep*, as ranges.

    The other probe copies must be stripped along with solvent, or they end up in the
    MMGBSA receptor.

    :param all_resids: every resid of the probe species.
    :param keep: the resid that stays as the ligand.
    :return: e.g. ``":279-299,301-315"``; empty string when nothing survives.
    """
    survivors = sorted(int(r) for r in all_resids if int(r) != int(keep))
    if not survivors:
        return ""
    ranges = []
    start = prev = survivors[0]
    for r in survivors[1:]:
        if r != prev + 1:
            ranges.append((start, prev))
            start = r
        prev = r
    ranges.append((start, prev))
    parts = [str(a) if a == b else f"{a}-{b}" for a, b in ranges]
    return ":" + ",".join(parts)


def topology_signature(universe):
    """What makes two topologies the same system, ignoring anything that is not identity.

    Deliberately excludes box dimensions and any file-level metadata: real replicas
    differ in ``BOX_DIMENSIONS`` and in Amber's ``%VERSION`` build stamp while being the
    same topology, so a byte or md5 comparison is wrong here.
    """
    atoms = universe.atoms
    return (
        len(atoms),
        tuple(atoms.names),
        tuple(atoms.types),
        tuple(np.round(atoms.charges, 6)),
        tuple(universe.residues.resnames),
        tuple(int(r) for r in universe.residues.resids),
    )


def check_single_topology(occupancies, open_universe=None):
    """Verify every occupancy record refers to the same topology; return its path.

    :raises ValueError: naming the paths and the first field that diverges.
    """
    if not occupancies:
        raise ValueError(
            "check_single_topology requires at least one occupancy record; "
            "got an empty list."
        )
    opener = open_universe or (lambda t: _open_universe(t))
    paths = []
    for occ in occupancies:
        if occ.topology not in paths:
            paths.append(occ.topology)
    if len(paths) == 1:
        return paths[0]

    signatures = {p: topology_signature(opener(p)) for p in paths}
    fields = ("n_atoms", "names", "types", "charges", "resnames", "resids")
    first = signatures[paths[0]]
    for p in paths[1:]:
        other = signatures[p]
        for name, a, b in zip(fields, first, other):
            if a != b:
                raise ValueError(
                    f"topologies do not match and their frames cannot share one "
                    f"trajectory: {paths[0]} and {p} differ in {name}. Restrict the "
                    f"selection to one simulation with --mmgbsa-source."
                )
    return paths[0]


def select_frames(occupancy, n_frames, seed=0, strategy="random"):
    """Choose frames for MMGBSA from those where this molecule occupied the hotspot.

    :param occupancy: a :class:`ProbeOccupancy` record.
    :param n_frames: how many frames to take; fewer available takes all of them.
    :param seed: RNG seed, so a run is reproducible.
    :param strategy: ``"random"`` today. ``"cluster"`` is the reserved extension point
        for conformational clustering and raises until it is implemented.
    :return: sorted frame indices.
    """
    if strategy == "cluster":
        raise NotImplementedError(
            "strategy='cluster' is reserved for conformational clustering; "
            "use strategy='random' until it lands."
        )
    if strategy != "random":
        raise ValueError(f"Unknown frame selection strategy {strategy!r}.")

    frames = sorted(occupancy.frames)
    if len(frames) <= n_frames:
        if len(frames) < n_frames:
            logger.info(
                "Only %d occupied frames available for %s resid %d in '%s'; "
                "requested %d.",
                len(frames), occupancy.probe_resname, occupancy.probe_resid,
                occupancy.source_label, n_frames,
            )
        return [int(f) for f in frames]
    rng = np.random.default_rng(seed)
    return sorted(int(f) for f in rng.choice(frames, size=n_frames, replace=False))


def write_frame_trajectory(selections, out_dcd, open_universe=None):
    """Write one trajectory holding only the selected frames.

    All atoms are written, never a subset, so the file still matches the prmtop
    atom-for-atom and only the frame count changes.

    :param selections: ``[(ProbeOccupancy, [frame, ...]), ...]``. Several entries pool
        into one file, which is valid only when they share a topology — call
        :func:`check_single_topology` first.
    :param out_dcd: output path; ``frames.json`` is written beside it.
    :return: manifest mapping each new frame index to its source and original frame.
    """
    import MDAnalysis as mda

    opener = open_universe or _open_universe
    records = []
    writer = None
    try:
        for occ, frames in selections:
            u = opener(occ.topology, occ.trajectory)
            if writer is None:
                writer = mda.Writer(out_dcd, n_atoms=u.atoms.n_atoms)
            for f in sorted(frames):
                u.trajectory[int(f)]
                writer.write(u.atoms)
                records.append({
                    "index": len(records),
                    "source_label": occ.source_label,
                    "original_frame": int(f),
                })
    finally:
        if writer is not None:
            writer.close()

    manifest = {
        "trajectory": os.path.abspath(out_dcd),
        "n_frames": len(records),
        "frames": records,
    }
    with open(os.path.join(os.path.dirname(os.path.abspath(out_dcd)), "frames.json"),
              "w") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Wrote %d selected frames to %s.", len(records), out_dcd)
    return manifest
