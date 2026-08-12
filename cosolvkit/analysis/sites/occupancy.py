#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Which probe molecule sat in which hotspot, in which source file and frames.
#

import logging

import numpy as np

from cosolvkit.analysis.core.models import ProbeOccupancy

logger = logging.getLogger(__name__)


def positions_to_voxel_indices(positions, origin, delta):
    """Map Cartesian positions to grid indices.

    Voxel centres sit at ``origin + i * delta`` — the same convention
    ``find_pocket_residues`` uses — so each cell spans +/- delta/2 about its centre and
    the conversion rounds to the nearest centre. ``np.floor`` would bias every lookup by
    half a voxel.

    :param positions: shape ``(n, 3)`` Cartesian coordinates in Angstroms.
    :param origin: shape ``(3,)`` grid origin.
    :param delta: shape ``(3,)`` voxel spacing.
    :return: shape ``(n, 3)`` integer indices, unclipped — callers must mask out-of-range.
    """
    return np.rint((np.asarray(positions, dtype=float) - origin) / delta).astype(int)


def build_label_volume(hotspots):
    """Paint each hotspot's mask into one integer volume, numbered from 1.

    One volume per cosolvent, never one shared across cosolvents: watershed masks are
    disjoint within a cosolvent but may overlap between them, and a shared volume would
    let one label silently overwrite another.

    :param hotspots: hotspots sharing a grid, all with ``voxel_mask``/``grid_origin``/
        ``grid_delta`` set.
    :return: ``(label_volume, ordered_hotspots)`` where ``label_volume == i + 1`` marks
        ``ordered_hotspots[i]``.
    :raises ValueError: if the hotspots do not share one grid.
    """
    if not hotspots:
        raise ValueError("build_label_volume requires at least one hotspot.")

    first = hotspots[0]
    shape = first.voxel_mask.shape
    for h in hotspots[1:]:
        if (h.voxel_mask.shape != shape
                or not np.allclose(h.grid_origin, first.grid_origin)
                or not np.allclose(h.grid_delta, first.grid_delta)):
            raise ValueError(
                f"Hotspot {h.site_id} is on a different grid from hotspot "
                f"{first.site_id}; masks cannot share a label volume."
            )

    vol = np.zeros(shape, dtype=np.int32)
    for i, h in enumerate(hotspots, start=1):
        vol[h.voxel_mask] = i
    return vol, list(hotspots)


def assert_inside_grid(protein_positions, origin, delta, shape, min_fraction=0.5):
    """Fail unless the protein actually sits inside the density grid.

    The grids live in the aligned, superposed frame. Handed an unaligned trajectory every
    voxel lookup falls outside the grid and the scan returns a silent zero, so this is
    checked once up front rather than debugged later.

    :raises ValueError: if fewer than *min_fraction* of atoms fall inside the grid.
    """
    idx = positions_to_voxel_indices(protein_positions, origin, delta)
    inside = np.all((idx >= 0) & (idx < np.asarray(shape)), axis=1)
    fraction = float(inside.mean()) if len(inside) else 0.0
    if fraction < min_fraction:
        raise ValueError(
            f"Only {fraction:.1%} of protein atoms fall inside the density grid "
            f"(need >= {min_fraction:.0%}). The trajectory is almost certainly not "
            f"aligned onto the frame the maps were built in — align it first."
        )


def _open_universe(topology, trajectory):
    """Indirection so tests can inject a synthetic Universe."""
    import MDAnalysis as mda
    return mda.Universe(topology, trajectory)


class OccupancyAnnotator:
    """Records which probe molecules occupied which hotspots, per source file.

    A molecule occupies a hotspot in a frame when at least ``min_atoms_in_blob`` of its
    heavy atoms fall inside that hotspot's voxel mask.
    """

    def __init__(self, simulations, stride=1, min_atoms_in_blob=1,
                 residue_cutoff=4.5, contact_cutoff=4.0, logger=None):
        self.simulations = list(simulations)
        self.stride = int(stride)
        self.min_atoms_in_blob = int(min_atoms_in_blob)
        self.residue_cutoff = float(residue_cutoff)
        self.contact_cutoff = float(contact_cutoff)
        self.logger = logger or logging.getLogger(__name__)

    def annotate(self, results):
        """Fill ``probe_occupancy`` and per-residue contacts on every hotspot in *results*.

        :param results: ``{cosolvent_name: [Hotspot, ...]}``, as produced by
            :meth:`HotspotDetector.detect_all` or loaded from a checkpoint.
        """
        for sim in self.simulations:
            targets = {c: results[c] for c in sim.cosolvents
                       if c in results and results[c]}
            if not targets:
                self.logger.info("No hotspots for the cosolvents in '%s'; skipping.",
                                 sim.label)
                continue
            universe = _open_universe(sim.topology, sim.trajectory)
            for cosolvent, hotspots in targets.items():
                self._scan(universe, sim, cosolvent, hotspots)

    # ------------------------------------------------------------------

    def _scan(self, u, sim, cosolvent, hotspots):
        from MDAnalysis.lib.distances import capped_distance, minimize_vectors

        label_vol, ordered = build_label_volume(hotspots)
        origin = np.asarray(ordered[0].grid_origin, dtype=float)
        delta = np.asarray(ordered[0].grid_delta, dtype=float)
        shape = np.asarray(label_vol.shape)

        protein = u.select_atoms("protein and not name H*")
        assert_inside_grid(protein.positions, origin, delta, label_vol.shape)

        probe = u.select_atoms(f"resname {cosolvent} and not name H*")
        if len(probe) == 0:
            self.logger.warning("No atoms with resname %s in '%s'.", cosolvent, sim.label)
            return
        self._assert_unique_resids(probe, sim, cosolvent)

        for h in ordered:
            if not h.pocket_residues:
                self._find_pocket_residues(u, h, origin, delta)

        res_groups = {
            h.site_id: {
                pr.resid: u.select_atoms(f"resindex {pr.resindex} and not name H*")
                for pr in h.pocket_residues
            }
            for h in ordered
        }
        pr_by_site = {h.site_id: {pr.resid: pr for pr in h.pocket_residues}
                      for h in ordered}

        probe_resids = probe.resids
        resid_to_resindex = dict(zip(probe.resids, probe.resindices))
        # {(site_index, resid): [frames]}
        hits = {}
        n_scanned = 0

        for ts in u.trajectory[::self.stride]:
            n_scanned += 1
            frame = int(ts.frame)
            ref = protein.positions.mean(axis=0)
            pos = minimize_vectors(probe.positions - ref, ts.dimensions) + ref

            idx = positions_to_voxel_indices(pos, origin, delta)
            valid = np.all((idx >= 0) & (idx < shape), axis=1)
            if not valid.any():
                continue
            labels = np.zeros(len(idx), dtype=np.int32)
            vi = idx[valid]
            labels[valid] = label_vol[vi[:, 0], vi[:, 1], vi[:, 2]]

            occupied = labels > 0
            if not occupied.any():
                continue
            pairs, counts = np.unique(
                np.stack([labels[occupied], probe_resids[occupied]], axis=1),
                axis=0, return_counts=True,
            )
            frame_molecules = {}
            for (label, resid), count in zip(pairs, counts):
                if count < self.min_atoms_in_blob:
                    continue
                hits.setdefault((int(label) - 1, int(resid)), []).append(frame)
                frame_molecules.setdefault(int(label) - 1, []).append(int(resid))

            for site_index, resids in frame_molecules.items():
                site = ordered[site_index]
                groups = res_groups[site.site_id]
                if not groups:
                    continue
                mol_mask = np.isin(probe_resids, resids)
                mol_pos = pos[mol_mask]
                mol_ids = probe_resids[mol_mask]
                for res_resid, res_ag in groups.items():
                    pair_idx, _ = capped_distance(
                        mol_pos, res_ag.positions,
                        max_cutoff=self.contact_cutoff, box=ts.dimensions,
                        return_distances=True,
                    )
                    if len(pair_idx) == 0:
                        continue
                    touching = np.unique(mol_ids[pair_idx[:, 0]])
                    pr = pr_by_site[site.site_id][res_resid]
                    for rid in touching:
                        (pr.cosolvent_contacts
                           .setdefault(sim.label, {})
                           .setdefault(cosolvent, {})
                           .setdefault(int(rid), [])
                           .append(frame))

        for (site_index, resid), frames in hits.items():
            site = ordered[site_index]
            site.probe_occupancy.append(ProbeOccupancy(
                source_label=sim.label,
                topology=sim.topology,
                trajectory=sim.trajectory,
                probe_resname=cosolvent,
                probe_resid=int(resid),
                probe_resindex=int(resid_to_resindex[resid]),
                frames=sorted(frames),
                n_frames_scanned=n_scanned,
                stride=self.stride,
            ))

        for h in ordered:
            for pr in h.pocket_residues:
                for by_probe in pr.cosolvent_contacts.values():
                    for mol_dict in by_probe.values():
                        for rid in mol_dict:
                            mol_dict[rid].sort()

        self.logger.info(
            "Occupancy for '%s'/%s: %d molecule-site records over %d frames.",
            sim.label, cosolvent, len(hits), n_scanned,
        )

    def _assert_unique_resids(self, probe, sim, cosolvent):
        """A PDB topology renumbers probes from 1, colliding with protein resids."""
        from cosolvkit.analysis.sites.properties import warn_if_fused_residues

        warn_if_fused_residues(probe, self.logger,
                               context=f"{sim.label}/{cosolvent}")
        n_molecules = len(np.unique(probe.resindices))
        n_resids = len(np.unique(probe.resids))
        if n_resids != n_molecules:
            raise ValueError(
                f"{cosolvent} in '{sim.label}' has {n_molecules} molecules but only "
                f"{n_resids} distinct resids, so resid cannot identify a molecule. "
                f"This is the signature of a PDB topology — point 'topology' at the "
                f"matching system.prmtop instead of {sim.topology}."
            )

    def _find_pocket_residues(self, u, hotspot, origin, delta):
        """Delegate to the existing calculator, which reads the current frame."""
        from cosolvkit.analysis.sites.properties import PocketPropertyCalculator

        calc = PocketPropertyCalculator.__new__(PocketPropertyCalculator)
        calc.universe = u
        calc.logger = self.logger
        PocketPropertyCalculator.find_pocket_residues(
            calc, hotspot, cutoff=self.residue_cutoff)
