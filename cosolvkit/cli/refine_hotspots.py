#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Annotate hotspots with probe occupancy and emit AutoPath refinement jobs.
#
# Usage:
#   refine_hotspots --config analysis.yaml --top-n 5 --mode both
#

import argparse
import logging
import os

import pandas as pd

from cosolvkit.analysis.config import AnalysisConfig
from cosolvkit.analysis.sites.detect import HotspotDetector
from cosolvkit.analysis.sites.occupancy import OccupancyAnnotator, apply_chain_reference

logger = logging.getLogger(__name__)


def build_parser():
    """Command-line surface for ``refine_hotspots``."""
    p = argparse.ArgumentParser(
        prog="refine_hotspots",
        description="Annotate detected hotspots with probe occupancy and generate "
                    "AutoPath MMGBSA / steered-MD jobs.",
    )
    p.add_argument("--config", required=True,
                   help="Analysis YAML the hotspots were detected from.")
    p.add_argument("--checkpoint", default=None,
                   help="Directory holding hotspot_checkpoints/ "
                        "(default: <out_path>/merged).")
    p.add_argument("--out", default=None,
                   help="Output directory (default: <out_path>/refine).")
    p.add_argument("--target", choices=["binding_sites", "hotspots"],
                   default="binding_sites",
                   help="What to refine (default: binding_sites).")
    p.add_argument("--top-n", type=int, default=5,
                   help="How many targets to refine (default: 5).")
    p.add_argument("--stride", type=int, default=1,
                   help="Trajectory stride for the occupancy scan (default: 1).")
    p.add_argument("--min-atoms", type=int, default=1,
                   help="Heavy atoms inside a blob that count as occupancy (default: 1).")
    p.add_argument("--gap-tolerance", type=int, default=0,
                   help="Sampled frames a molecule may miss without ending an episode.")
    p.add_argument("--annotate-only", action="store_true",
                   help="Stop after annotation; write no poses or jobs.")
    p.add_argument("--mode", choices=["mmgbsa", "smd", "both"], default="both",
                   help="Which refinement legs to generate (default: both).")
    p.add_argument("--submit", action="store_true",
                   help="sbatch the generated qfiles instead of only writing them.")
    p.add_argument("--slurm-template", default=None,
                   help="Optional SLURM template file. Placeholders {{NAME}}, {{SCRIPT}}, "
                        "{{WORKDIR}} and {{PYTHON}} are substituted; a template must cd "
                        "to {{WORKDIR}} itself. Omitted, a built-in block is written.")
    p.add_argument("--python-exe", dest="python_exe", default=None,
                   help="Interpreter the generated SLURM jobs run (default: the one "
                        "running this command). Must be able to import autopath and "
                        "MDAnalysis; a bare 'python' on a compute node usually cannot.")
    p.add_argument("--mmgbsa-n-frames", type=int, default=20,
                   help="Frames to extract per MMGBSA job (default: 20).")
    p.add_argument("--mmgbsa-n-molecules", type=int, default=1,
                   help="Occupying molecules to emit a job for, best first (default: 1). "
                        "Each gets its own subdirectory under the target's mmgbsa/.")
    p.add_argument("--mmgbsa-frame-strategy", choices=["random", "cluster"],
                   default="random",
                   help="How to pick frames within a record (default: random). "
                        "'cluster' is reserved and not implemented yet.")
    p.add_argument("--mmgbsa-source", default=None,
                   help="Restrict MMGBSA frame selection to this simulation label.")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for random frame selection (default: 0).")
    return p


def occupancy_rows(results, gap_tolerance=0):
    """Flatten ``{cosolvent: [Hotspot]}`` into one row per ProbeOccupancy record.

    *gap_tolerance* must be the value pose selection uses, or the episodes reported here
    disagree with the episode the chosen pose actually came from. ``topology`` and
    ``trajectory`` are carried on every row because a frame index is trajectory-local and
    therefore meaningless without the file it indexes.
    """
    rows = []
    for cosolvent, hotspots in results.items():
        for h in hotspots:
            for occ in h.probe_occupancy:
                episodes = occ.episodes(gap_tolerance)
                longest = occ.longest_episode(gap_tolerance)
                rows.append({
                    "site_id": h.site_id,
                    "cosolvent": cosolvent,
                    "source_label": occ.source_label,
                    "topology": occ.topology,
                    "trajectory": occ.trajectory,
                    "probe_resname": occ.probe_resname,
                    "probe_resid": occ.probe_resid,
                    "probe_resindex": occ.probe_resindex,
                    "n_frames_bound": occ.n_frames_bound,
                    "occupancy_fraction": round(occ.occupancy_fraction, 4),
                    "n_episodes": len(episodes),
                    "longest_start": longest[0] if longest else None,
                    "longest_end": longest[1] if longest else None,
                })
    return rows


def annotate(config, checkpoint_dir, args):
    """Load the checkpoint, run the occupancy scan, and persist the result."""
    cosolvents = []
    for sim in config.simulations:
        for c in sim.cosolvents:
            if c not in cosolvents:
                cosolvents.append(c)

    results = HotspotDetector.load_checkpoint(checkpoint_dir, cosolvents)
    annotator = OccupancyAnnotator(
        config.simulations,
        stride=args.stride,
        min_atoms_in_blob=args.min_atoms,
        logger=logger,
    )
    annotator.annotate(results)

    for sim in config.simulations:
        if not sim.chain_reference:
            continue
        import MDAnalysis as mda
        reference = mda.Universe(sim.chain_reference)
        topology = mda.Universe(sim.topology)
        for c in sim.cosolvents:
            if c in results:
                apply_chain_reference(results[c], reference, topology)

    HotspotDetector.save_checkpoint(results, checkpoint_dir)
    return results


def main(argv=None):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    config = AnalysisConfig.from_yaml(args.config)
    # Resolve once and write it back, so job generation reads the merged maps from the
    # same directory the hotspots were loaded from.
    args.checkpoint = checkpoint_dir = (args.checkpoint
                                        or os.path.join(config.out_path, "merged"))
    out_dir = args.out or os.path.join(config.out_path, "refine")
    os.makedirs(out_dir, exist_ok=True)

    results = annotate(config, checkpoint_dir, args)

    rows = occupancy_rows(results, gap_tolerance=args.gap_tolerance)
    csv_path = os.path.join(out_dir, "hotspot_occupancy.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info("Wrote %d occupancy records to %s.", len(rows), csv_path)

    if args.annotate_only:
        return 0

    from cosolvkit.cli.refine_hotspots_jobs import generate_jobs
    generate_jobs(config, results, out_dir, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
