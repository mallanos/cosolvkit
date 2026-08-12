#!/bin/bash
#SBATCH -e 6e22_collect.err
#SBATCH -o 6e22_collect.out
#SBATCH --time=1:00:00
#SBATCH --partition=forli-pro,forli
#SBATCH --cpus-per-task=2
#SBATCH --job-name="6e22_collect"

# Parse the finished MMGBSA runs and fold them back onto the hotspots: total interaction
# energy onto each Hotspot, per-residue decomposition onto its PocketResidues, plus two
# CSV summaries. Run after stage 2 of the MMGBSA has completed.
#
# Safe to run early or repeatedly: jobs without a FINAL_RESULTS_mmpbsa.dat yet are
# skipped with a log line, and re-collecting replaces results rather than duplicating
# them, so this can be re-submitted as the queue drains.
source ~/.bashrc
micromamba activate autopath

python -m cosolvkit.cli.refine_hotspots --config ../03_analysis/analysis.yaml \
                --out results \
                --collect
