#!/bin/bash
#SBATCH -e 6e22_refine.err
#SBATCH -o 6e22_refine.out
#SBATCH --time=8:00:00
#SBATCH --partition=forli-pro,forli
#SBATCH --cpus-per-task=4
#SBATCH --job-name="6e22_refine"

# Occupancy annotation + MMGBSA job generation. CPU-only: the scan is trajectory I/O
# and the generation just writes files. The MMGBSA itself runs from the per-target
# qfiles this produces; run_collect.q then folds the answers back onto the hotspots.
#
# The autopath env, not cosolvkit: the CLI imports autopath.ap_PLIP for the
# MMPBSA wrapper, and the cosolvkit env lacks waterdynamics.
source ~/.bashrc
micromamba activate autopath

CONFIG=../03_analysis/analysis.yaml

# One pass does both: annotate the checkpoint, then write the jobs. Split them with
# --annotate-only first if you want to inspect hotspot_occupancy.csv before committing
# to which sites to refine.
python -m cosolvkit.cli.refine_hotspots --config "$CONFIG" \
                --out results \
                --stride 5 \
                --target binding_sites --top-n 3 \
                --mode mmgbsa \
                --mmgbsa-n-frames 20

# Stage 1 of the MMGBSA is fast (seconds per target) — run it here rather than queueing
# another job. It writes each target's qfiles_mmgbsa/ and run_mmgbsa_batch.sh.
bash results/submit_all.sh

echo
echo "Stage 1 submitted. Nothing is computed until stage 2 is submitted per target:"
for d in results/*/; do
    [ -f "${d}run_mmgbsa_batch.sh" ] && echo "    cd ${d} && ./run_mmgbsa_batch.sh"
done
echo "Then collect:  python -m cosolvkit.cli.refine_hotspots --config $CONFIG --out results --collect"
