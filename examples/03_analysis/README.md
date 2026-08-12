# 03 — Analysis and Visualization

Final stage of the cosolvent-MD workflow: turn the production trajectory into AGFE
density maps and ranked hotspots with CosolvKit's YAML-driven analysis, then explore
them in the interactive dashboard. Chains off
[`../02_equilibration-and-production`](../02_equilibration-and-production) on the
`6E22 + benzene` system.

Unlike examples 01/02, this stage is **CPU-only**. Run it in the **`autopath`** env —
the `cosolvkit` env lacks `waterdynamics`, which survival probability needs, and
`run_analysis.q` activates `autopath` for the same reason.

## Prerequisites

Run examples 01 and 02 first. This example reads:
- **trajectory** — `../02_equilibration-and-production/6E22_benzene/MD/MD_6E22_benzene_rep1_aligned.dcd`
- **topology** — `../01_build_cosolvent_system/6E22_benzene/system.prmtop`
  (`chain_reference` points at `system.pdb` alongside it)

> **Use the prmtop, not the PDB.** The build writes both. A PDB's resid field is 4
> characters and wraps: in this system `system.pdb` gives **9,803 unique resids for
> 10,770 residues** (967 collisions) and numbers benzene 777-819 where the prmtop says
> 778-820. Any `resid`-based selection is then unreliable — pocket residues, per-residue
> contacts and MMGBSA ligand masks all depend on it. The prmtop also carries charges and
> types, which example 04 needs. Its one cost is chain IDs, which `chain_reference`
> restores.

Both paths are set in [`analysis.yaml`](analysis.yaml) and are resolved **relative to
that file's directory**, so the example is portable.

## Step 1 — Run the analysis

```bash
micromamba activate autopath
analyze_cosolvent_simulation -cfg analysis.yaml
# or submit to SLURM (CPU):
sbatch run_analysis.q
```

`analysis.yaml` is a tailored copy of CosolvKit's template. To regenerate the full,
fully-commented template from scratch:

```bash
analyze_cosolvent_simulation --generate-config analysis.yaml
```

Outputs (under `results/`, i.e. `03_analysis/results/`):

| Path | Purpose |
|------|---------|
| `results/benzene_rep1/averaged_trajectory.pdb` | protein reference (from `report.rmsf: true`); used by the dashboard |
| `results/merged/map_agfe_BEN.dx` | total benzene AGFE density map |
| `results/merged/map_agfe_<atomtype>_BEN.dx` | per-atom-type maps (`use_atomtypes: true`) |
| `results/merged/hotspot_sites_BEN.csv` | ranked hotspots (flat, one row per hotspot) |
| `results/merged/hotspot_sites_all.tsv` | the same rows for every probe, combined |
| `results/merged/hotspot_labels_BEN.dx` | hotspot label map (voxel value = site rank) |
| `results/merged/hotspot_checkpoints/` | saved detection (`checkpoint.save_hotspots: true`) |
| `results/benzene_rep1/` | per-simulation maps + RMSF/averaged structure |
| `results/*.pse` | PyMol session (`pymol.enabled: true`) |

`results/merged/` in this checked-in run also contains `hotspot_sites_BEN.json` and
`hotspot_sites_IMI.json`. Those are leftovers from an older run: the JSON export was
removed because it duplicated the checkpoint byte for byte and nothing read it. A fresh
run will not recreate them, and they can be deleted.

> **Key knob:** `hotspots.n_kt` (here `1.0`) sets what counts as a favorable voxel, as
> a multiple of kT: the effective cutoff is `-n_kt * kB * T`, i.e. **-0.60 kcal/mol** at
> 300 K. Pass that kcal/mol value — not `n_kt` — to the dashboard's `--agfe-cutoff`.

## Step 2 — Visualize in the dashboard

`visualize_hotspots` launches an interactive Dash app. It auto-detects the `merged/`
subdirectory and `averaged_trajectory.pdb` under the output dir. It is an interactive
server — run it directly (do **not** `sbatch` it).

```bash
micromamba activate autopath
visualize_hotspots -d results/ --agfe-cutoff -0.6
# then open http://localhost:8050
```

Requires the dashboard extras: `pip install dash dash-bio`.

**Running on an HPC node?** Bind all interfaces (the default) and forward the port from
your local machine:

```bash
# on the HPC node:
visualize_hotspots -d results/ --agfe-cutoff -2.0      # binds 0.0.0.0:8050

# in a LOCAL terminal (the dashboard prints this line automatically):
ssh -L 8050:localhost:8050 user@hpc-hostname -N
# then open http://localhost:8050 locally
```

Useful flags: `--port 8888` (change port), `-p some.pdb` (explicit reference PDB),
`--debug` (hot-reload).

## Multi-probe analysis (optional)

This example already analyses **two** probes (benzene and imidazole), so cross-probe
binding sites are detected: `binding_sites.enabled: true` groups each probe's hotspots
into shared pockets, ranked in `results/binding_sites.csv` with a pharmacophore profile
in `results/binding_sites_pharmacophore.json`. Add a third probe by appending another
entry under `simulations:`. (There is no `consensus` config section; grouping lives
under `binding_sites`.)

## Where this sits in the workflow

```
01 build  ->  system.prmtop + system.pdb + system.xml
02 equil + production  ->  MD/MD_6E22_benzene_rep1_aligned.dcd
03 analysis (analyze_cosolvent_simulation -cfg analysis.yaml)
            ->  results/merged/{map_agfe_*.dx, hotspot_sites_BEN.csv,
                                hotspot_checkpoints/}
03 dashboard (visualize_hotspots -d results/)
04 MMGBSA annotation (refine_hotspots) -> per-hotspot dG + per-residue decomposition
```
