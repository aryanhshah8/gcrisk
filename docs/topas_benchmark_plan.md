# TOPAS Benchmark Workflow

The repo now includes a real TOPAS/OpenTOPAS-RBE benchmark path rather than
just a placeholder comparison layer.

## What It Does

The benchmark uses a repo-owned proton-water phantom setup in:

- [topas/rbe_v79_proton_water/run.txt](/Users/aryan/Documents/Code/Astrobiology/gcr-dosimetry-pipeline/topas/rbe_v79_proton_water/run.txt)
- [topas/rbe_v79_proton_water/experiment.txt](/Users/aryan/Documents/Code/Astrobiology/gcr-dosimetry-pipeline/topas/rbe_v79_proton_water/experiment.txt)
- [topas/rbe_v79_proton_water/rbe_scorers.txt](/Users/aryan/Documents/Code/Astrobiology/gcr-dosimetry-pipeline/topas/rbe_v79_proton_water/rbe_scorers.txt)
- [topas/rbe_v79_proton_water/CellLineV79.txt](/Users/aryan/Documents/Code/Astrobiology/gcr-dosimetry-pipeline/topas/rbe_v79_proton_water/CellLineV79.txt)

It runs TOPAS with:

- a monoenergetic proton beam
- a water phantom scored along depth
- `ProtonLET`
- `DoseToWater`
- `RBE_Wedenberg`
- `RBE_McNamara`

## Main Command

```bash
python scripts/run_topas_rbe_benchmark.py
```

This command:

1. runs TOPAS in the repo-owned benchmark directory
2. writes raw scorer outputs into `topas/rbe_v79_proton_water/results/`
3. merges the outputs into:
   - [data/topas/proton_water_v79_reference.csv](/Users/aryan/Documents/Code/Astrobiology/gcr-dosimetry-pipeline/data/topas/proton_water_v79_reference.csv)
4. compares the merged CSV against `gcrisk.rbe`

## Follow-On Comparison

```bash
python scripts/compare_topas_rbe.py data/topas/proton_water_v79_reference.csv
```

## CSV Semantics

The merged reference CSV contains:

- `LETd_keV_um`: scored proton LETd profile from TOPAS
- `physical_dose_Gy`: scored physical dose profile from TOPAS
- `dose_Gy`: prescribed dose used by the TOPAS RBE scorer
- `alpha_beta_x_Gy`: cell-line alpha/beta used by the RBE scorer
- `RBE_wedenberg_ref`
- `RBE_mcnamara_ref`
- `dose_RBE_wedenberg_ref`
- `dose_RBE_mcnamara_ref`

The important subtlety is that `dose_Gy` is **not** the local physical dose
scored in the phantom. It is the prescribed dose parameter used when TOPAS
computes `RBE`.

## Current Reference Case

The checked-in benchmark uses:

- cell line: V79-style alpha/beta definition
- `alpha_beta_x_Gy = 1.412`
- prescribed dose: `4.0 Gy`
- 200 depth bins through a water sample

## What This Validates

- our Python Wedenberg implementation against OpenTOPAS-RBE
- our Python McNamara implementation against OpenTOPAS-RBE
- the dose/LET-to-RBE workflow for proton-water phantom data

## What This Does Not Validate

- the mixed-field GCR transport model against TOPAS
- the Mars mission dose path
- neutron or fragment transport on the space side
