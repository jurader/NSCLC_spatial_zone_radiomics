# Google Colab reanalysis package (Supplementary Table S3 version)

This package reproduces feature selection, final Cox model fitting, and C-index calculation for the max5, max10, and max20 analyses.

## Original-GTV radiomics sources

- NSCLC-Radiomics training: Le et al. Supplementary Table S1
- NSCLC-Radiomics testing: Le et al. Supplementary Table S2
- NSCLC-Radiogenomics validation: Le et al. Supplementary Table S3

The locally re-extracted `lung2_pyradiomics_851.csv` is not used in this version.

## Package contents

- `reanalyze_strict_common_portable.py`: portable reanalysis script
- `requirements-colab.txt`: Python dependencies
- `data/`: required clinical, original-radiomics, cohort-manifest, and spatial-zone feature tables
- `reference_results/`: expected principal results for verification
- `results/`: S3-based reference run and the default output location

The fixed cohorts contain 397 NSCLC-Radiomics development patients and 113 NSCLC-Radiogenomics external-validation patients.
