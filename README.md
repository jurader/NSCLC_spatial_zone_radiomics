# Reanalysis of NSCLC spatial-zone radiomics

This notebook recalculates feature selection, final Cox models, and C-indices for max5, max10, and max20 from CSV files of the spatial zone radiomics.

## File in this repository
- `reanalyze_spatial_zone_radiomics.ipynb`: Google Colab notebook

## File in other storage service
- `colab_reanalysis.zip` zip file which includes necessary files for running the ipynb file.  

https://filedn.com/lpAczQGgeBjkX6l7SpI5JJy/public_code/nsclc_spatial_zone_radiomics/colab_reanalysis.zip 

## Content in the zip file

- `reanalyze_strict_common_portable.py`: analysis code adapted to use relative paths
- `requirements-colab.txt`: principal package versions matching the validated environment
- `data/`: all required input files
- `reference_results/`: principal results from the original analysis for reproducibility checks
- `results/`: output directory for recalculated results

## Running the analysis in Google Colab

1. Open the ipynb file in Google Colab.
2. Run the cells in order.
3. After the max5, max10, and max20 analyses finish, review the comparison with the original results.
4. Use the final cell to download the recalculated results as a ZIP archive.

CT images and GTV masks are not required. This reanalysis starts from the CSV files of radiomics feature.

## Input files

- `data/clinical_outcome/Supplementary-Table-S1-Training-set-LUNG1-Radiomics.csv`
- `data/clinical_outcome/Supplementary-Table-S2-Testing-set-LUNG1-Radiomics.csv`
- `data/clinical_outcome/Supplementary-Table-S3-Validation-set-LUNG2-Radiogenomics.csv`
- `data/radiomics_851_lung2/lung2_pyradiomics_851.csv`
- `data/common_cohort_manifest/aligned_case_manifest.csv`
- `data/spatial_features/features_physical2mm.csv`
- `data/spatial_features/features_physical3mm.csv`
- `data/spatial_features/features_physical5mm.csv`

## Reproduced workflow

- Load the fixed 397-patient LUNG1 development cohort and 113-patient LUNG2 validation cohort.
- Construct the 10 analysis conditions.
- Perform training-fold median imputation, standardization, and clipping to `[-10, 10]` during cross-validation. No imaging-feature values are missing in these datasets, so the implemented imputation safeguard does not alter any values.
- Select the Elastic-Net Cox `l1_ratio` and `alpha` using stratified five-fold cross-validation.
- Rank nonzero-coefficient features by the absolute Elastic-Net coefficient and retain up to 5, 10, or 20 features.
- Fit the final models using `CoxPHFitter(penalizer=0.01, robust=True)`.
- Calculate risk scores as log partial hazards.
- Calculate Harrell's C-index.

