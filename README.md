# Reanalysis of NSCLC spatial-zone radiomics

This repository provides a Google Colab workflow for recalculating feature selection, final Cox models, and Harrell’s C-indices for the max5, max10, and max20 settings using previously saved spatial-zone radiomics CSV files.

## File in this repository
- `reanalyze_spatial_zone_radiomics.ipynb`: Google Colab notebook for running the reanalysis.

## Files hosted externally
- `colab_reanalysis.zip`: Zip file contains the files required to run the notebook, including the analysis script, package requirements, input data, reference results, and output directory.

The zip file is available from the following URL:

https://filedn.com/lpAczQGgeBjkX6l7SpI5JJy/public_code/nsclc_spatial_zone_radiomics/colab_reanalysis.zip 

## Content in the zip file

- `reanalyze_strict_common_portable.py`: analysis code adapted to use relative paths
- `requirements-colab.txt`: principal package versions matching the validated environment
- `data/`: all required input files
- `reference_results/`: principal results from the original analysis for reproducibility checks
- `results/`: output directory for recalculated results

## Running the analysis in Google Colab

1 Open the notebook in Google Colab.
2 Run the cells in order.
3 After the max5, max10, and max20 analyses are complete, review the comparison with the original results.
4 Run the final cell to download the recalculated results as a ZIP archive.

CT images and GTV masks are not required because this reanalysis starts from previously saved radiomics feature CSV files.

## Reproduced workflow

The workflow uses LUNG1 development and LUNG2 validation cohorts, constructs 10 analysis conditions, performs preprocessing within cross-validation, selects Elastic-Net Cox hyperparameters using stratified five-fold cross-validation, retains up to 5, 10, or 20 selected features, fits the final Cox models, and calculates log-partial-hazard risk scores and Harrell’s C-index.

