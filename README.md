# MIRTH: Metabolite Imputation via Rank Transformation and Harmonization

## Dependencies:

This method requires NumPy, Pandas, and PyTorch. We recommend using an Anaconda environment with Python>=3.7. To install PyTorch:
```
conda install pytorch torchvision -c pytorch
```

## Running MIRTH Imputation: 

Data available here: https://www.dropbox.com/sh/c2zquiqrzlg06a9/AAA-8_V2RGJQuSzUCs-dk7eba?dl=0

To run MIRTH imputation:
1) Save all metabolomics datasets as `.csv` files to a single folder. The header in each `.csv` file should consist of sample names; the first column should consist of feature names. Feature names should be harmonized across all datasets under consideration.
2) From the main directory of the repo, run:
```
python impute.py -in <data_folder>
```

Please note that imputation of the 9 metabolomics datasets (1727 samples x 1904 features) on which MIRTH was benchmarked takes ~5 minutes on a personal machine.

## Output:

MIRTH outputs the following data files:
- `raw_aggregate_data.csv`: the aggregate data matrix consisting of the merged raw datasets
- `normalized_aggregate_data.csv`: the TIC-normalized aggregate data matrix
- `ranked_aggregate_data.csv`: the rank-transformed aggregate data matrix
- `imputed_aggregate_data.csv`: the **imputed** rank-transformed aggregate data. **This is the main MIRTH output.**

Additionally, MIRTH outputs the sample and feature embedding matrices, `sample_embedding_W.csv` and `feature_embedding_H.csv`, used in the imputation.
Finally, when cross-validation is enabled, MIRTH outputs `cv_folds_scores.csv`, which logs the mean absolute error in each fold for each number of embedding dimensions sampled.


## Additional Options:

By default, the resulting imputed data will be saved to the directory `MIRTH_out`. This default can be changed with the `-rd` flag:
```
python impute.py -in <data_folder> -rd <output_folder>
```
The default number of embedding dimensions for imputation is 30 (previously identified as optimal for the benchmarked datasets). This default can be changed with the `-d` flag:
```
python impute.py -in <data_folder> -d <n_dims>
```
Cross-validation is disabled by default. Cross-validation will automatically find the optimal number of embedding dimensions for imputation at the expense of significantly increased runtime (approximately by a factor of the number of sampled embedding dimensions times the number of folds). Cross-validation can be enabled with the `-c` flag. The range of embedding dimensions that cross-validation considers is specified by the `-cd` flag. The number of folds is given by the `-cf` flag. Cross-validation overrides `-d` assignment.
```
python impute.py -in <data_folder> -c -cd <cv_dims_start> <cv_dims_stop> -cf <n_folds>
```
