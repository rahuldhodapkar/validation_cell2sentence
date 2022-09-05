# Validation of cell2sentence

As part of the validation for cell2sentence, we would like to replicate
the cell line experiment by Korunsky et al in the Harmony paper, and
show that rank ordering alone is sufficient to provide batch normalization


## Data
Data download links provided courtesy of 10x genomics:

jurkat  https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/jurkat
hek293t https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/293t
half    https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/jurkat:293t_50:50

the `Gene / cell matrix (filtered)` was downloaded for each of these datasets
and unpacked to the `data` directory for evaluation by cell2sentence.

