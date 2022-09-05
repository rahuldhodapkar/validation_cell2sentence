#!/usr/bin/env python
#
# Evaluate mixing of samples using dual cell multiplexing oligo (CMO) sample
#

import os
import scanpy as sc
import cell2sentence as cs
import anndata as ad

import matplotlib.pyplot as plt
import matplotlib as mpl
import umap

import pandas as pd
import numpy as np
import plotnine as pn

################################################################################
## Create Output Scaffolding
################################################################################

os.makedirs('./calc/batch', exist_ok=True)

################################################################################
## Load and Clean Data
################################################################################

adata = sc.read_10x_mtx(
    'data/multiplexed/pbmc_multi_cmo/',
    var_names='gene_symbols',
    cache=True)
adata_orig = adata

meta = pd.read_csv('./data/multiplexed/metadata.csv')

barcode2sample = {meta['cell_barcode'][i]: meta['feature_call'][i] for i in range(len(meta))}

################################################################################
## Perform Standard Clustering and Visualization
################################################################################
#
# Following tutorial at:
#   https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html
#
adata = adata_orig

adata.obs['sample'] = [barcode2sample[x] if x in barcode2sample else None for x in adata.obs_names]

sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata.raw = adata
adata = adata[:, adata.var.highly_variable]
#sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
#sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=20)
sc.tl.umap(adata)

sc.pl.umap(adata, color='sample')



