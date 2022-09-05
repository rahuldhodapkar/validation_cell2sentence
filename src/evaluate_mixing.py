#!/usr/bin/env python
#
# Evaluate mixing of samples with cell2sentence.
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

adata_jurkat = sc.read_10x_mtx(
    'data/batch/jurkat/hg19/',
    var_names='gene_symbols',
    cache=True)
adata_293t = sc.read_10x_mtx(
    'data/batch/293t/hg19/',
    var_names='gene_symbols',
    cache=True)
adata_5050 = sc.read_10x_mtx(
    'data/batch/jurkat_293t_mix/hg19/',
    var_names='gene_symbols',
    cache=True)

adata_combined = ad.concat({
    'jurkat': adata_jurkat,
    '293t': adata_293t,
    'jurkat_293t_mix': adata_5050
}, label='dataset', axis=0)

################################################################################
## Perform Standard Clustering and Visualization
################################################################################
#
# Following tutorial at:
#   https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html
#

adata = adata_combined

adata.var_names_make_unique()
adata.obs_names_make_unique()
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# annotate the group of mitochondrial genes as 'mt'
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], 
                           percent_top=None, log1p=False, inplace=True)
adata = adata[adata.obs.n_genes_by_counts < 2500, :]
adata = adata[adata.obs.pct_counts_mt < 5, :]


sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=1000, flavor='seurat_v3')
adata.raw = adata
adata = adata[:, adata.var.highly_variable]
#sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
#sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=20)
sc.tl.umap(adata)

sc.pl.umap(adata, color='dataset')

################################################################################
## Perform cell2sentence Clustering and Visualization
################################################################################
#
# Following tutorial at:
#   https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html
#

adata = adata_combined

adata.var_names_make_unique()
adata.obs_names_make_unique()
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# annotate the group of mitochondrial genes as 'mt'
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], 
                           percent_top=None, log1p=False, inplace=True)
adata = adata[adata.obs.n_genes_by_counts < 2500, :]
adata = adata[adata.obs.pct_counts_mt < 5, :]

adata_hvg = adata
sc.pp.normalize_total(adata_hvg, target_sum=1e4)
sc.pp.log1p(adata_hvg)
sc.pp.highly_variable_genes(adata_hvg, n_top_genes=1000, flavor='seurat_v3')

adata = adata[:, adata_hvg.var.highly_variable]

csdata = cs.transforms.csdata_from_adata(adata)
dist = csdata.create_distance_matrix(dist_type='damerau_levenshtein', prefix_len=20)
csdata.create_knn_graph(k=10)

reducer = umap.UMAP(metric='precomputed', n_components=2)
embedding = reducer.fit_transform(dist)

plot_df = pd.DataFrame({
    'UMAP1': embedding[:, 0],
    'UMAP2': embedding[:, 1],
    'dataset': adata.obs['dataset']
})

(pn.ggplot(plot_df, pn.aes(x='UMAP1', y='UMAP2', color='dataset'))
    + pn.geom_point())

