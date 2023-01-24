#!/usr/bin/env python
#
# @author Rahul Dhodapkar
#

import anndata as ad
import cell2sentence as cs
import scanpy as sc
import os
import umap
import sklearn.metrics as skmetrics
import pandas as pd
import plotnine as pn
import numpy as np

################################################################################
# Create Output Scaffolding
################################################################################

os.makedirs('./calc/pbmc_experiment', exist_ok=True)
os.makedirs('./fig/pbmc_experiment', exist_ok=True)

################################################################################
# Load Annotated PBMCs
################################################################################

reducer = umap.UMAP(metric='precomputed', random_state=42, n_components=1)

# load pre-processed anndata object from scanpy
adata_labeled = sc.datasets.pbmc3k_processed()
adata = sc.datasets.pbmc3k()
adata = adata[adata_labeled.obs_names,adata_labeled.var_names]
adata.obs = adata_labeled.obs
adata.obsm = adata_labeled.obsm
adata_dists = skmetrics.pairwise_distances(adata.obsm['X_pca'])
adata_umap1d = reducer.fit_transform(adata_dists)
np.savetxt('./calc/pbmc_experiment/raw_dist_mat.csv',
    adata_dists, delimiter=',')

# create cell2sentence object
csdata = cs.transforms.csdata_from_adata(adata)

# compute and plot 1d umaps for each distance of interest
distance_types = [
    'levenshtein',
    'damerau_levenshtein',
    'jaro',
    'jaro_winkler',
    'zlib_ncd'
]

for d in distance_types:
    print('===== PROCESSING ({}) ====='.format(d))
    mat = csdata.create_distance_matrix(dist_type=d)
    np.savetxt('./calc/pbmc_experiment/{}_dist_mat.csv'.format(d), mat, delimiter=',')
    tmp_umap1d = reducer.fit_transform(mat)
    plot_df = pd.DataFrame({
        'edit_UMAP'.format(d): np.ravel(tmp_umap1d),
        'orig_UMAP': np.ravel(adata_umap1d),
        'celltype': adata.obs['louvain']
    })
    plot_df.to_csv('./calc/pbmc_experiment/{}_umap_data.csv'.format(d),
        index=False)

print('All done!')
