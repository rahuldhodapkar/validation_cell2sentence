#!/usr/bin/env python
#
# Benchmark actual information loss by rank order transformation in single cell
# data sets from GEO.
#
# @author Rahul Dhodapkar
#

import sklearn.metrics as skmetrics
from scipy.stats import rankdata
from sklearn.feature_selection import mutual_info_regression
import sklearn.preprocessing as preproc
from sklearn.utils import shuffle
import sklearn.linear_model as lm
import sklearn.metrics as metrics
from scipy.stats import pearsonr

import scipy.optimize as optim

import anndata as ad
import scanpy as sc
import cell2sentence as cs
import numpy as np
import pandas as pd

import os
from tqdm import tqdm

import plotnine as pn

################################################################################
# Create Output Scaffolding
################################################################################

os.makedirs('./calc/info_experiment', exist_ok=True)
os.makedirs('./fig/info_experiment', exist_ok=True)

################################################################################
# Load Datasets
################################################################################

adata_labeled = sc.datasets.pbmc3k_processed()
adata = sc.datasets.pbmc3k()
adata = adata[adata_labeled.obs_names,adata_labeled.var_names]
adata.obs = adata_labeled.obs
adata.obsm = adata_labeled.obsm

################################################################################
# Calculate Rank Order Transformations
################################################################################

scale = preproc.StandardScaler(with_mean=False)

raw_X = adata.X.todense()
norm_X = np.log1p(
    np.diag(1/np.ravel(np.sum(raw_X, axis=1))) @ raw_X
)
norm_X = scale.fit_transform(np.asarray(norm_X))

#norm_X = adata_labeled.X
rank_X = np.zeros(shape=raw_X.shape)
for i in tqdm(range(adata.X.shape[0])):
    cols = np.ravel(range(adata.X.shape[1]))
    vals = np.ravel(adata.X[i,:].todense())
    cols, vals = shuffle(cols, vals)
    ranks = cols[np.argsort(-vals, kind='stable')]
    for j in range(len(ranks)):
        rank_X[i,ranks[j]] = j


ranked_norm_X = np.zeros(shape=raw_X.shape)
for i in tqdm(range(adata.X.shape[0])):
    cols = np.ravel(range(norm_X.shape[1]))
    vals = np.ravel(norm_X[i,:])
    cols, vals = shuffle(cols, vals)
    ranks = cols[np.argsort(-vals, kind='stable')]
    for j in range(len(ranks)):
        ranked_norm_X[i,ranks[j]] = j


plot_df = pd.DataFrame({
    'raw': np.ravel(raw_X),
    'norm': np.ravel(norm_X),
    'rank': np.ravel(rank_X),
    'rankednorm': np.ravel(ranked_norm_X),
    'lograw': np.log1p(np.ravel(raw_X)),
    'logrank': np.log1p(np.ravel(rank_X)),
    'lognorm': np.log1p(np.ravel(norm_X)),
    'logranknorm': np.log1p(np.ravel(ranked_norm_X))
})

################################################################################
# Generate Discriptive / Exploratory Plots
################################################################################

(pn.ggplot(plot_df.sample(10000),
    pn.aes(x='rankednorm', y='raw'))
    + pn.geom_point())

(pn.ggplot(plot_df.sample(10000),
    pn.aes(x='rankednorm', y='norm'))
    + pn.geom_point())

(pn.ggplot(plot_df.sample(10000),
    pn.aes(x='rank', y='raw'))
    + pn.geom_point())

(pn.ggplot(plot_df.sample(100000),
    pn.aes(x='logrank', y='lograw'))
    + pn.geom_point())

log_pow_model = lm.LinearRegression().fit(
    np.array(plot_df.loc[plot_df['logrank'] < 3,'logrank']).reshape(-1,1),
    plot_df.loc[plot_df['logrank'] < 3,'lograw'])

(pn.ggplot(plot_df.sample(100000),
    pn.aes(x='logrank', y='lograw'))
    + pn.geom_point()
    + pn.geom_abline(slope=log_pow_model.coef_,
                     intercept=log_pow_model.intercept_))

reg = lm.LinearRegression().fit(
    np.array(plot_df.loc[plot_df['logranknorm'] < 3,'logranknorm']).reshape(-1,1),
    plot_df.loc[plot_df['logranknorm'] < 3,'lognorm'])

(pn.ggplot(plot_df.sample(100000),
    pn.aes(x='logranknorm', y='lognorm'))
    + pn.geom_point()
    + pn.geom_abline(slope=reg.coef_,
                     intercept=reg.intercept_))

(pn.ggplot(plot_df.sample(10000),
    pn.aes(x='logranknorm', y='norm'))
    + pn.geom_point())

(pn.ggplot(plot_df.sample(10000),
    pn.aes(x='logranknorm', y='lognorm'))
    + pn.geom_point())

################################################################################
# Reconstruct Raw Expression from Rank Data
################################################################################
# power law from linear fit in log-log space.
#

# predict raw read values
raw_model = lm.LinearRegression().fit(
    np.array(plot_df.loc[plot_df['logrank'] < 3,'logrank']).reshape(-1,1),
    plot_df.loc[plot_df['logrank'] < 3,'lograw'])

rank_reconstructed_X = np.exp(
    raw_model.predict(np.asarray(plot_df['rank']).reshape(-1,1))
)

metrics.r2_score(np.asarray(plot_df['raw']), np.asarray(rank_reconstructed_X))
pearsonr(np.asarray(plot_df['raw']), np.asarray(rank_reconstructed_X))

# predict normed read values
norm_model = lm.LinearRegression().fit(
    np.array(plot_df.loc[plot_df['logranknorm'] < 3,'logranknorm']).reshape(-1,1),
    plot_df.loc[plot_df['logranknorm'] < 3,'lognorm'])

rank_reconstructed_X = np.exp(
    norm_model.predict(np.asarray(plot_df['logranknorm']).reshape(-1,1))
)

metrics.r2_score(np.asarray(plot_df['norm']), np.asarray(rank_reconstructed_X))
pearsonr(np.asarray(plot_df['norm']), np.asarray(rank_reconstructed_X))

################################################################################
# Fit Curves to Original Data
################################################################################

# the exponential here can be specified by two parameters, l and c.

np.random.seed(42)

samp_df = plot_df.sample(100)
x = samp_df['rank']
y = samp_df['raw']

popt, pcov = optim.curve_fit(
    lambda t, a, b, c: a * np.exp(b * t) + c, 
    x, y,
    p0=(50, -5, 10))


(pn.ggplot(plot_df.sample(10000),
    pn.aes(x='rank'))
    + pn.geom_histogram())


(pn.ggplot(plot_df.loc[plot_df['raw'] != 0,:].sample(10000),
    pn.aes(x='rank', y='rankednorm'))
    + pn.geom_point())

(pn.ggplot(plot_df.loc[plot_df['raw'] != 0,:].sample(10000),
    pn.aes(x='norm', y='raw'))
    + pn.geom_point())

(pn.ggplot(plot_df.loc[plot_df['raw'] != 0,:].sample(10000),
    pn.aes(x='rank', y='norm'))
    + pn.geom_point())



mutual_info_regression(
    np.ravel(adata.X[i,:].todense()).reshape(-1, 1),
    np.ravel(adata.X[i,:].todense()).reshape(-1, 1)
)




vals = np.zeros(adata.X.shape[0])
norm_vals = np.zeros(adata.X.shape[0])
for i in tqdm(range(adata.X.shape[0])):
    vals[i] = mutual_info_regression(
        np.ravel(adata.X[i,:].todense()).reshape(-1, 1),
        np.ravel(rankdata(adata.X[i,:].todense()))
    )
    norm_vals[i] = mutual_info_regression(
        np.ravel(adata.X[i,:].todense()).reshape(-1, 1),
        np.ravel(adata_labeled.X[i,:])
    )


plot_df = pd.DataFrame({
    'mutual_info':  np.append(
        vals, 
        norm_vals
    ),
    'label': np.append(
        np.repeat('rank', len(vals)),
        np.repeat('norm', len(norm_vals))
    ),
})
(pn.ggplot(plot_df, pn.aes(x='mutual_info', fill='label')) +
    pn.geom_histogram())

(pn.ggplot(plot_df, pn.aes(x='mutual_info')) +
    pn.geom_histogram())

