#!/usr/bin/env python
## mix_species_sentences.py
#
# Generate average pooling for sentences and project to UMAP
#

import os
import pandas as pd
import numpy as np
import gensim
import umap
import plotnine as pn
from tqdm import tqdm
import scipy as sp
import re

################################################################################
## Create Output Scaffolding
################################################################################

os.makedirs('./calc/translate', exist_ok=True)
os.makedirs('./fig/translate', exist_ok=True)

################################################################################
## Define Configuration Parameters
################################################################################

PREFIX_LEN = 10000

################################################################################
## Define Configuration Parameters
################################################################################

human_model = gensim.models.KeyedVectors.load_word2vec_format(
        './data/word2vec_embs/{}_gene_vec.bin'.format('hu'), binary=True)
human_emb = human_model.vectors
human_words = human_model.index_to_key

human_metadata = pd.read_csv('./data/Human_retina_combined_all_meta.csv', skiprows=1)
barcode2cluster = {human_metadata['TYPE'][i] : human_metadata['group'][i] for i in range(human_metadata.shape[0])}

human_barcodes = pd.read_csv('./data/valid_sentences/valid_barcodes.human', header = None)


mouse_metadata = pd.read_csv('./data/GSE118614_barcodes.tsv', sep='\t')
mouse_barcodes = pd.read_csv('./data/valid_sentences/valid_barcodes.mouse', header = None)
mousebarcode2cluster = {mouse_metadata['barcode'][i] : mouse_metadata['umap2_CellType'][i] for i in range(mouse_metadata.shape[0])}

mouse_model = gensim.models.KeyedVectors.load_word2vec_format(
        './data/word2vec_embs/{}_gene_vec.bin'.format('mu'), binary=True)
mouse_emb = mouse_model.vectors
mouse_words = mouse_model.index_to_key

mouse_gw_map = pd.read_csv('./calc/translate/gw_mu_hu_matrix.csv', index_col=0)
mouse_gw_matrix = mouse_gw_map.to_numpy()

# match human and macaque words
hu_words_stripped = np.array([x[6:] for x in human_words])
mu_words_stripped = np.array([x[6:] for x in mouse_words])

homolog_mu2hu = {}
for i, w in enumerate(mu_words_stripped):
    if w == '':
        continue   # skip first word ('')
    w = w.upper()
    #
    #
    match = np.where(hu_words_stripped == w)[0]
    if len(match) != 1:
        continue
    homolog_mu2hu[i] = int(match[0])


mapped_dists = []
unmapped_dists = []
for i, j in tqdm(homolog_mu2hu.items()):
    #print("{} -> {}".format(i, j))
    dist_unmapped = sp.spatial.distance.cosine(
        mouse_model.vectors[i,:], human_model.vectors[j,:])
    dist_mapped = sp.spatial.distance.cosine(
        np.sum(np.eye(1, len(mouse_model.index_to_key), i) @ mouse_gw_matrix @ human_model.vectors, axis=0),
        human_model.vectors[j,:])
    unmapped_dists.append(dist_unmapped)
    mapped_dists.append(dist_mapped)

[(mouse_words[x[0]], human_words[x[1]] for x in homolog_mu2hu.items()]

plot_df = pd.DataFrame({
    'dist': mapped_dists + unmapped_dists,
    'type': (['mapped'] * len(mapped_dists)) + (['unmapped'] * len(unmapped_dists))
})

(pn.ggplot(plot_df, pn.aes(x='type', y='dist')) +
    pn.geom_boxplot())

# now let's try with random pairs
np.random.seed(42)

random_pairs = []
while len(random_pairs) < len(homolog_mu2hu.items()):
    i = np.random.randint(len(mouse_words))
    j = np.random.randint(len(human_words))
    if i not in homolog_mu2hu or homolog_mu2hu[i] != j:
        random_pairs.append( (i,j) )



rand_mapped_dists = []
rand_unmapped_dists = []
for i, j in tqdm(random_pairs):
    #print("{} -> {}".format(i, j))
    dist_unmapped = sp.spatial.distance.cosine(
        mouse_model.vectors[i,:], human_model.vectors[j,:])
    dist_mapped = sp.spatial.distance.cosine(
        np.sum(np.eye(1, len(mouse_model.index_to_key), i) @ mouse_gw_matrix @ human_model.vectors, axis=0),
        human_model.vectors[j,:])
    rand_unmapped_dists.append(dist_unmapped)
    rand_mapped_dists.append(dist_mapped)


rand_plot_df = pd.DataFrame({
    'dist': rand_mapped_dists + rand_unmapped_dists,
    'type': (['mapped'] * len(mapped_dists)) + (['unmapped'] * len(unmapped_dists))
})

(pn.ggplot(rand_plot_df, pn.aes(x='type', y='dist')) +
    pn.geom_boxplot())

sp.stats.ranksums(x=mapped_dists, y=rand_mapped_dists)

merged_plot_df = pd.DataFrame({
    'dist': mapped_dists + rand_mapped_dists,
    'diffdist': ([mapped_dists[x] - unmapped_dists[x] for x in range(len(mapped_dists))] +
                 [rand_mapped_dists[x] - rand_unmapped_dists[x] for x in range(len(rand_mapped_dists))]),
    'type': (['homolog'] * len(mapped_dists)) + (['random'] * len(rand_mapped_dists))
})
(pn.ggplot(merged_plot_df, pn.aes(x='type', y='diffdist')) +
    pn.geom_violin())



