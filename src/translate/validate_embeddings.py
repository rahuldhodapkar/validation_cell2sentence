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



macaque_metadata = pd.read_csv('./data/Macaque_combined_meta.csv')
macaque_barcodes = pd.read_csv('./data/valid_sentences/valid_barcodes.macaque', header = None)
macaquebarcode2cluster = {macaque_metadata['barcode'][i]: macaque_metadata['celltype'][i] for i in range(macaque_metadata.shape[0])}

macaque_model = gensim.models.KeyedVectors.load_word2vec_format(
        './data/word2vec_embs/{}_gene_vec.bin'.format('ma'), binary=True)
macaque_emb = macaque_model.vectors
macaque_words = macaque_model.index_to_key

macaque_gw_map = pd.read_csv('./calc/translate/gw_ma_hu_matrix.csv', index_col=0)
macaque_gw_matrix = macaque_gw_map.to_numpy()

# match human and macaque words
hu_words_stripped = np.array([x[6:] for x in human_words])
ma_words_stripped = np.array([x[8:] for x in macaque_words])

homolog_ma2hu = {}
for i, w in enumerate(ma_words_stripped):
    if w == '':
        continue   # skip first word ('')
    #
    #
    match = np.where(hu_words_stripped == w)[0]
    if len(match) != 1:
        continue
    homolog_ma2hu[i] = int(match[0])


mapped_dists = []
unmapped_dists = []
for i, j in tqdm(homolog_ma2hu.items()):
    #print("{} -> {}".format(i, j))
    dist_unmapped = sp.spatial.distance.cosine(
        macaque_model.vectors[i,:], human_model.vectors[j,:])
    dist_mapped = sp.spatial.distance.cosine(
        np.sum(np.eye(1, len(macaque_model.index_to_key), i) @ macaque_gw_matrix @ human_model.vectors, axis=0),
        human_model.vectors[j,:])
    unmapped_dists.append(dist_unmapped)
    mapped_dists.append(dist_mapped)

extreme_low_distance = [list(homolog_ma2hu.items())[x] for x in range(len(mapped_dists)) if mapped_dists[x] < 1e-3]


plot_df = pd.DataFrame({
    'dist': mapped_dists + unmapped_dists,
    'type': (['mapped'] * len(mapped_dists)) + (['unmapped'] * len(unmapped_dists))
})

(pn.ggplot(plot_df, pn.aes(x='type', y='dist')) +
    pn.geom_boxplot())

# now let's try with random pairs
np.random.seed(42)

random_pairs = []
while len(random_pairs) < len(homolog_ma2hu.items()):
    i = np.random.randint(len(macaque_words))
    j = np.random.randint(len(human_words))
    if i not in homolog_ma2hu or homolog_ma2hu[i] != j:
        random_pairs.append( (i,j) )



rand_mapped_dists = []
rand_unmapped_dists = []
for i, j in tqdm(random_pairs):
    #print("{} -> {}".format(i, j))
    dist_unmapped = sp.spatial.distance.cosine(
        macaque_model.vectors[i,:], human_model.vectors[j,:])
    dist_mapped = sp.spatial.distance.cosine(
        np.sum(np.eye(1, len(macaque_model.index_to_key), i) @ macaque_gw_matrix @ human_model.vectors, axis=0),
        human_model.vectors[j,:])
    rand_unmapped_dists.append(dist_unmapped)
    rand_mapped_dists.append(dist_mapped)


extreme_low_distance_rand = [random_pairs[x] for x in range(len(rand_mapped_dists)) if rand_mapped_dists[x] < 1e-3]
[ (macaque_words[x[0]], human_words[x[1]]) for x in extreme_low_distance_rand ]


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
    pn.geom_boxplot())



