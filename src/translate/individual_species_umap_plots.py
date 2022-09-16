#!/usr/bin/env python
## mix_species_sentences.py
#
# Cluster each group alone and plot.
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
import ot

################################################################################
## Create Output Scaffolding
################################################################################

os.makedirs('./calc/translate', exist_ok=True)
os.makedirs('./fig/translate', exist_ok=True)

################################################################################
## Define Configuration Parameters
################################################################################

PREFIX_LEN = 10000


def simplify_cluster_names(clusters):
    simple_clusters = clusters
    simple_clusters[
        np.vectorize(lambda x:bool(re.compile('^ch_AC').match(x)))(simple_clusters)] = 'ch_Amacrine'
    simple_clusters[
        np.vectorize(lambda x:bool(re.compile('^ch_BP').match(x)))(simple_clusters)] = 'ch_Bipolar'
    simple_clusters[
        np.vectorize(lambda x:bool(re.compile('^ch_HC').match(x)))(simple_clusters)] = 'ch_Horizontal'
    simple_clusters[
        np.vectorize(lambda x:bool(re.compile('^ch_.*Cone').match(x)))(simple_clusters)] = 'ch_Cones'
    simple_clusters[
        np.vectorize(lambda x:bool(re.compile('^ch_MG').match(x)))(simple_clusters)] = 'ch_Muller'
    simple_clusters[
        np.vectorize(lambda x:bool(re.compile('^ch_OG').match(x)))(simple_clusters)] = 'ch_OligoDC'
    simple_clusters[
        np.vectorize(lambda x:bool(re.compile('^ch_RGC').match(x)))(simple_clusters)] = 'ch_RGC'
    ####################################
    # Human Simplify Clusters
    #
    simple_clusters[
        np.vectorize(lambda x:bool(re.compile('^hu_Gaba').match(x)))(simple_clusters)] = 'hu_Amacrine'
    simple_clusters[
        np.vectorize(lambda x:bool(re.compile('^hu_Gly').match(x)))(simple_clusters)] = 'hu_Amacrine'
    simple_clusters[
        np.vectorize(lambda x:bool(re.compile('^hu_H\\d').match(x)))(simple_clusters)] = 'hu_Horizontal'
    #
    simple_clusters[
        np.vectorize(lambda x:bool(re.compile('^hu_FMB').match(x)))(simple_clusters)] = 'hu_Bipolar'
    simple_clusters[
        np.vectorize(lambda x:bool(re.compile('^hu_IMB').match(x)))(simple_clusters)] = 'hu_Bipolar'
    simple_clusters[
        np.vectorize(lambda x:bool(re.compile('^hu_DB').match(x)))(simple_clusters)] = 'hu_Bipolar'
    simple_clusters[
        np.vectorize(lambda x:bool(re.compile('^hu_BB').match(x)))(simple_clusters)] = 'hu_Bipolar'
    simple_clusters[
        np.vectorize(lambda x:bool(re.compile('^hu_RB').match(x)))(simple_clusters)] = 'hu_Bipolar'
    simple_clusters[
        np.vectorize(lambda x:bool(re.compile('^hu_MG_').match(x)))(simple_clusters)] = 'hu_Bipolar'
    simple_clusters[
        np.vectorize(lambda x:bool(re.compile('^hu_PG_').match(x)))(simple_clusters)] = 'hu_Bipolar'
    simple_clusters[
        np.vectorize(lambda x:bool(re.compile('^hu_OFF').match(x)))(simple_clusters)] = 'hu_Bipolar'
    simple_clusters[
        np.vectorize(lambda x:bool(re.compile('^hu_RGC').match(x)))(simple_clusters)] = 'hu_RGC'
    #
    simple_clusters[
        np.vectorize(lambda x:bool(re.compile('^hu_.*Cones').match(x)))(simple_clusters)] = 'hu_Cones'
    ####################################
    # Mouse Simplify Clusters
    #
    simple_clusters[
        np.vectorize(lambda x:bool(re.compile('^mu_Amacrine').match(x)))(simple_clusters)] = 'mu_Amacrine'
    simple_clusters[
        np.vectorize(lambda x:bool(re.compile('^mu_Horizontal').match(x)))(simple_clusters)] = 'mu_Horizontal'
    simple_clusters[
        np.vectorize(lambda x:bool(re.compile('^mu_Retinal Ganglion').match(x)))(simple_clusters)] = 'mu_RGC'
    simple_clusters[
        np.vectorize(lambda x:bool(re.compile('^mu_Muller').match(x)))(simple_clusters)] = 'mu_Muller'
    simple_clusters[
        np.vectorize(lambda x:bool(re.compile('^mu_Bipolar').match(x)))(simple_clusters)] = 'mu_Bipolar'
    ####################################
    # Macaque Simplify Clusters
    #
    simple_clusters[
        np.vectorize(lambda x:bool(re.compile('^ma_PR').match(x)))(simple_clusters)] = 'ma_Photoreceptors'
    simple_clusters[
        np.vectorize(lambda x:bool(re.compile('^ma_BC').match(x)))(simple_clusters)] = 'ma_Bipolar'
    simple_clusters[
        np.vectorize(lambda x:bool(re.compile('^ma_RGC').match(x)))(simple_clusters)] = 'ma_RGC'
    simple_clusters[
        np.vectorize(lambda x:bool(re.compile('^ma_AC').match(x)))(simple_clusters)] = 'ma_Amacrine'
    simple_clusters[
        np.vectorize(lambda x:bool(re.compile('^ma_HC').match(x)))(simple_clusters)] = 'ma_Horizontal'
    simple_clusters[
        np.vectorize(lambda x:bool(re.compile('^ma_EpiImmune').match(x)))(simple_clusters)] = 'ma_NonNeuronal'
    #####################################
    ## Additional simplification as macaque does not contain rod vs cone annotations
    simple_clusters[
        np.vectorize(lambda x:bool(re.compile('^hu_Rods').match(x)))(simple_clusters)] = 'hu_Photoreceptors'
    simple_clusters[
        np.vectorize(lambda x:bool(re.compile('^hu_Cones').match(x)))(simple_clusters)] = 'hu_Photoreceptors'
    simple_clusters[
        np.vectorize(lambda x:bool(re.compile('^ch_Rods').match(x)))(simple_clusters)] = 'ch_Photoreceptors'
    simple_clusters[
        np.vectorize(lambda x:bool(re.compile('^ch_Cones').match(x)))(simple_clusters)] = 'ch_Photoreceptors'
    simple_clusters[
        np.vectorize(lambda x:bool(re.compile('^mu_Rods').match(x)))(simple_clusters)] = 'mu_Photoreceptors'
    simple_clusters[
        np.vectorize(lambda x:bool(re.compile('^mu_Cones').match(x)))(simple_clusters)] = 'mu_Photoreceptors'
    return simple_clusters


clusts_to_compare = [
    'Photoreceptors',
    'Horizontal',
    'Bipolar',
    'Amacrine',
    'RGC'
]

mixed_color_scale = ['#e69f00', '#56b4e9', '#009e73', '#d55e00', '#cc79a7']

################################################################################
## Load Human
################################################################################

human_model = gensim.models.KeyedVectors.load_word2vec_format(
        './data/word2vec_embs/{}_gene_vec.bin'.format('hu'), binary=True)
human_emb = human_model.vectors
human_words = human_model.index_to_key

human_metadata = pd.read_csv('./data/Human_retina_combined_all_meta.csv', skiprows=1)
barcode2cluster = {human_metadata['TYPE'][i] : human_metadata['group'][i] for i in range(human_metadata.shape[0])}

human_barcodes = pd.read_csv('./data/valid_sentences/valid_barcodes.human', header = None)


with open('./data/valid_sentences/valid.human', 'r') as fp:
    for num_human_sentences, line in enumerate(fp):
        pass

num_human_sentences += 1

sentence_embeddings = np.zeros(shape=(num_human_sentences, human_emb.shape[1]))
with open('./data/valid_sentences/valid.human', 'r') as fp:
    for n, line in enumerate(fp):
        toks = line.rstrip().split()
        num_toks_with_embeddings = 0
        vec = np.zeros(human_model.vectors.shape[1])
        for t in toks[:PREFIX_LEN]:
            if not t in human_model.key_to_index:
                continue
            num_toks_with_embeddings += 1
            vec += human_model.vectors[ human_model.key_to_index[t], : ]
        vec /= num_toks_with_embeddings
        sentence_embeddings[n, :] = vec

hu_cell_emb = sentence_embeddings
hu_dist_matrix = sp.spatial.distance.cdist(hu_cell_emb, hu_cell_emb, metric='cosine')

reducer = umap.UMAP(metric='precomputed')
embedding = reducer.fit_transform(hu_dist_matrix)

plot_df = pd.DataFrame({
    'umap1': embedding[:,0],
    'umap2': embedding[:,1],
    'barcode': np.squeeze(human_barcodes.to_numpy()),
    'raw_cluster': ['hu_' + barcode2cluster[x] for x in np.squeeze(human_barcodes.to_numpy())]
})

plot_df['sp_cluster'] = simplify_cluster_names(np.array(plot_df['raw_cluster']))
plot_df['cluster'] = [x[3:] for x in plot_df['sp_cluster']]

# plot neuronal types only
hu_neuron_dist_matrix = hu_dist_matrix[plot_df['cluster'].isin(clusts_to_compare),:][
                                       :, plot_df['cluster'].isin(clusts_to_compare)]
combined_umap_neuronal_red = reducer.fit_transform(hu_neuron_dist_matrix)

plot_df_neuronal = pd.DataFrame({
    'umap1': combined_umap_neuronal_red[:,0],
    'umap2': combined_umap_neuronal_red[:,1],
    'cluster': plot_df.loc[plot_df['cluster'].isin(clusts_to_compare),'cluster']
})

plot_df_neuronal['cluster'] = pd.Categorical(plot_df_neuronal['cluster'],
    categories=clusts_to_compare[::-1])

p = (pn.ggplot(plot_df_neuronal, pn.aes(x='umap1', y='umap2', fill='cluster', shape='species')) + 
    pn.geom_point(size=1.5, alpha=0.5, color='black', shape='^') +
    pn.scale_fill_manual(mixed_color_scale) +
    pn.theme_classic())
p.save('./fig/translate/hu_neurons_only_umap.png', height=8, width=8)

################################################################################
## Load Macaque
################################################################################
macaque_metadata = pd.read_csv('./data/Macaque_combined_meta.csv')
macaque_barcodes = pd.read_csv('./data/valid_sentences/valid_barcodes.macaque', header = None)
macaquebarcode2cluster = {macaque_metadata['barcode'][i]: macaque_metadata['celltype'][i] for i in range(macaque_metadata.shape[0])}

macaque_model = gensim.models.KeyedVectors.load_word2vec_format(
        './data/word2vec_embs/{}_gene_vec.bin'.format('ma'), binary=True)
macaque_emb = macaque_model.vectors
macaque_words = macaque_model.index_to_key

macaque_gw_map = pd.read_csv('./calc/translate/gw_ma_hu_matrix.csv', index_col=0)
macaque_gw_matrix = macaque_gw_map.to_numpy()
#macaque_gw_matrix /= np.reshape(np.sum(macaque_gw_matrix, axis=1), newshape=(macaque_gw_matrix.shape[0], 1))

with open('./data/valid_sentences/valid.macaque', 'r') as fp:
    for num_macaque_sentences, line in enumerate(fp):
        pass

num_macaque_sentences += 1


sentence_embeddings = np.zeros(shape=(num_macaque_sentences, macaque_emb.shape[1]))
with open('./data/valid_sentences/valid.macaque', 'r') as fp:
    for n, line in enumerate(tqdm(fp, total=num_macaque_sentences)):
        toks = line.rstrip().split()
        num_toks_with_embeddings = 0
        vec = np.zeros(macaque_model.vectors.shape[1])
        for t in toks[:PREFIX_LEN]:
            if not t in macaque_model.key_to_index:
                continue
            num_toks_with_embeddings += 1
            vec += macaque_model.vectors[ macaque_model.key_to_index[t], : ]
        vec /= num_toks_with_embeddings
        sentence_embeddings[n, :] = vec



ma_cell_emb = sentence_embeddings
ma_dist_matrix = sp.spatial.distance.cdist(ma_cell_emb, ma_cell_emb, metric='cosine')

plot_df = pd.DataFrame({
    'barcode': np.squeeze(macaque_barcodes.to_numpy()),
    'raw_cluster': ['ma_' + macaquebarcode2cluster[x] for x in np.squeeze(macaque_barcodes.to_numpy())]
})

plot_df['sp_cluster'] = simplify_cluster_names(np.array(plot_df['raw_cluster']))
plot_df['cluster'] = [x[3:] for x in plot_df['sp_cluster']]

# plot neuronal types only
ma_neuron_dist_matrix = ma_dist_matrix[plot_df['cluster'].isin(clusts_to_compare),:][
                                       :, plot_df['cluster'].isin(clusts_to_compare)]
combined_umap_neuronal_red = reducer.fit_transform(ma_neuron_dist_matrix)

plot_df_neuronal = pd.DataFrame({
    'umap1': combined_umap_neuronal_red[:,0],
    'umap2': combined_umap_neuronal_red[:,1],
    'cluster': plot_df.loc[plot_df['cluster'].isin(clusts_to_compare),'cluster']
})

plot_df_neuronal['cluster'] = pd.Categorical(plot_df_neuronal['cluster'],
    categories=clusts_to_compare[::-1])

p = (pn.ggplot(plot_df_neuronal, pn.aes(x='umap1', y='umap2', fill='cluster', shape='species')) + 
    pn.geom_point(size=1.5, alpha=0.5, color='black', shape='s') +
    pn.theme_classic())
p.save('./fig/translate/ma_neurons_only_umap.png', height=8, width=8)


################################################################################
## Load Mouse
################################################################################

mouse_metadata = pd.read_csv('./data/GSE118614_barcodes.tsv', sep='\t')
mouse_barcodes = pd.read_csv('./data/valid_sentences/valid_barcodes.mouse', header = None)
mousebarcode2cluster = {mouse_metadata['barcode'][i] : mouse_metadata['umap2_CellType'][i] for i in range(mouse_metadata.shape[0])}

mouse_model = gensim.models.KeyedVectors.load_word2vec_format(
        './data/word2vec_embs/{}_gene_vec.bin'.format('mu'), binary=True)
mouse_emb = mouse_model.vectors
mouse_words = mouse_model.index_to_key

mouse_gw_map = pd.read_csv('./calc/translate/gw_mu_hu_matrix.csv', index_col=0)
mouse_gw_matrix = mouse_gw_map.to_numpy()
# mouse_gw_matrix /= np.reshape(np.sum(mouse_gw_matrix, axis=1), newshape=(mouse_gw_matrix.shape[0], 1))


with open('./data/valid_sentences/valid.mouse', 'r') as fp:
    for num_mouse_sentences, line in enumerate(fp):
        pass


num_mouse_sentences += 1

sentence_embeddings = np.zeros(shape=(num_mouse_sentences, mouse_emb.shape[1]))
with open('./data/valid_sentences/valid.mouse', 'r') as fp:
    for n, line in enumerate(tqdm(fp, total=num_mouse_sentences)):
        toks = line.rstrip().split()
        num_toks_with_embeddings = 0
        vec = np.zeros(mouse_model.vectors.shape[1])
        for t in toks[:PREFIX_LEN]:
            if not t in mouse_model.key_to_index:
                continue
            num_toks_with_embeddings += 1
            vec += mouse_model.vectors[ mouse_model.key_to_index[t], : ]
        vec /= num_toks_with_embeddings
        sentence_embeddings[n, :] = vec

mu_cell_emb = sentence_embeddings
mu_dist_matrix = sp.spatial.distance.cdist(mu_cell_emb, mu_cell_emb, metric='cosine')

plot_df = pd.DataFrame({
    'barcode': np.squeeze(mouse_barcodes.to_numpy()),
    'raw_cluster': ['mu_' + mousebarcode2cluster[x] for x in np.squeeze(mouse_barcodes.to_numpy())]
})

plot_df['sp_cluster'] = simplify_cluster_names(np.array(plot_df['raw_cluster']))
plot_df['cluster'] = [x[3:] for x in plot_df['sp_cluster']]


# plot neuronal types only
mu_neuron_dist_matrix = mu_dist_matrix[plot_df['cluster'].isin(clusts_to_compare),:][
                                       :, plot_df['cluster'].isin(clusts_to_compare)]
combined_umap_neuronal_red = reducer.fit_transform(mu_neuron_dist_matrix)

plot_df_neuronal = pd.DataFrame({
    'umap1': combined_umap_neuronal_red[:,0],
    'umap2': combined_umap_neuronal_red[:,1],
    'cluster': plot_df.loc[plot_df['cluster'].isin(clusts_to_compare),'cluster']
})

plot_df_neuronal['cluster'] = pd.Categorical(plot_df_neuronal['cluster'],
    categories=clusts_to_compare[::-1])

p = (pn.ggplot(plot_df_neuronal, pn.aes(x='umap1', y='umap2', fill='cluster', shape='species')) + 
    pn.geom_point(size=1.5, alpha=0.5, color='black', shape='D') +
    pn.theme_classic())
p.save('./fig/translate/mu_neurons_only_umap.png', height=8, width=8)

################################################################################
## Load Chick
################################################################################

chick_metadata = pd.read_csv('./data/Chick_retina_atlas_meta.csv', skiprows=1)
chick_barcodes = pd.read_csv('./data/valid_sentences/valid_barcodes.chick', header = None)
chickbarcode2cluster = {chick_metadata['TYPE'][i] : chick_metadata['group'][i] for i in range(chick_metadata.shape[0])}


chick_model = gensim.models.KeyedVectors.load_word2vec_format(
        './data/word2vec_embs/{}_gene_vec.bin'.format('ch'), binary=True)
chick_emb = chick_model.vectors
chick_words = chick_model.index_to_key

chick_gw_map = pd.read_csv('./calc/translate/gw_ch_hu_matrix.csv', index_col=0)
chick_gw_matrix = chick_gw_map.to_numpy()
#chick_gw_matrix /= np.reshape(np.sum(chick_gw_matrix, axis=1), newshape=(chick_gw_matrix.shape[0], 1))


with open('./data/valid_sentences/valid.chick', 'r') as fp:
    for num_chick_sentences, line in enumerate(fp):
        pass

num_chick_sentences += 1

sentence_embeddings = np.zeros(shape=(num_chick_sentences, chick_emb.shape[1]))
with open('./data/valid_sentences/valid.chick', 'r') as fp:
    for n, line in enumerate(tqdm(fp, total=num_chick_sentences)):
        toks = line.rstrip().split()
        num_toks_with_embeddings = 0
        vec = np.zeros(chick_model.vectors.shape[1])
        for t in toks[:PREFIX_LEN]:
            if not t in chick_model.key_to_index:
                continue
            num_toks_with_embeddings += 1
            vec += chick_model.vectors[ chick_model.key_to_index[t], : ]
        vec /= num_toks_with_embeddings
        sentence_embeddings[n, :] = vec

ch_cell_emb = sentence_embeddings
ch_dist_matrix = sp.spatial.distance.cdist(ch_cell_emb, ch_cell_emb, metric='cosine')

plot_df = pd.DataFrame({
    'barcode': np.squeeze(chick_barcodes.to_numpy()),
    'raw_cluster': ['ch_' + chickbarcode2cluster[x] if x in chickbarcode2cluster else 'ch_Unlabeled' for x in np.squeeze(chick_barcodes.to_numpy())]
})

plot_df['sp_cluster'] = simplify_cluster_names(np.array(plot_df['raw_cluster']))
plot_df['cluster'] = [x[3:] for x in plot_df['sp_cluster']]


# plot neuronal types only
ch_neuron_dist_matrix = ch_dist_matrix[plot_df['cluster'].isin(clusts_to_compare),:][
                                       :, plot_df['cluster'].isin(clusts_to_compare)]
combined_umap_neuronal_red = reducer.fit_transform(ch_neuron_dist_matrix)

plot_df_neuronal = pd.DataFrame({
    'umap1': combined_umap_neuronal_red[:,0],
    'umap2': combined_umap_neuronal_red[:,1],
    'cluster': plot_df.loc[plot_df['cluster'].isin(clusts_to_compare),'cluster']
})

plot_df_neuronal['cluster'] = pd.Categorical(plot_df_neuronal['cluster'],
    categories=clusts_to_compare[::-1])

p = (pn.ggplot(plot_df_neuronal, pn.aes(x='umap1', y='umap2', fill='cluster', shape='species')) + 
    pn.geom_point(size=1.5, alpha=0.5, color='black', shape='o') +
    pn.scale_
    pn.theme_classic())
p.save('./fig/translate/ch_neurons_only_umap.png', height=8, width=8)


