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


mixed_color_scale = ['#e69f00', '#56b4e9', '#009e73', '#d55e00', '#cc79a7']

################################################################################
## Load Data
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
    'cluster': [barcode2cluster[x] for x in np.squeeze(human_barcodes.to_numpy())]
})

p = (pn.ggplot(plot_df, pn.aes(x='umap1', y='umap2', color='cluster')) + 
    pn.geom_point(size=0.01))
p.save('./fig/translate/hu_alone_cluster_umap.png')

################################################################################
## Load Macaque
################################################################################
#
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
            macaque_1hot = np.eye(1, len(macaque_model.index_to_key), macaque_model.key_to_index[t])
            num_toks_with_embeddings += np.sum(macaque_1hot @ macaque_gw_matrix)
            vec += np.squeeze(macaque_1hot @ macaque_gw_matrix @ human_model.vectors)
        vec /= num_toks_with_embeddings
        sentence_embeddings[n, :] = vec


ma_cell_emb = sentence_embeddings

# combine the cell embeddings to common UMAP.
combined_emb = np.concatenate((hu_cell_emb, ma_cell_emb), axis=0)
combined_dist_matrix = sp.spatial.distance.cdist(combined_emb, combined_emb, metric='cosine')

reducer = umap.UMAP(metric='precomputed')
combined_umap_red = reducer.fit_transform(combined_dist_matrix)

plot_df = pd.DataFrame({
    'umap1': combined_umap_red[:,0],
    'umap2': combined_umap_red[:,1],
    'raw_cluster': (['hu_' + barcode2cluster[x] for x in np.squeeze(human_barcodes.to_numpy())] +
               ['ma_' + macaquebarcode2cluster[x] for x in np.squeeze(macaque_barcodes.to_numpy())]),
    'species': (['human' for x in range(hu_cell_emb.shape[0])] +
                ['macaque' for x in range(ma_cell_emb.shape[0])])
})

plot_df['sp_cluster'] = simplify_cluster_names(np.array(plot_df['raw_cluster']))
plot_df['cluster'] = [x[3:] for x in plot_df['sp_cluster']]

p = (pn.ggplot(plot_df, pn.aes(x='umap1', y='umap2', color='species')) + 
    pn.geom_point(size=0.01))
p.save('./fig/translate/hu_ma_species_plot.png')

p = (pn.ggplot(plot_df, pn.aes(x='umap1', y='umap2', color='cluster', shape='species')) + 
    pn.geom_point(size=1, alpha=0.5))
p.save('./fig/translate/hu_ma_clusters_plot.png')

p = (pn.ggplot(plot_df.loc[plot_df['is_pr'],:], pn.aes(x='umap1', y='umap2', color='cluster', shape='species')) + 
    pn.geom_point(size=0.01))
p.save('./fig/translate/hu_ma_photoreceptors.png', height=7, width=9)

p = (pn.ggplot(plot_df.loc[plot_df['is_mgc'],:], pn.aes(x='umap1', y='umap2', color='cluster', shape='species')) + 
    pn.geom_point(size=0.01))
p.save('./fig/translate/hu_ma_muller.png', height=7, width=9)

p = (pn.ggplot(plot_df.loc[plot_df['is_rgc'],:], pn.aes(x='umap1', y='umap2', color='species')) + 
    pn.geom_point(size=0.01))
p.save('./fig/translate/hu_ma_rgc.png', height=7, width=9)


################################################################################
## Load Chick
################################################################################
#
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
            chick_1hot = np.eye(1, len(chick_model.index_to_key), chick_model.key_to_index[t])
            num_toks_with_embeddings += np.sum(chick_1hot @ chick_gw_matrix)
            vec += np.squeeze(chick_1hot @ chick_gw_matrix @ human_model.vectors)
        vec /= num_toks_with_embeddings
        sentence_embeddings[n, :] = vec


ch_cell_emb = sentence_embeddings

# combine the cell embeddings to common UMAP.
combined_emb = np.concatenate((hu_cell_emb, ch_cell_emb), axis=0)
combined_dist_matrix = sp.spatial.distance.cdist(combined_emb, combined_emb, metric='cosine')

reducer = umap.UMAP(metric='precomputed')
combined_umap_red = reducer.fit_transform(combined_dist_matrix)

plot_df = pd.DataFrame({
    'umap1': combined_umap_red[:,0],
    'umap2': combined_umap_red[:,1],
    'raw_cluster': (['hu_' + barcode2cluster[x] for x in np.squeeze(human_barcodes.to_numpy())] +
               ['ch_' + chickbarcode2cluster[x] if x in chickbarcode2cluster else 'ch_Unlabeled' for x in np.squeeze(chick_barcodes.to_numpy())]),
    'species': (['human' for x in range(hu_cell_emb.shape[0])] +
                ['chick' for x in range(ch_cell_emb.shape[0])])
})
plot_df['sp_cluster'] = simplify_cluster_names(np.array(plot_df['raw_cluster']))
plot_df['cluster'] = [x[3:] for x in plot_df['sp_cluster']]

plot_df['is_mgc'] = plot_df['cluster'].isin(['Muller'])
plot_df['is_rod'] = plot_df['cluster'].isin(['Rods'])


p = (pn.ggplot(plot_df, pn.aes(x='umap1', y='umap2', color='species')) + 
    pn.geom_point(size=0.01))
p.save('./fig/translate/hu_ch_species_plot.png')

p = (pn.ggplot(plot_df, pn.aes(x='umap1', y='umap2', color='cluster')) + 
    pn.geom_point(size=0.01))
p.save('./fig/translate/hu_ch_clusters_plot.png')

p = (pn.ggplot(plot_df.loc[plot_df['is_mgc'],:], pn.aes(x='umap1', y='umap2', color='cluster', shape='species')) + 
    pn.geom_point(size=0.01))
p.save('./fig/translate/hu_ch_mgc_plot.png', height=7, width=9)

p = (pn.ggplot(plot_df.loc[plot_df['is_rod'],:], pn.aes(x='umap1', y='umap2', color='cluster', shape='species')) + 
    pn.geom_point(size=1))
p.save('./fig/translate/hu_ch_rods_plot.png', height=7, width=9)

################################################################################
## Load Mouse
################################################################################
#
# Mouse data comes from a developmental atlas dataset
#
# GSE118614_barcodes.tsv
#

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
        for t in toks:
            if not t in mouse_model.key_to_index:
                continue
            mouse_1hot = np.eye(1, len(mouse_model.index_to_key), mouse_model.key_to_index[t])
            num_toks_with_embeddings += np.sum(mouse_1hot @ mouse_gw_matrix)
            vec += np.squeeze(mouse_1hot @ mouse_gw_matrix @ human_model.vectors)
        vec /= num_toks_with_embeddings
        sentence_embeddings[n, :] = vec

mu_cell_emb = sentence_embeddings

# combine the cell embeddings to common UMAP.
combined_emb = np.concatenate((hu_cell_emb, mu_cell_emb), axis=0)
combined_dist_matrix = sp.spatial.distance.cdist(combined_emb, combined_emb, metric='cosine')

reducer = umap.UMAP(metric='precomputed')
combined_umap_red = reducer.fit_transform(combined_dist_matrix)

plot_df = pd.DataFrame({
    'umap1': combined_umap_red[:,0],
    'umap2': combined_umap_red[:,1],
    'cluster': (['hu_' + barcode2cluster[x] for x in np.squeeze(human_barcodes.to_numpy())] +
               ['mu_' + mousebarcode2cluster[x] for x in np.squeeze(mouse_barcodes.to_numpy())]),
    'species': (['human' for x in range(hu_cell_emb.shape[0])] +
                ['mouse' for x in range(mu_cell_emb.shape[0])])
})

plot_df['is_rod'] = plot_df['cluster'].isin(['hu_Rods', 'mu_Rods'])
plot_df['is_rgc'] = plot_df['cluster'].isin(
    ['hu_RGC5', 'hu_RGC6', 'hu_RGC7', 'hu_RGC8', 'hu_RGC9', 'hu_RGC10', 'hu_RGC11', 'hu_RGC12',
    'mu_Retinal Ganglion Cells'])
plot_df['is_mgc'] = plot_df['cluster'].isin(['hu_Muller', 'mu_Muller Glia'])

p = (pn.ggplot(plot_df, pn.aes(x='umap1', y='umap2', color='species')) + 
    pn.geom_point(size=0.01))
p.save('./fig/translate/hu_mu_species_plot.png')

p = (pn.ggplot(plot_df, pn.aes(x='umap1', y='umap2', color='cluster')) + 
    pn.geom_point(size=0.01))
p.save('./fig/translate/hu_mu_cluster_plot.png')


p = (pn.ggplot(plot_df.loc[plot_df['is_rod'],:], pn.aes(x='umap1', y='umap2', color='cluster')) + 
    pn.geom_point(size=0.01))
p.save('./fig/translate/hu_mu_rod_plot.png')

p = (pn.ggplot(plot_df.loc[plot_df['is_rgc'],:], pn.aes(x='umap1', y='umap2', color='species')) + 
    pn.geom_point(size=0.01))
p.save('./fig/translate/hu_mu_rgc_plot.png', height=7, width=9)

p = (pn.ggplot(plot_df.loc[plot_df['is_mgc'],:], pn.aes(x='umap1', y='umap2', color='cluster', shape='species')) + 
    pn.geom_point(size=0.01))
p.save('./fig/translate/hu_mu_mgc_plot.png', height=7, width=9)


################################################################################
## Combine Mouse, Macaque, and Chick in the Human word2vec space
################################################################################

combined_emb = np.concatenate((hu_cell_emb, ch_cell_emb, mu_cell_emb, ma_cell_emb), axis=0)
combined_dist_matrix = sp.spatial.distance.cdist(combined_emb, combined_emb, metric='cosine')

reducer = umap.UMAP(metric='precomputed')
combined_umap_red = reducer.fit_transform(combined_dist_matrix)

plot_df = pd.DataFrame({
    'umap1': combined_umap_red[:,0],
    'umap2': combined_umap_red[:,1],
    'raw_cluster': (['hu_' + barcode2cluster[x] for x in np.squeeze(human_barcodes.to_numpy())] +
               ['ch_' + chickbarcode2cluster[x] if x in chickbarcode2cluster else 'ch_Unlabeled' for x in np.squeeze(chick_barcodes.to_numpy())] +
               ['mu_' + mousebarcode2cluster[x] for x in np.squeeze(mouse_barcodes.to_numpy())] +
               ['ma_' + macaquebarcode2cluster[x] for x in np.squeeze(macaque_barcodes.to_numpy())]),
    'species': (['human' for x in range(hu_cell_emb.shape[0])] +
                ['chick' for x in range(ch_cell_emb.shape[0])] +
                ['mouse' for x in range(mu_cell_emb.shape[0])] +
                ['macaque' for x in range(ma_cell_emb.shape[0])])
})

plot_df['sp_cluster'] = simplify_cluster_names(np.array(plot_df['raw_cluster']))
plot_df['cluster'] = [x[3:] for x in plot_df['sp_cluster']]

p = (pn.ggplot(plot_df, pn.aes(x='umap1', y='umap2', color='species')) + 
    pn.geom_point(size=0.01))
p.save('./fig/translate/all_species_umap.png', height=7, width=9)

p = (pn.ggplot(plot_df, pn.aes(x='umap1', y='umap2', color='cluster', shape='species')) + 
    pn.geom_point(size=0.5) +
    pn.theme_classic())
p.save('./fig/translate/all_cluster_umap.png', height=7, width=9)

clusts_to_compare = [
    'Photoreceptors',
    'Horizontal',
    'Bipolar',
    'Amacrine',
    'RGC'
]

# get subset with only key neuronal types
combined_dist_matrix_neuronal = combined_dist_matrix[plot_df['cluster'].isin(clusts_to_compare),:][
                                                     :, plot_df['cluster'].isin(clusts_to_compare)]
reducer_neuronal = umap.UMAP(metric='precomputed')                              
combined_umap_neuronal_red = reducer_neuronal.fit_transform(combined_dist_matrix_neuronal)

plot_df_neuronal = pd.DataFrame({
    'umap1': combined_umap_neuronal_red[:,0],
    'umap2': combined_umap_neuronal_red[:,1],
    'cluster': plot_df.loc[plot_df['cluster'].isin(clusts_to_compare),'cluster'],
    'species': plot_df.loc[plot_df['cluster'].isin(clusts_to_compare),'species']
})

plot_df_neuronal['cluster'] = pd.Categorical(plot_df_neuronal['cluster'],
    categories=clusts_to_compare[::-1])

p = (pn.ggplot(plot_df_neuronal, pn.aes(x='umap1', y='umap2', fill='cluster', shape='species')) + 
    pn.geom_point(size=1.5, alpha=0.5, color='black') +
    pn.theme_classic())
p.save('./fig/translate/all_cluster_umap_neurons.png', height=10, width=15)



##################
## compare key groups across species in embedded space.
##################

tags = ['hu', 'mu', 'ch', 'ma']

plot_df['cluster']

def calc_emd_matrix(tag1, tag2):
    """ Generate EMD matrix """
    emd_dist_matrix = np.zeros(shape=(len(clusts_to_compare), len(clusts_to_compare)))
    for i, c_from in enumerate(tqdm(clusts_to_compare)):
        for j, c_to in enumerate(clusts_to_compare):
            from_idxs = np.array(plot_df['sp_cluster']==tag1 + "_" + c_from)
            to_idxs = np.array(plot_df['sp_cluster']==tag2 + "_" + c_to)
            M = combined_dist_matrix[from_idxs,:][:,to_idxs]
            a = ot.unif(M.shape[0])
            b = ot.unif(M.shape[1])
            emd_dist_matrix[i,j] = ot.emd2(a,b,M, numItermax=1e7)
    return emd_dist_matrix




def gen_concordance_matrix(tag1, tag2):
    """ Build Concordance Matrix for word2vec space evaluation """
    concordance_matrix = np.zeros(shape=(len(clusts_to_compare), len(clusts_to_compare)))
    for i, c_from in enumerate(tqdm(clusts_to_compare)):
        within_c_from_dist_1 = sp.spatial.distance.cdist(
            combined_emb[plot_df['sp_cluster']==tag1 + "_" + c_from,:],
            combined_emb[plot_df['sp_cluster']==tag1 + "_" + c_from,:], metric='cosine')
        med_c_from_1 = np.argmin(within_c_from_dist_1.sum(axis=0))
        #
        within_c_from_dist_2 = sp.spatial.distance.cdist(
            combined_emb[plot_df['sp_cluster']==tag2 + "_" + c_from,:],
            combined_emb[plot_df['sp_cluster']==tag2 + "_" + c_from,:], metric='cosine')
        med_c_from_2 = np.argmin(within_c_from_dist_2.sum(axis=0))
        for j, c_to in enumerate(clusts_to_compare):
            # 
            within_c_to_dist_1 = sp.spatial.distance.cdist(
                combined_emb[plot_df['sp_cluster']==tag1 + "_" + c_to,:],
                combined_emb[plot_df['sp_cluster']==tag1 + "_" + c_to,:], metric='cosine')
            med_c_to_1 = np.argmin(within_c_to_dist_1.sum(axis=0))
            #
            within_c_to_dist_2 = sp.spatial.distance.cdist(
                combined_emb[plot_df['sp_cluster']==tag2 + "_" + c_to,:],
                combined_emb[plot_df['sp_cluster']==tag2 + "_" + c_to,:], metric='cosine')
            med_c_to_2 = np.argmin(within_c_to_dist_2.sum(axis=0))
            #
            concordance_matrix[i,j] = (
                1 - sp.spatial.distance.cosine(combined_emb[med_c_to_1,:] - combined_emb[med_c_from_1,:],
                                               combined_emb[med_c_to_2,:] - combined_emb[med_c_from_2,:]))
    return concordance_matrix




def gen_medioid_concordance(tag1, tag2):
    """ Compare mediod of 'from' in tag1 to 'to' in tag2 """
    concordance_matrix = np.zeros(shape=(len(clusts_to_compare), len(clusts_to_compare)))
    for i, c_from in enumerate(tqdm(clusts_to_compare)):
        within_c_from_dist_1 = sp.spatial.distance.cdist(
            combined_emb[plot_df['sp_cluster']==tag1 + "_" + c_from,:],
            combined_emb[plot_df['sp_cluster']==tag1 + "_" + c_from,:], metric='cosine')
        med_c_from_1 = np.argmin(within_c_from_dist_1.sum(axis=0))
        #
        for j, c_to in enumerate(clusts_to_compare):
            within_c_to_dist_2 = sp.spatial.distance.cdist(
                combined_emb[plot_df['sp_cluster']==tag2 + "_" + c_to,:],
                combined_emb[plot_df['sp_cluster']==tag2 + "_" + c_to,:], metric='cosine')
            med_c_to_2 = np.argmin(within_c_to_dist_2.sum(axis=0))
            #
            concordance_matrix[i,j] = (
                sp.spatial.distance.cosine(combined_emb[med_c_from_1,:], combined_emb[med_c_to_2,:]))
    return concordance_matrix



hu_mu_emd_matrix = calc_emd_matrix('hu', 'mu')
hu_ch_emd_matrix = calc_emd_matrix('hu', 'ch')
hu_ma_emd_matrix = calc_emd_matrix('hu', 'ma')

ma_hu_emd_matrix = calc_emd_matrix('ma', 'hu')
mu_hu_emd_matrix = calc_emd_matrix('mu', 'hu')
ch_hu_emd_matrix = calc_emd_matrix('ch', 'hu')


# show the distances between the correct matchings
emd_dist_df = pd.DataFrame({
    'dist': np.concatenate((
        np.diagonal(hu_ma_emd_matrix),
        np.diagonal(hu_mu_emd_matrix),
        np.diagonal(hu_ch_emd_matrix)), axis=0),
    'species': ((['macaque'] * len(clusts_to_compare))
                 + (['mouse'] * len(clusts_to_compare))
                 + (['chicken'] * len(clusts_to_compare))),
    'celltype': (clusts_to_compare * 3)
})

emd_all_df = pd.DataFrame({
    'dist': np.concatenate((
        np.ravel(hu_ma_emd_matrix),
        np.ravel(hu_mu_emd_matrix),
        np.ravel(hu_ch_emd_matrix)), axis=0),
    'species': ((['macaque'] * len(clusts_to_compare) ** 2)
                 + (['mouse'] * len(clusts_to_compare) ** 2)
                 + (['chicken'] * len(clusts_to_compare) ** 2))
})

# compare matching within species, validating cell embedding
def get_diagonal_diffs(emd_mat):
    emd_nodiag = np.copy(emd_mat)
    np.fill_diagonal(emd_nodiag, np.nan)
    diff_mat = emd_nodiag + (np.zeros(shape=emd_nodiag.shape) - np.diagonal(emd_mat)).T
    return diff_mat


sp.stats.ttest_1samp(
    a=np.ravel(get_diagonal_diffs(ma_hu_emd_matrix)[
        ~np.isnan(get_diagonal_diffs(ma_hu_emd_matrix))
    ]), popmean=0)

sp.stats.ttest_1samp(
    a=np.ravel(get_diagonal_diffs(mu_hu_emd_matrix)[
        ~np.isnan(get_diagonal_diffs(mu_hu_emd_matrix))
    ]), popmean=0)

sp.stats.ttest_1samp(
    a=np.ravel(get_diagonal_diffs(ch_hu_emd_matrix)[
        ~np.isnan(get_diagonal_diffs(ch_hu_emd_matrix))
    ]), popmean=0)

# now plot heatmaps using 1/emd
comb_heatmap_df = pd.DataFrame({
    'affinity': np.concatenate( (
        np.ravel(1/ma_hu_emd_matrix),
        np.ravel(1/mu_hu_emd_matrix),
        np.ravel(1/ch_hu_emd_matrix)
    ) ),
    'distance': np.concatenate( (
        np.ravel(ma_hu_emd_matrix),
        np.ravel(mu_hu_emd_matrix),
        np.ravel(ch_hu_emd_matrix)
    ) ),
    'from': pd.Categorical(np.concatenate(
        [[x] * len(clusts_to_compare) for x in clusts_to_compare] * 3),
        categories=clusts_to_compare),
    'to': pd.Categorical(
        clusts_to_compare * len(clusts_to_compare) * 3,
        categories=clusts_to_compare),
    'species': pd.Categorical(np.concatenate((
        ['macaque'] * len(clusts_to_compare)**2,
        ['mouse'] * len(clusts_to_compare)**2,
        ['chick'] * len(clusts_to_compare)**2
    )), categories=['macaque', 'mouse', 'chick'])
})

p = (pn.ggplot(comb_heatmap_df, pn.aes(y='from', x='to', fill='affinity')) +
    pn.geom_tile() +
    pn.facets.facet_grid('. ~ species') +
    pn.scales.scale_fill_cmap(cmap_name='magma') +
    pn.theme_classic())
p.save('./fig/translate/translation_heatmaps.png', width=18, height=6)


hu_ma_emd_matrix_nodiag - (
    np.zeros(shape=hu_ma_emd_matrix_nodiag.shape) 
    - np.diagonal(hu_ma_emd_matrix)).T
np.ravel()

sp.stats.ttest_rel(np.diagonal(hu_ma_emd_matrix), np.nanmean(hu_ma_emd_matrix_nodiag, axis=1))
sp.stats.ttest_rel(np.diagonal(hu_mu_emd_matrix), np.nanmean(hu_mu_emd_matrix_nodiag, axis=1))
sp.stats.ttest_rel(np.diagonal(hu_ch_emd_matrix), np.nanmean(hu_ch_emd_matrix_nodiag, axis=1))


# compare matchings across species, looking at conservation
(pn.ggplot(emd_dist_df, pn.aes(x='species', y='dist', color='species')) +
    pn.geom_point() +
    pn.geom_line() +
    pn.facets.facet_grid('. ~ celltype'))

(pn.ggplot(emd_all_df, pn.aes(x='species', y='dist')) +
    pn.geom_boxplot())

##################
## compute distnaces manually
##################

"""
hu_rod_mean = np.mean(combined_emb[plot_df['simple_cluster']=='hu_Rods',:], axis=0)
mu_rod_mean = np.mean(combined_emb[plot_df['simple_cluster']=='mu_Rods',:], axis=0)

hu_cone_mean = np.mean(combined_emb[plot_df['simple_cluster']=='hu_Cones',:], axis=0)
mu_cone_mean = np.mean(combined_emb[plot_df['simple_cluster']=='mu_Cones',:], axis=0)

hu_bipolar_mean = np.mean(combined_emb[plot_df['simple_cluster']=='hu_Bipolar',:], axis=0)
mu_bipolar_mean = np.mean(combined_emb[plot_df['simple_cluster']=='mu_Bipolar',:], axis=0)

hu_cone_mean = np.mean(combined_emb[plot_df['simple_cluster']=='hu_Cones',:], axis=0)
mu_cone_mean = np.mean(combined_emb[plot_df['simple_cluster']=='mu_Cones',:], axis=0)

hu_rgc_mean = np.mean(combined_emb[plot_df['simple_cluster']=='hu_RGC',:], axis=0)
mu_rgc_mean = np.mean(combined_emb[plot_df['simple_cluster']=='mu_RGC',:], axis=0)


sp.spatial.distance.cosine(hu_rod_mean + (mu_cone_mean - hu_cone_mean), mu_rod_mean)

sp.spatial.distance.cosine(hu_bipolar_mean, mu_bipolar_mean)
sp.spatial.distance.cosine(hu_bipolar_mean + (mu_cone_mean - hu_cone_mean), mu_bipolar_mean)
sp.spatial.distance.cosine(hu_rgc_mean + (mu_rod_mean - hu_rod_mean), mu_rgc_mean)


sp.spatial.distance.euclidean(hu_rod_mean - hu_cone_mean, mu_rod_mean - mu_cone_mean)


combined_emb[plot_df['simple_cluster']=='hu_Rods',:] - combined_emb[plot_df['simple_cluster']=='mu_Rods',:]

sp.spatial.distance.cosine(
    np.mean(combined_emb[plot_df['simple_cluster']=='hu_Rods',:], axis=0),
    np.mean(combined_emb[plot_df['simple_cluster']=='mu_Muller',:], axis=0))
"""

#######
## Additional Manual Snippets
######
"""
p = (pn.ggplot(plot_df, pn.aes(x='umap1', y='umap2', color='species')) + 
    pn.geom_point(size=0.01))
p.save('./fig/translate/hu_mu_ch_species_plot.png', height=7, width=9)

p =(pn.ggplot(plot_df, pn.aes(x='umap1', y='umap2', color='simple_cluster')) + 
    pn.geom_point(size=0.01))
p.save('./fig/translate/hu_mu_ch_cluster_plot.png', height=7, width=9)

p =(pn.ggplot(plot_df, pn.aes(x='umap1', y='umap2', color='cluster_nospecies')) + 
    pn.geom_point(size=0.01))
p.save('./fig/translate/hu_mu_ch_clusternospecies_plot.png', height=7, width=9)


p =(pn.ggplot(plot_df.loc[plot_df['cluster_nospecies'] == 'Amacrine',:], pn.aes(x='umap1', y='umap2', color='species')) + 
    pn.geom_point(size=0.01))
p.save('./fig/translate/hu_mu_ch_amacrine_plot.png', height=7, width=9)


(pn.ggplot(plot_df.loc[
    plot_df['cluster_nospecies'].isin([
        'Muller', 'Early RPCs', 'Late RPCs', 'Neurogenic Cells', 'Photoreceptor Precursors'
    ]),:], pn.aes(x='umap1', y='umap2', color='simple_cluster')) + 
    pn.geom_point(size=0.01))

(pn.ggplot(plot_df.loc[
    plot_df['cluster_nospecies'].isin([
        'OligoDC', 'Muller', 'Neurogenic Cells', 'Photoreceptor Precursors', 'DevRods'
    ]),:], pn.aes(x='umap1', y='umap2', color='simple_cluster')) + 
    pn.geom_point(size=0.01))

(pn.ggplot(plot_df.loc[
    plot_df['cluster_nospecies'].isin([
        'Muller', 'Neurogenic Cells', 'Photoreceptor Precursors', 'DevRods'
    ]),:], pn.aes(x='umap1', y='umap2', color='cluster_nospecies', shape='species')) + 
    pn.geom_point(size=1))



# zoom in on square with mixing
zoom_df = plot_df.loc[(plot_df['umap1'] > 8.5) & (plot_df['umap1'] < 11.5)
                      & (plot_df['umap2'] > -2.5) & (plot_df['umap2'] < 1),:]

p = (pn.ggplot(zoom_df, pn.aes(x='umap1', y='umap2', color='species')) + 
    pn.geom_point(size=0.01))
p.save('./fig/translate/zoom_hu_mu_ch_species_plot.png', height=7, width=9)


p = (pn.ggplot(zoom_df, pn.aes(x='umap1', y='umap2', color='cluster')) + 
    pn.geom_point(size=0.01))
p.save('./fig/translate/zoom_hu_mu_ch_cluster_plot.png', height=7, width=9)
"""


