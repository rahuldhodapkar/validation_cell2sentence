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
## Load Zebrafish
################################################################################
#

#zebrafish_metadata = pd.read_csv('./data/Chick_retina_atlas_meta.csv', skiprows=1)
#zebrafish_barcodes = pd.read_csv('./data/valid_sentences/valid_barcodes.zebrafish', header = None)
#zfbarcode2cluster = {chick_metadata['TYPE'][i] : chick_metadata['group'][i] for i in range(chick_metadata.shape[0])}



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
chick_gw_matrix /= np.reshape(np.sum(chick_gw_matrix, axis=1), newshape=(chick_gw_matrix.shape[0], 1))


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
            chick_1hot = np.eye(1, len(chick_model.index_to_key), chick_model.key_to_index[t])
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
    'cluster': (['hu_' + barcode2cluster[x] for x in np.squeeze(human_barcodes.to_numpy())] +
               ['ch_' + chickbarcode2cluster[x] if x in chickbarcode2cluster else 'ch_Unlabeled' for x in np.squeeze(chick_barcodes.to_numpy())]),
    'species': (['human' for x in range(hu_cell_emb.shape[0])] +
                ['chick' for x in range(ch_cell_emb.shape[0])])
})


plot_df['is_mgc'] = plot_df['cluster'].isin(['hu_Muller', 'ch_MG-1'])

p = (pn.ggplot(plot_df, pn.aes(x='umap1', y='umap2', color='species')) + 
    pn.geom_point(size=0.01))
p.save('./fig/translate/hu_ch_species_plot.png')

p = (pn.ggplot(plot_df, pn.aes(x='umap1', y='umap2', color='cluster')) + 
    pn.geom_point(size=0.01))
p.save('./fig/translate/hu_ch_clusters_plot.png')

p = (pn.ggplot(plot_df.loc[plot_df['is_mgc'],:], pn.aes(x='umap1', y='umap2', color='cluster', shape='species')) + 
    pn.geom_point(size=0.01))
p.save('./fig/translate/hu_ch_mgc_plot.png', height=7, width=9)

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
mouse_gw_matrix /= np.reshape(np.sum(mouse_gw_matrix, axis=1), newshape=(mouse_gw_matrix.shape[0], 1))


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
            num_toks_with_embeddings += 1
            mouse_1hot = np.eye(1, len(mouse_model.index_to_key), mouse_model.key_to_index[t])
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

p = (pn.ggplot(plot_df.loc[plot_df['is_rgc'],:], pn.aes(x='umap1', y='umap2', color='cluster', shape='species')) + 
    pn.geom_point(size=0.01))
p.save('./fig/translate/hu_mu_rgc_plot.png', height=7, width=9)

p = (pn.ggplot(plot_df.loc[plot_df['is_mgc'],:], pn.aes(x='umap1', y='umap2', color='cluster', shape='species')) + 
    pn.geom_point(size=0.01))
p.save('./fig/translate/hu_mu_mgc_plot.png', height=7, width=9)


################################################################################
## Combine Mouse and Chick in the Human word2vec space
################################################################################

combined_emb = np.concatenate((hu_cell_emb, ch_cell_emb, mu_cell_emb), axis=0)
combined_dist_matrix = sp.spatial.distance.cdist(combined_emb, combined_emb, metric='cosine')

reducer = umap.UMAP(metric='precomputed')
combined_umap_red = reducer.fit_transform(combined_dist_matrix)

plot_df = pd.DataFrame({
    'umap1': combined_umap_red[:,0],
    'umap2': combined_umap_red[:,1],
    'cluster': (['hu_' + barcode2cluster[x] for x in np.squeeze(human_barcodes.to_numpy())] +
               ['ch_' + chickbarcode2cluster[x] if x in chickbarcode2cluster else 'ch_Unlabeled' for x in np.squeeze(chick_barcodes.to_numpy())] +
               ['mu_' + mousebarcode2cluster[x] for x in np.squeeze(mouse_barcodes.to_numpy())]),
    'species': (['human' for x in range(hu_cell_emb.shape[0])] +
                ['chick' for x in range(ch_cell_emb.shape[0])] +
                ['mouse' for x in range(mu_cell_emb.shape[0])])
})

########
## Re-label clusters
########

simple_clusters = np.unique(plot_df['cluster'])
simple_clusters[ simple_clusters ]

# Chicken Simplify Clusters
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

# Human Simplify Clusters
simple_clusters[
    np.vectorize(lambda x:bool(re.compile('^hu_Gaba').match(x)))(simple_clusters)] = 'hu_Amacrine'
simple_clusters[
    np.vectorize(lambda x:bool(re.compile('^hu_Gly').match(x)))(simple_clusters)] = 'hu_Amacrine'
simple_clusters[
    np.vectorize(lambda x:bool(re.compile('^hu_H\\d').match(x)))(simple_clusters)] = 'hu_Horizontal'

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

simple_clusters[
    np.vectorize(lambda x:bool(re.compile('^hu_.*Cones').match(x)))(simple_clusters)] = 'hu_Cones'

# mouse
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

simple_name_map_df = pd.DataFrame({
    'old_name': np.unique(plot_df['cluster']),
    'new_name': simple_clusters
})

cluster2simple_name = {simple_name_map_df['old_name'][i] : simple_name_map_df['new_name'][i] \
                            for i in range(simple_name_map_df.shape[0])}

plot_df['simple_cluster'] = [cluster2simple_name[x] for x in plot_df['cluster']]

plot_df['cluster_nospecies'] = [x[3:] for x in plot_df['simple_cluster']]

#######
##
######

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


##################
## compare key groups across species in embedded space.
##################

clusts_to_compare = [
    'Rods', 'Cones', 'Bipolar', 'RGC',
    'Horizontal', 'Amacrine', 'Muller'
]

tags = ['hu', 'mu', 'ch']

plot_df['cns'] = plot_df['cluster_nospecies']

# build directional heatmap
def build_heatmap_matrix(tag1, tag2):


tag1 = 'hu'
tag2 = 'mu'

def gen_concordance_matrix(tag1, tag2):
    """ Build Concordance Matrix for word2vec space evaluation """
    concordance_matrix = np.zeros(shape=(len(clusts_to_compare), len(clusts_to_compare)))
    for i, c_from in enumerate(tqdm(clusts_to_compare)):
        within_c_from_dist_1 = sp.spatial.distance.cdist(
            combined_emb[plot_df['simple_cluster']==tag1 + "_" + c_from,:],
            combined_emb[plot_df['simple_cluster']==tag1 + "_" + c_from,:], metric='cosine')
        med_c_from_1 = np.argmin(within_c_from_dist_1.sum(axis=0))
        #
        within_c_from_dist_2 = sp.spatial.distance.cdist(
            combined_emb[plot_df['simple_cluster']==tag2 + "_" + c_from,:],
            combined_emb[plot_df['simple_cluster']==tag2 + "_" + c_from,:], metric='cosine')
        med_c_from_2 = np.argmin(within_c_from_dist_2.sum(axis=0))
        for j, c_to in enumerate(clusts_to_compare):
            # 
            within_c_to_dist_1 = sp.spatial.distance.cdist(
                combined_emb[plot_df['simple_cluster']==tag1 + "_" + c_to,:],
                combined_emb[plot_df['simple_cluster']==tag1 + "_" + c_to,:], metric='cosine')
            med_c_to_1 = np.argmin(within_c_to_dist_1.sum(axis=0))
            #
            within_c_to_dist_2 = sp.spatial.distance.cdist(
                combined_emb[plot_df['simple_cluster']==tag2 + "_" + c_to,:],
                combined_emb[plot_df['simple_cluster']==tag2 + "_" + c_to,:], metric='cosine')
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
            combined_emb[plot_df['simple_cluster']==tag1 + "_" + c_from,:],
            combined_emb[plot_df['simple_cluster']==tag1 + "_" + c_from,:], metric='cosine')
        med_c_from_1 = np.argmin(within_c_from_dist_1.sum(axis=0))
        #
        for j, c_to in enumerate(clusts_to_compare):
            within_c_to_dist_2 = sp.spatial.distance.cdist(
                combined_emb[plot_df['simple_cluster']==tag2 + "_" + c_to,:],
                combined_emb[plot_df['simple_cluster']==tag2 + "_" + c_to,:], metric='cosine')
            med_c_to_2 = np.argmin(within_c_to_dist_2.sum(axis=0))
            #
            concordance_matrix[i,j] = (
                sp.spatial.distance.cosine(combined_emb[med_c_from_1,:], combined_emb[med_c_to_2,:]))
    return concordance_matrix


hu_mu_med_concordance = gen_medioid_concordance('hu', 'ch')
pd.DataFrame(hu_mu_med_concordance, index=clusts_to_compare, columns=clusts_to_compare)
np.diagonal(hu_mu_med_concordance)

sp.stats.ttest_1samp(a=np.diagonal(hu_mu_med_concordance), popmean=0)

hu_mu_concord_mat = gen_concordance_matrix('hu', 'mu')

hu_mu_concord_plot_df = pd.DataFrame(hu_mu_concord_mat, 
                                     index=clusts_to_compare, 
                                     columns=clusts_to_compare).melt()

##################
## compute distnaces manually
##################


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

