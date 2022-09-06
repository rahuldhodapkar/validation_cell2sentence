#!/usr/bin/env python
#
# Use classic optimal transport after anchoring embeddings with gene homologs
#

import gensim
import os
import numpy as np

import pandas as pd

from sklearn.neighbors import kneighbors_graph
import scipy as sp
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D  # noqa
import ot
import re

import igraph as ig

################################################################################
## Create Output Scaffolding
################################################################################

os.makedirs('./calc/translate', exist_ok=True)
os.makedirs('./fig/translate', exist_ok=True)

################################################################################
## Load Data
################################################################################

np.random.seed(42)

short_tags = ['hu', 'mu', 'zf', 'ch', 'ma']

for t_from in range(len(short_tags)):
    #for t_to in range(len(short_tags)):
    t_to = 0
    if t_from == 0:
        continue
    #    if i == j:
    #        continue
    tag1 = short_tags[t_from]
    tag2 = short_tags[t_to]
    print("Mapping {} -> {}".format(tag1, tag2))
    #
    model1 = gensim.models.KeyedVectors.load_word2vec_format(
        './data/word2vec_embs/{}_gene_vec.bin'.format(tag1), binary=True)
    emb1 = model1.vectors
    words1 = model1.index_to_key
    #
    model2 = gensim.models.KeyedVectors.load_word2vec_format(
        './data/word2vec_embs/{}_gene_vec.bin'.format(tag2), binary=True)
    emb2 = model2.vectors
    words2 = model2.index_to_key
    #
    emb1 -= emb1.mean(axis=0)
    emb2 -= emb2.mean(axis=0)
    # and scale
    emb1 /= np.linalg.norm(emb1, axis=1)[:,None]
    emb2 /= np.linalg.norm(emb2, axis=1)[:,None]
    #
    #
    prefix_1 = re.match('^([A-Za-z]*_)', words1[1]).group(1)
    prefix_2 = re.match('^([A-Za-z]*_)', words2[1]).group(1)
    #
    words1_stripped = np.array([x.replace(prefix_1, '').upper() for x in words1])
    words2_stripped = np.array([x.replace(prefix_2, '').upper() for x in words2])
    #
    homologs = {}
    for i, w in enumerate(words1_stripped):
        if w == '':
            continue   # skip first word ('')
        #
        #
        match = np.where(words2_stripped == w)[0]
        if len(match) != 1:
            continue
        homologs[i] = int(match[0])
    #
    M = np.ones((len(words1), len(words2)))
    #
    for i, j in homologs.items():
        M[i, j] = 0
    #
    #
    C1 = sp.spatial.distance.cdist(emb1, emb1, metric='cosine')
    C2 = sp.spatial.distance.cdist(emb2, emb2, metric='cosine')
    #
    #########
    # Normalize Distances
    #########
    # mean
    C1 /= C1.mean()
    C2 /= C2.mean()
    #
    p = ot.unif(emb1.shape[0])
    q = ot.unif(emb2.shape[0])
    #
    gw0, log0 = ot.gromov.fused_gromov_wasserstein(
        M, C1, C2, p, q, loss_fun='square_loss', alpha=0.5, verbose=True, log=True)
    #gw0, log0 = ot.gromov.entropic_gromov_wasserstein(
    #    C1, C2, p, q, loss_fun='square_loss', epsilon=1e-1, verbose=True, log=True)
    #gw0, log0 = ot.partial.partial_gromov_wasserstein(
    #    C1, C2, p, q, m=0.5, verbose=True, log=True, numItermax=200)
    #gw0, log0 = ot.gromov.pointwise_gromov_wasserstein(C1, C2, p, q, loss, max_iter=100,
    #                                               log=True)
    save_df = pd.DataFrame(
        gw0, index=pd.Index(words1), columns=pd.Index(words2)
    )
    #log_df = pd.DataFrame(log0)
    save_df.to_csv('./calc/translate/gw_{}_{}_matrix.csv'.format(tag1, tag2))
    #log_df.to_csv('./calc/translate/log_gw_{}_{}.csv'.format(tag1, tag2))


