#!/usr/bin/env python
#
# Use Gromov optimal transport to directly translate based on word2vec embedding
#

import gensim
import os
import numpy as np

import pandas as pd

import scipy as sp
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D  # noqa
import ot

################################################################################
## Create Output Scaffolding
################################################################################

os.makedirs('./calc/translate', exist_ok=True)
os.makedirs('./fig/translate', exist_ok=True)

################################################################################
## Load Data
################################################################################

def loss(x, y):
    return np.abs(x - y)


np.random.seed(42)

short_tags = ['hu', 'mu', 'zf', 'ch', 'ma']

for i in range(len(short_tags)):
    #for j in range(len(short_tags)):
    j = 0
    if i == 0:
        continue
    #    if i == j:
    #        continue
    tag1 = short_tags[i]
    tag2 = short_tags[j]
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
    #########
    ## Normalize Embeddings (scale and center)
    #########
    # center
    #
    emb1 -= emb1.mean(axis=0)
    emb2 -= emb2.mean(axis=0)
    # and scale
    emb1 /= np.linalg.norm(emb1, axis=1)[:,None]
    emb2 /= np.linalg.norm(emb2, axis=1)[:,None]
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
    gw0, log0 = ot.gromov.gromov_wasserstein(
        C1, C2, p, q, loss_fun='square_loss', verbose=True, log=True)
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

#gw, log = ot.gromov.entropic_gromov_wasserstein(
#    C1, C2, p, q, 'square_loss', epsilon=5e-4, log=True, verbose=True)


# query particular genes
'''
src_idx = 5
words1[src_idx]

tgts = np.eye(1, len(words1), src_idx) @ gw0
np.argsort(-tgts)

tgts[0, 774]
words2[774]
'''