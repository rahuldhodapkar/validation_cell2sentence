#!/usr/bin/env python
#
#
#

import gensim

species_short_names = ['ch', 'hu', 'ma', 'mu', 'zf']
species_long_names = [
'chick',
'human',
'macaque',
'mouse',
'zebrafish'
]

for i in range(len(species_short_names)):
    wv = gensim.models.KeyedVectors.load_word2vec_format(
        "./data/word2vec_embs/{}_gene_vec.bin".format(species_short_names[i]), binary=True)
    trim_prefix_len = len(species_long_names[i]) + 1
    wv_gene_names = gensim.models.KeyedVectors(vector_size=200)
    wv_gene_names.add_vectors(
        [x[trim_prefix_len:] for x in wv.index_to_key],
        wv.vectors
    )
    wv_gene_names.save('./calc/gene_names_{}_retina.wv'.format(species_long_names[i]))
