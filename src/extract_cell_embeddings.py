#!/usr/bin/env python
## extract_gene_embeddings.py
#
# Extract cell embeddings by average pooling the embedded representations for cell sentences.
#

import torch
import sys
import argparse
import re
import numpy as np
from collections import OrderedDict

from tqdm import tqdm

import xlm
from xlm.utils import AttrDict
from xlm.data.dictionary import Dictionary
from xlm.model.transformer import TransformerModel

import os

# create output directory
os.makedirs('./examine_embeddings/', exist_ok=True)

################################
## load language model
################################

parser = parser = argparse.ArgumentParser(description="Translate sentences")
params = parser.parse_args()

# set defaults
params.batch_size = 32

reloaded = torch.load('/home/rd389/scratch60/xsm/XLM/dumped/cross_species/hvg_shortcell_bak/best-valid_mlm_ppl.pth')
model_params = AttrDict(reloaded['params'])

for name in ['n_words', 'bos_index', 'eos_index', 'pad_index', 'unk_index', 'mask_index']:
    setattr(params, name, getattr(model_params, name))

dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])

encoder = TransformerModel(model_params, dico, is_encoder=True, with_output=True).cuda().eval()
encoder.load_state_dict(reloaded['encoder'])

################################
## define cell embedding extraction function
################################

def get_cell_embeddings(src_sent, species_name):
    cell_embeddings = np.zeros( (len(src_sent), 512) )
    params.src_id = model_params.lang2id[species_name]
    for i in tqdm(range(0, len(src_sent), params.batch_size)):
        # prepare batch
        word_ids = [torch.LongTensor([dico.index(w) for w in s.strip().split()])
                    for s in src_sent[i:i + params.batch_size]]
        lengths = torch.LongTensor([len(s) + 2 for s in word_ids])
        batch = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(params.pad_index)
        batch[0] = params.eos_index
        for j, s in enumerate(word_ids):
            if lengths[j] > 2:  # if sentence not empty
                batch[1:lengths[j] - 1, j].copy_(s)
            batch[lengths[j] - 1, j] = params.eos_index
        langs = batch.clone().fill_(params.src_id)
        # encode source batch and translate it
        encoded = encoder('fwd', x=batch.cuda(), lengths=lengths.cuda(), langs=langs.cuda(), causal=False)
        encoded = encoded.transpose(0, 1)
        #
        enc_array = encoded.detach().cpu().numpy()
        enc_array = np.mean(enc_array, axis=1)
        cell_embeddings[i:i+enc_array.shape[0],:] = enc_array
    return cell_embeddings

################################
## load data
################################
languages = ['human', 'mouse', 'chick', 'zebrafish']

for l in languages:
    validation_sentences = None
    with open('./xlm_outpath/valid.{}'.format(l), 'r') as f:
        validation_sentences = [line.rstrip() for line in f]
    #
    #
    cell_embeddings = get_cell_embeddings(validation_sentences, l)
    np.savetxt('./examine_embeddings/{}_cell_embeddings.csv'.format(l), cell_embeddings, delimiter=',')


print('All done!')

