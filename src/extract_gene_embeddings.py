#!/usr/bin/env python
## extract_gene_embeddings.py
#
# Extract gene embeddings using single-word sentences for all words in combined species dictionary
# and export to an external file for inspection.
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

###############################
## load vocabulary and get words
###############################

vocb_file_path = './xlm_outpath/vocab'

word2freq = OrderedDict()
with open(vocb_file_path, 'r') as f:
    for l in f:
        word2freq[l.split()[0]] = int(l.split()[1])

# ***TODO*** known inefficiency, this can be batched.
gene_embeddings = np.zeros( (len(word2freq.keys()), 512) )

for i, w in enumerate(tqdm(word2freq.keys())):
    species = re.match(r'^([A-Za-z]*)_', w).group(1)
    batch = torch.LongTensor(1,1).fill_(dico.index(w))
    lengths = torch.LongTensor([1])
    langs = batch.clone().fill_(model_params.lang2id[species])
    encoded = encoder('fwd', x=batch.cuda(), lengths=lengths.cuda(), langs=langs.cuda(), causal=False)
    gene_embeddings[i,:] = encoded.detach().cpu().numpy()[0,0,:]


np.savetxt('./examine_embeddings/gene_embeddings.csv', gene_embeddings, delimiter=',')

print('All done!')

