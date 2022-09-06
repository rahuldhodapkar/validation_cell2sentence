#!/usr/bin/env python
#
# Generate metadata for macaque data. Designed to be run from the 
# cell2sentence root directory.
#

import sys
import os
from pathlib import Path
import re 
import pandas as pd
import numpy as np

from tqdm import tqdm

data_dir = './data'
tag = 'macaque'
files = Path("{}/{}".format(data_dir, tag)).glob('*')
file_names = [str(f) for f in files]

file_df = pd.DataFrame({
    'fn': file_names,
    'celltype': [re.search(r'Macaque_fov_([A-Za-z]*)', fn).group(1) for fn in file_names]
})

barcodes = []
celltypes = []
for i in tqdm(range(file_df.shape[0])):
    with open(file_df['fn'][i]) as f:
        first_line = f.readline()
    ct_barcodes = first_line.rstrip().split(',')[1:]
    barcodes += ct_barcodes
    celltypes += [file_df['celltype'][i]] * len(ct_barcodes)


macaque_meta_df = pd.DataFrame({
    'barcode': barcodes,
    'celltype': celltypes
})

macaque_meta_df.to_csv('./calc/Macaque_combined_meta.csv')
