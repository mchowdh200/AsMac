from pprint import pprint
from AsMac_model import AsMac
from AsMac_utility import one_hot, SeqIteratorDataset

import faiss
import torch
from torch.utils.data import DataLoader

import numpy as np
torch.set_printoptions(profile="full")
np.set_printoptions(precision=10)

## load pretrained model
embed_dim = 300 # number of kernel sequences
kernel_size = 20 # kernel length
learning_rate=1e-3 # learning rate
net = AsMac(4, embed_dim, kernel_size)
net_state_dict = torch.load('model/final.pt')
net.load_state_dict(net_state_dict)


## get embeddings, add to faiss index
fasta = 'data/test_subset.fa'
dataset = SeqIteratorDataset(fasta)
dataloader = DataLoader(dataset=dataset, batch_size=2,)
index = faiss.IndexFlatL2(embed_dim)
with torch.no_grad():
    for records in dataloader:
        # for id in records['id']: # vector of fasta sequence IDs
        seq_oh = one_hot(records['seq'])

        # rows are embeddings
        embeddings = net.get_embeddings(seq_oh) \
                        .detach().numpy().astype(np.float32)

        # add the vectors (rows) to the index
        index.add(embeddings) 

        pprint(index.search(embeddings, 5))

## test nearest neighbor query
# TODO need to add a dict that associates a fasta sequence record with the
# integer ID from the FAISS index.

