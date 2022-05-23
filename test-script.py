from pprint import pprint
from AsMac_model import AsMac
from AsMac_utility import one_hot, SeqIteratorDataset

import torch
from torch.utils.data import DataLoader

import numpy as np
torch.set_printoptions(profile="full")
np.set_printoptions(precision=10)

## load pretrained model
# ====================================================================
embed_dim = 300 # number of kernel sequences
kernel_size = 20 # kernel length
learning_rate=1e-3 # learning rate
net = AsMac(4, embed_dim, kernel_size)
net_state_dict = torch.load('model/final.pt')
net.load_state_dict(net_state_dict)

fasta = 'data/test_subset.fa'
dataset = SeqIteratorDataset(fasta)
dataloader = DataLoader(dataset=dataset, batch_size=4, )
with torch.no_grad():
    for records in dataloader:
        pprint(records['id'])
        seq_oh = one_hot(records['seq'])
        pprint(net.get_embeddings(seq_oh).detach().numpy().astype(np.float64))

