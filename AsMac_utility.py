from __future__ import annotations
import gzip
import math

import random
from typing import Optional
random.seed(1)

import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset, IterableDataset, DataLoader
from Bio import SeqIO

from AsMac_model import AsMac

def load_pretrained(model: str) -> AsMac:
    embed_dim = 300 # number of kernel sequences
    kernel_size = 20 # kernel length
    learning_rate=1e-3 # learning rate
    net = AsMac(4, embed_dim, kernel_size)
    net_state_dict = torch.load(model)
    net.load_state_dict(net_state_dict)
    return net

def one_hot(ss, basis={'A': 0, 'T': 1, 'C': 2, 'G': 3}):
    feature_list = []
    for s in ss:
        feature = np.zeros([4, len(s)])
        for i, c in enumerate(s):
            if c not in ['A', 'T', 'G', 'C']:
                continue
            else:
                feature[basis[c], i] = 1
        feature_list.append(feature)
    return feature_list


def time_cost(seconds):

    hour = math.floor(seconds / 3600)
    minute = math.floor((seconds - 3600 * hour) / 60)
    second = int(seconds - 3600 * hour - 60 * minute)

    out_put = '%2ih:%2im:%2is' % (hour, minute, second)
    return out_put


class MSELoss(torch.nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, output, dist_m):
        MSE = torch.mean(torch.pow(output - dist_m, 2))
        # print(torch.pow(output - dist_m, 2))
        # print(MSE)
        return MSE

class SeqIteratorDataset(IterableDataset):
    """
    Simple dataset without target lables that transforms
    a fast{a|q} sequence to one hot representation and faciliates
    batch loading by a torch DataLoader.
    """
    def __init__(self, paths: list[str], # paths to seq files
                 format: str = 'fasta',  # fast{a|q}
                 gzipped: bool = False,
                 alphabet: Optional[str] = None): # used for one-hot encoding
        self.paths = paths
        self.format = format
        self.gzipped = gzipped
        self.alphabet = alphabet
        if alphabet:
            self.encoder = {c: i for i, c in enumerate(alphabet)}

    def one_hot(self, seq: str) -> np.ndarray:
        """
        one-hot encoder for single seq.
        """

        # convert seq to indices (unknown tokens = -1)
        seq_ind = np.array([self.encoder.get(c, -1) for c in seq.upper()])
        encoded = np.zeros((len(self.encoder), len(seq)))

        # uknown tokens will be all zeros
        cond = seq_ind >= 0
        encoded[seq_ind[cond], cond] = 1
        return  encoded
        
    def __iter__(self):
        """
        Use Biopython's fasta iterator. TODO use pyfastx instead
        """
        # 1. if length of list is 1 do normal stuff
        # 2. else make a chain of iterators
        if len(self.paths) < 1:
            raise ValueError('paths must be a list of 1 or more file paths')
        i = 0 # used to keep track of seq number an map to seq id
        for p in self.paths:
            handle = gzip.open(p, 'rt') if self.gzipped else open(p, 'r')
            S = SeqIO.parse(handle, format=self.format)
            for record in S:
                yield {'index': i, 'id': record.id, 'file': p,
                       'seq': self.one_hot(str(record.seq))
                       if self.alphabet else str(record.seq)}
                i += 1
def makeDataLoader(seqit: SeqIteratorDataset, batch_size: int=64, num_workers: int=64):
    """
    Simple factory for dataloader.
    """
    return DataLoader(seqit, shuffle=False,
                      batch_size=batch_size,
                      num_workers=num_workers,
                      collate_fn=lambda x: x)

def formatBatchMetadata(batch: list[dict]) -> list[str]:
    """
    return metadata of the batch as list of tab delimitted strings.
    """
    return ['\t'.join([str(b['index']), b['file'], b['id']])
            for b in batch]


class SeqDataset(Dataset):

    def __init__(self, seq_dir, align_dir, l):
        self.l = l
        self.seq_dict, self.seq_list, self.cnt = self.read_seq(seq_dir)
        self.M = self.load_distance_matrix(align_dir)

    def __getitem__(self, index):
        if self.l >= self.cnt:
            return self.seq_list, self.M
        else:
            chosen_seq_list, chosen_M = self.sample_seq()
            return chosen_seq_list, chosen_M

    def __len__(self):
        return self.l

    def read_seq(self, seq_dir):
        f_s = open(seq_dir, 'r')
        seq_dict = dict()
        seq_list = []
        cnt = 0
        while 1:
            ind = f_s.readline()[1:-1]
            if not ind:
                break
            seq = f_s.readline()[:-1]
            seq_dict[ind] = [seq, cnt]
            seq_list.append(seq)
            cnt += 1
        return seq_dict, seq_list, cnt


    def load_distance_matrix(self, align_dir):

        f_a = open(align_dir, 'r')
        M = np.zeros([self.cnt, self.cnt])
        while 1:
            name = f_a.readline()
            if not name:
                break
            ind_pair = name.split(' ')[0]
            ind_1 = self.seq_dict[ind_pair.split('-')[0]][1]
            ind_2 = self.seq_dict[ind_pair.split('-')[1]][1]
            dist = np.float64(name.split(' ')[-1])
            if dist == 0:
                print('here', name)
            M[ind_1, ind_2] = dist
            M[ind_2, ind_1] = dist
        return M

    def sample_seq(self):
        chosen_ind = sorted(random.sample(self.seq_dict.keys(), self.l))
        chosen_M = np.zeros([self.l, self.l])
        chosen_seq_list = []
        cnt_chosen = 0
        for ind in chosen_ind:
            chosen_seq_list.append(self.seq_dict[ind][0])
            cnt_chosen += 1

        for i in range(self.l - 1):
            for j in range(i + 1, self.l):
                ind1 = self.seq_dict[chosen_ind[i]][1]
                ind2 = self.seq_dict[chosen_ind[j]][1]
                chosen_M[i, j] = self.M[ind1, ind2]
                chosen_M[j, i] = self.M[ind1, ind2]

        return chosen_seq_list, chosen_M
