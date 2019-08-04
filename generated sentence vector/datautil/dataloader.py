import numpy as np
import os
import torch


class Instance(object):
    def __init__(self, words=None, tag=None):
        self.words = words
        self.tag = tag


class Batch(object):
    def __init__(self, wd_src=None, tgt=None, non_pad_mask=None):
        self.wd_src = wd_src
        self.tgt = tgt
        self.non_pad_mask = non_pad_mask


def load_dataset(path):
    assert os.path.exists(path)

    insts = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as fin:
        for line in fin:
            tag, src_sent = line.strip().split('|||')
            tokens = src_sent.strip().split()
            insts.append(Instance(tokens, tag.strip()))
    return insts


def batch_iter(dataset, batch_size, wd_vocab, shuffle=True):
    if shuffle:
        np.random.shuffle(dataset)

    nb_batch = int(np.ceil(len(dataset) / batch_size))

    for i in range(nb_batch):
        batch_data = dataset[i*batch_size: (i+1)*batch_size]
        if shuffle:
            np.random.shuffle(batch_data)

        batch_size = len(batch_data)
        max_seq_len = max(len(seq) for seq in batch_data)
        wd_idxs = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        for i, wd_seq in enumerate(batch_data):
            seq_len = len(wd_seq)
            wd_idxs[i, :seq_len] = torch.tensor(wd_vocab.word2index(wd_seq))

        yield wd_idxs


def batch_iter(dataset, batch_size, wd_vocab, shuffle=True, device=torch.device('cpu')):
    if shuffle:
        np.random.shuffle(dataset)

    nb_batch = int(np.ceil(len(dataset) / batch_size))

    for i in range(nb_batch):
        batch_data = dataset[i*batch_size: (i+1)*batch_size]
        if shuffle:
            np.random.shuffle(batch_data)

        yield batch_gen(batch_data, wd_vocab, device)


def batch_gen(batch_data, wd_vocab, device=torch.device('cpu')):
    batch_size = len(batch_data)
    max_seq_len = max(len(inst.words) for inst in batch_data)

    wd_idxs = torch.zeros((batch_size, max_seq_len), dtype=torch.long, device=device)

    for i, inst in enumerate(batch_data):
        seq_len = len(inst.words)
        wd_idxs[i, :seq_len] = torch.tensor(wd_vocab.extword2index(inst.words))

    non_pad_mask = wd_idxs.ne(0)

    return Batch(wd_idxs, non_pad_mask=non_pad_mask)





