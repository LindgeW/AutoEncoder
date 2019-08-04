import numpy as np
import os
from datautil.dataloader import load_dataset
from collections import Counter


def create_vocab(path):
    wd_counter = Counter()
    # tag_counter = Counter()
    dataset = load_dataset(path)
    for inst in dataset:
        wd_counter.update(inst.words)
        # tag_counter[inst.tag] += 1
    return WordVocab(wd_counter)


class WordVocab(object):
    def __init__(self, wd_counter, min_count=5):
        super(WordVocab, self).__init__()
        self.PAD = 0
        self.UNK = 1

        self._wd2freq = dict((wd, freq) for wd, freq in wd_counter.items() if freq > min_count)

        self._wd2idx = dict((wd, idx+2) for idx, wd in enumerate(self._wd2freq.keys()))
        self._wd2idx['<pad>'] = self.PAD
        self._wd2idx['<unk>'] = self.UNK
        self._idx2wd = dict((idx, wd) for wd, idx in self._wd2idx.items())

        self._extwd2idx = dict()
        self._extidx2wd = dict()

    def get_embedding_weights(self, embed_path):
        assert os.path.exists(embed_path)
        vec_tab = dict()

        vec_size = 0
        with open(embed_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                tokens = line.strip().split()
                wd, vec = tokens[0], tokens[1:]
                if vec_size == 0:
                    vec_size = len(vec)
                vec_tab[wd] = np.asarray(vec, dtype=np.float32)

        oov = 0
        for wd in self._wd2idx.keys():
            if wd not in vec_tab:
                oov += 1
        oov_ratio = 100 * (oov - 2) / (len(self._wd2idx) - 2)
        print('oov ratio:', oov_ratio)

        self._extwd2idx = dict((wd, idx+2) for idx, wd in enumerate(vec_tab.keys()))
        self._extwd2idx['<pad>'] = self.PAD
        self._extwd2idx['<unk>'] = self.UNK
        self._extidx2wd = dict((idx, wd) for wd, idx in self._extwd2idx.items())
        vocab_size = len(self._extwd2idx)

        embedding_weights = np.zeros((vocab_size, vec_size), dtype=np.float32)
        for idx, wd in self._extidx2wd.items():
            if idx != self.PAD and idx != self.UNK:
                embedding_weights[idx] = vec_tab[wd]

        embedding_weights[self.UNK] = np.mean(embedding_weights, 0) / np.std(embedding_weights)

        return embedding_weights

    def word2index(self, wds):
        if isinstance(wds, list):
            return [self._wd2idx.get(w, self.UNK) for w in wds]
        else:
            return self._wd2idx.get(wds, self.UNK)

    def index2word(self, idxs):
        if isinstance(idxs, list):
            return [self._idx2wd.get(i) for i in idxs]
        else:
            return self._idx2wd.get(idxs)

    def extword2index(self, wds):
        if isinstance(wds, list):
            return [self._extwd2idx.get(w, self.UNK) for w in wds]
        else:
            return self._extwd2idx.get(wds, self.UNK)

    def extindex2word(self, idxs):
        if isinstance(idxs, list):
            return [self._extidx2wd.get(i) for i in idxs]
        else:
            return self._extidx2wd.get(idxs)

    @property
    def vocab_size(self):
        return len(self._wd2idx)

    @property
    def extvocab_size(self):
        return len(self._extwd2idx)
