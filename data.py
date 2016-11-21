import numpy as np
import os
import pandas as pd
import pickle
from utils import *

class bern_emb_data():
    def __init__(self, cs, ns, n_minibatch, L):
        assert cs%2 == 0
        self.cs = cs
        self.ns = ns
        self.n_minibatch = n_minibatch
        self.L = L
        url = 'http://mattmahoney.net/dc/'
        filename = maybe_download(url, 'text8.zip', 31344016)
        words = read_data(filename)
        self.build_dataset(words)
        self.batch = self.batch_generator()

    def build_dataset(self, words):
        count = [['UNK', -1]]
        count.extend(collections.Counter(words).most_common(self.L - 1))
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = list()
        unk_count = 0
        for word in words:
            if word in dictionary:
              index = dictionary[word]
            else:
              index = 0 
              unk_count += 1
            data.append(index)
        count[0][1] = unk_count
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        self.data = np.array(data)
        self.count = count
        self.dictionary = dictionary
        self.labels = [reverse_dictionary[x] for x in range(len(reverse_dictionary))]
        unigram_dist = np.array([1.0*i for ii, i in count])
        unigram_dist = (unigram_dist/unigram_dist.sum())**(3.0/4)
        self.unigram = unigram_dist/unigram_dist.sum()


    def batch_generator(self):
        batch_size = self.n_minibatch + self.cs
        data = self.data
        while True:
            if data.shape[0] < batch_size:
                data = np.hstack([data, self.data])
                if data.shape[0] < batch_size:
                    continue
            words = data[:batch_size]
            data = data[batch_size:]
            yield words
    
    def feed(self, placeholder):
        return {placeholder: self.batch.next()}
