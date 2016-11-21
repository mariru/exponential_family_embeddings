import edward as ed
import numpy as np
import os
import pickle
import tensorflow as tf

from edward.models import Normal, Bernoulli
from sklearn.manifold import TSNE
from utils import *

class bern_emb_model():
    def __init__(self, d, K):
        self.K = K

        # Data Placeholder
        self.words = tf.placeholder(tf.int32, shape = (d.n_minibatch + d.cs))
        self.placeholders = self.words

        # Index Masks
        self.p_mask = tf.range(d.cs/2,d.n_minibatch + d.cs/2)
        rows = tf.tile(tf.expand_dims(tf.range(0, d.cs/2),[0]), [d.n_minibatch, 1])
        columns = tf.tile(tf.expand_dims(tf.range(0, d.n_minibatch), [1]), [1, d.cs/2])
        self.ctx_mask = tf.concat(1,[rows+columns, rows+columns +d.cs/2+1])

        # Embedding vectors
        self.rho = tf.Variable(tf.random_normal([d.L, self.K])/self.K)

        # Context vectors
        self.alpha = tf.Variable(tf.random_normal([d.L, self.K])/self.K)

        # Taget words 
        self.p_idx = tf.gather(self.words, self.p_mask)
        self.p_rho = tf.squeeze(tf.gather(self.rho, self.p_idx))
        
        # Negative samples
        unigram_logits = tf.tile(tf.expand_dims(tf.log(tf.constant(d.unigram)), [0]), [d.n_minibatch, 1])
        self.n_idx = tf.multinomial(unigram_logits, d.ns)
        self.n_rho = tf.gather(self.rho, self.n_idx)

        # Context
        self.ctx_idx = tf.squeeze(tf.gather(self.words, self.ctx_mask))
        self.ctx_alphas = tf.gather(self.alpha, self.ctx_idx)
        ctx_sum = tf.reduce_sum(self.ctx_alphas,[1])

        # Natural parameter
        p_eta = tf.expand_dims(tf.reduce_sum(tf.mul(self.p_rho, ctx_sum),-1),1)
        n_eta = tf.reduce_sum(tf.mul(self.n_rho, tf.tile(tf.expand_dims(ctx_sum,1),[1,d.ns,1])),-1)
        
        # Conditional likelihood
        self.y_pos = Bernoulli(logits = p_eta)
        self.y_neg = Bernoulli(logits = n_eta)

        # Hallucinated data
        self.data = {self.y_pos: tf.ones((d.n_minibatch, 1)), self.y_neg: tf.zeros((d.n_minibatch, d.ns))}


    def dump(self, fname):
            dat = {'rho':  self.rho.eval(),
                   'alpha':  self.alpha.eval()}
            pickle.dump( dat, open( fname, "a+" ) )

    def plot_params(self, dir_name, labels):
        plot_only = len(labels)

	tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        low_dim_embs_alpha2 = tsne.fit_transform(self.alpha.eval()[:plot_only])
        plot_with_labels(low_dim_embs_alpha2[:plot_only], labels[:plot_only], dir_name + '/alpha.eps')

        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        low_dim_embs_rho2 = tsne.fit_transform(self.rho.eval()[:plot_only])
        plot_with_labels(low_dim_embs_rho2[:plot_only], labels[:plot_only], dir_name + '/rho.eps')





