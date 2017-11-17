import numpy as np
import os
import pickle
import tensorflow as tf

from tensorflow.contrib.distributions import Normal, Bernoulli
from tensorflow.contrib.tensorboard.plugins import projector
from sklearn.manifold import TSNE
from utils import *

class bern_emb_model():
    def __init__(self, d, K, sig, sess, logdir):
        self.K = K
        self.sig = sig
        self.sess = sess
        self.logdir = logdir

        with tf.name_scope('model'):
            # Data Placeholder
            with tf.name_scope('input'):
                self.placeholders = tf.placeholder(tf.int32)
                self.words = self.placeholders
            

            # Index Masks
            with tf.name_scope('context_mask'):
                self.p_mask = tf.cast(tf.range(d.cs/2, d.n_minibatch + d.cs/2),tf.int32)
                rows = tf.cast(tf.tile(tf.expand_dims(tf.range(0, d.cs/2),[0]), [d.n_minibatch, 1]),tf.int32)
                columns = tf.cast(tf.tile(tf.expand_dims(tf.range(0, d.n_minibatch), [1]), [1, d.cs/2]),tf.int32)
                self.ctx_mask = tf.concat([rows+columns, rows+columns +d.cs/2+1], 1)

            with tf.name_scope('embeddings'):
                # Embedding vectors
                self.rho = tf.Variable(tf.random_normal([d.L, self.K])/self.K, name='rho')

                # Context vectors
                self.alpha = tf.Variable(tf.random_normal([d.L, self.K])/self.K, name='alpha')


                with tf.name_scope('priors'):
                    prior = Normal(loc = 0.0, scale = self.sig)
                    self.log_prior = tf.reduce_sum(prior.log_prob(self.rho) + prior.log_prob(self.alpha))

            with tf.name_scope('natural_param'):
                # Taget and Context Indices
                with tf.name_scope('target_word'):
                    self.p_idx = tf.gather(self.words, self.p_mask)
                    self.p_rho = tf.squeeze(tf.gather(self.rho, self.p_idx))
                
                # Negative samples
                with tf.name_scope('negative_samples'):
                    unigram_logits = tf.tile(tf.expand_dims(tf.log(tf.constant(d.unigram)), [0]), [d.n_minibatch, 1])
                    self.n_idx = tf.multinomial(unigram_logits, d.ns)
                    self.n_rho = tf.gather(self.rho, self.n_idx)

                with tf.name_scope('context'):
                    self.ctx_idx = tf.squeeze(tf.gather(self.words, self.ctx_mask))
                    self.ctx_alphas = tf.gather(self.alpha, self.ctx_idx)


                # Natural parameter
                ctx_sum = tf.reduce_sum(self.ctx_alphas,[1])
                self.p_eta = tf.expand_dims(tf.reduce_sum(tf.multiply(self.p_rho, ctx_sum),-1),1)
                self.n_eta = tf.reduce_sum(tf.multiply(self.n_rho, tf.tile(tf.expand_dims(ctx_sum,1),[1, d.ns, 1])),-1)
            
            # Conditional likelihood
            self.y_pos = Bernoulli(logits = self.p_eta)
            self.y_neg = Bernoulli(logits = self.n_eta)

            self.ll_pos = tf.reduce_sum(self.y_pos.log_prob(1.0)) 
            self.ll_neg = tf.reduce_sum(self.y_neg.log_prob(0.0))

            self.log_likelihood = self.ll_pos + self.ll_neg
            
            scale = 1.0*d.N/d.n_minibatch
            self.loss = - (scale * self.log_likelihood + self.log_prior)

            # Training
            optimizer = tf.train.AdamOptimizer()
            self.train = optimizer.minimize(self.loss)
            with self.sess.as_default():
                tf.global_variables_initializer().run()
            variable_summaries('rho', self.rho)
            variable_summaries('alpha', self.alpha)
            with tf.name_scope('objective'):
                tf.summary.scalar('loss', self.loss)
                tf.summary.scalar('priors', self.log_prior)
                tf.summary.scalar('ll_pos', self.ll_pos)
                tf.summary.scalar('ll_neg', self.ll_neg)
            self.summaries = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(self.logdir, self.sess.graph)
            self.saver = tf.train.Saver()
            config = projector.ProjectorConfig()

            alpha = config.embeddings.add()
            alpha.tensor_name = 'model/embeddings/alpha'
            alpha.metadata_path = '../vocab.tsv'
            rho = config.embeddings.add()
            rho.tensor_name = 'model/embeddings/rho'
            rho.metadata_path = '../vocab.tsv'
            projector.visualize_embeddings(self.train_writer, config)

        
    def dump(self, fname):
        with self.sess.as_default():
            dat = {'rho':  self.rho.eval(),
                   'alpha':  self.alpha.eval()}
        pickle.dump( dat, open( fname, "a+" ) )

    def plot_params(self, dir_name, labels):
        plot_only = len(labels)

        with self.sess.as_default():
            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            low_dim_embs_alpha2 = tsne.fit_transform(self.alpha.eval()[:plot_only])
            plot_with_labels(low_dim_embs_alpha2[:plot_only], labels[:plot_only], dir_name + '/alpha.eps')

            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            low_dim_embs_rho2 = tsne.fit_transform(self.rho.eval()[:plot_only])
            plot_with_labels(low_dim_embs_rho2[:plot_only], labels[:plot_only], dir_name + '/rho.eps')





