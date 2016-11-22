from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import os
import pickle
import tensorflow as tf

from data import *
from models import *
from args import *

args, dir_name = parse_args()
os.makedirs(dir_name)

# DATA
d = bern_emb_data(args.cs, args.ns, args.mb, args.L)

# MODEL
m = bern_emb_model(d, args.K)

# MAP INFERENCE
inference = ed.MAP({}, data = m.data)

sess = ed.get_session()
inference.initialize()
tf.initialize_all_variables().run()

# TRAINING
train_loss = np.zeros(args.n_iter)

for i in range(args.n_iter):
    for ii in range(args.n_epochs):
        sess.run([inference.train], feed_dict=d.feed(m.placeholders))
    _, train_loss[i] = sess.run([inference.train, inference.loss], feed_dict=d.feed(m.placeholders))
    print("iteration {:d}/{:d}, train loss: {:0.3f}\n".format(i, args.n_iter, train_loss[i])) 

print('training finished. Results are saved in '+dir_name)
m.dump(dir_name+"/variational.dat")
m.plot_params(dir_name, d.labels[:500])
