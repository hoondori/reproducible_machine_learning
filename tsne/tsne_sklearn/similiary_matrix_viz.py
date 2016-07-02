# This is not my creation
# It is simply the mimic of the following blog post from oreilly
# https://www.oreilly.com/learning/an-illustrated-introduction-to-the-t-sne-algorithm

import numpy as np
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist

import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits # MNIST
from sklearn.preprocessing import scale

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities,
                                    _kl_divergence)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
sns.set_palette("muted")
sns.set_context("notebook",font_scale=1.5, rc={"lines.linewidth": 2.5})

# load data
digits = load_digits()

# reorder data points according to handwritten numbers
X = np.vstack([digits.data[digits.target==i]
               for i in range(10)])
y = np.hstack([digits.target[digits.target==i]
               for i in range(10)])


def _joint_probabilities_constant_sigma(D, sigma):
    P = np.exp(-D**2/2 * sigma**2)
    P /= np.sum(P, axis=1)
    return P

# pairwise distances between all data points
D = pairwise_distances(X,squared=True)

# Similarity with constant sigma
P_constant = _joint_probabilities_constant_sigma(D, .002)

# Similarity with variable sigma
P_binary = _joint_probabilities(D, 30., False)

# output of this function needs to be reshaped to a square matrix
P_binary_s = squareform(P_binary)

# plot this similarity matrix
plt.figure(figsize=(12,4))
pal = sns.light_palette("blue",as_cmap=True)

plt.subplot(131)
plt.imshow(D[::10, ::10], interpolation='none',cmap=pal)
plt.axis('off')
plt.title("Distance matrix", fontdict={'fontsize': 16})

plt.subplot(132)
plt.imshow(P_constant[::10,::10],interpolation='none',cmap=pal)
plt.axis('off')
plt.title("$p_{j|i}$ (constant $\sigma$)",fontdict={'fontsize': 16})

plt.subplot(133)
plt.imshow(P_binary_s[::10,::10],interpolation='none',cmap=pal)
plt.axis('off')
plt.title("$p_{j|i}$ (variable $\sigma$)",fontdict={'fontsize': 16})

plt.savefig('images/similarity-generated.png', dpi=120)