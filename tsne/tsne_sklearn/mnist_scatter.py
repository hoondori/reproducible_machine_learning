
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

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

import seaborn as sns
sns.set_style("darkgrid")
sns.set_palette("muted")
sns.set_context("notebook",font_scale=1.5, rc={"lines.linewidth": 2.5})

# load data
digits = load_digits()
#print digits.data.shape
print(digits['DESCR'])

# visualize original mnist data
nrows, ncols = 2, 5
plt.figure(figsize=(6,3))
plt.gray()
for i in range(nrows*ncols):
    ax = plt.subplot(nrows, ncols, i+1)
    ax.matshow(digits.images[i,...])
    plt.xticks([]); plt.yticks([])
    plt.title(digits.target[i])
plt.savefig('images/digis-generated.png',dpi=150)

# reorder data points according to handwritten numbers
X = np.vstack([digits.data[digits.target==i]
               for i in range(10)])
y = np.hstack([digits.target[digits.target==i]
               for i in range(10)])

# run t-sne algorithm on the dataset
RS = 20160701 # Random state
digits_proj = TSNE(random_state=RS).fit_transform(X)

# utility function to display transformed data
def scatter(x,colors):

    # choose color from palette
    palette = np.array(sns.color_palette("hls",10))

    # create a scatter plot
    f = plt.figure(figsize=(8,8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0],x[:,1],lw=0,s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25,25)
    plt.ylim(-25,25)
    ax.axis('off')
    ax.axis('tight')

    # add labels for each digit
    txts = []
    for i in range(10):
        # position of each label
        xtext, ytext = np.median(x[colors==i,:],axis=0)
        txt = ax.text(xtext,ytext,str(i),fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5,foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    return f, ax, sc, txts

# her is the result
scatter(digits_proj,y)
plt.savefig('images/digits_tsne-generated.png',dpi=120)