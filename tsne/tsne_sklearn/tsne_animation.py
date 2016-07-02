# This is not my creation
# It is simply the mimic of the following blog post from oreilly
# https://www.oreilly.com/learning/an-illustrated-introduction-to-the-t-sne-algorithm

import numpy as np
from numpy import linalg

# We import sklearn.
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities,
                                    _kl_divergence)
from sklearn.utils.extmath import _ravel

# We'll generate an animation with matplotlib and moviepy.
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
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

# This list will contain the positions of the map points at every iteration of gradient descent
positions = []
def _gradient_descent(objective, p0, it, n_iter, n_iter_without_progress=30,
                      momentum=0.5, learning_rate=1000.0, min_gain=0.01,
                      min_grad_norm=1e-7, min_error_diff=1e-7, verbose=0,
                      args=[]):
    p = p0.copy().ravel()
    update = np.zeros_like(p)
    gains = np.ones_like(p)
    error = np.finfo(np.float).max
    best_error = np.finfo(np.float).max
    best_iter = 0

    for i in range(it, n_iter):
        # we save the current position
        positions.append(p.copy())
        new_error, grad = objective(p, *args)
        error_diff = np.abs(new_error - error)
        error = new_error
        grad_norm = linalg.norm(grad)

        if error < best_error:
            best_error = error
            best_iter = i
        elif i - best_iter > n_iter_without_progress:
            break
        if min_grad_norm >= grad_norm:
            break
        if min_error_diff >= error_diff:
            break

        inc = update * grad >= 0.0
        dec = np.invert(inc)
        gains[inc] += 0.05
        gains[dec] *= 0.95
        np.clip(gains,min_gain,np.inf)
        grad *= gains
        update = momentum * update - learning_rate * grad
        p += update
    return p, error, i

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

# replace original gradient descent with custojmized one
sklearn.manifold.t_sne._gradient_descent = _gradient_descent

# run t-sne algorithm on the dataset
RS = 20160701 # Random state
X_proj = TSNE(random_state=RS).fit_transform(X)
X_iter = np.dstack(position.reshape(-1,2)
                   for position in positions)

# make movie
f, ax, sc, txts = scatter(X_iter[...,-1],y)

def make_frame_mpl(t):
    i = int(t*40)
    x = X_iter[...,i]
    sc.set_offsets(x)
    for j,txt in zip(range(10),txts):
        xtext,ytext = np.median(x[y==j, :], axis =0 )
        txt.set_x(xtext)
        txt.set_y(ytext)
    return mplfig_to_npimage(f)

animation = mpy.VideoClip(make_frame_mpl,duration=X_iter.shape[2]/40.)
animation.write_gif("images/digits_tsne-generated.gif", fps=20)


