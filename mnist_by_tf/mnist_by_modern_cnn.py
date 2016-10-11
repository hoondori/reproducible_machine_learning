"""Simple tutorial following the TensorFlow example of a Convolutional Network.
Parag K. Mital, Jan. 2016"""
# %% imports
import numpy as np
import tensorflow as tf
import matplotlib
import utils
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from libs import batch_norm
from libs import lrelu
from libs import conv2d
from libs import linear


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '../data/mnist/', 'Directory for storing data')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

# %% we can visualize any one of the images by reshaping it to a 28x28 image
#plt.imshow(np.reshape(mnist.train.images[100, :], (28, 28)), cmap='gray')
#plt.waitforbuttonpress()

# modeling
# 1@28x28 ->
# conv by [5x5,1to16] -> 16@14x14
# conv by [5x5,16to16] -> 16@7x7
# fc by [7*7*16 x 1024] -> 1024
# fc by [1024,10] -> 10
# softmax
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

# %% We add a new type of placeholder to denote when we are training.
# This will be used to change the way we compute the network during
# training/testing.
is_training = tf.placeholder(tf.bool, name='is_training')

# %% We'll convert our MNIST vector data to a 4-D tensor:
# N x W x H x C
x_tensor = tf.reshape(x,[-1,28,28,1])

# %% We'll use a new method called  batch normalization.
# This process attempts to "reduce internal covariate shift"
# which is a fancy way of saying that it will normalize updates for each
# batch using a smoothed version of the batch mean and variance
# The original paper proposes using this before any nonlinearities
h_1 = lrelu(batch_norm(conv2d(x_tensor, 32, name='conv1'),
                       is_training, scope='bn1'), name='lrelu1')
h_2 = lrelu(batch_norm(conv2d(h_1, 64, name='conv2'),
                       is_training, scope='bn2'), name='lrelu2')
h_3 = lrelu(batch_norm(conv2d(h_2, 64, name='conv3'),
                       is_training, scope='bn3'), name='lrelu3')
h_3_flat = tf.reshape(h_3, [-1,64*4*4])
h_4 = linear(h_3_flat, 10)
y_pred = tf.nn.softmax(h_4)

# %% Define loss/eval/training functions
cross_entropy = -tf.reduce_sum(y * tf.log(y_pred))
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# %% We now create a new session to actually perform the initialization the
# variables:
sess = tf.Session()
sess.run(tf.initialize_all_variables())

# %% We'll train in minibatches and report accuracy:
n_epochs = 10
batch_size = 100
for epoch_i in range(n_epochs):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={
        x: batch_xs, y: batch_ys, is_training: True})
    print(sess.run(accuracy,
               feed_dict={
                   x: mnist.validation.images,
                   y: mnist.validation.labels,
                   is_training: False
               }))

# final error
print(sess.run(accuracy,{x: mnist.test.images, y: mnist.test.labels, is_training: False}))

