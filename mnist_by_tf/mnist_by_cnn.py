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

# %% Since x is currently [batch, height*width], we need to reshape to a
# 4-D tensor to use it in a convolutional graph.  If one component of
# `shape` is the special value -1, the size of that dimension is
# computed so that the total size remains constant.  Since we haven't
# defined the batch dimension's shape yet, we use -1 to denote this
# dimension should not change size.
x_tensor = tf.reshape(x, [-1, 28, 28, 1]) # batchSize,height,width,depth

# %% We'll setup the first convolutional layer
# Weight matrix is [height x width x input_channels x output_channels]
W_conv1 = tf.Variable(tf.random_normal([5,5,1,16], mean=0.0, stddev=0.01))
b_conv1 = tf.Variable(tf.random_normal([16], mean=0.0, stddev=0.01)) # dim of output channel

# %% Now we can build a graph which does the first layer of convolution:
# we define our stride as batch x height x width x channels
# instead of pooling, we use strides of 2 and more layers
# with smaller filters.
h_conv1 = tf.nn.relu(tf.nn.conv2d(input=x_tensor,filter=W_conv1,strides=[1,2,2,1],padding='SAME') + b_conv1)

# %% And just like the first layer, add additional layers to create
# a deep net
W_conv2 = tf.Variable(tf.random_normal([5,5,16,16], mean=0.0, stddev=0.01))
b_conv2 = tf.Variable(tf.random_normal([16], mean=0.0, stddev=0.01)) # dim of output channel
h_conv2 = tf.nn.relu(tf.nn.conv2d(input=h_conv1,filter=W_conv2,strides=[1,2,2,1],padding='SAME') + b_conv2)

# %% We'll now reshape so we can connect to a fully-connected layer:
h_conv2_flat = tf.reshape(h_conv2, [-1, 7 * 7 * 16])

# %% Create a fully-connected layer:
W_fc1 = tf.Variable(tf.random_normal([7*7*16,1024], mean=0.0, stddev=0.01))
b_fc1 = tf.Variable(tf.random_normal([1024], mean=0.0, stddev=0.01))
h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat,W_fc1)+b_fc1)

# %% We can add dropout for regularizing and to reduce overfitting like so:
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

# %% And finally our softmax layer:
W_fc2 = tf.Variable(tf.random_normal([1024, 10], mean=0.0, stddev=0.01))
b_fc2 = tf.Variable(tf.random_normal([10], mean=0.0, stddev=0.01))
y_pred = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

# %% Define loss/eval/training functions
cross_entropy = -tf.reduce_sum(y*tf.log(y_pred))
optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)

# %% Monitor accuracy
correct_prediction = tf.equal( tf.argmax(y_pred,1), tf.argmax(y,1) )
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

# Train
n_epoch = 5
n_batch = 100
sess = tf.Session()
sess.run(tf.initialize_all_variables())
for i in range(n_epoch):
    batch_xs, batch_ys = mnist.train.next_batch(n_batch)
    sess.run(optimizer,{x: batch_xs, y: batch_ys, keep_prob: 0.5})

    # validation error
    print(sess.run(accuracy,{x: mnist.validation.images, y: mnist.validation.labels, keep_prob: 1.0}))

# final error
print(sess.run(accuracy,{x: mnist.test.images, y: mnist.test.labels,  keep_prob: 1.0}))

# %% Let's take a look at the kernels we've learned
W = sess.run(W_conv1)
plt.imshow(utils.montage(W / np.max(W)), cmap='coolwarm')
plt.waitforbuttonpress()