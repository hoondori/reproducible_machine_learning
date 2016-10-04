"""Simple tutorial for using TensorFlow to compute a linear regression.
Parag K. Mital, Jan. 2016"""
# %% imports
import numpy as np
import tensorflow as tf
import matplotlib
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

# setup input and output dimension
n_input = 784
n_output = 10

# model : softmax(Wx+b)
x = tf.placeholder(tf.float32,[None,n_input])
y = tf.placeholder(tf.float32,[None,10])

W = tf.Variable(tf.zeros([n_input, n_output]))
b = tf.Variable(tf.zeros([n_output]))
y_ = tf.nn.softmax(tf.matmul(x,W)+b)


# Define loss and optimizer
cross_entropy = -tf.reduce_sum( y * tf.log(y_) )
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# accuracy : top1 in y == top1 in y_
correct_prediction = tf.equal( tf.argmax(y_,1), tf.argmax(y,1) )
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))


# Train
n_epoch = 10
n_batch = 100
sess = tf.Session()
sess.run(tf.initialize_all_variables())
for i in range(n_epoch):
    batch_xs, batch_ys = mnist.train.next_batch(n_batch)
    sess.run(optimizer,{x: batch_xs, y: batch_ys})

    # validation error
    print(sess.run(accuracy,{x: mnist.validation.images, y: mnist.validation.labels}))

# final error
print(sess.run(accuracy,{x: mnist.test.images, y: mnist.test.labels}))



