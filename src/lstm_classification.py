
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np

import datalayer as dl


class ClassificationLstm:
    def __init__(self, n_features, n_hidden, n_classes, forget_bias=1.0):
        """lstm neural network for classification
        Args:
            n_hidden (int): the number of neuron in lstm
            n_classes (int): the number of classes
            forget_bias (float): forget bias for lstm. default value is 1.0
        """
        self._n_features = n_features
        self._n_hidden = n_hidden
        self._n_classes = n_classes
        self._forget_bias = forget_bias
        self._weight = tf.Variable(tf.random_normal([n_hidden, n_classes]), name="classification_lstm_weight")
        self._bias = tf.Variable(tf.random_normal([n_classes]), name="classification_lstm_bias")

    def run_lstm(self, x, seq_len, max_seq_len):

        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [-1, self._n_features])
        x = tf.split(0, max_seq_len, x)

        # Define a lstm cell with tensorflow
        lstm_cell = rnn_cell.BasicLSTMCell(self._n_hidden, forget_bias=self._forget_bias)

        # Get lstm cell output
        outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seq_len)

        outputs = tf.pack(outputs)
        outputs = tf.transpose(outputs, [1, 0, 2])

        batch_size = tf.shape(outputs)[0]
        index = tf.range(0, batch_size) * max_seq_len + (seq_len - 1)
        outputs = tf.gather(tf.reshape(outputs, [-1, self._n_hidden]), index)

        return tf.matmul(outputs, self._weight) + self._bias


# read data
print("read data")
data_path = "../data"
data = dl.driverdata(data_path)

# Parameters
n_cross_validation = 4
learning_rate = 0.001
training_iters = 1000
batch_size = 12
display_step = 10

# Network Parameters
n_features = data.get_num_features()
max_seq_len = data.get_max_seq_len()
n_hidden = 128 # the number of hidden neurons in lstm
n_classes = 2 # two classes: expert vs inexpert
forget_bias = 1.0 # forget bias for lstm

# tf Graph input
x = tf.placeholder("float", [None, max_seq_len, n_features], name="x-input-data")
y = tf.placeholder("float", [None, n_classes], name="y-output-label")
seq_len = tf.placeholder(tf.int32, [None])

print("create lstm nn: {} features, {} sequence length, {} hidden neurons, {} classes".format(n_features, max_seq_len, n_hidden, n_classes))
cl = ClassificationLstm(n_features, n_hidden, n_classes)
pred = cl.run_lstm(x, seq_len, max_seq_len)

# Define loss
with tf.name_scope("cost") as scope:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    tf.scalar_summary("cost", cost)

# Minimize
with tf.name_scope("train") as scope:
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
with tf.name_scope("evaluate") as scope:
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.scalar_summary("accuracy", accuracy)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    merged = tf.merge_all_summaries()

    for i in range(n_cross_validation):
        logfilename = "./logs/%d_hidden/group%d" % (n_hidden, i)
        summary_writer = tf.train.SummaryWriter(logfilename, sess.graph_def)

        print("build train and test data")
        cross_validation_data = data.get_cross_validation_input(n_cross_validation, i)

        train_data = cross_validation_data["train"]["data"]
        train_label = cross_validation_data["train"]["label"]
        train_seq_len = cross_validation_data["train"]["seq_len"]

        test_data = cross_validation_data["test"]["data"]
        test_label = cross_validation_data["test"]["label"]
        test_seq_len = cross_validation_data["test"]["seq_len"]

        print("start learning")
        sess.run(init)

        # Keep training until reach max iterations
        step = 1
        while step * batch_size < training_iters:
            # Run optimization op (backprop)
            sess.run(optimizer, \
                    feed_dict={x: train_data, y: train_label, seq_len: train_seq_len})
            if step % display_step == 0:
                # Calculate batch accuracy and loss
                summary, loss, acc = \
                        sess.run([merged, cost, accuracy], \
                        feed_dict={x: train_data, y: train_label, seq_len: train_seq_len})
                summary_writer.add_summary(summary, step*batch_size)
                print("Iter " + str(step*batch_size) + \
                        ", Minibatch Loss= " + "{:.6f}".format(loss) + \
                        ", Training Accuracy= " + "{:.5f}".format(acc))
            step += 1

        print("Optimization Finished!")
        # test network with test data
        print("Testing Accuracy:", \
                sess.run(accuracy, \
                feed_dict={x: test_data, y: test_label, seq_len: test_seq_len}))

