
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np

import datalayer as dl

import pdb

# read data
print("read data")
data_path = "../data"
data = dl.driverdata(data_path) # data.input_data, data.get_cross_validation_input(num_groups, group)

# print("length " + str(len(data.input_data)))
# print("length " + str(len(data.input_data["expert"])))
# print("length " + str(len(data.input_data["inexpert"])))


# Parameters
n_cross_validation = 4
learning_rate = 0.001
training_iters = 1000
batch_size = 12
display_step = 10

# Network Parameters
n_input = 23 # the number of filtered column # 101 # the number of column
n_steps = 100 # 2629 # timesteps # 500 #
n_hidden = 128 # 128 # hidden layer num of features
n_classes = 2 # MNIST total classes (0-9 digits)


def cutData(data, num_row):
    features = []
    label = []
    for d in data:
        # features.append(np.array(d["features"])[:n_steps,:])
        features.append(d["features"][:n_steps,:])
        label.append(d["label"])
    return [np.array(features), np.array(label)]

def createTrainTestData(cross_validation_data):
    train_features, train_label = cutData(cross_validation_data["train"], n_steps)
    test_features, test_label = cutData(cross_validation_data["test"], n_steps)
    return [train_features, train_label, test_features, test_label]



# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input], name="x-input-data")
y = tf.placeholder("float", [None, n_classes], name="y-output-label")

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]), name="output_weight")
}

biases = {
    'out': tf.Variable(tf.random_normal([n_classes]), name="output_bias")
}

tf.histogram_summary("output_weight", weights["out"])
tf.histogram_summary("output_bias", biases["out"])


print("create lstm nn")
def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


pred = RNN(x, weights, biases)

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
        logfilename = "./logs/%d_rows/%d_hidden/group%d" % (n_steps, n_hidden, i)
        summary_writer = tf.train.SummaryWriter(logfilename, sess.graph_def)

        print("build train and test data")
        train_data, train_label, test_data, test_label = createTrainTestData(data.get_cross_validation_input(n_cross_validation, i))

        print("start learning")
        sess.run(init)

        # Keep training until reach max iterations
        step = 1
        while step * batch_size < training_iters:
            # print("step " + str(step))
            batch_x = train_data
            batch_y = train_label

            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            if step % display_step == 0:
                # Calculate batch accuracy and loss
                summary, loss, acc = \
                        sess.run([merged, cost, accuracy], feed_dict={x: batch_x, y: batch_y})
                summary_writer.add_summary(summary, step*batch_size)
                print("Iter " + str(step*batch_size) + \
                        ", Minibatch Loss= " + "{:.6f}".format(loss) + \
                        ", Training Accuracy= " + "{:.5f}".format(acc))
            step += 1
        print("Optimization Finished!")

        # Calculate accuracy for 128 mnist test images
        print("Testing Accuracy:", \
                sess.run(accuracy, feed_dict={x: test_data, y: test_label}))

