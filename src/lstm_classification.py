
import argparse
import datetime

import tensorflow as tf
import numpy as np

import datalayer as dl
import util.outpututil as ou


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

        self._hist_weight = tf.summary.histogram("weight", self._weight)
        self._hist_bias = tf.summary.histogram("bias", self._bias)

    def run_lstm(self, x, seq_len, max_seq_len):

        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [-1, self._n_features])
        x = tf.split(0, max_seq_len, x)

        # Define a lstm cell with tensorflow
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self._n_hidden, forget_bias=self._forget_bias)

        # Get lstm cell output
        outputs, states = tf.nn.rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seq_len)

        outputs = tf.pack(outputs)
        outputs = tf.transpose(outputs, [1, 0, 2])

        batch_size = tf.shape(outputs)[0]
        index = tf.range(0, batch_size) * max_seq_len + (seq_len - 1)
        outputs = tf.gather(tf.reshape(outputs, [-1, self._n_hidden]), index)

        return tf.matmul(outputs, self._weight) + self._bias


def classify_drivers(input_data_path, num_hidden, logdir):
    # start time
    start_time = datetime.datetime.now()

    # log output
    report = ou.ConsoleFileOutput(logdir + "/report.txt")

    # read data
    report.output("read data")
    data = dl.driverdata(input_data_path)

    # Parameters
    n_cross_validation = 4
    learning_rate = 0.001
    training_iters = 1000
    batch_size = 12
    display_step = 10

    # Network Parameters
    n_features = data.get_num_features()
    max_seq_len = data.get_max_seq_len()
    n_hidden = num_hidden # the number of hidden neurons in lstm
    n_classes = 2 # two classes: expert vs inexpert
    forget_bias = 1.0 # forget bias for lstm

    # tf Graph input
    x = tf.placeholder("float", [None, max_seq_len, n_features], name="x-input-data")
    y = tf.placeholder("float", [None, n_classes], name="y-output-label")
    seq_len = tf.placeholder(tf.int32, [None])

    report.output("create lstm nn: {} features, {} sequence length, {} hidden neurons, {} classes".format(n_features, max_seq_len, n_hidden, n_classes))
    cl = ClassificationLstm(n_features, n_hidden, n_classes)
    pred = cl.run_lstm(x, seq_len, max_seq_len)

    # Define loss
    with tf.name_scope("cost") as scope:
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
        tf.summary.scalar("cost", cost)

    # Minimize
    with tf.name_scope("train") as scope:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    with tf.name_scope("evaluate") as scope:
        correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        report.output("Cross Validation Start")
        merged = tf.summary.merge_all() # tf.merge_all_summaries()

        for i in range(n_cross_validation):
            logtensorboard = "{}/groupid{}".format(logdir, i)
            summary_writer = tf.summary.FileWriter(logtensorboard, sess.graph)
            #tf.summary.FileWriter(logtensorboard, sess.graph_def) #tf.train.SummaryWriter

            report.output("build train and test data")
            cross_validation_data = data.get_cross_validation_input(n_cross_validation, i)

            train_data = cross_validation_data["train"]["data"]
            train_label = cross_validation_data["train"]["label"]
            train_seq_len = cross_validation_data["train"]["seq_len"]

            test_data = cross_validation_data["test"]["data"]
            test_label = cross_validation_data["test"]["label"]
            test_seq_len = cross_validation_data["test"]["seq_len"]

            report.output("start learning")
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
                    logstr = "Iter {}".format(step*batch_size) + \
                            ", Minibatch Loss= {:.6f}".format(loss) + \
                            ", Training Accuracy= {:.5f}".format(acc)
                    report.output(logstr)
                step += 1

            report.output("Optimization Finished!")
            # test network with test data
            report.output("Testing Accuracy: {}".format( \
                    sess.run(accuracy, \
                    feed_dict={x: test_data, \
                    y: test_label, seq_len: test_seq_len})))
                    
    report.output("Cross Validation end")

    # end time
    end_time = datetime.datetime.now()
    time_diff = end_time - start_time
    hours, seconds = divmod(time_diff.seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    report.output("timestamp {}:{}:{}".format(hours, minutes, seconds))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_data_path", help="input data path")
    parser.add_argument("-n", "--num_hidden", help="the number of hidden neurons in LSTM", type=int, default=64)
    parser.add_argument("-l", "--logdir", help="log directory", default="./logs/default")

    args = parser.parse_args()
    input_data_path = args.input_data_path
    num_hidden = args.num_hidden
    logdir = args.logdir

    classify_drivers(input_data_path, num_hidden, logdir)

if __name__ == "__main__":
    main()

