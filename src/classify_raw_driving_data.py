import argparse
import datetime

import tensorflow as tf
import numpy as np

import datalayer as dl
import util.outpututil as ou

from solution_layer import *


def classify_drivers(input_data_path, num_hidden, logdir):
    # start time
    start_time = datetime.datetime.now()

    # log output
    report = ou.ConsoleFileOutput(logdir + "/report.txt")

    # read data
    report.output("read data")
    data = dl.driverdata(input_data_path)
    n_features = data.get_num_features()
    max_seq_len = data.get_max_seq_len()

    # Parameters
    n_cross_validation = 4
    learning_rate = 0.001
    ae_training_iters = 500
    training_iters = 1000
    batch_size = 12
    ae_display_step = 100
    display_step = 10


    ## Autoencoder

    # Autoencoder Parameters
    n_final_features = 25
    num_neurons_in_layer = [data.get_num_features(), 75, 50, n_final_features]
    # num_neurons_in_layer = [n_features, n_final_features]

    # Autoencoder Network
    report.output("create multi-layer autoencoder: {}".format(num_neurons_in_layer))
    x_raw = tf.placeholder("float", [max_seq_len, n_features], name="x-raw-input-data")
    final_encoded_x = x_raw
    saes = list()
    encoders = list()
    ae_costs = list()
    ae_optimizers = list()

    for i in range(len(num_neurons_in_layer)-1):
        saes.append(SingleAutoencoder(num_neurons_in_layer[i], num_neurons_in_layer[i+1]))
        encoded_x = saes[i].encode(final_encoded_x)
        decoded_x = saes[i].decode(encoded_x)
        ae_costs.append(tf.reduce_mean(tf.pow(final_encoded_x - decoded_x, 2)))
        ae_optimizers.append(tf.train.RMSPropOptimizer(learning_rate).minimize(ae_costs[i]))

        weight = tf.placeholder("float", [num_neurons_in_layer[i], num_neurons_in_layer[i+1]], name="encoder{}_weight".format(i))
        bias = tf.placeholder("float", [num_neurons_in_layer[i+1]], name="encoder{}_bias".format(i))
        encoders.append({"weight": weight, "bias": bias})
        final_encoded_x = tf.nn.sigmoid(tf.add(tf.matmul(final_encoded_x, weight), bias))

    whole_encoded_x = x_raw
    for sae in saes:
        whole_encoded_x = tf.nn.sigmoid(tf.add(tf.matmul(whole_encoded_x, sae.encoder["weight"]), sae.encoder["bias"]))

    whole_decoded_x = whole_encoded_x
    for i in range(len(saes)):
        whole_decoded_x = tf.add(tf.matmul(whole_decoded_x, saes[-i-1].decoder["weight"]), saes[-i-1].decoder["bias"])

    whole_ae_cost = tf.reduce_mean(tf.pow(x_raw - whole_decoded_x, 2))
    whole_ae_optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(whole_ae_cost)


    ## LSTM

    # LSTM Parameters
    lstm_features = n_final_features
    lstm_hidden = num_hidden # the number of hidden neurons in lstm
    n_classes = 2 # two classes: expert vs inexpert
    forget_bias = 1.0 # forget bias for lstm

    # LSTM Network
    x = tf.placeholder("float", [None, max_seq_len, lstm_features], name="x-input-data")
    y = tf.placeholder("float", [None, n_classes], name="y-output-label")
    seq_len = tf.placeholder(tf.int32, [None])

    report.output("create lstm nn: {} features, {} sequence length, {} hidden neurons, {} classes".format(lstm_features, max_seq_len, lstm_hidden, n_classes))
    cl = LstmClassification(lstm_features, lstm_hidden, n_classes)
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


    ##
    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        report.output("Cross Validation Start")
        merged = tf.summary.merge_all() # tf.merge_all_summaries()

        for i in range(n_cross_validation):
            report.output("\n[ start cross validation {} ]".format(i))

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

            ## Autoencoder
            for j in range(len(num_neurons_in_layer)-1):
                report.output("start autoencoder {}".format(j))
                step = 0
                while step < ae_training_iters:
                    for td in train_data:
                        feed_dict = {x_raw: td}
                        for k, sae in enumerate(saes):
                            feed_dict[encoders[k]["weight"]] = sess.run(sae.encoder["weight"])
                            feed_dict[encoders[k]["bias"]] = sess.run(sae.encoder["bias"])
                        # sess.run(ae_optimizers[i], feed_dict={x_raw: td})
                        sess.run(ae_optimizers[j], feed_dict=feed_dict)
                        if step == 0 or step % ae_display_step == ae_display_step-1:
                            # loss = sess.run(ae_costs[0], feed_dict={x_raw: td})
                            loss = sess.run(ae_costs[j], feed_dict=feed_dict)
                            logstr = "Layer {}".format(j) + ", Iter {}".format(step) + ", Minibatch Loss= {:.6f}".format(loss)
                            report.output(logstr)
                    step += 1

                report.output("Cross Validation {} Layer {} Optimization Finished".format(i, j))
                # test autoencoder layer with test data
                for td in test_data:
                    feed_dict = {x_raw: td}
                    for k, sae in enumerate(saes):
                        feed_dict[encoders[k]["weight"]] = sess.run(sae.encoder["weight"])
                        feed_dict[encoders[k]["bias"]] = sess.run(sae.encoder["bias"])
                    report.output("Layer {} Testing Accuracy: {}".format(j, sess.run(ae_costs[j], feed_dict=feed_dict)))

            report.output("start whole autoencoder")
            step = 0
            while step < ae_training_iters:
                for td in train_data:
                    feed_dict = {x_raw: td}
                    sess.run(whole_ae_optimizer, feed_dict={x_raw: td})
                    if step == 0 or step % ae_display_step == ae_display_step-1:
                        loss = sess.run(whole_ae_cost, feed_dict=feed_dict)
                        logstr = "Whole AE" + ", Iter {}".format(step) + ", Minibatch Loss= {:.6f}".format(loss)
                        report.output(logstr)
                step += 1

            report.output("Cross Validation {} Autoencoder Optimization Finished".format(i))
            # test autoencoder with test data
            for td in test_data:
                report.output("Autoencoder Testing Accuracy: {}".format(sess.run(whole_ae_cost, feed_dict={x_raw: td})))


            ## build train test data from autoencoder
            encoded_train_data = list()
            for td in train_data:
                feed_dict = {x_raw: td}
                for k, sae in enumerate(saes):
                    feed_dict[encoders[k]["weight"]] = sess.run(sae.encoder["weight"])
                    feed_dict[encoders[k]["bias"]] = sess.run(sae.encoder["bias"])
                encoded_train_data.append(sess.run(final_encoded_x, feed_dict=feed_dict))

            encoded_test_data = list()
            for td in test_data:
                feed_dict = {x_raw: td}
                for k, sae in enumerate(saes):
                    feed_dict[encoders[k]["weight"]] = sess.run(sae.encoder["weight"])
                    feed_dict[encoders[k]["bias"]] = sess.run(sae.encoder["bias"])
                encoded_test_data.append(sess.run(final_encoded_x, feed_dict=feed_dict))


            logtensorboard = "{}/groupid{}".format(logdir, i)
            summary_writer = tf.summary.FileWriter(logtensorboard, sess.graph)
            step = 0
            while step * batch_size < training_iters:
                # Run optimization op (backprop)
                sess.run(optimizer, \
                        feed_dict={x: encoded_train_data, y: train_label, seq_len: train_seq_len})
                if step % display_step == 0:
                    # Calculate batch accuracy and loss
                    summary, loss, acc = \
                            sess.run([merged, cost, accuracy], \
                            feed_dict={x: encoded_train_data, y: train_label, seq_len: train_seq_len})
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
                    feed_dict={x: encoded_test_data, \
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

