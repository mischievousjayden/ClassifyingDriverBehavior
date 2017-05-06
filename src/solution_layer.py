import tensorflow as tf
import numpy as np


class LstmClassification:
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


class SingleAutoencoder:
    def __init__(self, n_features, n_hidden):
        self._n_features = n_features
        self._n_hidden = n_hidden
        self.encoder = {
                "weight": tf.Variable(tf.random_normal([n_features, n_hidden]), name="encoder_weight"),
                "bias": tf.Variable(tf.random_normal([n_hidden]), name="encoder_bias")
            }
        self.decoder = {
                "weight": tf.Variable(tf.random_normal([n_hidden, n_features]), name="decoder_weight"),
                "bias": tf.Variable(tf.random_normal([n_features]), name="decoder_bias")
            }

    def encode(self, x):
        encoded_x = tf.nn.sigmoid(tf.add(tf.matmul(x, self.encoder["weight"]), self.encoder["bias"]))
        return encoded_x

    def decode(self, encoded_x):
        decoded_x = tf.add(tf.matmul(encoded_x, self.decoder["weight"]), self.decoder["bias"])
        return decoded_x


class MultilayerAutoencoder:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.num_layers = len(num_neurons)-1
        self.encoders = list()
        self.decoders = list()

        for i in range(self.num_layers):
            self.encoders.append({
                    "weight": tf.Variable(tf.random_normal([num_neurons[i], num_neurons[i+1]]), name="encoder{}_weight".format(i)),
                    "bias": tf.Variable(tf.random_normal([num_neurons[i+1]]), name="encoder{}_bias".format(i))
                })
            self.decoders.append({
                    "weight": tf.Variable(tf.random_normal([num_neurons[self.num_layers-i], num_neurons[self.num_layers-i-1]]), name="decoder{}_weight".format(i)),
                    "bias": tf.Variable(tf.random_normal([num_neurons[self.num_layers-i-1]]), name="decoder{}_bias".format(i))
                })

    def encode(self, x):
        encoded_x = x
        for encoder in self.encoders:
            encoded_x = tf.nn.sigmoid(tf.add(tf.matmul(encoded_x, encoder["weight"]), encoder["bias"]))
            # encoded_x = tf.nn.tanh(tf.add(tf.matmul(encoded_x, encoder["weight"]), encoder["bias"]))
        return encoded_x

    def decode(self, encoded_x):
        decoded_x = encoded_x
        for decoder in self.decoders:
            # decoded_x = tf.nn.sigmoid(tf.add(tf.matmul(decoded_x, decoder["weight"]), decoder["bias"]))
            # decoded_x = tf.nn.tanh(tf.add(tf.matmul(decoded_x, decoder["weight"]), decoder["bias"]))
            decoded_x = tf.add(tf.matmul(decoded_x, decoder["weight"]), decoder["bias"])
        return decoded_x


