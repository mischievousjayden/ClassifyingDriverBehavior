
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np

import datalayer as dl


# read data
print("read data")
data_path = "../data"
data = dl.driverdata(data_path) # data.expert_data, data.inexpert_data

# for d in data.expert_data:
#     print(np.shape(d))
#
# for d in data.inexpert_data:
#     print(np.shape(d))


# Parameters
learning_rate = 0.001
training_iters = 100
batch_size = 6
display_step = 10

# Network Parameters
n_input = 101 # MNIST data input (img shape: 28*28)
n_steps = 100 # 2629 # timesteps # 500 #
n_hidden = 256 # hidden layer num of features
n_classes = 2 # MNIST total classes (0-9 digits)



print("build train and test data")
def createTrainTestData(expert_data, inexpert_data):
    train_data = []
    train_label = []
    test_data = []
    test_label = []

    for i in range(4):
        if(i != 3):
            train_data.append(np.array(expert_data[i])[:n_steps,:])
            train_label.append([1,0])
        else:
            test_data.append(np.array(expert_data[i])[:n_steps,:])
            test_label.append([1,0])

    for i in range(4):
        if(i != 3):
            train_data.append(np.array(inexpert_data[i])[:n_steps,:])
            train_label.append([0,1])
        else:
            test_data.append(np.array(inexpert_data[i])[:n_steps,:])
            test_label.append([0,1])

    return [train_data, train_label, test_data, test_label]

train_data, train_label, test_data, test_label = createTrainTestData(data.expert_data, data.inexpert_data)






# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


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

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    print("start learning")
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        # print("step " + str(step))
        batch_x = train_data
        batch_y = train_label

        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
        # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_data, y: test_label}))

