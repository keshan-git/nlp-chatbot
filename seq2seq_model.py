import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import MultiRNNCell

# Create placeholders for the input and the targets
def model_input():
    inputs = tf.placeholder(tf.int32, shape=[None, None], name='inputs')
    targets = tf.placeholder(tf.int32, shape=[None, None], name='targets')

    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    return inputs, targets, lr, keep_prob


# Pre process the target
def get_batch_targets(targets, batch_size=10):
    sos_vec = tf.fill([batch_size, 1], 2)  # <SOS>
    _targets = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])
    batch_targets = tf.concat([sos_vec, _targets], axis=1)
    return batch_targets


# Create the encoder RNN layer
def encoder_rnn(rnn_input, rnn_size, num_of_layers, keep_prob, sequence_length):
    _lstm = BasicLSTMCell(rnn_size)
    lstm = DropoutWrapper(_lstm, input_keep_prob=keep_prob)

    cell = MultiRNNCell([lstm] * num_of_layers)
    _, state = tf.contrib.nn.bidirectional_dynamic_rnn(cell_fv=cell, cell_bv=cell,
                                                               sequence_length=sequence_length, inputs=rnn_input,
                                                               dtype=tf.float32)
    return state


