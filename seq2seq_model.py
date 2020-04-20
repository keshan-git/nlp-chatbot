import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, DropoutWrapper, MultiRNNCell
from tensorflow.contrib import seq2seq


# Create placeholders for the input and the targets
def model_input():
    inputs = tf.placeholder(tf.int32, shape=[None, None], name='inputs')
    targets = tf.placeholder(tf.int32, shape=[None, None], name='targets')

    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    return inputs, targets, lr, keep_prob


# Pre process the target
def batch_targets(targets, batch_size, sos_id):
    sos_vec = tf.fill([batch_size, 1], sos_id)
    _targets = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])
    return tf.concat([sos_vec, _targets], axis=1)


# Create the encoder RNN layer
def encoder_rnn(rnn_input, rnn_size, num_of_layers, keep_prob, sequence_length):
    _lstm = BasicLSTMCell(rnn_size)
    lstm = DropoutWrapper(_lstm, input_keep_prob=keep_prob)

    cell = MultiRNNCell([lstm] * num_of_layers)
    output, state = tf.contrib.nn.bidirectional_dynamic_rnn(cell_fv=cell,
                                                            cell_bv=cell,
                                                            sequence_length=sequence_length,
                                                            inputs=rnn_input,
                                                            dtype=tf.float32)
    return state


# Decoding the training set
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope,
                        output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])

    attention_keys, attention_values, attention_score_function, attention_construct_function \
        = seq2seq.prepare_attention(attention_states,
                                    attention_option="bahdanau",
                                    num_units=decoder_cell.output_size)

    training_decoder_function = seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                   attention_keys,
                                                                   attention_values,
                                                                   attention_score_function,
                                                                   attention_construct_function,
                                                                   name="attn_dec_train")

    decoder_output, _, _ = seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                       training_decoder_function,
                                                       decoder_embedded_input,
                                                       sequence_length,
                                                       scope=decoding_scope)

    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)


# Decoding the test/validation set
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words,
                    decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])

    attention_keys, attention_values, attention_score_function, attention_construct_function \
        = seq2seq.prepare_attention(attention_states,
                                    attention_option="bahdanau",
                                    num_units=decoder_cell.output_size)

    test_decoder_function = seq2seq.attention_decoder_fn_inference(output_function,
                                                                   encoder_state[0],
                                                                   attention_keys,
                                                                   attention_values,
                                                                   attention_score_function,
                                                                   attention_construct_function,
                                                                   decoder_embeddings_matrix,
                                                                   sos_id,
                                                                   eos_id,
                                                                   maximum_length,
                                                                   num_words,
                                                                   name="attn_dec_inf")

    test_predictions, _, _ = seq2seq.dynamic_rnn_decoder(decoder_cell, test_decoder_function, scope=decoding_scope)
    return test_predictions


# Creating the decoder RNN
def decoder_rnn(decoder_embedded_inputs, decoder_embedding_matrix, encoder_state, num_words, sequence_length, rnn_size,
                num_of_layers, sos_id, eos_id, keep_prob, batch_size):
    # perform on decoding scope
    with tf.variable_scope("decoding") as decoding_scope:
        _lstm = BasicLSTMCell(rnn_size)
        lstm = DropoutWrapper(_lstm, input_keep_prob=keep_prob)

        cell = MultiRNNCell([lstm] * num_of_layers)

        # Initialize weights and biases
        weights = tf.truncated_normal_initializer(stddev=0.1)
        biases = tf.zeros_initializer()

        # define output function
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope=decoding_scope,
                                                                      weights_initializers=weights,
                                                                      biases_initializers=biases)

        training_prediction = decode_training_set(encoder_state, cell, decoder_embedded_inputs, sequence_length,
                                                  decoding_scope, output_function, keep_prob, batch_size)

        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state, cell, decoder_embedding_matrix, sos_id, eos_id,
                                           sequence_length - 1, num_words, decoding_scope, output_function, keep_prob,
                                           batch_size)

    return training_prediction, test_predictions


# Build the model
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words,
                  encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, sos_id, eos_id):
    # build encoder rnn
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,
                                                              encoder_embedding_size,
                                                              initializer=tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)

    # build decoder rnn
    batch_targets = get_batch_targets(targets, batch_size, sos_id)
    decoder_embeddings_matrix = tf.variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, batch_targets)

    # predictions
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix,
                                                         encoder_state, questions_num_words, sequence_length, rnn_size,
                                                         num_layers, sos_id, eos_id, keep_prob, batch_size)
    return training_predictions, test_predictions
