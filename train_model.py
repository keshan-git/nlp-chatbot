import os
import time

import tensorflow as tf
from tensorflow.contrib import seq2seq
import numpy as np
import logging

from parameters import max_question_size, encoding_embedding_size, decoding_embedding_size, \
    rnn_size, num_of_layers, epochs, keep_probability, learning_rate_decay, min_learning_rate, \
    batch_index_check_training_loss, early_stopping_stop, sos_id, eos_id, pad_id, batch_size, train_validation_split, \
    param_learning_rate, weights_file_name
from seq2seq_model import model_input, seq2seq_model

logger = logging.getLogger(__name__)


# prepare the session with placeholders
def prepare_session(vocabulary_size):
    logger.info('Preparing Tensorflow session')
    # Define a session
    tf.reset_default_graph()
    session = tf.InteractiveSession()

    # load the model inputs
    inputs, targets, lr, keep_prob = model_input()

    # setting the sequence length
    sequence_length = tf.placeholder_with_default(max_question_size, None, name='sequence_length')

    # getting the shape of the input tensor
    input_shape = tf.shape(inputs)

    logger.info('Initializing Seq2Seq model')
    # getting the training and test predictions
    training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]), targets, keep_prob, batch_size,
                                                           sequence_length, vocabulary_size, vocabulary_size,
                                                           encoding_embedding_size, decoding_embedding_size,
                                                           rnn_size, num_of_layers, sos_id, eos_id)

    # Setting the Loss Error, the Optimizer and Gradient Clipping
    with tf.name_scope("optimization"):
        loss_error = seq2seq.sequence_loss(training_predictions, targets, tf.ones([input_shape[0], sequence_length]))

        # Optimizer
        _optimizer = tf.train.AdamOptimizer(param_learning_rate)

        # Gradient clipping
        gradients = _optimizer.compute_gradients(loss_error)
        clipped_gradients = [(tf.clip_by_value(gradient_tensor, -5., 5.), gradient_variable)
                             for gradient_tensor, gradient_variable in gradients if gradient_tensor is not None]

        optimizer = _optimizer.apply_gradients(clipped_gradients)

    logger.info('Tensorflow session is ready')
    return session, optimizer, loss_error, inputs, targets, lr, keep_prob, sequence_length, test_predictions


# Padding the sequence with the <PAD> token
def apply_padding(batch_of_sequences):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [pad_id] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]


# Splitting the data into batches of questions and answers
def split_batches(questions, answers, size):
    for batch_index in range(0, len(questions) // size):
        start_index = batch_index * size
        questions_in_batch = questions[start_index: start_index + size]
        answers_in_batch = answers[start_index: start_index + size]

        padded_questions_in_batch = np.array(apply_padding(questions_in_batch))
        padded_answers_batch = np.array(apply_padding(answers_in_batch))

        yield padded_questions_in_batch, padded_answers_batch


# Splitting the questions and answers into training and validation sets
def split_training_validation(questions, answers):
    logger.info('Splitting training and validation sets, split ration={0}'.format(train_validation_split))
    training_validation_split = int(len(questions) * train_validation_split)
    training_question = questions[training_validation_split:]
    training_answers = answers[training_validation_split:]

    validation_question = questions[:training_validation_split]
    validation_answers = answers[:training_validation_split]

    return training_question, training_answers, validation_question, validation_answers


# Start training
def start_training(session, optimizer, loss_error, training_questions, training_answers,
                   validation_questions, validation_answers, inputs, targets, lr, keep_prob, sequence_length):
    # Training
    batch_index_check_validation_loss = (len(training_questions) // batch_size // 2) - 1
    total_training_loss_error = 0
    list_validation_loss_error = []
    early_stopping_check = 0
    learning_rate = param_learning_rate
    session.run(tf.global_variables_initializer())
    logger.info('Start training the model')

    for epoch in range(1, epochs + 1):
        for batch_index, (questions, answers) \
                in enumerate(split_batches(training_questions, training_answers, batch_size)):
            starting_time = time.time()
            _, batch_training_loss_error = session.run([optimizer, loss_error], {inputs: questions,
                                                                                 targets: answers,
                                                                                 lr: learning_rate,
                                                                                 sequence_length: answers.shape[1],
                                                                                 keep_prob: keep_probability})
            total_training_loss_error += batch_training_loss_error
            batch_time = time.time() - starting_time

            if batch_index % batch_index_check_training_loss == 0:
                logger.info(
                    'Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time: {:d} s'.format(
                        epoch, epochs, batch_index, len(training_questions) // batch_size,
                        total_training_loss_error / batch_index_check_training_loss,
                        int(batch_time * batch_index_check_training_loss)))
                total_training_loss_error = 0

            if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
                total_validation_loss_error = 0
                starting_time = time.time()

                for batch_index_validation, (validation_question, validation_answer) \
                        in enumerate(split_batches(validation_questions, validation_answers, batch_size)):
                    batch_validation_loss_error = session.run(loss_error, {inputs: validation_question,
                                                                           targets: validation_answer,
                                                                           lr: learning_rate,
                                                                           sequence_length: validation_answer.shape[1],
                                                                           keep_prob: 1.0})
                    total_validation_loss_error += batch_validation_loss_error

                batch_time = time.time() - starting_time
                avg_validation_loss_error = total_validation_loss_error / (len(validation_question) / batch_size)
                logger.info('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(
                    avg_validation_loss_error, int(batch_time)))

                learning_rate *= learning_rate_decay
                if learning_rate < min_learning_rate:
                    learning_rate = min_learning_rate

                list_validation_loss_error.append(avg_validation_loss_error)
                last_min_validation_loss_error = min(list_validation_loss_error)
                if avg_validation_loss_error <= last_min_validation_loss_error:
                    logger.info('Improvement over last validation step from {:>6.3f} to {:>6.3f}'
                                .format(last_min_validation_loss_error, avg_validation_loss_error))
                    early_stopping_check = 0

                    saver = tf.train.Saver()
                    saver.save(session, os.path.join(os.getcwd(), "weights" + os.sep + weights_file_name))
                else:
                    logger.info('No new improvement over last validation step from {:>6.3f} to {:>6.3f}'
                                .format(last_min_validation_loss_error, avg_validation_loss_error))
                    early_stopping_check += 1

                    if early_stopping_check == early_stopping_stop:
                        break

        if early_stopping_check == early_stopping_stop:
            logger.info("Reached the stop check during the training")
            break
