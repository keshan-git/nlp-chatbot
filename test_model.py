import os

import tensorflow as tf
import numpy as np

from parameters import pad_id, batch_size, weights_file_name
from pre_processor import encode_question


# Loading the weights and running the session
def load_session():
    session_file_path = "weights" + os.sep + weights_file_name
    session = tf.InteractiveSession()
    session.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(session, session_file_path)
    return session


def predict_answer(session, test_predictions, question, word2id):
    _question = encode_question(question, word2id)
    _question = _question + [pad_id] * (20 - len(_question))

    _batch = np.zeros([batch_size, 20])
    _batch[0] = _question

    predicted_answer = session.run(test_predictions, {inputs: _batch, keep_prob: 0.5})[0]

    answer = ''
    for i in np.argmax(predicted_answer, 1):
        answer += word2id[i]

    return answer


