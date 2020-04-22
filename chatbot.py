import logging.config

from pre_processor import read_files, clean_data, generate_embedding
from test_model import predict_answer, load_session
from train_model import prepare_session, split_training_validation, start_training

logging.config.fileConfig(fname='log_config.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main():
    logger.info('Starting chatbot module')

    # load and pre process data set
    lines, conversations, contractions = read_files()
    clean_questions, clean_answers = clean_data(lines, conversations, contractions)
    questions2id, answers2id, word2id, id2word = generate_embedding(clean_questions, clean_answers)

    logger.info('Starting model with {0} questions {1} answers and vocabulary size={2}'.format(len(questions2id),
                                                                                               len(answers2id),
                                                                                               len(word2id)))

    logger.info('Preparing training session')
    session, optimizer, loss_error, inputs, targets, lr, keep_prob, sequence_length, test_predictions = prepare_session(len(word2id))

    t_question, t_answers, v_question, v_answers = split_training_validation(questions2id, answers2id)
    logger.info('Training/validation split is done, training set={0} validation set={1}'.format(len(t_question),
                                                                                                len(v_question)))
    logger.info('Starting training session')
    start_training(session, optimizer, loss_error,
                   t_question, t_answers, v_question, v_answers,
                   inputs, targets, lr, keep_prob, sequence_length)
    logger.info('Training session completed')

    logger.info('Starting chatbot with existing session')
    loaded_session = load_session()

    # Setting up the chat
    while True:
        question = input("You: ")
        if question == 'Goodbye':
            break

        answer = predict_answer(loaded_session, test_predictions, question, word2id)
        print('Chatbot: ' + answer)


if __name__ == "__main__":
    main()
