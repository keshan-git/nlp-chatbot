import logging.config

from pre_processor import read_files, clean_data, generate_embedding

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


if __name__ == "__main__":
    main()
