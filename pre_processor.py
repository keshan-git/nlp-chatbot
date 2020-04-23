import nltk
import re
import ast
from string import punctuation
import logging

# Import the data-set
from parameters import column_split_symbol, frequent_threshold, TOKEN_PAD, TOKEN_SOS, TOKEN_EOS, TOKEN_UNK, \
    max_question_size, unk_id

logger = logging.getLogger(__name__)


# load data sets from files
def read_files():
    logger.info('Start importing data sets')
    lines = open('dataset/movie_lines.txt', encoding='utf8', errors='ignore').read().split(sep='\n')
    conversations = open('dataset/movie_conversations.txt', encoding='utf8', errors='ignore').read().split(sep='\n')
    contractions = open('dataset/contractions.txt', encoding='utf8', errors='ignore').read().split(sep='\n')
    logger.info('Data sets are imported')

    return lines, conversations, contractions


# separate data-set in to questions and answers and clean questions and answers
def clean_data(lines, conversations, contractions):
    logger.info('Building internal data structure from the data set')
    # Create the dictionary that map each line to its line id, key=column1 value=column5
    id2line = {line.split(column_split_symbol)[0]: line.split(column_split_symbol)[4] for line in lines if
               len(line.split(column_split_symbol)) == 5}

    # Create list of conversations
    conversation_ids = [ast.literal_eval(conversation.split(column_split_symbol)[-1])
                        for conversation in conversations[:-1]]

    # Separate the lines in to questions and answers; answer[i] is answer to the question[i]
    # ex: ['L194', 'L195', 'L196', 'L197'] L194 is question L195 answer
    _lines = [id2line[_idx] for conversation in conversation_ids for _idx in conversation][:-1]
    questions = _lines[0:][::2]
    answers = _lines[1:][::2]

    # Create contractions dictionary; used in cleaning
    contractions_mapping = {line.split(":")[0]: line.split(":")[1].strip() for line in contractions}

    logger.info('Start cleaning questions and answers')
    clean_questions = {idx: _clean_text(question, contractions_mapping) for idx, question in enumerate(questions)}
    clean_answers = {idx: _clean_text(answer, contractions_mapping) for idx, answer in enumerate(answers)}

    return clean_questions, clean_answers


# create word2id embedding and questions / answer vectors with word embeddings
def generate_embedding(clean_questions, clean_answers):
    # flat sentences to words
    logger.info('Generating bag of words')
    bag_of_words = [word for question in clean_questions.values() for word in question.split()] + \
                   [word for answers in clean_answers.values() for word in answers.split()]

    # Create word distribution, so that filtering less frequent words is possible
    logger.info('Calculating word frequency')
    frequent_distribution = nltk.FreqDist(bag_of_words)

    _len = len(frequent_distribution)
    logger.info('Selecting top {0}% words out of total {1} words'.format(frequent_threshold * 100, _len))
    selected_words = frequent_distribution.most_common(int(_len * frequent_threshold))

    # Create word2id mapping - Word embedding
    logger.info('Creating word2id for the selected words')
    tokens = [TOKEN_PAD, TOKEN_SOS, TOKEN_EOS, TOKEN_UNK]
    word2id = {word[0]: i for i, word in enumerate(selected_words, start=len(tokens)+1)}
    word2id.update({token: i for i, token in enumerate(tokens, start=1)})

    # Create id2word mapping - Inverse word mapping
    logger.info('Creating id2word inverse mapping')
    id2word = {idx: word for word, idx in word2id.items()}

    # Convert words in questions/answers to idx based on word2id
    logger.info('Converting sentence in to [ids] based on word2id mapping')
    questions2id = {idx: [word2id.get(word, word2id[TOKEN_UNK]) for word in question.split()]
                    for idx, question in clean_questions.items()}

    answers2id = {idx: [word2id.get(word, word2id[TOKEN_UNK]) for word in answer.split()]
                  for idx, answer in clean_answers.items()}

    # Sort questions by the length
    logger.info('Start sorting questions based on the word counts (=length)')
    sorted_questions2id = []
    sorted_answers2id = []
    for idx, question in sorted(questions2id.items(), key=lambda item: len(item[1])):
        if len(question) > max_question_size:
            break

        sorted_questions2id.append(question)

        # Add EOS token to all the answers
        answers2id[idx].append(word2id[TOKEN_EOS])
        sorted_answers2id.append(answers2id[idx])

    return sorted_questions2id, sorted_answers2id, word2id, id2word


# clean text lines
def _clean_text(text, contractions_mapping):
    text = text.lower()

    # expand contractions
    contractions_pattern = re.compile('({})'.format('|'.join(contractions_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)
    text = contractions_pattern.sub(lambda m: contractions_mapping.get(m.group(0)), text)

    # remove all punctuations
    text = re.sub(r"[{}]".format(punctuation), "", text)
    return text


# Converting new questions from string to lists of encoding int
def encode_question(question, word2id):
    _question = _clean_text(question)
    return [word2id.get(word, unk_id) for word in question.split()]
