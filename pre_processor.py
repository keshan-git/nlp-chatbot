import nltk
import re
import ast
from string import punctuation

# Define special tokens
TOKEN_PAD = '<PAD>'
TOKEN_SOS = '<SOS>'
TOKEN_EOS = '<EOS>'
TOKEN_UNK = '<UNK>'

column_split_symbol = ' +++$+++ '

# Hyper parameters
frequent_threshold = 0.9  # to 90% of the world will be selected to generate the word2id
max_question_size = 25  # Questions containing more than 25 words will be discarded

# Import the data-set
print('Start importing data sets')
lines = open('dataset/movie_lines.txt', encoding='utf8', errors='ignore').read().split(sep='\n')
conversations = open('dataset/movie_conversations.txt', encoding='utf8', errors='ignore').read().split(sep='\n')
contractions = open('dataset/contractions.txt', encoding='utf8', errors='ignore').read().split(sep='\n')
print('Data sets are imported')

print('Building internal data structure from the data set')
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


# clean text lines
def clean_text(text):
    text = text.lower()

    # expand contractions
    contractions_pattern = re.compile('({})'.format('|'.join(contractions_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)
    text = contractions_pattern.sub(lambda m: contractions_mapping.get(m.group(0)), text)

    # remove all punctuations
    text = re.sub(r"[{}]".format(punctuation), "", text)
    return text


print('Start cleaning questions and answers')
clean_questions = {idx: clean_text(question) for idx, question in enumerate(questions)}
clean_answers = {idx: clean_text(answer) for idx, answer in enumerate(answers)}

# flat sentences to words
print('Generating bag of words')
bag_of_words = [word for question in clean_questions.values() for word in question.split()] + \
               [word for answers in clean_answers.values() for word in answers.split()]

# Create word distribution, so that filtering less frequent words is possible
print('Calculating word frequency')
frequent_distribution = nltk.FreqDist(bag_of_words)

_len = len(frequent_distribution)
print('Selecting top {0}% words out of total {1} words'.format(frequent_threshold * 100, _len))
selected_words = frequent_distribution.most_common(int(_len * frequent_threshold))

# Create word2id mapping - Word embedding
print('Creating word2id for the selected words')
tokens = [TOKEN_PAD, TOKEN_SOS, TOKEN_EOS, TOKEN_UNK]
word2id = {word[0]: i for i, word in enumerate(selected_words, start=len(tokens)+1)}
word2id.update({token: i for i, token in enumerate(tokens, start=1)})

# Create id2word mapping - Inverse word mapping
print('Creating id2word inverse mapping')
id2word = {idx: word for word, idx in word2id.items()}

# Convert words in questions/answers to idx based on word2id
print('Converting sentence in to [ids] based on word2id mapping')
questions2id = {idx: [word2id.get(word, word2id[TOKEN_UNK]) for word in question.split()]
                for idx, question in clean_questions.items()}

answers2id = {idx: [word2id.get(word, word2id[TOKEN_UNK]) for word in answer.split()]
              for idx, answer in clean_answers.items()}

# Sort questions by the length
print('Start sorting questions based on the word counts (=length)')
sorted_questions2id = []
sorted_answers2id = []
for idx, question in sorted(questions2id.items(), key=lambda item: len(item[1])):
    if len(question) > max_question_size:
        break

    sorted_questions2id.append(question)

    # Add EOS token to all the answers
    answers2id[idx].append(word2id[TOKEN_EOS])
    sorted_answers2id.append(answers2id[idx])
