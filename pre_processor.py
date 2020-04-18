import numpy as np
import tensorflow as tf
import re
import time
import ast
from string import punctuation

column_split_symbol = ' +++$+++ '

# Import the data-set
lines = open('dataset/movie_lines.txt', encoding='utf8', errors='ignore').read().split(sep='\n')
conversations = open('dataset/movie_conversations.txt', encoding='utf8', errors='ignore').read().split(sep='\n')
contractions = open('dataset/contractions.txt', encoding='utf8', errors='ignore').read().split(sep='\n')

# Create the dictionary that map each line to its line id, key=column1 value=column5
id2line = {line.split(column_split_symbol)[0]: line.split(column_split_symbol)[4] for line in lines if
           len(line.split(column_split_symbol)) == 5}

# Create list of conversations
conversation_ids = [ast.literal_eval(conversation.split(column_split_symbol)[-1])
                    for conversation in conversations[:-1]]

# Separate the lines in to questions and answers; answer[i] is answer to the question[i]
# ex: ['L194', 'L195', 'L196', 'L197'] L194 is question L195 answer
_lines = [id2line[_idx] for conversation in conversation_ids for _idx in conversation]
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


clean_questions = [clean_text(question) for question in questions]
clean_answers = [clean_text(answers) for answer in answers]