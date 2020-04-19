# Define special tokens
TOKEN_PAD = '<PAD>'
TOKEN_SOS = '<SOS>'
TOKEN_EOS = '<EOS>'
TOKEN_UNK = '<UNK>'

column_split_symbol = ' +++$+++ '

# Hyper parameters
frequent_threshold = 0.9  # to 90% of the world will be selected to generate the word2id
max_question_size = 25  # Questions containing more than 25 words will be discarded
