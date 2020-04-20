# Define special tokens
TOKEN_PAD = '<PAD>'
TOKEN_SOS = '<SOS>'
TOKEN_EOS = '<EOS>'
TOKEN_UNK = '<UNK>'

column_split_symbol = ' +++$+++ '
pad_id = 1
sos_id = 2
eos_id = 3
unk_id = 4

# Hyper parameters
frequent_threshold = 0.9  # to 90% of the world will be selected to generate the word2id
max_question_size = 25  # Questions containing more than 25 words will be discarded

train_validation_split = 0.15
epochs = 100
batch_size = 64
rnn_size = 512
num_of_layers = 3  # layers in encoder RNN, layers in decoder RNN
encoding_embedding_size = 512
decoding_embedding_size = 512
param_learning_rate = 0.01
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5  # 1 - dropout rate

batch_index_check_training_loss = 10
early_stopping_stop = 1000
