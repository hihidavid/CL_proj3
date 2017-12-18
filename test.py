import sys
from optparse import OptionParser
import codecs
import string
import random
import math
import time
import utils
from nltk.metrics.distance import edit_distance
import numpy as np

import data
from model import *


def generateRandomBatch(input, batch_size):
    random.shuffle(input)
    for i in range(0, len(input), batch_size):
        yield input[i:i + batch_size]


def process_batch(batch, max_len=20):
    batch_pairs, batch_len = [(var_from_word_padded(input_lang, w0, max_len), var_from_word_padded(output_lang, w1, max_len))
            for (w0, w1) in batch], \
            [(len(w0) + 1, len(w1) + 1) for (w0, w1) in batch]
    input_batch, output_batch = zip(*batch_pairs)
    input_batch_len, output_batch_len = zip(*batch_len)
    input_max_len = max(input_batch_len)
    output_max_len = max(output_batch_len)
    max_len = max(input_max_len, output_max_len)
    input_batch = np.array(input_batch)[:,:input_max_len]
    output_batch = np.array(output_batch)[:, :output_max_len]
    input_batch_len = np.array(input_batch_len)
    output_batch_len = np.array(output_batch_len)
    idx = np.argsort(-input_batch_len)
    return input_batch[idx, :], output_batch[idx, :], input_batch_len[idx], output_batch_len[idx]


def indexesFromWord(lang, word):
    return [lang.char2index[char] for char in list(word)]


# padded to l+1
def var_from_word_padded(lang, word, l):
    l = l - len(word) if l > len(word) else 0
    indexes = indexesFromWord(lang, word)
    indexes.append(EOS_token)
    indexes.extend([0] * l)
    return indexes
    # result = Variable(torch.LongTensor(indexes).view(-1, 1))
    # if use_cuda:
    #     return result.cuda()
    # else:
    #     return result


#################################################################################
# PUT IT ALL TOGETHER
#################################################################################
if __name__ == "__main__":

    file_path = 'data/en_bg.train.txt'
    # model parameters
    hidden_size = 256

    # training hyperparameters
    learn_rate = 0.01
    n_epoch = 3
    batch_size = 5
    learning_rate = 0.01

    # how verbose
    printfreq = 1000
    plotfreq = 100

    # STEP 1: read in and prepare training data
    input_lang, output_lang, pairs = data.prepareTrainData(file_path, 'en', 'bg', reverse=True)
    batch_data = generateRandomBatch(pairs, batch_size)

    # STEP 2: define and train sequence to sequence model
    encoder = EncoderRNN(input_lang.n_chars, hidden_size, bidir=True, n_layers=1)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_chars, 1, dropout_p=0.1, bidir=True)

    encoder = encoder.cuda() if use_cuda else encoder
    decoder = decoder.cuda() if use_cuda else decoder

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    # TODO: start for loop over batch_data

    # process input output data
    input_batch, output_batch, input_batch_len, output_batch_len = process_batch(next(batch_data))
    input_batch_var = Variable(torch.LongTensor(input_batch))
    input_batch_var = input_batch_var.cuda() if use_cuda else input_batch_var

    criterion = nn.NLLLoss()

    h0 = encoder.initHidden(batch_size)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # hn (2, 5, 256)
    encoder_output, hn = encoder(input_batch_var, input_batch_len)
    decoder_hidden = hn

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_output, decoder_hidden, decoder_attention = decoder(
        decoder_input, decoder_hidden, encoder_output, encoder_outputs)
    loss = criterion(decoder_output, target_variable[di])
    decoder_input = target_variable[di]

    # =============================================================================================================
    # [20, 256]
    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    # output (seq_len, batch, hidden_size * num_directions) [1, 1, 256]
    # h_n (num_layers * num_directions, batch, hidden_size) [1, 1, 256]
    encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
    encoder_outputs[ei] = encoder_output[0][0]

    pass


