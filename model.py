import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
SOS_token = 0
EOS_token = 1


################
# An Encoder model
################
class EncoderRNN(nn.Module):
    # http://pytorch.org/docs/master/nn.html
    def __init__(self, input_size, hidden_size, bidir=False, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.n_dir = 2 if bidir else 1
        self.hidden_size = hidden_size
        # input_size: 85, hidden_size: 256
        # <class 'torch.nn.modules.sparse.Embedding'>
        # Embedding(num_embeddings, embedding_dim)
        self.embedding = nn.Embedding(input_size, hidden_size)
        # <class 'torch.nn.modules.rnn.GRU'>
        # GRU(input_size, hidden_size, num_layers=1)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, bias=True,
                          batch_first=True, bidirectional=bidir)


    # input (batch, max_seq_len) variable
    def forward(self, input, input_len):
        # input: LongTensor(N,W), output: (N,W,256)
        embedded = self.embedding(input)
        input_batch_packed = nn.utils.rnn.pack_padded_sequence(embedded, input_len, batch_first=True)
        # output, hn = gru(input, h0)
        output, hn = self.gru(input_batch_packed)
        return output, hn

    def initHidden(self, batch):
        # h_0 (num_layers * num_directions, batch, hidden_size)
        # may not need to call this. Defaults to zero according to doc
        result = Variable(torch.zeros(self.n_layers*self.n_dir, batch, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result



# RNN with attention and dropout
# as illustrated here http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#attention-decoder
class AttnDecoderRNN(nn.Module):
    # 256, 31, 1
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=20, bidir=False):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.n_dir = 2 if bidir else 1

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=n_layers, bias=True,
                          batch_first=True, bidirectional=bidir)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_output, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        # emb: [1, 1, 256], hidden [1, 1, 256], cat: [1, 512], attn: [1, 20]
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)))
        # attn_unsqueeze: [1, 1, 20], encoder_outputs_unsqueeze: [1, 20, 256], attn_applied: [1, 1, 256]
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        # output: [1, 512]
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        # combine: [1, 256], output: [1, 1, 256]
        output = self.attn_combine(output).unsqueeze(0)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


#################################################################################
# TRAINING
#################################################################################


# Train the model on one example
def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=20):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    # [20, 256]
    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0
    # TODO: for loop not needed.

    for ei in range(input_length):
        # output (seq_len, batch, hidden_size * num_directions) [1, 1, 256]
        # h_n (num_layers * num_directions, batch, hidden_size) [1, 1, 256]
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    # TODO: change teacher forcing
    teacher_forcing_ratio = 0.5

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        # TODO: for loop
        for di in range(target_length):
            # decoder_output: [1, 31], hidden: [1, 1, 256], attention: [1, 20]
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            # input: (minibatch, Class) [1, 31], target: (minibatch) [1]
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]

    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            # [1, 1]
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])

            if ni == EOS_token:
                break

    # use autograd to backpropagate loss
    loss.backward()
    # update model parameters
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


# train 1 epoch
def trainEpoch(pairs, input_lang, output_lang, encoder, decoder, n_iters, print_every=100, plot_every=1000,
               learning_rate=0.01):
    plot_losses = []
    print_loss_total = 0  # reset every print_every
    plot_loss_total = 0  # reset every print_every

    # TODO: training data generation

    # TODO: other loss function? MSE
    criterion = nn.NLLLoss()

    # now proceed one iteration at a time
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        # train on one example
        loss = train(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (
            utils.timeSince(start, float(iter) / float(n_iters)), iter, float(iter) / float(n_iters) * 100,
            print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / float(plot_every)
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    # plot the learning curve
    utils.showPlot(plot_losses)

