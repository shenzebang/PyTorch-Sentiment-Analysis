import torch
import torch.nn as nn
import torch.nn.functional as F
from utilities import count_parameters
from torchtext.legacy import data

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, 10)

        self.decoder = nn.Linear(10, output_dim)

        self.dropout = nn.Dropout(dropout)

        # to be defined by the model personalization algorithm
        self.head_keys = None
        self.representation_keys = None

    def forward(self, text, text_lengths):
        # text = [sent len, batch size]

        embedded = self.dropout(self.embedding(text))

        # embedded = [sent len, batch size, emb dim]

        # pack sequence
        # lengths need to be on CPU!
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'))

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        # output = [sent len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors

        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout

        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        # hidden = [batch size, hid dim * num directions]

        return self.decoder(self.fc(hidden))


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, 10)

        self.decoder = nn.Linear(10, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        # text = [batch size, sent len]

        embedded = self.embedding(text)

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))

        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.decoder(self.fc(cat))


def get_model(args, TEXT):
    if args.model == "rnn": # define the RNN model
        INPUT_DIM = len(TEXT.vocab)
        OUTPUT_DIM = 1
        BIDIRECTIONAL = True
        PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

        model = RNN(INPUT_DIM,
                    args.EMBEDDING_DIM,
                    args.HIDDEN_DIM,
                    OUTPUT_DIM,
                    args.N_LAYERS,
                    BIDIRECTIONAL,
                    args.DROPOUT,
                    PAD_IDX)

    elif args.model == "cnn": # define the CNN model
        INPUT_DIM = len(TEXT.vocab)
        N_FILTERS = 100
        FILTER_SIZES = [3, 4, 5]
        OUTPUT_DIM = 1
        PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

        model = CNN(INPUT_DIM, args.EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, args.DROPOUT, PAD_IDX)
    else:
        raise NotImplementedError

    print(f'The model has {count_parameters(model):,} trainable parameters')

    return model

def get_keys(model):
    head_keys = ['decoder.weight', 'decoder.bias']
    representation_keys = []
    for key in model.state_dict().keys():
        if key not in head_keys:
            # if key not in head_keys and key != 'embedding.weight':
            representation_keys.append(key)

    return head_keys, representation_keys

def get_TEXT(args):
    if args.model == "rnn":
        TEXT = data.Field(tokenize='spacy',
                          tokenizer_language='en_core_web_sm',
                          include_lengths=True)
    elif args.model == "cnn":
        TEXT = data.Field(tokenize='spacy',
                          tokenizer_language='en_core_web_sm',
                          batch_first=True,
                          include_lengths=True)
    else:
        raise NotImplementedError

    return TEXT