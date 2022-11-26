import torch
from torchtext.legacy import data
from options import args_parser
from torchtext.legacy import datasets
import random
from model import RNN
from utilities import count_parameters, binary_accuracy, epoch_time, TEXT, LABEL
import torch.optim as optim
import torch.nn as nn
import time


def train(model, iterator, optimizer, criterion, N_local_epoch=1, ft=False):
    epoch_loss = 0
    epoch_acc = 0

    if not ft:
        model.train()

    for _ in range(N_local_epoch):
        for batch in iterator:
            optimizer.zero_grad()

            text, text_lengths = batch.text

            predictions = model(text, text_lengths).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return model.state_dict(), epoch_loss / len(iterator) / N_local_epoch, epoch_acc / len(iterator) / N_local_epoch


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text

            predictions = model(text, text_lengths).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def SA_single(args, device):


    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)


    TEXT.build_vocab(train_data,
                     max_size=args.MAX_VOCAB_SIZE,
                     vectors="glove.6B.100d",
                     unk_init=torch.Tensor.normal_)

    LABEL.build_vocab(train_data)

    train_data, valid_data = train_data.split()

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=args.BATCH_SIZE,
        sort_within_batch=True,
        device=device)

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

    print(f'The model has {count_parameters(model):,} trainable parameters')

    pretrained_embeddings = TEXT.vocab.vectors

    print(pretrained_embeddings.shape)

    model.embedding.weight.data.copy_(pretrained_embeddings)

    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

    model.embedding.weight.data[UNK_IDX] = torch.zeros(args.EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(args.EMBEDDING_DIM)

    print(model.embedding.weight.data)

    # ===== training =====
    print("#" * 10)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    N_EPOCHS = 5

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        _, train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut2-model.pt')

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    model.load_state_dict(torch.load('tut2-model.pt'))

    test_loss, test_acc = evaluate(model, test_iterator, criterion)

    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')


if __name__ == '__main__':
    args = args_parser()

    torch.manual_seed(args.SEED)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    SA_single(args, device)
