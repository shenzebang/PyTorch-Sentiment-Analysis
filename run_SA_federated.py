import torch
from torchtext.legacy import data
from options import args_parser
from torchtext.legacy import datasets
import random
from model import RNN
from utilities import count_parameters, binary_accuracy, epoch_time, splits_federated
import torch.optim as optim
import torch.nn as nn
import time
import copy

from run_SA_single import evaluate, train

def SA_federated(args, device):
    TEXT = data.Field(tokenize='spacy',
                      tokenizer_language='en_core_web_sm',
                      include_lengths=True)

    LABEL = data.LabelField(dtype=torch.float)

    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

    _train_datasets_federated = splits_federated(train_data, args.N_clients)

    train_datasets_federated = []
    valid_datasets_federated = []
    for train_dataset in _train_datasets_federated:
        train_dataset, valid_dataset = train_dataset.split(split_ratio=0.9)
        train_datasets_federated.append(train_dataset)
        valid_datasets_federated.append(valid_dataset)

    test_datasets_federated = splits_federated(test_data, args.N_clients)



    TEXT.build_vocab(train_data,
                     max_size=args.MAX_VOCAB_SIZE,
                     vectors="glove.6B.100d",
                     unk_init=torch.Tensor.normal_)

    LABEL.build_vocab(train_data)

    clients_iterators = []
    for train_data, valid_data, test_data in zip(train_datasets_federated, valid_datasets_federated, test_datasets_federated):
        train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_size=args.BATCH_SIZE,
            sort_within_batch=True,
            device=device)
        clients_iterators.append((train_iterator, valid_iterator, test_iterator))

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

    criterion = nn.BCEWithLogitsLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    N_EPOCHS = 5

    best_valid_loss = float('inf')

    sd_global = model.state_dict()

    for epoch in range(N_EPOCHS):

        start_time = time.time()



        train_loss = 0
        train_acc = 0
        valid_loss = 0
        valid_acc = 0

        sd_locals = []
        for client_iterators in clients_iterators:
            train_iterator, valid_iterator, _ = client_iterators

            sd_local = copy.deepcopy(sd_global)
            model.load_state_dict(sd_local)
            optimizer = optim.Adam(model.parameters())

            sd_local, train_loss_local, train_acc_local = train(model, train_iterator, optimizer, criterion, args.N_local_epoch)
            valid_loss_local, valid_acc_local = evaluate(model, valid_iterator, criterion)

            train_loss += train_loss_local
            train_acc += train_acc_local
            valid_loss += valid_loss_local
            valid_acc += valid_acc_local
            sd_locals.append(sd_local)

        train_loss /= args.N_clients
        train_acc /= args.N_clients
        valid_loss /= args.N_clients
        valid_acc /= args.N_clients

        for key in sd_global.keys():
            sd_global[key] = torch.mean(torch.stack([sd_local[key] for sd_local in sd_locals], dim=0), dim=0)


        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(sd_global, 'tut2-model.pt')

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    model.load_state_dict(torch.load('tut2-model.pt'))

    test_loss = 0
    test_acc = 0
    for client_iterators in clients_iterators:
        _, _, test_iterator = client_iterators
        test_loss_local, test_acc_local = evaluate(model, test_iterator, criterion)
        test_loss += test_loss_local
        test_acc += test_acc_local

    test_loss /= args.N_clients
    test_acc /= args.N_clients

    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')


if __name__ == '__main__':
    args = args_parser()

    random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    SA_federated(args, device)
