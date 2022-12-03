import torch
from torchtext.legacy import data
from options import args_parser
from torchtext.legacy import datasets
import random
from model import get_model, get_keys, get_TEXT
from utilities import count_parameters, binary_accuracy, epoch_time
import torch.optim as optim
import torch.nn as nn
import time
import copy

from run_SA_single import evaluate, train
from utilities_data import LOAD_DATASET_FEDEATED
from algorithms_federated import ALGORITHMS, TO_CANDIDATE, train_over_keys
from scheduler import Scheduler

import numpy as np

def train_on_federated_datasets(args, model, clients_iterators):
    # ===== training =====
    print("#" * 10 + f" start training on {args.dataset} using {args.algorithm} " + "#" * 10)

    N_clients = len(clients_iterators)
    print("#" * 10 + f"There are {N_clients} clients. " + "#" * 10)

    n_sample_clients_train = np.asarray([len(client_iterators[0].dataset) for client_iterators in clients_iterators])
    n_sample_clients_test = np.asarray([len(client_iterators[2].dataset) for client_iterators in clients_iterators])
    # weight_clients = weight_clients / np.sum(weight_clients)

    criterion = nn.BCEWithLogitsLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    best_valid_loss = float('inf')

    sd_global = model.state_dict()
    sd_locals = [copy.deepcopy(sd_global) for _ in range(N_clients)]

    algorithm = ALGORITHMS[args.algorithm]
    to_candidate = TO_CANDIDATE[args.algorithm]
    scheduler_fl = Scheduler("exponential-fixed", args.scheduler, N_clients=N_clients, participating_rate=args.participating_rate,
                             N_init_clients=args.N_init_clients, double_every=args.double_every)
    for epoch in range(args.N_global_rounds):
        start_time = time.time()
        ### train ###
        train_loss = 0
        train_acc = 0

        index_activated = scheduler_fl.step()
        index_activated = np.sort(index_activated)
        # print(f"In epoch {epoch}, the activated clients are {index_activated}")

        clients_iterators_activated = [clients_iterators[index] for index in index_activated]
        sd_locals_activated = [sd_locals[index] for index in index_activated]

        weight_clients = n_sample_clients_train / np.sum(n_sample_clients_train[index_activated])

        for cid, client_iterators, sd_local in zip(index_activated, clients_iterators_activated, sd_locals_activated):
            train_iterator, _, _ = client_iterators

            sd_local, train_loss_local, train_acc_local = algorithm(model, sd_global, sd_local, train_iterator, criterion,
                                                                    args.N_local_epoch, lr=args.lr)
            train_loss += train_loss_local * weight_clients[cid]
            train_acc += train_acc_local * weight_clients[cid]
            sd_locals[cid] = sd_local

        for key in sd_global.keys():
            sd_global[key] = sd_global[key] * (1 - args.lr_g) \
                             + args.lr_g * torch.sum(torch.stack([sd_locals[cid][key] * weight_clients[cid] for cid in index_activated], dim=0), dim=0)

        torch.save(sd_global, 'tut2-model.pt')

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        ### validate ###
        valid_loss = 0
        valid_acc = 0
        if args.validate:
            weight_clients = n_sample_clients_test / np.sum(n_sample_clients_test)
            for cid, (client_iterators, sd_local) in enumerate(zip(clients_iterators, sd_locals)):
                # train_iterator, valid_iterator, _ = client_iterators
                train_iterator, _, valid_iterator = client_iterators

                sd_candidate = to_candidate(model, sd_global, sd_local)
                model.load_state_dict(sd_candidate)

                if args.N_ft_epoch > 0:
                    optimizer = optim.Adam(model.parameters(), lr=1e-2)
                    train_over_keys(model, train_iterator, optimizer, criterion, args.N_ft_epoch, model.head_keys)

                valid_loss_local, valid_acc_local = evaluate(model, valid_iterator, criterion)

                valid_loss += valid_loss_local * weight_clients[cid]
                valid_acc += valid_acc_local * weight_clients[cid]

        # if valid_loss < best_valid_loss or True:
        #     best_valid_loss = valid_loss
        #     torch.save(sd_global, 'tut2-model.pt')

        sd_global = torch.load('tut2-model.pt')

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s | Simulated time {scheduler_fl.total_simulated_time: .2f}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    ### test ###
    sd_test = torch.load('tut2-model.pt')
    test_loss = 0
    test_acc = 0
    weight_clients = n_sample_clients_test / np.sum(n_sample_clients_test)
    for cid, (client_iterators, sd_local) in enumerate(zip(clients_iterators, sd_locals)):
        train_iterator, _, test_iterator = client_iterators

        # model.load_state_dict(copy.deepcopy(sd_test))

        sd_candidate = to_candidate(model, sd_test, sd_local)
        model.load_state_dict(sd_candidate)

        if args.N_ft_epoch > 0:
            optimizer = optim.Adam(model.parameters(), lr=1e-2)
            train_over_keys(model, train_iterator, optimizer, criterion, args.N_ft_epoch, model.head_keys)

        test_loss_local, test_acc_local = evaluate(model, test_iterator, criterion)
        test_loss += test_loss_local * weight_clients[cid]
        test_acc += test_acc_local * weight_clients[cid]

    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')


def SA_federated(args, device):
    # prepare the iterators of the clients
    TEXT = get_TEXT(args)
    LABEL = data.LabelField(dtype=torch.float)

    load_dataset = LOAD_DATASET_FEDEATED[args.dataset]
    train_datasets_federated, valid_datasets_federated, test_datasets_federated = load_dataset(args, TEXT, LABEL)

    clients_iterators = []
    for train_data, valid_data, test_data in zip(train_datasets_federated, valid_datasets_federated,
                                                 test_datasets_federated):
        train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_size=args.BATCH_SIZE,
            sort_within_batch=True,
            device=device)
        clients_iterators.append((train_iterator, valid_iterator, test_iterator))

    model = get_model(args, TEXT)

    head_keys, representation_keys = get_keys(model)
    model.head_keys = head_keys
    model.representation_keys = representation_keys
    print(f"The head keys for fine-tuning are {model.head_keys}")

    pretrained_embeddings = TEXT.vocab.vectors
    print(pretrained_embeddings.shape)
    model.embedding.weight.data.copy_(pretrained_embeddings)

    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    model.embedding.weight.data[UNK_IDX] = torch.zeros(args.EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(args.EMBEDDING_DIM)

    # start training
    train_on_federated_datasets(args, model, clients_iterators)


if __name__ == '__main__':
    args = args_parser()

    random.seed(args.SEED)
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    SA_federated(args, device)
