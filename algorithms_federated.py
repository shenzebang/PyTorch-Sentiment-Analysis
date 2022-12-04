from run_SA_single import train
import torch
import torch.nn as nn
import torch.optim as optim
import copy

def train_over_keys(model, iterator, optimizer, criterion, N_epoch, keys):
    if N_epoch == 0:
        return model.state_dict(), -1, -1

    for name, param in model.named_parameters():
        if name in keys:
            param.requires_grad = True
        else:
            param.requires_grad = False

    new_sd, loss, acc = train(model, iterator, optimizer, criterion, N_epoch)

    return new_sd, loss, acc


def fedavg(model: nn.Module, train_iterator, criterion, N_local_epoch, lr=1e-2):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    all_keys = model.ft_keys + model.non_ft_keys
    new_sd, loss, acc = train_over_keys(model, train_iterator, optimizer, criterion, N_local_epoch, all_keys)

    return new_sd, loss, acc

def fedrep(model: nn.Module, train_iterator, criterion, N_local_epoch, N_head_epoch=5, lr=1e-2):
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    train_over_keys(model, train_iterator, optimizer, criterion, N_head_epoch, model.ft_keys)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=.5)
    sd_new, loss, acc = train_over_keys(model, train_iterator, optimizer, criterion, N_local_epoch, model.non_ft_keys)

    return sd_new, loss, acc


def to_candidate_fedavg(model, sd_global, sd_local):
    for key in model.representation_keys:
        sd_local[key] = sd_global[key]
    return sd_local

def to_candidate_fedrep(model, sd_global, sd_local):
    for key in model.head_keys:
        sd_global[key] = copy.deepcopy(sd_local[key])
    return sd_global

ALGORITHMS = {'fedavg': fedavg,
              'fedrep': fedrep,
              'lg'    : fedrep,
              }

TO_CANDIDATE = {'fedavg': to_candidate_fedavg,
                'fedrep': to_candidate_fedrep,
}