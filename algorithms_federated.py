from run_SA_single import train
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

    return train(model, iterator, optimizer, criterion, N_epoch)


def fedavg(model: nn.Module, sd_global, sd_local, train_iterator, criterion, N_local_epoch, lr=1e-3):
    sd_local = copy.deepcopy(sd_global)
    model.load_state_dict(sd_local)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    all_keys = model.representation_keys + model.head_keys
    return train_over_keys(model, train_iterator, optimizer, criterion, N_local_epoch, all_keys)


def fedrep(model: nn.Module, sd_global, sd_local, train_iterator, criterion, N_local_epoch, N_head_epoch=5, lr=1e-3):
    _sd_local = copy.deepcopy(sd_global)
    for key in model.head_keys:
        _sd_local[key] = sd_local[key]

    model.load_state_dict(_sd_local)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_over_keys(model, train_iterator, optimizer, criterion, N_head_epoch, model.head_keys)

    optimizer = optim.Adam(model.parameters())
    return train_over_keys(model, train_iterator, optimizer, criterion, N_local_epoch, model.representation_keys)



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
              }

TO_CANDIDATE = {'fedavg': to_candidate_fedavg,
                'fedrep': to_candidate_fedrep,
}