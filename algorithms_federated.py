from run_SA_single import train
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from collections import OrderedDict
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

def fedrep(model: nn.Module, train_iterator, criterion, N_local_epoch, lr=1e-2):
    N_head_epoch = 5
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    train_over_keys(model, train_iterator, optimizer, criterion, N_head_epoch, model.ft_keys)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=.5)
    sd_new, loss, acc = train_over_keys(model, train_iterator, optimizer, criterion, N_local_epoch, model.non_ft_keys)

    return sd_new, loss, acc

HFMAML_alpha = .005
HFMAML_delta = .001

def hfmaml(model: nn.Module, train_iterator, criterion, N_local_epoch, lr=1e-2):

    for ep in range(N_local_epoch):
        for batch in train_iterator:
            sd = model.state_dict()
            model.zero_grad()
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
            loss = criterion(predictions, batch.label)
            grad_at_sd = torch.autograd.grad(loss, tuple(model.parameters()))
            sd_temp = OrderedDict() # sd_temp is sd after a gradient step
            for kid, key in enumerate(sd.keys()):
                sd_temp[key] = sd[key] - HFMAML_alpha * grad_at_sd[kid]

            # compute the gradient at sd_temp
            model.load_state_dict(sd_temp)
            model.zero_grad()
            predictions = model(text, text_lengths).squeeze(1)
            loss = criterion(predictions, batch.label)
            grad_at_sd_temp = torch.autograd.grad(loss, tuple(model.parameters()))

            # compute the Hessian-vector product via gradient difference

            # named_param_pg is param + delta * grad_at_named_param_temp
            sd_pg = OrderedDict()
            for kid, key in enumerate(sd.keys()):
                sd_pg[key] = sd[key] + HFMAML_delta * grad_at_sd_temp[kid]
            model.load_state_dict(sd_pg)
            model.zero_grad()
            predictions = model(text, text_lengths).squeeze(1)
            loss = criterion(predictions, batch.label)
            grad_pg = torch.autograd.grad(loss, tuple(model.parameters()))

            # named_param_mg is param - delta * grad_at_named_param_temp
            sd_mg = OrderedDict()
            for kid, key in enumerate(sd.keys()):
                sd_mg[key] = sd[key] - HFMAML_delta * grad_at_sd_temp[kid]
            model.load_state_dict(sd_mg)
            model.zero_grad()
            predictions = model(text, text_lengths).squeeze(1)
            loss = criterion(predictions, batch.label)
            grad_mg = torch.autograd.grad(loss, tuple(model.parameters()))

            # hvp: hessian vector product
            hvp = OrderedDict()
            for kid, key in enumerate(sd.keys()):
                hvp[key] = (grad_pg[kid] - grad_mg[kid]) / 2. / HFMAML_delta


            # update the variable
            for kid, key in enumerate(sd.keys()):
                sd[key] = sd[key] - lr * (grad_at_sd_temp[kid] - HFMAML_alpha * hvp[key])

            model.load_state_dict(sd)

    return model.state_dict(), 0, 0



def adapt(model, train_iterator, criterion, N_ft_epoch):
    if N_ft_epoch > 0:
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        train_over_keys(model, train_iterator, optimizer, criterion, N_ft_epoch, model.ft_keys)

def no_adapt(model, train_iterator, criterion, N_ft_epoch):
    pass

def adapt_hfmaml(model, train_iterator, criterion, N_ft_epoch):
    if N_ft_epoch > 0:
        sd = model.state_dict()
        grad_at_sd = tuple()
        for batch in train_iterator:
            model.zero_grad()
            text, text_lengths = batch.text

            predictions = model(text, text_lengths).squeeze(1)
            loss = criterion(predictions, batch.label)
            grad_at_sd = torch.autograd.grad(loss, tuple(model.parameters()))
            break

        sd_new = OrderedDict()
        for kid, key in enumerate(sd.keys()):
            sd_new[key] = sd[key] - HFMAML_alpha * grad_at_sd[kid]

        model.load_state_dict(sd_new)

def to_candidate_fedavg(model, sd_global, sd_local):
    for key in model.representation_keys:
        sd_local[key] = sd_global[key]
    return sd_local

def to_candidate_fedrep(model, sd_global, sd_local):
    for key in model.head_keys:
        sd_global[key] = copy.deepcopy(sd_local[key])
    return sd_global

ALGORITHMS = {
    'fedavg'    : fedavg,
    'fedavg-ft' : fedavg,
    'fedrep'    : fedrep,
    'lg'        : fedrep,
    'hfmaml'    : hfmaml,
}


ADAPT = {
    'fedavg'    : no_adapt,
    'fedavg-ft' : adapt,
    'fedrep'    : adapt,
    'lg'        : adapt,
    'hfmaml'    : adapt_hfmaml,
}
