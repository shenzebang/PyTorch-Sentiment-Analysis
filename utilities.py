import torch
from torchtext.legacy import data
from torchtext.legacy.data import Dataset
from typing import Tuple
from torchtext.data.utils import RandomShuffler
import numpy as np
import time


TEXT = data.Field(tokenize='spacy',
                      tokenizer_language='en_core_web_sm',
                      include_lengths=True)

LABEL = data.LabelField(dtype=torch.float)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc

def splits_federated(dataset: Dataset, N_clients: int) -> Tuple[Dataset]:
    examples = dataset.examples
    N_examples = len(examples)

    rng = RandomShuffler()

    randperm = rng(range(N_examples))

    # decide the number of examples per client
    len_per_clients = [N_examples // N_clients] * N_clients
    for i in range(N_examples % N_clients):
        len_per_clients[i] += 1

    indices = []
    i_start = 0
    for len_per_client in len_per_clients:
        indices.append(randperm[i_start: i_start+len_per_client])
        i_start += len_per_client

    datasets = tuple([examples[i] for i in index] for index in indices)

    # convert to an instance of Dataset
    splits = [Dataset(d, dataset.fields) for d in datasets]
    for split in splits:
        split.sort_key = dataset.sort_key

    splits = tuple(splits)

    return splits
