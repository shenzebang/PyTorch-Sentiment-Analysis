#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--SEED', type=int, default=400)
    parser.add_argument('--EMBEDDING_DIM', type=int, default=100)
    parser.add_argument('--HIDDEN_DIM', type=int, default=256)
    parser.add_argument('--N_LAYERS', type=int, default=2)
    parser.add_argument('--N_clients', type=int, default=10)
    parser.add_argument('--N_local_epoch', type=int, default=5)
    parser.add_argument('--N_ft_epoch', type=int, default=2)
    parser.add_argument('--N_global_rounds', type=int, default=5)
    parser.add_argument('--BATCH_SIZE', type=int, default=64)
    parser.add_argument('--DROPOUT', type=float, default=.5)
    parser.add_argument('--MAX_VOCAB_SIZE', type=int, default=25000)
    parser.add_argument('--leaf_dir', type=str, default="/Github/leaf",
                        help="directory to the leaf repository (relative to ~)")
    parser.add_argument('--dataset', type=str, choices=["imdb", "sent140"])
    parser.add_argument('--algorithm', type=str, choices=["fedavg", "fedrep"])

    args = parser.parse_args()
    return args
