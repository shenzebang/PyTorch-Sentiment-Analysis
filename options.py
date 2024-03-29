#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, choices=["imdb", "sent140", "sent140_homo"])
    parser.add_argument('--algorithm', type=str, choices=["fedavg-ft", "fedrep", "lg", "hfmaml", "fedavg"])
    parser.add_argument('--model', type=str, choices=["rnn", "cnn"])
    parser.add_argument('--mode', type=str, choices=["exponential-fixed", "fast-and-slow", "uniform"], default="exponential-fixed")


    parser.add_argument('--SEED', type=int, default=400)
    parser.add_argument('--EMBEDDING_DIM', type=int, default=100)
    parser.add_argument('--HIDDEN_DIM', type=int, default=256)
    parser.add_argument('--N_LAYERS', type=int, default=2)
    parser.add_argument('--N_clients', type=int, default=80)

    parser.add_argument('--N_local_epoch', type=int, default=5)
    parser.add_argument('--N_ft_epoch', type=int, default=2)
    parser.add_argument('--N_global_rounds', type=int, default=5)
    parser.add_argument('--participating_rate', type=float, default=.2)
    parser.add_argument('--BATCH_SIZE', type=int, default=10)
    parser.add_argument('--DROPOUT', type=float, default=.5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_g', type=float, default=1.)
    parser.add_argument('--MAX_VOCAB_SIZE', type=int, default=25000)
    parser.add_argument('--leaf_dir', type=str, default="/Github/leaf",
                        help="directory to the leaf repository (relative to ~)")

    parser.add_argument('--validate', action="store_true")

    parser.add_argument('--scheduler', type=str, choices=["all", "flanp"])
    parser.add_argument('--N_init_clients', type=int, default=40)
    parser.add_argument('--double_every', type=int, default=1)

    parser.add_argument('--description', type=str, default='')
    args = parser.parse_args()
    return args
