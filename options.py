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
    parser.add_argument('--BATCH_SIZE', type=int, default=64)
    parser.add_argument('--DROPOUT', type=float, default=.5)
    parser.add_argument('--MAX_VOCAB_SIZE', type=int, default=25000)

    args = parser.parse_args()
    return args
