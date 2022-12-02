from torchtext import legacy
from utilities import splits_federated
import torch
import json
from itertools import chain
import os

import re

def load_IMDB_federated(args, TEXT, LABEL):
    train_data, test_data = legacy.datasets.IMDB.splits(TEXT, LABEL)

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

    return train_datasets_federated, valid_datasets_federated, test_datasets_federated


digit_to_pos_neg = {0: 'neg', 1: 'pos'}


def remove_links_mentions(tweet):
    https_link_re_pattern = "https?:\/\/t.co/[\w]+"
    http_link_re_pattern = "http?:\/\/t.co/[\w]+"
    mention_re_pattern = "@\w+"
    tweet = re.sub(http_link_re_pattern, "", tweet)
    tweet = re.sub(https_link_re_pattern, "", tweet)
    tweet = re.sub(mention_re_pattern, "", tweet)
    return tweet.lower()

class SENT140_SINGLE_USER(legacy.data.Dataset):
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, user_data_and_label, text_field, label_field, **kwargs):
        fields = [('text', text_field), ('label', label_field)]

        examples = self.to_examples(user_data_and_label, fields)

        super(SENT140_SINGLE_USER, self).__init__(examples, fields, **kwargs)

    def to_examples(self, user_data_and_label, fields):
        user_datas = user_data_and_label['x']
        user_labels = user_data_and_label['y']
        user_examples = []
        for user_data, user_label in zip(user_datas, user_labels):
            user_examples.append(legacy.data.Example.fromlist([remove_links_mentions(user_data[4]), digit_to_pos_neg[user_label]], fields))

        return user_examples

def load_sent140_federated(args, TEXT, LABEL):
    sent140_dir = os.path.expanduser('~') + args.leaf_dir + "/data/sent140/data/"

    sent140_dir_train = sent140_dir + "train/all_data_niid_3_keep_50_train_9.json"
    sent140_dir_test = sent140_dir + "test/all_data_niid_3_keep_50_test_9.json"

    data_train = json.load(open(sent140_dir_train))
    data_test = json.load(open(sent140_dir_test))

    user_names = data_train['users']
    _sent140_multiple_users_train = []
    sent140_multiple_users_test = []
    for uid, user_name in enumerate(user_names):
        user_data_and_label_train = data_train['user_data'][user_name]
        user_data_and_label_test = data_test['user_data'][user_name]
        _sent140_multiple_users_train.append(SENT140_SINGLE_USER(user_data_and_label_train, TEXT, LABEL))
        sent140_multiple_users_test.append(SENT140_SINGLE_USER(user_data_and_label_test, TEXT, LABEL))

    all_examples = list(chain.from_iterable([user.examples for user in _sent140_multiple_users_train]))
    sent140_global_train = legacy.data.Dataset(all_examples, _sent140_multiple_users_train[0].fields)

    TEXT.build_vocab(sent140_global_train,
                     max_size=25000,
                     vectors="glove.6B.100d",
                     unk_init=torch.Tensor.normal_)

    LABEL.build_vocab(_sent140_multiple_users_train[0])

    print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
    print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")

    sent140_multiple_users_valid = []
    sent140_multiple_users_train = []
    for uid, sent140_single_user_train in enumerate(_sent140_multiple_users_train):
        train_data, valid_data = sent140_single_user_train.split(0.9)
        sent140_multiple_users_train.append(train_data)
        sent140_multiple_users_valid.append(valid_data)

    return sent140_multiple_users_train, sent140_multiple_users_valid, sent140_multiple_users_test




LOAD_DATASET_FEDEATED = {
    'imdb': load_IMDB_federated,
    'sent140': load_sent140_federated
}