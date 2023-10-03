import matplotlib.pyplot as plt
import matplotlib as mpl

import csv
import numpy as np

import os

mpl.rcParams['savefig.pad_inches'] = 0

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 25

def find_first(xs, target):
    for i in range(xs.shape[0]):
        if xs[i] > target:
            return i


# settings = (
#     (2, 100, "cifar10"),
#     (2, 500, "cifar10"),
#     (5, 100, "cifar10"),
#     (5, 100, "cifar100"),
#     (20, 100, "cifar100"),
# )

settings = (
    # (5, 100, "cifar10"),
    (2, 83, "sent140"),
)

# settings = (
#     # (5, 500),
#     # (2, 500),
#     (2, 94, "emnist"),
#     (5, 94, "emnist"),
#     (2, 470, "emnist"),
#     (5, 470, "emnist")
# )


# settings = (
#     (2, 100, "mnist"),
# )
# datasets = ["cifar100"]
# communication_times = (0, 1000, 10000, 100000)
# communication_times = (0, 1, 10, 100)
# shards = [2, 5]
# methods = ["ffgd", "fed-avg", "scaffold"]
# methods = ["fedrep-double-10-10-s1", "fedrep-full-s1", "fedavg-double-10-10-s1", "fedavg-full-s1"]
# methods = ["fedrep-full-s1", "fedavg-double-10-10-s1", "fedavg-full-s1"]
# methods = ["fedavg-double-10-10-s1-FT", "fedrep-double-10-10-s1-FT", "fedrep-full-s1-FT", "fedavg-full-s1-FT",
#                 "fedavg-double-10-10-s1-global", "fedavg-full-s1-global"]

# methods = ["flanp"]
# methods = ["FedRep"]
# methods_label = {"flanp" : "FLANP", "FedRep": "FedRep"}
colors = {
    'lg_flanp'      : 'blue',
    'lg_all'        : 'darkblue',
    'fedrep_flanp'  : 'red',
    'fedrep_all'    : 'darkred'
}
first = True
methods = [f"fedrep_flanp", f"fedrep_all", f"lg_flanp", f"lg_all",]
fast_slow_ratios = ['0.9', '0.7']
communication_times = (0, 1, 10, 100)
for communication_time in communication_times:
    for shard, N, dataset in settings:
        min_comm = 10000000
        for method in methods:
            color = colors[method]
            for fast_slow_ratio in fast_slow_ratios:
                file_name = f'save/fast-and-slow-{fast_slow_ratio}/sent140_{method}_fast-and-slow_.csv'
                i = 0
                comm = []
                accu = []
                with open(file_name, 'r') as csvfile:
                    plots = csv.reader(csvfile, delimiter=',')
                    for idx, row in enumerate(plots):
                        if i == 0: # ignore the title (the first row)
                            i += 1
                            continue
                        comm.append(float(row[0]) + idx * communication_time)
                        accu.append(float(row[1]))
                    min_comm = min(min_comm, max(comm))

                accu = np.asarray(accu)
                comm = np.asarray(comm)
                linestyle = 'solid' if fast_slow_ratio == fast_slow_ratios[0] else 'dashed'
                slow_ratio = {'0.9': '10%', '0.7': '30%'}
                if 'fedrep' in method:
                    if 'flanp' in method:
                        label = f'FedRep-SRPFL-{slow_ratio[fast_slow_ratio]}'
                    else:
                        label = f'FedRep-{slow_ratio[fast_slow_ratio]}'
                        # accu_mean -= 10
                elif 'fedavg' in method:
                    if 'flanp' in method:
                        label = 'FLANP'
                    else:
                        label = 'FedAvg'
                    if 'ft' in method:
                        label += '-FT'
                    # else:
                        # label += '-global'
                elif 'lg' in method:
                    if 'flanp' in method:
                        label = f'LG-SRPFL-{slow_ratio[fast_slow_ratio]}'
                    else:
                        label = f'LG-FedAvg-{slow_ratio[fast_slow_ratio]}'
                        # accu_mean -= 10
                elif 'hfmaml' in method:
                    label = 'HFMAML'
                else:
                    raise NotImplementedError


                plt.plot(comm, accu, label=label, color=color, linewidth=5.0, linestyle=linestyle)

            # plt.fill_between(comm, accu_mean - accu_std, accu_mean + accu_std, facecolor=color, alpha=0.15)
            # print(comm[find_first(accu_mean, 0.495)])
            # plt.plot(comm[1:], accu[1:], label=method)

            # ax = plt.gca()



        plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        plt.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
        # plt.ticklabel_format(axis='x', style='sci',)
        plt.xticks(np.arange(0, int(min_comm)+1, int(min_comm/4)).astype(int), size=BIGGER_SIZE)
        plt.yticks(np.arange(start=0.6, stop=0.83, step=0.05), size=BIGGER_SIZE)
        plt.xlabel(f'Stragglers % (C.T. = {communication_time})', size=BIGGER_SIZE)
        plt.xlim(0, min_comm)
        plt.ylabel('Testing Accuracy', size=BIGGER_SIZE)
        if dataset == 'cifar10':
            _dataset = 'CIFAR10'
        elif dataset == 'cifar100':
            _dataset = 'CIFAR100'
        elif dataset == 'emnist':
            _dataset = 'EMNIST'
        elif dataset == 'sent140':
            _dataset = 'SENT140'
        else:
            raise NotImplementedError
        plt.title(f'{_dataset}, M={N}, Shard={shard}', size=BIGGER_SIZE)
        # plt.xticks()
        plt.yticks(size=BIGGER_SIZE)
        if first:
            plt.legend()
            plt.legend(fontsize='x-large')
            first = False
        # plt.yticks(np.arange(0, 11, 400))
        # plt.xticks(np.arange(0, 2001, 400))

        save_directory = './plot/'
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        plt.savefig(f'plot/test_acc_time_{dataset}_{N}_shard{shard}_fast-and-slow_{communication_time}.pdf',
                        bbox_inches='tight')
        plt.show()


# plt.errorbar(ffgd_comm[1:], ffgd_accu[1:], yerr=ffgd_yerr[1:], label='ffgd-0.3')
# plt.plot(sgd_time[1:], sgd_loss[1:], label='Adam')
# ax.set_yscale('log')
# plt.yscale('log')


