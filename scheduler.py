from typing import List
import numpy as np


class Scheduler:
    def __init__(self, mode: str, strategy: str, N_clients: int, participating_rate: float = 1,
                 N_init_clients: int = 10, double_every: int = 1):
        self.N_clients = N_clients
        self.participating_rate = participating_rate
        self.N_activate = max(1, int(N_clients * participating_rate))

        if mode == "uniform":
            # mode == uniform: clients have a uniform running time
            self.simulated_running_time = np.ones(N_clients)
        elif mode == "exponential-fixed":
            # mode == exponential-fixed: clients have an exponentially distributed running time.
            self.simulated_running_time = np.random.exponential(1, N_clients)
        else:
            raise NotImplementedError

        self.strategy = strategy

        self.double_every = double_every # double m_flanp after every other "double_every" rounds
        self.double_counter = double_every
        self.m_flanp = min(N_init_clients, self.N_activate)

        self.total_simulated_time = 0

    def step(self) -> List[int]:
        # sample the activated clients
        index_activated = np.random.choice(self.N_clients, self.N_activate, replace=False)

        simulated_running_time_activated = self.simulated_running_time[index_activated]
        if self.strategy == "all":
            # strategy == all: all activated clients participate
            simulated_time = np.max(simulated_running_time_activated)
            index_participate = index_activated
        elif self.strategy == "flanp":
            # strategy == flanp: only the fastest activated clients participate
            self.m_flanp = min(self.m_flanp * 2, self.N_activate) if self.double_counter == 0 else self.m_flanp
            self.double_counter = self.double_every if self.double_counter == 0 else self.double_counter - 1
            running_time_ordering = np.argsort(simulated_running_time_activated)
            index_participate = index_activated[running_time_ordering[:self.m_flanp]]
            simulated_time = np.sort(simulated_running_time_activated)[self.m_flanp-1]
        else:
            raise NotImplementedError

        self.total_simulated_time = self.total_simulated_time + simulated_time

        return index_participate
