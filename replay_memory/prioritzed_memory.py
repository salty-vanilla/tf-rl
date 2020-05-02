import sys
import numpy as np
sys.path.append(__file__)
import random
from collections import namedtuple
from sumtree import SumTree
from base_memory import Memory
from replay_memory import Transition


class PrioritizedMemory(Memory):
    def __init__(self, capacity,
                 epsilon=0.01,
                 alpha=0.6,
                 beta=0.4,
                 beta_increment=0.001):
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta                       
        self.beta_increment = beta_increment

        self.capacity = capacity
        self.tree = SumTree(self.capacity)

    def _compute_priority(self, loss):
        return (np.abs(loss) + self.epsilon) ** self.alpha

    def push(self, *args):
        priority = self.tree.max()
        priority = 1 if priority <= 0 else priority
        self.tree.add(priority, Transition(*args))

    def sample(self, batch_size):
        batch = []
        indices = []
        weights = np.empty(batch_size, dtype='float32')
        self.beta += self.beta_increment
        beta = np.minimum(1., self.beta)
        total = self.tree.total()
        for i, r in enumerate(np.random.uniform(0, total, (batch_size, ))):
            index, priority, data = self.tree.get(r)
            batch.append(data)
            indices.append(index)
            weights[i] = (self.capacity*priority/total) ** (-beta)

        return batch, indices, weights/weights.max()

    def update(self, index, loss):
        priority = self._compute_priority(loss)
        self.tree.update(index, priority)

    def __len__(self):
        return self.tree.n_entries

