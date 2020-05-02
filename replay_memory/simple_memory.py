import sys
import numpy as np
sys.path.append(__file__)
import random
from collections import namedtuple
from base_memory import Memory
from replay_memory import Transition


class SimpleMemory(Memory):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.randint(0, len(self), size=(batch_size, ))
        batch = [self.memory[i] for i in indices]
        weights = np.ones(shape=(batch_size, ))
        return batch, indices, weights

    def __len__(self):
        return len(self.memory)