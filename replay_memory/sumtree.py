# Refrence
# https://github.com/rlcode/per

import numpy as np


class SumTree:
    position = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2*capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.leaf_start_index= capacity - 1
        self.n_entries = 0

    def _propagate(self, index, change):
        parent = (index-1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, index, s):
        left = 2*index + 1
        right = left + 1

        if left >= len(self.tree):
            return index

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left]) 
    
    def total(self):
        return self.tree[0]

    def add(self, p, data):
        index = self.position + self.leaf_start_index

        self.data[self.position] = data
        self.update(index, p)

        self.position += 1

        if self.position >= self.capacity:
            self.position = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1
        
    def update(self, index, p):
        change = p - self.tree[index]

        self.tree[index] = p
        self._propagate(index, change)
        
    def get(self, s):
        index = self._retrieve(0, s)
        data_index = index - self.capacity + 1

        return (index, self.tree[index], self.data[data_index])

    def max(self):
        return self.tree[self.leaf_start_index:].max()
    