from abc import abstractmethod


class Memory:
    def __init__(self, capacity):
        self.capacity = capacity

    @abstractmethod
    def push(self, *args):
        pass

    @abstractmethod
    def sample(self, batch_size):
        pass

    def update(self, index, loss):
        pass

    @abstractmethod
    def __len__(self):
        pass