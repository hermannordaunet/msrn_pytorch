import random

import numpy as np

from collections import namedtuple

# Memory for Experience Replay
class ReplayMemory(object):
    def __init__(self, capacity, batch_size=32):
        self.capacity = capacity
        self.batch_size = batch_size
        self.memory = list()
        self.position = 0

        self.Transition = namedtuple(
            "Transition", ("state", "action", "reward", "next_state", "done")
        )

    def add(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(
                None
            )  # if we haven't reached full capacity, we append a new transition
        self.memory[self.position] = self.Transition(*args)
        self.position = (
            self.position + 1
        ) % self.capacity  # e.g if the capacity is 100, and our position is now 101, we don't append to
        # position 101 (impossible), but to position 1 (its remainder), overwriting old data


    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(
                None
            )  # if we haven't reached full capacity, we append a new transition
        self.memory[self.position] = self.Transition(*args)
        self.position = (
            self.position + 1
        ) % self.capacity  # e.g if the capacity is 100, and our position is now 101, we don't append to
        # position 101 (impossible), but to position 1 (its remainder), overwriting old data

    def sample(self):
        indexes = np.random.choice(range(len(self.memory)), self.batch_size, replace=False)
        # return np.random.choice(self.memory, size=self.batch_size)

        experiences = [self.memory[i] for i in indexes]

        return experiences

    def __len__(self):
        return len(self.memory)