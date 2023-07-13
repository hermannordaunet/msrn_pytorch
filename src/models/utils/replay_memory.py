import numpy as np

from collections import namedtuple

# Local imports
from visualization.visualize import plot_grid_based_perception

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
        # plot_grid_based_perception(*args[0], title="state inside memory add")
        self.position = (
            self.position + 1
        ) % self.capacity  # e.g if the capacity is 100, and our position is now 101, we don't append to
        # position 101 (impossible), but to position 1 (its remainder), overwriting old data

    def sample(self, indexes=None):
        if indexes is None:
            indexes = np.random.choice(
                range(len(self.memory)), self.batch_size, replace=False
            )
        # return np.random.choice(self.memory, size=self.batch_size)

        # CRITICAL: Think this is a bit to slow
        experiences = [self.memory[i] for i in indexes]
        # experiences = (self.memory[i] for i in indexes)
        # experiences = np.array(self.memory)[indexes]
        # experiences = np.array(self.memory, dtype=object)[indexes]

        return experiences

    def __len__(self):
        return len(self.memory)
