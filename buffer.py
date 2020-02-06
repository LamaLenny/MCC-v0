import random


class Buffer:
    def __init__(self, cap):
        self.cap = cap
        self.mem = []
        self.pos = -1  # позиция последнего записанного элемента

    def __len__(self):
        return len(self.mem)

    def add(self, element):
        if len(self.mem) < self.cap:
            self.mem.append(None)
        new_pos = (self.pos + 1) % self.cap
        self.mem[new_pos] = element
        self.pos = new_pos

    def sample(self, batch_size):
        return random.sample(self.mem, batch_size)

    def __getitem__(self, k):
        return self.mem[(self.pos + 1 + k) % self.cap]