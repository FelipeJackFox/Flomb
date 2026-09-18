"""Small first-click-safe Minesweeper. Policies receive only visible cells."""
import numpy as np


class Minesweeper:
    def __init__(self, seed, size=5, mines=3):
        if not 0 < mines < size * size - 1:
            raise ValueError('Invalid mine count')
        self.size, self.mine_count = size, mines
        self.visible = np.full(size * size, -1, dtype=np.int8)
        self._mines = np.zeros(size * size, dtype=bool)
        self.done, self.won = False, False
        first = (size // 2) * size + size // 2
        candidates = np.delete(np.arange(size * size), first)
        self._mines[np.random.default_rng(seed).choice(candidates, mines, replace=False)] = True
        self._counts = np.array([sum(self._mines[j] for j in self.neighbors(i))
                                 for i in range(size * size)], dtype=np.int8)
        self.step(first)  # Identical automatic center opening for every policy.

    def neighbors(self, i):
        r, c = divmod(int(i), self.size)
        return [rr * self.size + cc
                for rr in range(max(0, r - 1), min(self.size, r + 2))
                for cc in range(max(0, c - 1), min(self.size, c + 2))
                if (rr, cc) != (r, c)]

    def observation(self):
        # 0 = hidden; 1..9 = revealed counts 0..8. Never access _mines here.
        return np.eye(10, dtype=np.float32)[self.visible + 1].reshape(-1)

    def legal_mask(self):
        return self.visible == -1

    def step(self, action):
        action = int(action)
        if self.done or not 0 <= action < self.size ** 2 or self.visible[action] != -1:
            raise ValueError('Illegal action')
        if self._mines[action]:
            self.done = True
            return -1.0
        before = int(np.sum(self.visible >= 0))
        pending = [action]
        while pending:
            i = pending.pop()
            if self.visible[i] >= 0 or self._mines[i]:
                continue
            self.visible[i] = self._counts[i]
            if self._counts[i] == 0:
                pending.extend(self.neighbors(i))
        self.won = int(np.sum(self.visible >= 0)) == self.size ** 2 - self.mine_count
        self.done = self.won
        revealed = int(np.sum(self.visible >= 0)) - before
        return (1.0 if self.won else 0.0) + 0.05 * revealed
