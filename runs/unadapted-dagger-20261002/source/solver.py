"""Visible-only reference. No environment object is accepted by this module."""
from dataclasses import dataclass
from itertools import product
from math import comb
import numpy as np


@dataclass
class Analysis:
    safe: set
    mines: set
    mine_probability: dict
    probability_method: str

    def classify(self, action):
        if action in self.safe:
            return 'proven_safe'
        if action in self.mines:
            return 'proven_mine'
        return 'guess'


def deduce(constraints):
    """Sound zero/full and subset deductions; bounded closure may miss deductions."""
    safe, mines = set(), set()
    current = {frozenset(c): int(n) for c, n in constraints if c}
    for _ in range(64):
        reduced = {}
        for cells, n in current.items():
            n -= len(cells & mines)
            cells = cells - safe - mines
            if n < 0 or n > len(cells):
                raise ValueError('Inconsistent visible constraints')
            if cells:
                if cells in reduced and reduced[cells] != n:
                    raise ValueError('Inconsistent visible constraints')
                reduced[cells] = n
        ns = set().union(*(set(c) for c, n in reduced.items() if n == 0))
        nm = set().union(*(set(c) for c, n in reduced.items() if n == len(c)))
        if ns & nm:
            raise ValueError('Contradictory deductions')
        if ns or nm:
            safe |= ns
            mines |= nm
            current = reduced
            continue
        additions = {}
        items = list(reduced.items())
        for a, na in items:
            for b, nb in items:
                if a < b:
                    diff = b - a
                    if diff not in reduced and (len(diff) <= 8 or nb - na in (0, len(diff))):
                        additions[diff] = nb - na
                        if len(reduced) + len(additions) >= 512:
                            break
            if len(reduced) + len(additions) >= 512:
                break
        if not additions:
            return safe, mines, reduced
        current = reduced | additions
    return safe, mines, current


def analyze(observation, legal_mask, size, mine_count, exact_limit=12):
    obs = np.asarray(observation).reshape(size * size, 10)
    legal = np.asarray(legal_mask, dtype=bool).reshape(size * size)
    if not np.all((obs == 0) | (obs == 1)) or not np.all(obs.sum(axis=1) == 1):
        raise ValueError('Expected unpadded one-hot visible observation')
    visible = obs.argmax(axis=1) - 1
    if not np.array_equal(legal, visible == -1):
        raise ValueError('Legal mask must match hidden cells')
    hidden = set(map(int, np.flatnonzero(legal)))
    constraints = [(hidden, mine_count)]
    for i in np.flatnonzero(~legal):
        r, c = divmod(int(i), size)
        neighbors = {rr*size+cc for rr in range(max(0, r-1), min(size, r+2))
                     for cc in range(max(0, c-1), min(size, c+2))
                     if (rr, cc) != (r, c)} & hidden
        if neighbors:
            constraints.append((neighbors, int(visible[i])))
        elif visible[i] != 0:
            raise ValueError('Invalid revealed clue')
    safe, mines, _ = deduce(constraints)
    unknown = hidden - safe - mines
    remaining = mine_count - len(mines)
    local = [(frozenset(c) - safe - mines, n - len(set(c) & mines))
             for c, n in constraints[1:]]
    local = [(c, n) for c, n in local if c]
    frontier = sorted(set().union(*(set(c) for c, _ in local)))
    outside = unknown - set(frontier)
    probabilities = {i: 0. for i in safe} | {i: 1. for i in mines}
    method = 'global_density_heuristic'
    if len(frontier) <= exact_limit:
        indices = {cell: j for j, cell in enumerate(frontier)}
        checks = [([indices[c] for c in cells], n) for cells, n in local]
        total, weighted_outside = 0, 0
        weighted = [0] * len(frontier)
        for bits in product((0, 1), repeat=len(frontier)):
            rest = remaining - sum(bits)
            if not 0 <= rest <= len(outside):
                continue
            if any(sum(bits[j] for j in js) != n for js, n in checks):
                continue
            weight = comb(len(outside), rest)
            total += weight
            weighted_outside += weight * rest
            for j, bit in enumerate(bits):
                weighted[j] += weight * bit
        if not total:
            raise ValueError('No consistent mine placements')
        for j, cell in enumerate(frontier):
            probabilities[cell] = weighted[j] / total
            if weighted[j] == 0:
                safe.add(cell)
            elif weighted[j] == total:
                mines.add(cell)
        for cell in outside:
            probabilities[cell] = weighted_outside / (total * len(outside))
            if weighted_outside == 0:
                safe.add(cell)
            elif weighted_outside == total * len(outside):
                mines.add(cell)
        method = 'exact_uniform_consistent_boards'
    else:
        probabilities.update({i: remaining / len(unknown) for i in unknown})
    return Analysis(safe, mines, probabilities, method)


def choose(observation, legal_mask, size, mine_count, rng, exact_limit=12):
    info = analyze(observation, legal_mask, size, mine_count, exact_limit)
    candidates = sorted(info.safe)
    if not candidates:
        possible = sorted(set(map(int, np.flatnonzero(legal_mask))) - info.mines)
        if not possible:
            raise ValueError('No non-mine legal action; episode should have ended')
        minimum = min(info.mine_probability[i] for i in possible)
        candidates = [i for i in possible if info.mine_probability[i] <= minimum + 1e-12]
    action = int(rng.choice(candidates))
    return action, info
