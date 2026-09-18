"""Deterministic easy/medium/large schedule and visible-only fixed-canvas adapter."""
import numpy as np

GROUPS = ('small', 'medium', 'large')
SIZES = {'small': (5, 7), 'medium': (9, 12), 'large': (16,)}


def mixture(episode, total):
    if total < 2 or not 0 <= episode < total:
        raise ValueError('Invalid curriculum position')
    t = min(1., episode / (0.7 * (total - 1)))
    return np.array([.6 * (1-t), .3 * (1-t), .1 + .9*t])


def draw_board(rng, episode, total):
    probs = mixture(episode, total)
    group = GROUPS[int(rng.choice(3, p=probs))]
    size = int(rng.choice(SIZES[group]))
    low, high = {'small': (.08, .16), 'medium': (.10, .19), 'large': (.12, .22)}[group]
    mines = int(rng.integers(max(1, round(size*size*low)), round(size*size*high)+1))
    return group, size, mines, probs


def encode(env, canvas=16):
    if env.size > canvas:
        raise ValueError('Board exceeds canvas')
    # Padding is all zeros, distinct from the one-hot hidden-cell channel.
    board = np.zeros((canvas, canvas, 10), np.float32)
    board[:env.size, :env.size] = env.observation().reshape(env.size, env.size, 10)
    mask = np.zeros((canvas, canvas), bool)
    mask[:env.size, :env.size] = env.legal_mask().reshape(env.size, env.size)
    return board.reshape(-1), mask.reshape(-1)


def decode(action, size, canvas=16):
    r, c = divmod(int(action), canvas)
    if not (0 <= r < size and 0 <= c < size):
        raise ValueError('Action outside real board')
    return r*size+c


def reward_step(env, action):
    before = int(np.sum(env.visible >= 0))
    original_reward = env.step(action)
    if original_reward == -1:
        return -1.
    revealed = int(np.sum(env.visible >= 0)) - before
    # Total shaping <= .25 per game, irrespective of board area.
    return float(env.won) + .25 * revealed / (env.size**2 - env.mine_count)
