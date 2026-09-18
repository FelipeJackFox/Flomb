"""NumPy QR-DQN primitives; no environment or connectome side effects."""
from __future__ import annotations
import copy
import numpy as np


class DuelingQuantileHead:
    """Linear dueling head. Gradients are ordered Wv, bv, Wa, ba."""
    def __init__(self, features=1024, actions=256, quantiles=16, seed=0, lr=3e-4):
        self.features, self.actions, self.quantiles = features, actions, quantiles
        self.lr = float(lr)
        rng = np.random.default_rng(seed)
        scale = 0.01 / np.sqrt(features)
        self.params = [rng.normal(0, scale, (features, quantiles)).astype(np.float32),
                       np.zeros(quantiles, np.float32),
                       rng.normal(0, scale, (features, actions * quantiles)).astype(np.float32),
                       np.zeros(actions * quantiles, np.float32)]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.step = 0

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.features:
            raise ValueError('x must have shape [batch, features]')
        wv, bv, wa, ba = self.params
        value = x @ wv + bv
        advantage = (x @ wa + ba).reshape(-1, self.actions, self.quantiles)
        z = value[:, None, :] + advantage - advantage.mean(axis=1, keepdims=True)
        return z, x

    def backward(self, cache, dz):
        x = cache
        dz = np.asarray(dz, dtype=np.float32)
        if dz.shape != (len(x), self.actions, self.quantiles):
            raise ValueError('dz shape differs from output')
        dv = dz.sum(axis=1)
        da = (dz - dz.mean(axis=1, keepdims=True)).reshape(len(x), -1)
        dx = dv @ self.params[0].T + da @ self.params[2].T
        return dx, [x.T @ dv, dv.sum(axis=0), x.T @ da, da.sum(axis=0)]

    def apply(self, grads, clip_norm=10.0):
        if len(grads) != len(self.params):
            raise ValueError('four gradients required')
        norm = float(np.sqrt(sum(float(np.sum(np.square(g, dtype=np.float64))) for g in grads)))
        if not np.isfinite(norm):
            raise ValueError('nonfinite gradient')
        scale = min(1.0, float(clip_norm) / max(norm, 1e-12))
        self.step += 1
        for p, m, v, raw in zip(self.params, self.m, self.v, grads):
            if raw.shape != p.shape:
                raise ValueError('gradient shape mismatch')
            g = raw * scale
            m *= 0.9
            m += 0.1 * g
            v *= 0.999
            v += 0.001 * g * g
            p -= self.lr * (m / (1 - 0.9 ** self.step)) / (np.sqrt(v / (1 - 0.999 ** self.step)) + 1e-8)
        return norm

    def state_dict(self):
        result = {'features': np.array(self.features), 'actions': np.array(self.actions),
                  'quantiles': np.array(self.quantiles), 'lr': np.array(self.lr), 'step': np.array(self.step)}
        for name in ('params', 'm', 'v'):
            result.update({f'{name}_{i}': x.copy() for i, x in enumerate(getattr(self, name))})
        return result

    def load_state_dict(self, state):
        if tuple(int(state[k]) for k in ('features', 'actions', 'quantiles')) != (self.features, self.actions, self.quantiles):
            raise ValueError('head dimensions do not match')
        for name in ('params', 'm', 'v'):
            values = [np.asarray(state[f'{name}_{i}'], dtype=np.float32) for i in range(4)]
            if any(x.shape != y.shape for x, y in zip(values, getattr(self, name))):
                raise ValueError('head state shape mismatch')
            setattr(self, name, [x.copy() for x in values])
        self.lr, self.step = float(state['lr']), int(state['step'])

    def copy_from(self, other):
        self.load_state_dict(other.state_dict())
        return self

    def clone(self):
        return DuelingQuantileHead(self.features, self.actions, self.quantiles, lr=self.lr).copy_from(self)


def quantile_huber(pred, target, weights=None):
    """Pairwise QR Huber loss, mean over batch and both quantile axes.

    Error priorities are unweighted mean absolute pairwise TD errors.
    Targets are detached by caller; returned gradient is only wrt pred.
    """
    pred, target = np.asarray(pred), np.asarray(target)
    if pred.ndim != 2 or target.ndim != 2 or len(pred) != len(target):
        raise ValueError('pred and target must be [batch, quantiles]')
    b, n = pred.shape
    w = np.ones(b) if weights is None else np.asarray(weights)
    if w.shape != (b,) or b == 0:
        raise ValueError('weights must be [batch], nonempty batch')
    delta = target[:, None, :] - pred[:, :, None]
    abs_delta = np.abs(delta)
    huber = np.where(abs_delta <= 1, 0.5 * delta ** 2, abs_delta - 0.5)
    tau = (np.arange(n) + 0.5) / n
    asymmetry = np.abs(tau[None, :, None] - (delta < 0))
    per_sample = np.mean(asymmetry * huber, axis=(1, 2))
    grad = -np.sum(asymmetry * np.clip(delta, -1, 1), axis=2) * w[:, None] / (b * n * target.shape[1])
    return float(np.mean(per_sample * w)), grad.astype(pred.dtype), abs_delta.mean(axis=(1, 2))


def double_dqn_targets(rewards, dones, next_online, next_target, next_legal, gamma=0.99):
    """Select masked online-mean argmax, evaluate target distribution.

    gamma can be scalar or a [batch] n-step discount. Terminal rows may
    have no legal actions; nonterminal rows without legal actions are errors.
    """
    online, target = np.asarray(next_online), np.asarray(next_target)
    legal, done = np.asarray(next_legal, bool), np.asarray(dones, bool)
    rewards = np.asarray(rewards)
    b = len(online)
    if online.ndim != 3 or online.shape != target.shape or legal.shape != online.shape[:2] or done.shape != (b,) or rewards.shape != (b,):
        raise ValueError('inconsistent target shapes')
    if np.any(~done & ~legal.any(axis=1)):
        raise ValueError('nonterminal state has no legal action')
    selected = np.argmax(np.where(legal, online.mean(axis=2), -np.inf), axis=1)
    result = np.repeat(rewards[:, None], target.shape[2], axis=1).astype(np.result_type(target.dtype, np.float32))
    active = np.flatnonzero(~done)
    discount = np.broadcast_to(np.asarray(gamma), (b,))
    result[active] += discount[active, None] * target[active, selected[active], :]
    return result


class PrioritizedReplay:
    """Proportional PER, stratified sum-tree sampling; items are owned copies.

    add/update receive raw positive TD magnitudes (alpha applied internally).
    State is pickle-serializable, including item arrays and NumPy RNG state.
    """
    def __init__(self, capacity, seed=0, alpha=0.6, epsilon=1e-6):
        if capacity <= 0 or not 0 <= alpha <= 1 or epsilon <= 0:
            raise ValueError('invalid replay parameters')
        self.capacity, self.alpha, self.epsilon = int(capacity), float(alpha), float(epsilon)
        self.leaves = 1 << (self.capacity - 1).bit_length()
        self.tree = np.zeros(2 * self.leaves, dtype=np.float64)
        self.items = [None] * self.capacity
        self.position, self.size, self.max_priority = 0, 0, 1.0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def _set(self, index, priority):
        priority = max(abs(float(priority)), self.epsilon)
        if not np.isfinite(priority):
            raise ValueError('nonfinite replay priority')
        self.max_priority = max(self.max_priority, priority)
        node = self.leaves + index
        change = priority ** self.alpha - self.tree[node]
        while node:
            self.tree[node] += change
            node //= 2

    def add(self, item, priority=None):
        index = self.position
        self._set(index, self.max_priority if priority is None else priority)
        self.items[index] = copy.deepcopy(item)
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return index

    def sample(self, batch, beta=0.4):
        if not self.size or batch <= 0 or not 0 <= beta <= 1:
            raise ValueError('empty replay or invalid sampling arguments')
        total = self.tree[1]
        masses = (np.arange(batch) + self.rng.random(batch)) * (total / batch)
        indices = np.empty(batch, dtype=np.int64)
        for j, mass in enumerate(masses):
            node = 1
            while node < self.leaves:
                left = node * 2
                if mass < self.tree[left]:
                    node = left
                else:
                    mass -= self.tree[left]
                    node = left + 1
            indices[j] = node - self.leaves
        probs = self.tree[self.leaves + indices] / total
        weights = (self.size * probs) ** -beta
        weights /= weights.max()
        return [self.items[i] for i in indices], indices, weights.astype(np.float32)

    def update_priorities(self, indices, priorities):
        if len(indices) != len(priorities):
            raise ValueError('priority count mismatch')
        for i, p in zip(indices, priorities):
            if not 0 <= int(i) < self.size:
                raise IndexError('replay index out of range')
            self._set(int(i), p)

    def state_dict(self):
        return copy.deepcopy({k: getattr(self, k) for k in ('capacity', 'alpha', 'epsilon', 'leaves', 'tree', 'items', 'position', 'size', 'max_priority')} | {'rng_state': self.rng.bit_generator.state})

    def load_state_dict(self, state):
        if int(state['capacity']) != self.capacity:
            raise ValueError('replay capacity mismatch')
        for k in ('alpha', 'epsilon', 'leaves', 'tree', 'items', 'position', 'size', 'max_priority'):
            setattr(self, k, copy.deepcopy(state[k]))
        self.rng.bit_generator.state = copy.deepcopy(state['rng_state'])
