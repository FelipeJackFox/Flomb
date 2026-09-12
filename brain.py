"""Sparse rate-network policy. Anatomical graph; artificial input/output interfaces.

One learned positive gain per presynaptic neuron scales its outgoing edges.
No claim of spiking physiology, individual-synapse learning, or biological I/O.
"""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy import sparse


class BrainPolicy:
    def __init__(self, graph, seed=17, inputs=250, actions=25, readouts=1024, cycles=3, sparse_workers=1):
        self.graph = graph
        self.transpose = graph.T.tocsr()
        self._pool = None
        self.configure_compute(sparse_workers)
        self.cycles = cycles
        n = graph.shape[0]
        rng = np.random.default_rng(seed)
        self.input_index = rng.integers(inputs, size=(n, 4), dtype=np.int32)
        self.input_sign = rng.choice(np.array([-1., 1.], np.float32), size=(n, 4))
        self.extra_input_index = np.empty((n, 0), np.int32)
        self.extra_input_sign = np.empty((n, 0), np.float32)
        eligible = np.flatnonzero(np.diff(graph.indptr) > 0)
        self.output_index = rng.choice(eligible, min(readouts, len(eligible)), replace=False)
        self.log_gain = np.zeros(n, np.float32)
        self.readout = rng.normal(0, 0.05, (len(self.output_index), actions)).astype(np.float32)
        self.bias = np.zeros(actions, np.float32)
        self.feature_scale = 10.0
        self.m = [np.zeros_like(p) for p in self.parameters()]
        self.v = [np.zeros_like(p) for p in self.parameters()]
        self.updates = 0

    def configure_compute(self, sparse_workers=1):
        """Change execution only; sparse rows retain their summation order."""
        if not isinstance(sparse_workers, int) or sparse_workers < 1:
            raise ValueError('sparse_workers must be a positive integer')
        self.close()
        self.sparse_workers = sparse_workers
        self._parts = {}
        if sparse_workers > 1:
            for name, matrix in [('forward', self.graph), ('backward', self.transpose)]:
                edges = np.linspace(0, matrix.shape[0], min(sparse_workers, matrix.shape[0])+1, dtype=int)
                parts = []
                for first, last in zip(edges[:-1], edges[1:]):
                    lo, hi = matrix.indptr[first], matrix.indptr[last]
                    # Share data/indices; only the small rebased row pointer is allocated.
                    part = sparse.csr_matrix((matrix.data[lo:hi], matrix.indices[lo:hi],
                                              matrix.indptr[first:last+1]-lo),
                                             shape=(last-first, matrix.shape[1]), copy=False)
                    parts.append(part)
                self._parts[name] = parts
            self._pool = ThreadPoolExecutor(max_workers=sparse_workers)

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def _matvec(self, vector, backward=False):
        if self._pool is None:
            return (self.transpose if backward else self.graph) @ vector
        parts = self._parts['backward' if backward else 'forward']
        return np.concatenate(list(self._pool.map(lambda part: part @ vector, parts)))

    def activation_cache_bytes(self):
        """Array payload upper bound for one cached decision (including probabilities)."""
        return ((self.cycles+2)*self.log_gain.nbytes
                + len(self.output_index)*self.log_gain.dtype.itemsize + self.bias.nbytes)

    def parameters(self):
        return [self.log_gain, self.readout, self.bias]

    def forward(self, observation, legal_mask, cache=False):
        gain = np.exp(self.log_gain)
        drive = (observation[self.input_index] * self.input_sign).sum(axis=1)
        if self.extra_input_index.shape[1]:
            drive += (observation[self.extra_input_index] * self.extra_input_sign).sum(axis=1)
        h = np.tanh(drive)
        states = [h]
        for _ in range(self.cycles):
            h = np.tanh(self._matvec(gain * h))
            states.append(h)
        features = h[self.output_index] * self.feature_scale
        logits = features @ self.readout + self.bias
        logits = np.where(legal_mask, logits, -np.inf)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        if not np.isfinite(probs).all():
            raise FloatingPointError('Nonfinite action probabilities')
        return probs, (states, features, gain) if cache else None

    def backward(self, cache, dlogits):
        states, features, gain = cache
        dw = np.outer(features, dlogits)
        dh = np.zeros_like(self.log_gain)
        dh[self.output_index] = (self.readout @ dlogits) * self.feature_scale
        dg = np.zeros_like(self.log_gain)
        for k in range(self.cycles, 0, -1):
            dz = dh * (1 - states[k] ** 2)
            routed = self._matvec(dz, backward=True)
            dg += routed * states[k - 1] * gain
            dh = routed * gain
        return [dg, dw, dlogits.copy()]

    def apply(self, gradients, freeze_internal=False):
        if not all(np.isfinite(g).all() for g in gradients):
            raise FloatingPointError('Nonfinite gradients')
        norm = np.sqrt(sum(float(np.sum(g.astype(np.float64) ** 2)) for g in gradients))
        self.updates += 1
        for i, (p, g) in enumerate(zip(self.parameters(), gradients)):
            if i == 0 and freeze_internal:
                continue
            g = g * min(1., 5. / max(norm, 1e-12))
            self.m[i] = .9 * self.m[i] + .1 * g
            self.v[i] = .999 * self.v[i] + .001 * g * g
            m = self.m[i] / (1 - .9 ** self.updates)
            v = self.v[i] / (1 - .999 ** self.updates)
            p += (0.001 if i == 0 else 0.003) * m / (np.sqrt(v) + 1e-8)
        np.clip(self.log_gain, -1., 1., out=self.log_gain)
        return float(norm)

    def save(self, path):
        np.savez_compressed(path, log_gain=self.log_gain, readout=self.readout, bias=self.bias,
                            input_index=self.input_index, input_sign=self.input_sign,
                            output_index=self.output_index, cycles=self.cycles,
                            extra_input_index=self.extra_input_index, extra_input_sign=self.extra_input_sign,
                            feature_scale=self.feature_scale, updates=self.updates,
                            **{f'm{i}': m for i, m in enumerate(self.m)},
                            **{f'v{i}': v for i, v in enumerate(self.v)})

    def load(self, path):
        with np.load(path) as z:
            for key in ('log_gain', 'readout', 'bias', 'input_index', 'input_sign', 'output_index'):
                setattr(self, key, z[key].copy())
            self.cycles, self.feature_scale = int(z['cycles']), float(z['feature_scale'])
            self.extra_input_index = z['extra_input_index'].copy() if 'extra_input_index' in z else np.empty((len(self.log_gain), 0), np.int32)
            self.extra_input_sign = z['extra_input_sign'].copy() if 'extra_input_sign' in z else np.empty((len(self.log_gain), 0), np.float32)
            self.updates = int(z['updates'])
            self.m = [z[f'm{i}'].copy() for i in range(3)]
            self.v = [z[f'v{i}'].copy() for i in range(3)]

    def expand_canvas(self, old_size=5, size=16, seed=901):
        """Embed old I/O in upper-left corner, add inputs only outside it.

        A padded old board has exactly the original neural drive and action logits.
        Optimizer moments restart for the new curriculum; original checkpoint is untouched.
        """
        if self.bias.size != old_size ** 2 or size <= old_size or self.extra_input_index.size:
            raise ValueError('Expected unexpanded square policy')
        cell, channel = self.input_index // 10, self.input_index % 10
        self.input_index = ((cell // old_size * size + cell % old_size) * 10 + channel).astype(np.int32)
        rng = np.random.default_rng(seed)
        cells = np.arange(size * size)
        extra_cells = cells[(cells // size >= old_size) | (cells % size >= old_size)]
        shape = (len(self.log_gain), 8)
        self.extra_input_index = (rng.choice(extra_cells, size=shape) * 10 + rng.integers(10, size=shape)).astype(np.int32)
        self.extra_input_sign = rng.choice(np.array([-.5, .5], np.float32), size=shape)
        old_columns = np.arange(old_size ** 2)
        columns = old_columns // old_size * size + old_columns % old_size
        # Neutral new locations; original logits remain unchanged on the old board.
        weights = rng.normal(0, .05, (len(self.output_index), size ** 2)).astype(np.float32)
        bias = np.zeros(size ** 2, np.float32)
        weights[:, columns], bias[columns] = self.readout, self.bias
        self.readout, self.bias = weights, bias
        self.m = [np.zeros_like(p) for p in self.parameters()]
        self.v = [np.zeros_like(p) for p in self.parameters()]
        self.updates = 0
