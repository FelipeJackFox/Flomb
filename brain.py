"""Sparse rate-network policy. Anatomical graph; artificial input/output interfaces.

One learned positive gain per presynaptic neuron scales its outgoing edges.
No claim of spiking physiology, individual-synapse learning, or biological I/O.
"""
import numpy as np


class BrainPolicy:
    def __init__(self, graph, seed=17, inputs=250, actions=25, readouts=1024, cycles=3):
        self.graph = graph
        self.transpose = graph.T.tocsr()
        self.cycles = cycles
        n = graph.shape[0]
        rng = np.random.default_rng(seed)
        self.input_index = rng.integers(inputs, size=(n, 4), dtype=np.int32)
        self.input_sign = rng.choice(np.array([-1., 1.], np.float32), size=(n, 4))
        eligible = np.flatnonzero(np.diff(graph.indptr) > 0)
        self.output_index = rng.choice(eligible, min(readouts, len(eligible)), replace=False)
        self.log_gain = np.zeros(n, np.float32)
        self.readout = rng.normal(0, 0.05, (len(self.output_index), actions)).astype(np.float32)
        self.bias = np.zeros(actions, np.float32)
        self.feature_scale = 10.0
        self.m = [np.zeros_like(p) for p in self.parameters()]
        self.v = [np.zeros_like(p) for p in self.parameters()]
        self.updates = 0

    def parameters(self):
        return [self.log_gain, self.readout, self.bias]

    def forward(self, observation, legal_mask, cache=False):
        gain = np.exp(self.log_gain)
        h = np.tanh((observation[self.input_index] * self.input_sign).sum(axis=1))
        states = [h]
        for _ in range(self.cycles):
            h = np.tanh(self.graph @ (gain * h))
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
            routed = self.transpose @ dz
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
                            feature_scale=self.feature_scale, updates=self.updates,
                            **{f'm{i}': m for i, m in enumerate(self.m)},
                            **{f'v{i}': v for i, v in enumerate(self.v)})

    def load(self, path):
        with np.load(path) as z:
            for key in ('log_gain', 'readout', 'bias', 'input_index', 'input_sign', 'output_index'):
                setattr(self, key, z[key].copy())
            self.cycles, self.feature_scale = int(z['cycles']), float(z['feature_scale'])
            self.updates = int(z['updates'])
            self.m = [z[f'm{i}'].copy() for i in range(3)]
            self.v = [z[f'v{i}'].copy() for i in range(3)]
