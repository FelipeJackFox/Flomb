"""Bounded read-only real-graph CSR batching probe; never changes training."""
import json, os, platform, resource, signal, time
from pathlib import Path
import numpy as np
import scipy
from scipy import sparse

def timeout(*_):
    raise TimeoutError('60 second probe limit')
signal.signal(signal.SIGALRM, timeout)
signal.alarm(60)
root = Path(__file__).resolve().parents[1]
graph = sparse.load_npz(root / 'data/processed/graph.npz')
rng = np.random.default_rng(918)
results = []
for batch in [1, 4, 8, 16]:
    x = rng.normal(size=(graph.shape[1], batch)).astype(np.float32)
    reference = np.column_stack([graph @ x[:, i] for i in range(batch)])
    actual = graph @ x
    np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-6)
    timings = {'loop': [], 'spmm': []}
    for _ in range(3):
        for name in ['loop', 'spmm']:
            start = time.perf_counter()
            out = np.column_stack([graph @ x[:, i] for i in range(batch)]) if name == 'loop' else graph @ x
            timings[name].append(time.perf_counter() - start)
    med = {name: float(np.median(values)) for name, values in timings.items()}
    results.append({'batch': batch, **med, 'speedup': med['loop']/med['spmm'],
                    'max_abs_error': float(np.max(np.abs(reference-actual))),
                    'allclose': True, 'raw_seconds': timings})
result = {'platform': platform.platform(), 'numpy': np.__version__, 'scipy': scipy.__version__,
          'shape': graph.shape, 'nnz': graph.nnz, 'dtype': str(graph.dtype),
          'scope': 'single forward CSR operation; excludes encoding, gradients, optimizer, replay and checkpoint I/O; other training active',
          'peak_rss_native': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          'results': results}
path = root / 'benchmarks/batch_probe_result.json'
path.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
signal.alarm(0)
