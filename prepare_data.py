"""Verify official downloads and build the full retained neuronal sparse graph."""
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import urllib.request
import numpy as np
import pyarrow.feather as feather
from scipy import sparse

BASE = 'https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/'
FILES = {
    'annotations': 'body-annotations-male-cns-v1.0-minconf-0.5.feather',
    'neurotransmitters': 'body-neurotransmitters-male-cns-v1.0.feather',
    'edges': 'connectome-weights-male-cns-v1.0-minconf-0.5.feather',
}


def main():
    provenance = {'source': 'https://male-cns.janelia.org/download/',
                  'license': 'CC-BY, as linked by the source portal', 'files': {}}
    for key, filename in FILES.items():
        path = Path(f'data/raw/{key}.feather')
        if not path.exists():
            partial = path.with_suffix('.feather.part')
            if not partial.exists():
                urllib.request.urlretrieve(BASE + filename, partial)
            path = partial
        with urllib.request.urlopen(urllib.request.Request(BASE + filename, method='HEAD')) as r:
            length = int(r.headers['Content-Length'])
            remote_hashes = ','.join(r.headers.get_all('x-goog-hash', []))
        if path.stat().st_size != length:
            raise ValueError(f'Incomplete file {path}')
        with path.open('rb') as f:
            md5 = base64.b64encode(hashlib.file_digest(f, 'md5').digest()).decode()
        if 'md5=' + md5 not in remote_hashes:
            raise ValueError(f'Cloud MD5 verification failed: {path}')
        with path.open('rb') as f:
            sha = hashlib.file_digest(f, 'sha256').hexdigest()
        target = Path(f'data/raw/{key}.feather')
        if path != target:
            path.rename(target)
        provenance['files'][key] = {'url': BASE + filename, 'bytes': length,
                                    'sha256': sha, 'cloud_md5_verified': True}
        print('Verified', key, length, flush=True)
    annotations = feather.read_table('data/raw/annotations.feather')
    ids = annotations['bodyId'].to_numpy()
    classes = annotations['superclass'].to_pylist()
    statuses = annotations['status'].to_pylist()
    keep = np.array([bool(c) and s != 'Glia' for c, s in zip(classes, statuses)])
    ids = np.sort(ids[keep])
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Duplicate neuron IDs')
    np.save('data/processed/neuron_ids.npy', ids)
    nt = feather.read_table('data/raw/neurotransmitters.feather', columns=['body', 'consensus_nt'])
    nt_map = dict(zip(nt['body'].to_pylist(), nt['consensus_nt'].to_pylist()))
    names = [nt_map.get(int(i)) for i in ids]
    signs = np.array([-1 if x in ('gaba', 'glutamate', 'histamine') else 1 for x in names], np.float32)
    np.save('data/processed/neurotransmitter_signs.npy', signs)
    edges = feather.read_table('data/raw/edges.feather', memory_map=True)
    print('Edge schema', edges.schema, flush=True)
    pre_col = next(x for x in ('body_pre', 'bodyId_pre', 'body-pre') if x in edges.column_names)
    post_col = next(x for x in ('body_post', 'bodyId_post', 'body-post') if x in edges.column_names)
    weight_col = next(x for x in ('weight', 'syn_count', 'synapse_count') if x in edges.column_names)
    rows, cols, weights = [], [], []
    for batch in edges.select([pre_col, post_col, weight_col]).to_batches(max_chunksize=1_000_000):
        pre, post, w = (batch.column(i).to_numpy() for i in range(3))
        a, b = np.searchsorted(ids, pre), np.searchsorted(ids, post)
        valid = (a < len(ids)) & (b < len(ids))
        valid &= (ids[np.minimum(a, len(ids) - 1)] == pre)
        valid &= (ids[np.minimum(b, len(ids) - 1)] == post)
        cols.append(a[valid].astype(np.int32))
        rows.append(b[valid].astype(np.int32))
        weights.append(w[valid].astype(np.float32))
    rows, cols, weights = np.concatenate(rows), np.concatenate(cols), np.concatenate(weights)
    contacts = int(weights.astype(np.float64).sum())
    graph = sparse.csr_matrix((weights * signs[cols], (rows, cols)), shape=(len(ids), len(ids)))
    # Preserve all released edges among retained neurons. Normalize incoming absolute mass.
    incoming = np.asarray(abs(graph).sum(axis=1)).ravel()
    graph = sparse.diags(1 / np.maximum(incoming, 1)) @ graph
    graph = graph.astype(np.float32).tocsr()
    graph.sort_indices()
    sparse.save_npz('data/processed/graph.npz', graph)
    provenance.update(neurons=len(ids), edges=int(graph.nnz), contacts=contacts,
                      raw_edge_rows=edges.num_rows,
                      retained_edge_rows=len(weights),
                      node_policy='Assigned nonempty superclass and status != Glia; all such neurons retained',
                      normalization='Sum of absolute incoming weights per neuron; no edge pruning',
                      nt_counts=dict(Counter(str(x) for x in names)),
                      sign_assumption='gaba/glutamate/histamine negative; all others positive, including unknowns; receptor dynamics not modeled',
                      graph_bytes=int(graph.data.nbytes + graph.indices.nbytes + graph.indptr.nbytes))
    Path('data/processed/provenance.json').write_text(json.dumps(provenance, indent=2))
    print(json.dumps({k: provenance[k] for k in ('neurons', 'edges', 'contacts', 'graph_bytes')}, indent=2))


if __name__ == '__main__':
    main()
