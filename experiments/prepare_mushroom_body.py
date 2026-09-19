"""Extract the mushroom-body learning circuit with raw synapse counts from the MaleCNS edge list."""
import json,re
from pathlib import Path
import numpy as np
import pyarrow.feather as feather
import pyarrow.compute as pc
import pyarrow as pa
from scipy import sparse

OUT=Path('data/processed/mushroom_body.npz')
GROUPS=dict(KC=r'^KC',MBON=r'^MBON',PAM=r'^PAM',PPL1=r'^PPL1',APL=r'^APL')

def main():
    t=feather.read_table('data/raw/annotations.feather',columns=['bodyId','type','class','somaSide'])
    body=t['bodyId'].to_numpy();ty=np.array([x or '' for x in t['type'].to_pylist()],dtype=object);cl=np.array([x or '' for x in t['class'].to_pylist()],dtype=object)
    retained=np.load('data/processed/neuron_ids.npy');keep=np.isin(body,retained)
    sets={k:np.sort(body[keep&np.array([bool(re.match(p,x)) for x in ty])]) for k,p in GROUPS.items()}
    edges=feather.read_table('data/raw/edges.feather',memory_map=True)
    into_kc=edges.filter(pc.is_in(edges['body_post'],value_set=pa.array(sets['KC'])))
    alpn=set(body[keep&(cl=='ALPN')].tolist());pre=into_kc['body_pre'].to_numpy()
    sets['PN']=np.array(sorted(set(pre.tolist())&alpn),dtype=np.int64)
    def block(a,b,table=None):
        tb=table if table is not None else edges
        tb=tb.filter(pc.and_(pc.is_in(tb['body_pre'],value_set=pa.array(sets[a])),pc.is_in(tb['body_post'],value_set=pa.array(sets[b]))))
        return sparse.csr_matrix((tb['weight'].to_numpy().astype(np.float32),(np.searchsorted(sets[b],tb['body_post'].to_numpy()),np.searchsorted(sets[a],tb['body_pre'].to_numpy()))),shape=(len(sets[b]),len(sets[a])))
    mats={'PN_KC':block('PN','KC',into_kc)}
    for a,b in [('KC','MBON'),('PAM','MBON'),('PPL1','MBON'),('KC','APL'),('APL','KC')]:mats[f'{a}_{b}']=block(a,b)
    lookup=dict(zip(body.tolist(),ty.tolist()))
    np.savez_compressed(OUT,**{f'ids_{k}':v for k,v in sets.items()},**{f'types_{k}':np.array([lookup[i] for i in v.tolist()]) for k,v in sets.items()},
        **{f'{k}_{part}':getattr(m,part) for k,m in mats.items() for part in ('data','indices','indptr')},**{f'{k}_shape':np.array(m.shape) for k,m in mats.items()})
    summary={k:int(len(v)) for k,v in sets.items()}|{k:dict(edges=int(m.nnz),synapses=int(m.sum())) for k,m in mats.items()}
    Path('data/processed/mushroom_body.json').write_text(json.dumps(summary,indent=1));print(json.dumps(summary,indent=1))

if __name__=='__main__':main()
