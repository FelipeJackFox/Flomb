"""Paired, untrained escape-circuit diagnostic; never touches Minesweeper runs.

Reference engine is pinned, unmodified except import wiring. FlyWire assets are
community exports, hash checked; not independently rebuilt from original EM data.
Rate model ports local incoming normalization and 3/6/12 tanh updates, unit gains.
Rates are an engineering input, not raw vision. Negative motion is gated upstream.
"""
from pathlib import Path
import argparse, hashlib, json, time, types, sys
import numpy as np
from scipy import sparse

ROOT=Path(__file__).resolve().parents[1]
REF=ROOT/'reference/escape-functional'
OUT=ROOT/'runs/escape-functional-001'


def load():
    meta=json.loads((REF/'meta.json').read_text())
    for name,info in meta['files'].items():
        assert hashlib.sha256((REF/name).read_bytes()).hexdigest()==info['sha256']
    n,e=meta['n'],meta['nnz']; b=(REF/'connectome.bin').read_bytes()
    ptr=np.frombuffer(b,'<i4',n+1).copy()
    ix=np.frombuffer(b,'<i4',e,(n+1)*4).copy()
    for a,z in zip(ptr[:-1],ptr[1:]): ix[a:z]=np.cumsum(ix[a:z])
    w=np.frombuffer(b,'<i2',e,(n+1+e)*4).astype(np.float32)
    assert ptr[-1]==e and ix.min()>=0 and ix.max()<n
    graph=sparse.csr_matrix((w,ix,ptr),shape=(n,n)) # source rows
    nb=(REF/'neurons.bin').read_bytes()
    ids=np.frombuffer(nb,'<i8',n)
    tc=np.frombuffer(nb,'<u2',n,n*20)
    for kind in ('LC4','LPLC2','DNp01'):
        assert set(np.flatnonzero(tc==meta['cell_types'].index(kind)))==set(meta['lesionable'][kind])
    assert ids[meta['lesionable']['DNp01']].tolist()==meta['dnp01_root_ids']
    return meta,graph


def engine_class():
    # Execute reviewed pinned source in isolated namespaces; no math changes.
    module=types.ModuleType('escape_reference_lif');sys.modules[module.__name__]=module
    exec(compile((REF/'lif.py').read_text(),str(REF/'lif.py'),'exec'),module.__dict__)
    ns=dict(LIFParams=module.LIFParams,DEFAULT=module.DEFAULT)
    src=(REF/'lif_engine.py').read_text().replace('from brain.neuron_models.lif import LIFParams, DEFAULT','')
    exec(compile(src,str(REF/'lif_engine.py'),'exec'),ns)
    return ns['LIFEngine']


def stimulus(meta,kind,duration=300.):
    rf=meta['receptive_fields']; idx=np.array(rf['LC4']['idx']+rf['LPLC2']['idx'])
    az=np.array(rf['LC4']['az']+rf['LPLC2']['az']);el=np.array(rf['LC4']['el']+rf['LPLC2']['el'])
    sigma=np.clip(rf['LC4']['r']+rf['LPLC2']['r'],5,60)
    cos=np.cos(np.deg2rad(el))*np.cos(np.deg2rad(az-45))
    distance=np.rad2deg(np.arccos(np.clip(cos,-1,1)))
    t=np.arange(round(duration/.1))*.1
    speed={'looming':250.,'receding':-250.,'static':0.}[kind]
    d=50-speed*(t-20)/1000
    active=(t>=20)&(d>0)
    safe=np.maximum(d,1e-9)
    theta=np.rad2deg(np.arctan(5/safe))
    expansion=np.maximum(0,np.rad2deg(5*speed/(safe**2+25)))
    a=150*expansion/(expansion+300)
    b=150*theta/(theta+25)*expansion/(expansion+10)
    rate=np.concatenate([np.repeat(a[:,None],len(rf['LC4']['idx']),1),np.repeat(b[:,None],len(rf['LPLC2']['idx']),1)],axis=1)
    rate*=np.exp(-np.maximum(0,distance[None]-theta[:,None])**2/(2*sigma[None]**2))
    rate[~active]=0
    return t,idx,rate


def run_lif(meta,graph,kind,silenced,seed,normalization=False):
    E=engine_class();w=graph
    if normalization:
        mass=np.asarray(abs(w).sum(axis=0)).ravel()
        w=(w@sparse.diags(1/np.maximum(mass,1))).tocsr()
    engine=E(types.SimpleNamespace(w=w,n=w.shape[0]),seed=seed)
    t,idx,rates=stimulus(meta,kind)
    engine.silence([i for k in silenced for i in meta['lesionable'][k]])
    gf=np.array(meta['lesionable']['DNp01']); traces=[];spike_times=[];old=0
    start=time.perf_counter()
    for k,rate in enumerate(rates):
        # Changing rates must NOT reset refractory counters on every time step.
        if k==0:engine.set_poisson(idx,rate)
        else:engine._poi_p=rate*.0001
        spk=engine.step()
        if np.isin(gf,spk).any():spike_times.append(float(t[k]))
        if (k+1)%50==0:
            count=int(engine.spike_counts[gf].sum());traces.append((count-old)/len(gf)/.005);old=count
    return dict(model='lif_normalized' if normalization else 'lif_reference',condition=kind,silenced=list(silenced),seed=seed,
                dnp01_spikes=int(engine.spike_counts[gf].sum()),first_spike_ms=spike_times[0] if spike_times else None,
                active_neurons=int(np.count_nonzero(engine.spike_counts)),input_spikes=int(engine.spike_counts[idx].sum()),
                peak_hz=max(traces),trace_hz=traces,seconds=time.perf_counter()-start)


def run_rate(meta,graph,kind,silenced,cycles):
    w=graph.T.tocsr(); mass=np.asarray(abs(w).sum(1)).ravel()
    w=(sparse.diags(1/np.maximum(mass,1))@w).tocsr()
    sil=[i for k in silenced for i in meta['lesionable'][k]]
    t,idx,rates=stimulus(meta,kind);gf=meta['lesionable']['DNp01'];trace=[]
    for rate in rates[49::50]:
        drive=np.zeros(meta['n'],np.float32);drive[idx]=rate/150
        state=np.tanh(drive)
        for _ in range(cycles):
            source=state.copy();source[sil]=0
            state=np.tanh(w@source+.15*drive)
        trace.append(float(state[gf].mean()))
    return dict(model=f'rate_{cycles}',condition=kind,silenced=list(silenced),seed=None,
                trace_activation=trace,peak_activation=max(trace))


def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=3);a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'completed.json').exists():raise SystemExit('Completed run exists; preserve it.')
    meta,g=load();results=[]
    conditions=[('looming',()),('static',()),('receding',()),('looming',('LC4',)),('looming',('LPLC2',)),('looming',('LC4','LPLC2'))]
    (OUT/'config.json').write_text(json.dumps(dict(seeds=a.seeds,neurons=meta['n'],edges=meta['nnz'],duration_ms=300,source=json.loads((REF/'sources.json').read_text()),limitations=['Community-exported graph, original raw download not independently rebuilt','Engineering looming encoder; static/receding zero-input is by construction','Rate activations and LIF Hz are different units','Rate control uses unit gains, not trained Minesweeper checkpoint']),indent=2))
    for kind,sil in conditions:
        for seed in range(a.seeds):
            r=run_lif(meta,g,kind,sil,seed);results.append(r)
            print({k:v for k,v in r.items() if not k.startswith('trace')},flush=True)
        for cycles in (3,6,12):results.append(run_rate(meta,g,kind,sil,cycles))
        (OUT/'results.json').write_text(json.dumps(results,indent=2))
    for seed in range(a.seeds):
        r=run_lif(meta,g,'looming',(),seed,True);results.append(r);print(r['model'],seed,r['dnp01_spikes'],flush=True)
    (OUT/'results.json').write_text(json.dumps(results,indent=2))
    (OUT/'completed.json').write_text(json.dumps(dict(completed=True,experiments=len(results))))

if __name__=='__main__':main()
