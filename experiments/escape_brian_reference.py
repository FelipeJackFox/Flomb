"""Independent Brian2 execution of Shiu equations on the same exported graph.

External pre-generated Bernoulli events are applied in Brian2's synapses phase,
matching PoissonInput scheduling. This is deliberately not rescheduled to match
the community engine. Same rates, seed, nodes and signed weights as paired test.
"""
import json,time
import numpy as np
import brian2 as b
from experiments.escape_functional import load,stimulus,OUT


def run(cut=False):
    b.start_scope();b.prefs.codegen.target='numpy';b.defaultclock.dt=.1*b.ms
    meta,w=load();n=meta['n'];t,idx,rates=stimulus(meta,'looming')
    rng=np.random.default_rng(0);events=rng.random(rates.shape)<rates*.0001
    ns=dict(v0=-52*b.mV,vr=-52*b.mV,vt=-45*b.mV,tm=20*b.ms,tau=5*b.ms)
    neu=b.NeuronGroup(n,'''dv/dt=(v0-v+g)/tm : volt (unless refractory)
dg/dt=-g/tau : volt (unless refractory)
rfc : second''',threshold='v>vt',reset='v=vr;g=0*mV',refractory='rfc',method='linear',namespace=ns)
    neu.v=-52*b.mV;neu.g=0*b.mV;neu.rfc=2.2*b.ms;neu.rfc[idx]=0*b.ms
    syn=b.Synapses(neu,neu,'w:volt',on_pre='g+=w',delay=1.8*b.ms)
    pre=np.repeat(np.arange(n),np.diff(w.indptr));syn.connect(i=pre,j=w.indices)
    weights=w.data.copy()
    if cut:weights[np.isin(pre,idx)]=0
    syn.w=weights*.275*b.mV
    @b.network_operation(dt=.1*b.ms,when='synapses',order=0)
    def drive():
        k=int(round(float(b.defaultclock.t/b.ms)*10))
        if k<len(events):
            fired=idx[events[k]];neu.v[fired]+=68.75*b.mV
    mon=b.SpikeMonitor(neu)
    start=time.perf_counter();net=b.Network(neu,syn,drive,mon);net.run(300*b.ms)
    gf=np.array(meta['lesionable']['DNp01']);times=np.array(mon.t/b.ms);who=np.array(mon.i);gt=times[np.isin(who,gf)]
    hist=np.histogram(gt,bins=np.arange(0,305,5))[0]/len(gf)/.005
    return dict(model='brian2_shiu_equations',seed=0,cut_both=cut,dnp01_spikes=int(len(gt)),first_spike_ms=float(gt[0]) if len(gt) else None,trace_hz=hist.tolist(),seconds=time.perf_counter()-start,version=b.__version__)

if __name__=='__main__':
    results=[]
    for cut in (False,True):
        r=run(cut);results.append(r);print({k:v for k,v in r.items() if k!='trace_hz'},flush=True)
        (OUT/'brian2_reference.json').write_text(json.dumps(results,indent=2))
