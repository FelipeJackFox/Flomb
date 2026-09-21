"""Dopamine-gated learning in the real mushroom-body wiring (rate model, no spikes, no backprop).

Senses -> projection neurons -> Kenyon cells (sparse, APL-like inhibition) -> MBONs -> approach/avoid valence.
Only KC->MBON synapses change, with a local three-factor rule: presynaptic KC activity x dopamine reaching
that MBON's compartment. Heat on a mine drives PPL1; sugar on a safe tile drives PAM.
"""
from pathlib import Path
import numpy as np
from scipy import sparse
from experiments.fly_senses import ODORANTS

DATA=Path('data/processed/mushroom_body.npz')

def load(name,z):return sparse.csr_matrix((z[f'{name}_data'],z[f'{name}_indices'],z[f'{name}_indptr']),shape=tuple(z[f'{name}_shape'])).toarray().astype(np.float32)

class MushroomBody:
    def __init__(self,seed=0,rule='plain',shuffled=False,sparsity=.05,eta=.05,beta=30.,background=0.,tuning=0.,wiring_seed=None,context_reference=False,punish_gain=1.,inhibition='topk'):
        z=np.load(DATA,allow_pickle=False);rng=np.random.default_rng(seed);self.rule,self.eta,self.beta,self.sparsity=rule,eta,beta,sparsity
        pn_kc,kc_mbon,pam,ppl1=load('PN_KC',z),load('KC_MBON',z),load('PAM_MBON',z),load('PPL1_MBON',z)
        kc_apl,apl_kc=load('KC_APL',z),load('APL_KC',z).T   # both (APL, KC): what each Kenyon cell gives to, and receives from, each APL neuron
        self.context_reference=context_reference;self.punish_gain=punish_gain   # heat teaches faster than sugar (aversive one-trial learning)
        if shuffled:
            # Control: same synapse counts per KC and per MBON, partners drawn at random. The permutations come from their OWN
            # generator so that a real and a shuffled fly with the same seed share odour signature and tuning (paired control).
            wiring=np.random.default_rng([seed if wiring_seed is None else wiring_seed,7919])
            # shuffled=True keeps its original meaning (both stages, drawn in the original order). Finer controls:
            # 'input' only PN->KC, 'output' only KC->MBON, 'random' = same number of synapses and total weight, no degree structure at all.
            if shuffled in (True,'input','random'):
                pn_kc=np.stack([wiring.permutation(row) for row in pn_kc]) if shuffled!='random' else wiring.permutation(pn_kc.ravel()).reshape(pn_kc.shape)
            if shuffled in (True,'output','random'):
                kc_mbon=np.stack([wiring.permutation(row) for row in kc_mbon]) if shuffled!='random' else wiring.permutation(kc_mbon.ravel()).reshape(kc_mbon.shape)
            if shuffled in (True,'random'):order=wiring.permutation(kc_apl.shape[1]);kc_apl,apl_kc=kc_apl[:,order],apl_kc[:,wiring.permutation(apl_kc.shape[1])]
        types=z['types_PN'];self.glomeruli=sorted(set(types.tolist()));channel=np.array([self.glomeruli.index(t) for t in types.tolist()])
        # Engineering choice: every odorant has a fixed random glomerular signature (about 10% of glomeruli, graded).
        self.signature=(np.abs(rng.normal(size=(ODORANTS,len(self.glomeruli))))*(rng.random((ODORANTS,len(self.glomeruli)))<.1)).astype(np.float32)
        g=len(self.glomeruli);self.background=background*(rng.random(g)<.3).astype(np.float32)  # the arena's own smell, present on every tile
        self.plume_scale=np.array([2.,8.],np.float32);self.plume_k=np.exp(rng.uniform(np.log(.03),np.log(1.),size=(2,g))).astype(np.float32);self.plume_on=(rng.random((2,g))<.5).astype(np.float32)
        # Optional band-pass concentration tuning (lateral inhibition makes real PN responses non-monotonic): each glomerulus
        # answers one odour around a preferred log-concentration. tuning = width in log units; 0 keeps plain Hill curves.
        self.tuning=tuning;self.tuned_odour=rng.integers(2,size=g);self.tuned_mu=np.where(self.tuned_odour==0,rng.uniform(np.log(.06),np.log(4.),size=g),rng.uniform(np.log(.8),np.log(8.5),size=g)).astype(np.float32)
        # Raw three-odour senses (clue N, covered k, own pheromone m): separate generator so older configurations stay bit-identical.
        raw=np.random.default_rng([seed,4242]);self.raw_odour=raw.integers(3,size=g);self.raw_mu=raw.uniform(np.log(.8),np.log(8.5),size=g).astype(np.float32)
        pair=np.random.default_rng([seed,4444]);self.pair_channel=pair.integers(7,size=g);self.pair_mu=pair.uniform(np.log(.8),np.log(8.5),size=g).astype(np.float32);self.pair_geometry=(pair.random((64,g))<.5).astype(np.float32)*3
        far=np.random.default_rng([seed,4343]);self.far_odour=far.integers(3,size=g);self.far_mu=far.uniform(np.log(.8),np.log(90.),size=g).astype(np.float32)
        self.to_pn=np.eye(len(self.glomeruli),dtype=np.float32)[channel].T            # glomerulus -> its sister PNs
        self.pn_kc=(pn_kc/np.maximum(pn_kc.sum(1,keepdims=True),1)).T                   # (PN, KC), each KC's inputs sum to 1
        self.kc_mbon=kc_mbon/np.maximum(kc_mbon.sum(1,keepdims=True),1)                 # (MBON, KC)
        self.exists=(kc_mbon>0);self.baseline=0.
        punish,reward=ppl1.sum(1),pam.sum(1);total=np.maximum(punish+reward,1)
        # Compartment logic read from the wiring: MBONs bathed by PPL1 drive approach, those bathed by PAM drive avoidance.
        self.sign=np.where(punish+reward==0,0.,np.where(punish>reward,1.,-1.)).astype(np.float32)
        self.d_punish=(punish/total*(punish>0))[:,None].astype(np.float32);self.d_reward=(reward/total*(reward>0))[:,None].astype(np.float32)
        self.mbon_types=z['types_MBON']
        # Optional real feedback inhibition: each APL neuron sums Kenyon activity through its real KC->APL synapses and inhibits each
        # Kenyon cell through its real APL->KC synapses. One global gain is calibrated so that, on average, `sparsity` of the cells stay active.
        self.inhibition=inhibition;self._to_apl=(kc_apl/np.maximum(kc_apl.sum(1,keepdims=True),1)).astype(np.float32);self._from_apl=(apl_kc/np.maximum(apl_kc.mean(),1e-9)).astype(np.float32);self._apl_gain=0.
        if inhibition=='apl':
            probe=np.random.default_rng([seed,555]).integers(1,9,size=(150,3)).astype(np.float32);probe[:,2]=np.minimum(probe[:,2]-1,probe[:,1]);lo,hi=0.,1e4
            for _ in range(40):
                self._apl_gain=(lo+hi)/2;active=float((self.kenyon(probe)>0).mean());lo,hi=(self._apl_gain,hi) if active>sparsity else (lo,self._apl_gain)
        # Speed: weights live transposed (KC x MBON) so the active Kenyon cells are contiguous rows, and the valence read-out is kept as
        # one vector u[i] = sum_j sign_j * C_ji * (w_ji - 1), refreshed only for the Kenyon cells whose synapses just changed.
        # Only 15% of KC-MBON pairs are real synapses, so plastic weights are stored sparsely, grouped by Kenyon cell (CSR).
        kc,mbon=np.nonzero(self.exists.T);self._ptr=np.concatenate([[0],np.cumsum(np.bincount(kc,minlength=self.exists.shape[1]))]).astype(np.int64);self._mbon=mbon.astype(np.int64);self._kc=kc.astype(np.int64)
        self._signed=(self.kc_mbon*self.sign[:,None]).T[kc,mbon].astype(np.float64);self.w=np.ones_like(self.kc_mbon)
    @property
    def w(self):
        dense=np.ones(self.exists.shape,np.float32);dense[self._mbon,self._kc]=self._flat;return dense
    @w.setter
    def w(self,value):self._flat=np.asarray(value,np.float32)[self._mbon,self._kc].copy();self._u=np.bincount(self._kc,self._signed*(self._flat-1.),minlength=self.exists.shape[1])
    def _synapses_of(self,active):
        """Flat positions of every real synapse of the active Kenyon cells, and which active cell each belongs to."""
        start,count=self._ptr[active],self._ptr[active+1]-self._ptr[active];owner=np.repeat(np.arange(len(active)),count)
        return np.arange(count.sum())-np.repeat(np.cumsum(count)-count,count)+np.repeat(start,count),owner
    def glomerular(self,stimulus):
        if stimulus.shape[1]==ODORANTS:return stimulus@self.signature          # symbolic tiles: a bag of odorants
        if stimulus.shape[1]==7:   # remembered pair: six concentrations (B then A) on their own receptors, plus the proprioceptive 'how I moved' class
            ch=self.pair_channel;c=stimulus[:,np.minimum(ch,5)];conc=3*np.exp(-(np.log(np.maximum(c,1e-6))-self.pair_mu[None])**2/(2*(self.tuning or .35)**2))*(c>0)
            return np.where(ch[None]==6,self.pair_geometry[stimulus[:,6].astype(np.int64)],conc)
        if stimulus.shape[1]==4:   # near sniff (flag 0) or far field (flag 1): different receptors, same band-pass coding
            far=stimulus[:,3:]>0;c=np.where(far,stimulus[:,:3][:,self.far_odour],stimulus[:,:3][:,self.raw_odour]);mu=np.where(far,self.far_mu[None],self.raw_mu[None])
            return 3*np.exp(-(np.log(np.maximum(c,1e-6))-mu)**2/(2*(self.tuning or .35)**2))*(c>0)
        if stimulus.shape[1]==3:
            c=stimulus[:,self.raw_odour];return 3*np.exp(-(np.log(np.maximum(c,1e-6))-self.raw_mu[None])**2/(2*(self.tuning or .35)**2))*(c>0)
        if self.tuning:
            logc=np.log(np.maximum(stimulus[:,self.tuned_odour],1e-6));return 3*np.exp(-(logc-self.tuned_mu[None])**2/(2*self.tuning**2))*(stimulus[:,self.tuned_odour]>0)
        # plume: two odours at graded concentration; glomeruli differ in sensitivity (Hill curves), as real receptors do
        c=stimulus[:,:,None]/self.plume_scale[None,:,None];return ((c**2/(c**2+self.plume_k[None]**2))*self.plume_on[None]).sum(1)*3+self.background
    def kenyon(self,odor):
        drive=self.glomerular(odor);pn=drive**1.5/(1+drive**1.5+(.1*drive.sum(1,keepdims=True))**1.5)   # antennal-lobe style divisive normalisation
        h=(pn@self.to_pn)@self.pn_kc;k=max(1,int(h.shape[1]*self.sparsity))
        if self.inhibition=='apl':
            x=h;a=np.zeros((len(x),self._to_apl.shape[0]),np.float32)
            for _ in range(60):h=np.maximum(x-self._apl_gain*(a@self._from_apl),0);a=.7*a+.3*(h@self._to_apl.T)   # damped settling of the KC <-> APL loop
            return h/np.maximum(h.sum(1,keepdims=True),1e-9)*k*.1
        threshold=np.partition(h,-k,axis=1)[:,-k][:,None]
        h=np.maximum(h-threshold,0);return h/np.maximum(h.sum(1,keepdims=True),1e-9)*k*.1
    def valence(self,h):return (h@self._u).astype(np.float32)   # change from the naive fly, which is neutral to every smell
    def valence_reference(self,h):return (h@(self.kc_mbon*(self.w-1)).T)@self.sign   # original formula, kept for tests
    def learn(self,h,mine,value):
        """h: the chosen tile's KC pattern. mine -> PPL1 (heat); safe -> PAM (sugar).

        plain: active synapses relax toward 1, or toward 1-D where dopamine D reaches their compartment,
               so each synapse ends up tracking how often its KC preceded that outcome.
        rpe:   MBON feedback makes dopamine signal surprise; an omitted expected outcome potentiates (relief).
        """
        active=np.flatnonzero(h>0);at,owner=self._synapses_of(active);pre=(h[active]/h.max())[owner];w=self._flat[at];j=self._mbon[at]
        if self.rule=='plain':
            w+=self.eta*pre*((1-(self.d_punish if mine else self.d_reward))[j,0]-w)
        else:
            safe=1/(1+np.exp(-(self.beta*value+self.baseline)))
            # context_reference: the tonic term learns ONLY on odourless tiles, so valence 0 means 'as safe as an odourless tile'
            gain=self.punish_gain if mine else 1.
            if not self.context_reference:self.baseline+=self.eta*((0. if mine else 1.)-safe)   # the shared tonic term is never sped up: it would drag every smell below the dread threshold   # tonic context term: how safe an odourless tile is
            w-=self.eta*gain*pre*(self.d_punish*((1. if mine else 0.)-(1-safe))+self.d_reward*((0. if mine else 1.)-safe))[j,0]
        w=np.clip(w,0,2).astype(np.float32);self._flat[at]=w;self._u[active]=np.bincount(owner,self._signed[at]*(w-1.),minlength=len(active))
    def learn_context(self,mine):
        """An odourless tile: only the tonic context term can learn."""
        safe=1/(1+np.exp(-self.baseline));self.baseline+=self.eta*((0. if mine else 1.)-safe)
    def state(self):return dict(w=self.w.copy())
