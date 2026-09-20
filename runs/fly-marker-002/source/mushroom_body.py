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
    def __init__(self,seed=0,rule='plain',shuffled=False,sparsity=.05,eta=.05,beta=30.,background=0.,tuning=0.,wiring_seed=None,context_reference=False):
        z=np.load(DATA,allow_pickle=False);rng=np.random.default_rng(seed);self.rule,self.eta,self.beta,self.sparsity=rule,eta,beta,sparsity
        pn_kc,kc_mbon,pam,ppl1=load('PN_KC',z),load('KC_MBON',z),load('PAM_MBON',z),load('PPL1_MBON',z)
        self.context_reference=context_reference
        if shuffled:
            # Control: same synapse counts per KC and per MBON, partners drawn at random. The permutations come from their OWN
            # generator so that a real and a shuffled fly with the same seed share odour signature and tuning (paired control).
            wiring=np.random.default_rng([seed if wiring_seed is None else wiring_seed,7919])
            pn_kc=np.stack([wiring.permutation(row) for row in pn_kc]);kc_mbon=np.stack([wiring.permutation(row) for row in kc_mbon])
        types=z['types_PN'];self.glomeruli=sorted(set(types.tolist()));channel=np.array([self.glomeruli.index(t) for t in types.tolist()])
        # Engineering choice: every odorant has a fixed random glomerular signature (about 10% of glomeruli, graded).
        self.signature=(np.abs(rng.normal(size=(ODORANTS,len(self.glomeruli))))*(rng.random((ODORANTS,len(self.glomeruli)))<.1)).astype(np.float32)
        g=len(self.glomeruli);self.background=background*(rng.random(g)<.3).astype(np.float32)  # the arena's own smell, present on every tile
        self.plume_scale=np.array([2.,8.],np.float32);self.plume_k=np.exp(rng.uniform(np.log(.03),np.log(1.),size=(2,g))).astype(np.float32);self.plume_on=(rng.random((2,g))<.5).astype(np.float32)
        # Optional band-pass concentration tuning (lateral inhibition makes real PN responses non-monotonic): each glomerulus
        # answers one odour around a preferred log-concentration. tuning = width in log units; 0 keeps plain Hill curves.
        self.tuning=tuning;self.tuned_odour=rng.integers(2,size=g);self.tuned_mu=np.where(self.tuned_odour==0,rng.uniform(np.log(.06),np.log(4.),size=g),rng.uniform(np.log(.8),np.log(8.5),size=g)).astype(np.float32)
        self.to_pn=np.eye(len(self.glomeruli),dtype=np.float32)[channel].T            # glomerulus -> its sister PNs
        self.pn_kc=(pn_kc/np.maximum(pn_kc.sum(1,keepdims=True),1)).T                   # (PN, KC), each KC's inputs sum to 1
        self.kc_mbon=kc_mbon/np.maximum(kc_mbon.sum(1,keepdims=True),1)                 # (MBON, KC)
        self.exists=(kc_mbon>0);self.w=np.ones_like(self.kc_mbon);self.baseline=0.
        punish,reward=ppl1.sum(1),pam.sum(1);total=np.maximum(punish+reward,1)
        # Compartment logic read from the wiring: MBONs bathed by PPL1 drive approach, those bathed by PAM drive avoidance.
        self.sign=np.where(punish+reward==0,0.,np.where(punish>reward,1.,-1.)).astype(np.float32)
        self.d_punish=(punish/total*(punish>0))[:,None].astype(np.float32);self.d_reward=(reward/total*(reward>0))[:,None].astype(np.float32)
        self.mbon_types=z['types_MBON']
    def glomerular(self,stimulus):
        if stimulus.shape[1]==ODORANTS:return stimulus@self.signature          # symbolic tiles: a bag of odorants
        if self.tuning:
            logc=np.log(np.maximum(stimulus[:,self.tuned_odour],1e-6));return 3*np.exp(-(logc-self.tuned_mu[None])**2/(2*self.tuning**2))*(stimulus[:,self.tuned_odour]>0)
        # plume: two odours at graded concentration; glomeruli differ in sensitivity (Hill curves), as real receptors do
        c=stimulus[:,:,None]/self.plume_scale[None,:,None];return ((c**2/(c**2+self.plume_k[None]**2))*self.plume_on[None]).sum(1)*3+self.background
    def kenyon(self,odor):
        drive=self.glomerular(odor);pn=drive**1.5/(1+drive**1.5+(.1*drive.sum(1,keepdims=True))**1.5)   # antennal-lobe style divisive normalisation
        h=(pn@self.to_pn)@self.pn_kc;k=max(1,int(h.shape[1]*self.sparsity));threshold=np.partition(h,-k,axis=1)[:,-k][:,None]
        h=np.maximum(h-threshold,0);return h/np.maximum(h.sum(1,keepdims=True),1e-9)*k*.1
    def valence(self,h):return (h@(self.kc_mbon*(self.w-1)).T)@self.sign   # change from the naive fly, which is neutral to every smell
    def learn(self,h,mine,value):
        """h: the chosen tile's KC pattern. mine -> PPL1 (heat); safe -> PAM (sugar).

        plain: active synapses relax toward 1, or toward 1-D where dopamine D reaches their compartment,
               so each synapse ends up tracking how often its KC preceded that outcome.
        rpe:   MBON feedback makes dopamine signal surprise; an omitted expected outcome potentiates (relief).
        """
        active=h>0;pre=(h[active]/h.max())[None];w=self.w[:,active]
        if self.rule=='plain':
            w+=self.eta*pre*((1-(self.d_punish if mine else self.d_reward))-w)
        else:
            safe=1/(1+np.exp(-(self.beta*value+self.baseline)))
            # context_reference: the tonic term learns ONLY on odourless tiles, so valence 0 means 'as safe as an odourless tile'
            if not self.context_reference:self.baseline+=self.eta*((0. if mine else 1.)-safe)   # tonic context term: how safe an odourless tile is
            w-=self.eta*pre*(self.d_punish*((1. if mine else 0.)-(1-safe))+self.d_reward*((0. if mine else 1.)-safe))
        self.w[:,active]=np.where(self.exists[:,active],np.clip(w,0,2),1.)
    def learn_context(self,mine):
        """An odourless tile: only the tonic context term can learn."""
        safe=1/(1+np.exp(-self.baseline));self.baseline+=self.eta*((0. if mine else 1.)-safe)
    def state(self):return dict(w=self.w.copy())
