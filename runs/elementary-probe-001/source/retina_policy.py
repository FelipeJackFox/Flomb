"""Engineering interface using annotated optic columns, with a shared cell decoder.

No anatomical or physiological fidelity claim: hex coordinates are affinely mapped
to a square display, both eyes see the same board, and neuron dynamics are rate units.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.spatial import cKDTree
from experiments.plastic_sparse import PlasticMessage

INPUT_TYPES=('L1','L2','L3')
OUTPUT_TYPES=('Mi1','Mi4','Mi9','Tm1','Tm2','Tm4','Tm9','Tm20','T1','C3')


def build_mapping():
    import pyarrow.feather as feather
    ids=np.load('data/processed/neuron_ids.npy')
    table=feather.read_table('data/raw/annotations.feather',columns=['bodyId','assignedOlHex1','assignedOlHex2','type'])
    body=table['bodyId'].to_numpy()
    index=np.searchsorted(ids,body)
    keep=(index<len(ids)) & (ids[np.minimum(index,len(ids)-1)]==body)
    xy=np.stack([table['assignedOlHex1'].to_numpy(),table['assignedOlHex2'].to_numpy()],axis=1)
    keep &= np.isfinite(xy).all(1)
    types=np.array(table['type'].to_pylist(),dtype=object)
    xy=xy[keep];index=index[keep];types=types[keep]
    xy=(xy-xy.min(0))/(xy.max(0)-xy.min(0))
    input_mask=np.isin(types,INPUT_TYPES)
    result=dict(input_index=index[input_mask].astype(np.int64),input_xy=xy[input_mask].astype(np.float32),
                input_channel=np.array([INPUT_TYPES.index(t) for t in types[input_mask]],np.int64),
                input_types=INPUT_TYPES,output_types=OUTPUT_TYPES,outputs={})
    for size in (5,7):
        query=np.array([(c/(size-1),r/(size-1)) for r in range(size) for c in range(size)])
        indices,weights=[],[]
        for kind in OUTPUT_TYPES:
            mask=types==kind
            distance,nearest=cKDTree(xy[mask]).query(query,k=4)
            weight=1/(distance+.02);weight/=weight.sum(1,keepdims=True)
            indices.append(index[mask][nearest]);weights.append(weight)
        result['outputs'][size]=(np.stack(indices,axis=1).astype(np.int64),np.stack(weights,axis=1).astype(np.float32))
    return result


class RetinaPolicy(nn.Module):
    def __init__(self,operator,mapping,plastic=True,cycles=3):
        super().__init__()
        self.operator,self.cycles=operator,cycles
        self.register_buffer('base',torch.from_numpy(operator.base.copy()))
        self.register_buffer('input_index',torch.from_numpy(mapping['input_index']))
        self.register_buffer('input_xy',torch.from_numpy(mapping['input_xy']))
        self.register_buffer('input_channel',torch.from_numpy(mapping['input_channel']))
        for size,(indices,weights) in mapping['outputs'].items():
            self.register_buffer(f'out_{size}',torch.from_numpy(indices))
            self.register_buffer(f'weight_{size}',torch.from_numpy(weights))
        self.plastic=plastic
        if plastic:
            self.edge_log_gain=nn.Parameter(torch.zeros(len(operator.base)))
        self.node_log_gain=nn.Parameter(torch.zeros(operator.shape[0]))
        self.encoder=nn.Sequential(nn.Conv2d(12,16,3,padding=1),nn.Tanh(),nn.Conv2d(16,3,1),nn.Tanh())
        self.decoder=nn.Sequential(nn.Linear(len(OUTPUT_TYPES)+2,32),nn.Tanh(),nn.Linear(32,1))

    def forward(self,x,context,ablation='normal'):
        if ablation not in ('normal','zero','shuffle'):
            raise ValueError(ablation)
        batch=len(x)
        sizes=(context[:,0]*16).round().long()
        if not all(int(s) in (5,7) for s in sizes):
            raise ValueError('This diagnostic only maps 5x5 and 7x7')
        grid=x.reshape(batch,16,16,10).permute(0,3,1,2)
        encoded=self.encoder(torch.cat([grid,context[:,:,None,None].expand(-1,-1,16,16)],dim=1))
        encoded=encoded*(grid.sum(1,keepdim=True)!=0)
        # Affine display coordinates within the active board, not the padded canvas.
        sample_grid=2*self.input_xy[None]*(sizes[:,None,None]-1)/15-1
        samples=F.grid_sample(encoded,sample_grid[:,:,None],align_corners=True).squeeze(-1)
        sensory=samples[:,self.input_channel,torch.arange(len(self.input_channel))]
        drive=x.new_zeros(self.operator.shape[0],batch).index_copy(0,self.input_index,sensory.T)
        state=torch.tanh(drive)
        values=self.base*(self.edge_log_gain.clamp(-2,2).exp() if self.plastic else 1)
        gain=self.node_log_gain.clamp(-1,1).exp()[:,None]
        if ablation!='zero':
            for _ in range(self.cycles):
                state=torch.tanh(PlasticMessage.apply(values,gain*state,self.operator)+.15*drive)
        if ablation=='zero':
            state=state*0
        elif ablation=='shuffle':
            # Deterministic permutation for all games; does not change parameter values.
            permutation=torch.randperm(len(state),generator=torch.Generator().manual_seed(917))
            state=state[permutation]
        logits=x.new_zeros(batch,256)
        for size in (5,7):
            selected=torch.nonzero(sizes==size).flatten()
            if not len(selected):
                continue
            indices=getattr(self,f'out_{size}')
            weights=getattr(self,f'weight_{size}')
            pooled=(state[indices][:,:,:,selected]*weights[:,:,:,None]).sum(2).permute(2,0,1)
            ctx=context[selected,None,:].expand(-1,size*size,-1)
            scores=self.decoder(torch.cat([pooled*10,ctx],-1)).squeeze(-1)
            positions=torch.tensor([(r*16+c) for r in range(size) for c in range(size)])
            expanded=x.new_zeros(len(selected),256).scatter(1,positions[None].expand(len(selected),-1),scores)
            logits=logits.index_copy(0,selected,expanded)
        return logits
