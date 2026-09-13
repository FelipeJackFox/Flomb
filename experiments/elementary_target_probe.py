"""Separate neighborhood access from the existing single-target decoder."""
import json,pickle
import numpy as np
from experiments.elementary_probe import OUT,probe
import torch

def main():
 torch.set_num_threads(1)
 assert (OUT/'completed.json').exists()
 data=pickle.loads((OUT/'dataset.pkl').read_bytes());results=json.loads((OUT/'results.json').read_text())
 for name in results:
  a=np.load(OUT/f'{name}-features.npz');fs={}
  mean=a['train_cycle3'].mean(0)
  _,_,vt=np.linalg.svd(a['train_cycle3']-mean,full_matrices=False)
  projection=vt[:10].T
  for split,rows in data.items():
   patches=a[split+'_cycle3'].reshape(-1,9,10)
   offsets=[(r['target'][0]-r['center'][0]+1)*3+(r['target'][1]-r['center'][1]+1) for r in rows]
   fs[split]={'cycle3_target':patches[np.arange(len(rows)),offsets], 'neighborhood_pca10':(a[split+'_cycle3']-mean)@projection}
  results[name].update(probe(fs,data))
 (OUT/'results.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
