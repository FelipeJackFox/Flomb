import struct,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
for name in ['desk','chairDesk','computerScreen','computerMouse','computerKeyboard']:
 b=(Path('research_assets/kenney')/(name+'.glb')).read_bytes();sz=struct.unpack_from('<I',b,12)[0];g=json.loads(b[20:20+sz]);start=28+sz;buf=b[start:];pts=[];prims=[]
 def walk(i,parent):
  n=g['nodes'][i];t=np.eye(4)
  if 'matrix' in n:t=np.array(n['matrix']).reshape(4,4).T
  else:
   t[:3,:3]=Rotation.from_quat(n.get('rotation',[0,0,0,1])).as_matrix()@np.diag(n.get('scale',[1,1,1]));t[:3,3]=n.get('translation',[0,0,0])
  w=parent@t
  if 'mesh' in n:
   for p in g['meshes'][n['mesh']]['primitives']:
    a=g['accessors'][p['attributes']['POSITION']];v=g['bufferViews'][a['bufferView']];off=v.get('byteOffset',0)+a.get('byteOffset',0);stride=v.get('byteStride',12);x=np.ndarray((a['count'],3),dtype='<f4',buffer=buf,offset=off,strides=(stride,4));x=(w@np.c_[x,np.ones(len(x))].T).T[:,:3];pts.append(x);prims.append({'node':n.get('name'), 'mat':g['materials'][p['material']].get('name'),'min':x.min(0).tolist(),'max':x.max(0).tolist()})
  for c in n.get('children',[]):walk(c,w)
 for i in g['scenes'][g.get('scene',0)]['nodes']:walk(i,np.eye(4))
 x=np.concatenate(pts);print(json.dumps({'name':name,'min':x.min(0).tolist(),'max':x.max(0).tolist(),'dims':np.ptp(x,0).tolist(),'prims':prims}))
