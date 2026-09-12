"""Orthographic asset QA render; renders source triangles, no model generation."""
from pathlib import Path
import struct,json,sys
import numpy as np
from PIL import Image,ImageDraw
path=Path(sys.argv[1]);b=path.read_bytes();sz=struct.unpack_from('<I',b,12)[0];g=json.loads(b[20:20+sz]);buf=b[28+sz:];tri=[];colors=[]
def read(ai):
 a=g['accessors'][ai];v=g['bufferViews'][a['bufferView']];dims={'VEC3':3,'SCALAR':1}[a['type']];dt={5126:'<f4',5125:'<u4',5123:'<u2'}[a['componentType']];width=np.dtype(dt).itemsize;return np.ndarray((a['count'],dims),dtype=dt,buffer=buf,offset=v.get('byteOffset',0)+a.get('byteOffset',0),strides=(v.get('byteStride',dims*width),width)).copy()
def walk(i,parent):
 n=g['nodes'][i];t=np.eye(4)
 if 'matrix' in n:t=np.array(n['matrix']).reshape(4,4).T
 else:
  x,y,z,w=n.get('rotation',[0,0,0,1]);t[:3,:3]=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])@np.diag(n.get('scale',[1,1,1]));t[:3,3]=n.get('translation',[0,0,0])
 if len(sys.argv)>2 and n.get('name') in ['wing_left','wing_right']:
  for axis,angle in [('z',1.5),('x',.7),('y',-1.)]:
   c,s=np.cos(angle),np.sin(angle)
   r={'x':[[1,0,0],[0,c,-s],[0,s,c]],'y':[[c,0,s],[0,1,0],[-s,0,c]],'z':[[c,-s,0],[s,c,0],[0,0,1]]}[axis]
   t[:3,:3]=t[:3,:3]@r
 w=parent@t
 if 'mesh'in n:
  for p in g['meshes'][n['mesh']]['primitives']:
   v=read(p['attributes']['POSITION']);v=(w@np.c_[v,np.ones(len(v))].T).T[:,:3];f=read(p['indices']).reshape(-1,3);tri.append(v[f]);col=g['materials'][p['material']]['pbrMetallicRoughness']['baseColorFactor'];colors.extend([col]*len(f))
 for c in n.get('children',[]):walk(c,w)
for i in g['scenes'][g.get('scene',0)]['nodes']:walk(i,np.eye(4))
t=np.concatenate(tri);c=np.array(colors);view=np.array([1.3,-1,.8]) if 'flybody' in str(path) else np.array([1.3,.8,1]);view/=np.linalg.norm(view);up=np.array([0,0,1]) if 'flybody' in str(path) else np.array([0,1,0]);right=np.cross(up,view);right/=np.linalg.norm(right);up=np.cross(view,right);xy=t@np.array([right,up]).T;lo=xy.min((0,1));hi=xy.max((0,1));xy=(xy-(lo+hi)/2)*min(1050/(hi[0]-lo[0]),700/(hi[1]-lo[1]))+np.array([600,430]);xy[:,:,1]=860-xy[:,:,1]
n=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-14);shade=.48+.52*np.abs(n@view);cols=np.clip(c[:,:3]*shade[:,None]*255,0,255).astype(int);im=Image.new('RGB',(1200,860),(26,36,48));d=ImageDraw.Draw(im)
for i in np.argsort((t@view).mean(1)):
 color=tuple(cols[i]);alpha=c[i,3]
 if alpha<1:color=tuple((cols[i]*alpha+np.array([26,36,48])*(1-alpha)).astype(int))
 d.polygon([tuple(pt) for pt in xy[i]],fill=color)
out=path.with_suffix('.fold.qa.png' if len(sys.argv)>2 else '.qa.png');im.save(out);print(out)
