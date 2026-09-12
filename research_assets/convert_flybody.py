"""Convert existing Flybody OBJ + MJCF hierarchy to GLB; no new anatomy."""
from pathlib import Path
import json,struct,xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
SRC=Path('/tmp/mosca-fruta-assets/flybody')
OUT=Path('/tmp/mosca-fruta-assets/flybody.glb')
root=ET.parse(SRC/'fruitfly.xml').getroot()
g={'asset':{'version':'2.0','generator':'Mosca Fruta MJCF asset conversion','copyright':'Google DeepMind and HHMI Janelia; Apache-2.0'},'scene':0,'scenes':[{'nodes':[0]}],'nodes':[{'name':'flybody_Zup_Xforward','children':[]}],'meshes':[],'materials':[],'accessors':[],'bufferViews':[],'buffers':[{}]};blob=bytearray()
materials={}
for mat in root.find('asset').findall('material'):
 rgba=[float(x) for x in mat.get('rgba','1 1 1 1').split()]
 materials[mat.get('name')]=len(g['materials'])
 g['materials'].append({'name':mat.get('name'),'pbrMetallicRoughness':{'baseColorFactor':rgba,'metallicFactor':0,'roughnessFactor':0.55},'doubleSided':True,**({'alphaMode':'BLEND'} if rgba[3]<1 else {})})
def acc(arr,typ,component,target):
 while len(blob)%4:blob.append(0)
 off=len(blob);blob.extend(arr.tobytes());view=len(g['bufferViews']);g['bufferViews'].append({'buffer':0,'byteOffset':off,'byteLength':arr.nbytes,'target':target})
 a={'bufferView':view,'componentType':component,'count':len(arr),'type':typ}
 if typ=='VEC3':a.update(min=arr.min(axis=0).tolist(),max=arr.max(axis=0).tolist())
 g['accessors'].append(a);return len(g['accessors'])-1
meshfiles={m.get('name'):m.get('file') for m in root.find('asset').findall('mesh')}
loaded={};allworld=[];report=[]
def load(name,mat):
 key=(name,mat)
 if key in loaded:return loaded[key]
 vs=[];faces=[]
 for ln in (SRC/meshfiles[name]).read_text().splitlines():
  if ln.startswith('v '):vs.append([float(x) for x in ln.split()[1:4]])
  elif ln.startswith('f '):
   ids=[int(x.split('/')[0])-1 for x in ln.split()[1:]]
   for k in range(1,len(ids)-1):faces.append([ids[0],ids[k],ids[k+1]])
 v=np.asarray(vs,dtype='<f4')*.1;f=np.asarray(faces,dtype='<u4');n=np.zeros_like(v)
 fn=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
 for i in range(3):np.add.at(n,f[:,i],fn)
 n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-15)
 idx=len(g['meshes']);g['meshes'].append({'name':name,'primitives':[{'attributes':{'POSITION':acc(v,'VEC3',5126,34962),'NORMAL':acc(n,'VEC3',5126,34962)},'indices':acc(f.reshape(-1),'SCALAR',5125,34963),'material':materials[mat]}]});loaded[key]=idx
 report.append({'mesh':name,'vertices':len(v),'triangles':len(f)})
 return idx

def transform(e):
 p=np.fromstring(e.get('pos','0 0 0'),sep=' ');q=np.fromstring(e.get('quat','1 0 0 0'),sep=' ');q/=np.linalg.norm(q);xyzw=q[[1,2,3,0]]
 m=np.eye(4);m[:3,:3]=Rotation.from_quat(xyzw).as_matrix();m[:3,3]=p
 return p.tolist(),xyzw.tolist(),m

def recurse(e,parent,parentmat):
 p,q,t=transform(e);idx=len(g['nodes']);g['nodes'].append({'name':e.get('name','body'),'translation':p,'rotation':q,'children':[]});g['nodes'][parent]['children'].append(idx);world=parentmat@t
 for geom in e.findall('geom'):
  name=geom.get('mesh')
  if not name or 'collision' in geom.get('name',''):continue
  p,q,t=transform(geom);mi=load(name,geom.get('material','body'));gi=len(g['nodes']);g['nodes'].append({'name':'mesh_'+geom.get('name',name),'mesh':mi,'translation':p,'rotation':q});g['nodes'][idx]['children'].append(gi)
  posacc=g['accessors'][g['meshes'][mi]['primitives'][0]['attributes']['POSITION']];view=g['bufferViews'][posacc['bufferView']];v=np.frombuffer(blob,dtype='<f4',count=posacc['count']*3,offset=view['byteOffset']).reshape(-1,3).copy();allworld.append((world@t@np.c_[v,np.ones(len(v))].T).T[:,:3])
 for child in e.findall('body'):recurse(child,idx,world)
for body in root.find('worldbody').findall('body'):recurse(body,0,np.eye(4))
g['buffers'][0]['byteLength']=len(blob)
j=json.dumps(g,separators=(',',':')).encode();j+=b' '*((-len(j))%4);blob+=b'\x00'*((-len(blob))%4)
OUT.write_bytes(struct.pack('<III',0x46546c67,2,12+8+len(j)+8+len(blob))+struct.pack('<II',len(j),0x4e4f534a)+j+struct.pack('<II',len(blob),0x004e4942)+blob)
v=np.concatenate(allworld);summary={'file':str(OUT),'bytes':OUT.stat().st_size,'nodes':len(g['nodes']),'meshes':len(report),'triangles':sum(r['triangles'] for r in report),'bounds_min':v.min(0).tolist(),'bounds_max':v.max(0).tolist(),'axes':'MJCF: Z up, X forward; Three root rotateX(-pi/2) for Y up','preservation':'OBJ vertices/triangles and MJCF hierarchy preserved; OBJ scale .1 from MJCF applied; geometric qpos=0 only; no dynamics','parts':report}
OUT.with_suffix('.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='parts'},indent=2))
