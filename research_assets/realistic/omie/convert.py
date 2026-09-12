import bpy,json,pathlib
from mathutils import Vector
root=pathlib.Path(__file__).resolve().parent
src=root/'Office Cubicle/ComputerSetup'
out=root/'glb';out.mkdir(exist_ok=True)
report={}
for label in ['Monitor','Keyboard','Mouse','Computer']:
 bpy.ops.wm.read_factory_settings(use_empty=True)
 bpy.ops.import_scene.fbx(filepath=str(src/'Models'/('SM_'+label+'.fbx')))
 mat=bpy.data.materials.new('Omie_ComputerSetup_PBR');mat.use_nodes=True
 nodes=mat.node_tree.nodes;links=mat.node_tree.links;bsdf=nodes.get('Principled BSDF')
 for typ,input in [('BaseColor','Base Color'),('Metallic','Metallic'),('Roughness','Roughness'),('Normal','Normal')]:
  n=nodes.new('ShaderNodeTexImage');n.image=bpy.data.images.load(str(src/'Textures'/('Mat_ComputerSetup_'+typ+'.png')))
  if typ!='BaseColor': n.image.colorspace_settings.name='Non-Color'
  if typ=='Normal':
   normal=nodes.new('ShaderNodeNormalMap');links.new(n.outputs['Color'],normal.inputs['Color']);links.new(normal.outputs['Normal'],bsdf.inputs[input])
  else: links.new(n.outputs['Color'],bsdf.inputs[input])
 for obj in bpy.context.scene.objects:
  if obj.type=='MESH': obj.data.materials.clear();obj.data.materials.append(mat)
 # Preserve model geometry, center horizontally and put on floor in Blender Z-up.
 objects=[o for o in bpy.context.scene.objects if o.type=='MESH']
 pts=[o.matrix_world@Vector(v) for o in objects for v in o.bound_box]
 mn=Vector(tuple(min(p[i] for p in pts) for i in range(3)));mx=Vector(tuple(max(p[i] for p in pts) for i in range(3)))
 center=Vector(((mn.x+mx.x)/2,(mn.y+mx.y)/2,mn.z))
 for obj in objects: obj.location-=center
 bpy.context.view_layer.update()
 bpy.ops.export_scene.gltf(filepath=str(out/(label.lower()+'.glb')),export_format='GLB',export_yup=True)
 bpy.ops.export_scene.gltf(filepath=str(out/(label.lower()+'.gltf')),export_format='GLTF_SEPARATE',export_yup=True)
 report[label]={'original_bounds_blender':[list(mn),list(mx)],'size':list(mx-mn),'normalized':'center XY, base Z=0 before Y-up GLB export'}
print(json.dumps(report,indent=2));(out/'conversion.json').write_text(json.dumps(report,indent=2))
