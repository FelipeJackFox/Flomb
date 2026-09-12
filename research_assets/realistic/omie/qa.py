import bpy,pathlib,math,json
from mathutils import Vector
root=pathlib.Path(__file__).resolve().parent
bpy.ops.wm.read_factory_settings(use_empty=True)
for label,pos in [('monitor',(0,0,0)),('keyboard',(0,-.32,0)),('mouse',(.4,-.3,0)),('computer',(-.55,0,0))]:
 bpy.ops.import_scene.gltf(filepath=str(root/'glb'/(label+'.glb')))
 for o in bpy.context.selected_objects:
  o.location+=Vector(pos)
  if label=='monitor':
   o.rotation_mode='XYZ';o.rotation_euler.z+=math.pi
scene=bpy.context.scene
scene.render.engine='CYCLES';scene.cycles.samples=20
scene.world=bpy.data.worlds.new('World');scene.world.use_nodes=True;scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.25,.25,.25,1)
for xyz,power,size in [((1,-2,3),250,3),((-2,-1,2),200,3)]:
 bpy.ops.object.light_add(type='AREA',location=xyz);bpy.context.object.data.energy=power;bpy.context.object.data.shape='DISK';bpy.context.object.data.size=size
bpy.ops.object.camera_add(location=(1.2,-2,1.1));cam=bpy.context.object;cam.rotation_euler=(Vector((0,0,.2))-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.type='ORTHO';cam.data.ortho_scale=1.9;scene.camera=cam
scene.render.resolution_x=1200;scene.render.resolution_y=900;scene.render.resolution_percentage=100
scene.render.filepath=str(root/'qa.png');bpy.ops.render.render(write_still=True)
