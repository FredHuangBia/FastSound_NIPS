'''
	Given unit size shapes, scale it to 0.2 for both obj and ply using blender
	Usage: blender -b -P resize.py
'''

import bpy
import os

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
inFolder = "/data/vision/billf/object-properties/sound/sound/primitives/data/v4s/shapes/unit_shapes/"
outRootFolder = "/data/vision/billf/object-properties/sound/sound/primitives/data/v4s/shapes/"

# obj_list = list(range(14))
obj_list = list(range(14,18))

scene = bpy.context.scene

def CreateDir(path):
    if not os.path.exists(path):
        os.makedirs(path)

for i in obj_list:
	obj = str(i)+'.orig.obj'
	print("==================================================")
	print("Processing %s" %obj)
	bpy.ops.import_scene.obj(filepath = os.path.join(inFolder,str(i),obj))

	my_object = bpy.context.scene.objects[(obj[0:-4])]
	my_object.select = True
	scene.objects.active = my_object

	my_object.scale = (0.1, 0.1, 0.1)
	bpy.ops.object.transform_apply(scale=True)

	outFolder = outRootFolder+str(i)
	CreateDir(outFolder)
	bpy.ops.export_mesh.ply(filepath=os.path.join(outFolder,str(i)+'.ply'), use_mesh_modifiers=False, use_normals=False, use_uv_coords=False, use_colors=False, axis_forward='Y', axis_up='Z', global_scale=1.0)
	bpy.ops.export_scene.obj(filepath=os.path.join(outFolder,str(i)+'.orig.obj'), use_selection=True, use_animation=False, use_mesh_modifiers=False, use_edges=False, use_smooth_groups=False, use_smooth_groups_bitflags=False, use_normals=False, use_uvs=False, use_materials=False, use_triangles=True, use_nurbs=False, use_vertex_groups=False, use_blen_objects=False, group_by_object=False, group_by_material=False, keep_vertex_order=False, axis_forward='-Z', axis_up='Y', global_scale=1.0, path_mode='AUTO')

	# bpy.ops.object.mode_set(mode='OBJECT')
	bpy.ops.object.select_all(action='SELECT')
	bpy.ops.object.delete(use_global=False)
	print("finished processing %s" %obj)

