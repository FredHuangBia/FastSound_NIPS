#!/usr/bin/python
import bpy, os, subprocess

inFolder = "/data/vision/billf/object-properties/sound/sound/primitives/data/v2.3/"
newFolder = "/data/vision/billf/object-properties/sound/sound/primitives/data/v2b/"
scripts = "/data/vision/billf/object-properties/sound/sound/primitives/scripts/single_shape_v3/"

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

num = [125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 126, 120, 120]
mat0 = [0]
mat1 = [0,1,2,3,4,5,6,7,8,9]

def get_volume(pathToPly, plyName):
	bpy.ops.object.select_all(action='SELECT')
	bpy.ops.object.delete()
	print("==================================================")
	print("Processing object %s.ply" %plyName)
	inPath = os.path.join(pathToPly, plyName+'.ply')
	bpy.ops.import_mesh.ply(filepath = inPath)
	my_object = bpy.context.scene.objects[0]
	my_object.select = True
	bpy.context.scene.objects.active = my_object

	bpy.ops.mcell.init_cellblender()
	bpy.ops.mcell.meshalyzer()
	# normal_status = bpy.context.scene.mcell.meshalyzer['normal_status']

	# if normal_status != "Outward Facing Normals":
	# 	print("WARNING: the mesh has incorrect normals!")
	# 	bpy.ops.mesh.print3d_clean_non_manifold()
	# 	bpy.ops.mcell.meshalyzer()
	# 	normal_status = bpy.context.scene.mcell.meshalyzer['normal_status']
	# 	assert(normal_status == "Outward Facing Normals"), "ERROR: bad mesh!"

	volume = bpy.context.scene.mcell.meshalyzer['volume']
	assert(volume > 0), "ERROR: zero volume mesh!"
	print(volume)
	volumeOut = open(os.path.join(pathToPly,"volume.txt"),'w')
	volumeOut.write("%.3f" %volume)
	volumeOut.close()
	objPath = os.path.join(pathToPly, plyName+'.orig.obj')
	bpy.ops.export_scene.obj(filepath=objPath, axis_forward='-Z', axis_up='Y', use_selection=True, use_animation=False, use_mesh_modifiers=False, use_edges=False, use_smooth_groups=False, use_smooth_groups_bitflags=False, use_normals=False, use_uvs=False, use_materials=False, use_triangles=True, use_nurbs=False, use_vertex_groups=False, use_blen_objects=False, group_by_object=False, group_by_material=False, keep_vertex_order=False, global_scale=1.0, path_mode='ABSOLUTE')


def migrate():
	counter = 0
	record = open(newFolder+'directories.txt','w')
	for i in range(14):
		for j in range(num[i]):
			os.chdir(os.path.join(inFolder,'%d/%d'%(i,j)))
			get_volume(os.getcwd(), '%d-%d' %(i,j))
			for u in mat0:
				for v in mat1:
					cmd = scripts + 'migrate.sh' + ' %d %d %d %d %d' %(i,j,u,v,counter)
					subprocess.call(cmd, shell=True)
					record.write("%d\t%d/%d/mat-%d-%d\n" %(counter, i, j, u, v))
					record.flush()
					counter += 1

if __name__ == '__main__':
	# pathToPly = '/data/vision/billf/object-properties/sound/sound/primitives/data/v2.3/0/0/'
	# plyName = '0-0'
	# get_volume(pathToPly, plyName)
	migrate()
	



