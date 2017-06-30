#!/usr/bin/python
import bpy, os, subprocess

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()


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

	volume = bpy.context.scene.mcell.meshalyzer['volume']
	assert(volume > 0), "ERROR: zero volume mesh!"
	print(volume)
	volumeOut = open(os.path.join(pathToPly,"volume.txt"),'w')
	volumeOut.write("%.3f" %volume*125)
	volumeOut.close()

for i in [14,15,16,17]:
	ROOT = '/data/vision/billf/object-properties/sound/sound/primitives/data/v4/shapes'
	get_volume(os.path.join(ROOT, str(i)), '%d'%i)



