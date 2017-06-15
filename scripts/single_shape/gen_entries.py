import os

ROOT = '/data/vision/billf/object-properties/sound/sound/primitives/scripts/single_shape/'
fileOut = open(ROOT+'run6/sound_entries.txt','w')

obj0 = 101	# static floor
obj1 = 0	# moving object

smat0, smat1, smat2 = 10,10,10 # need new material for floor

for scene0 in range(6):
	for scene1 in range(7): 
		for mat0 in range(1): 
			for mat1 in range(6):
				for mat2 in range(6):
					cmd = '%d %d %d %d %d %d %d %d %d %d\n' %(scene0, scene1, obj0, smat0, smat1, smat2, obj1, mat0, mat1, mat2)
					fileOut.write(cmd)
