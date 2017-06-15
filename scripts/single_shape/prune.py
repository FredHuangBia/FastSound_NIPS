import os

ROOT = '/data/vision/billf/object-properties/sound/sound/primitives/result/single_shape/v1/'

for scene0 in range(6):
	for scene1 in range(7):
		os.chdir(ROOT+'scene-%d-%d'%(scene0,scene1))
		for mat0 in range(6):
			for mat1 in range(6):
				for mat2 in range(6):
					soundPath = ROOT+'scene-%d-%d/'%(scene0,scene1)+'mat-10-10-10-%d-%d-%d/'%(mat0,mat1,mat2)+'sound.wav'
					if os.path.exists(soundPath) is False:
						print('scene-%d-%d/'%(scene0,scene1)+'mat-10-10-10-%d-%d-%d'%(mat0,mat1,mat2))