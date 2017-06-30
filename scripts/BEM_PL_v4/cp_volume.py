from subprocess import call
import os

root = '/data/vision/billf/object-properties/sound/sound/primitives/data/v4s/shapes/unit_shapes/'
dst_root = '/data/vision/billf/object-properties/sound/sound/primitives/data/v4s/BEMs/'


for i in range(14,18):
	for j in range(10):
		cmd = 'cp %s %s' %(os.path.join(root,str(i),'volume.txt'), os.path.join(dst_root,'%02d%d'%(i,j)))
		call(cmd,shell=True)