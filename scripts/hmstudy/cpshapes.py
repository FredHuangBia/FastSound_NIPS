import subprocess
ROOT = '/data/vision/billf/object-properties/sound/sound/primitives/data/v2b/'
shape = '/data/vision/billf/object-properties/sound/sound/primitives/data/hmstudy/shape/obj/'

prim = [62,187,312,437,553,679,805,931,1057,1183,1309,1435,1593,1713]

for i,j in enumerate(prim):
	cmd = 'cp %s%d/%d.orig.obj %s%d.obj' %(ROOT,j*10,j*10,shape,i+1)
	# print(cmd)
	subprocess.call(cmd,shell=True)