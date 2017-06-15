import os, subprocess

ROOT_DIR = '/data/vision/billf/object-properties/sound/sound/primitives/data/v2.3/'
objName = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
subNum = [125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 126, 120, 120]

for i in objName:
	for j in range(subNum[i]):
		os.chdir(ROOT_DIR+'%d/%d'%(i,j))
		cmd = 'rm -r mat*'
		print(ROOT_DIR+'%d/%d'%(i,j))
		subprocess.call(cmd,shell=True)
