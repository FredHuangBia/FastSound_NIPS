import os, subprocess

ROOT = '/data/vision/billf/object-properties/sound/sound/primitives/data/hmstudy/audio/'
HMSTUDY = '/data/vision/billf/object-properties/sound/sound/primitives/data/hmstudy/'
os.chdir(HMSTUDY)
for j in range(1,101):
	cmd = 'cp %s %s' %(ROOT+'%03d'%(j)+'/sound.wav', './wav/%03d.wav'%j)
	subprocess.call(cmd,shell=True)

# [1,3,8,9,14,16,17,18,20,25,29,30,33,35,41,46,47,52,53,54,55,58,62,65,66,68,70,72,76,78,81,85,86,88,92,93,98]