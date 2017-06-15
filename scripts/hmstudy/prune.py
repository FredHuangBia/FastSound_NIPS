import subprocess

# bad = [1,3,8,9,14,16,17,18,20,25,29,30,33,35,41,46,47,52,53,54,55,58,62,65,66,68,70,72,76,78,81,85,86,88,92,93,98]
# bad = [14,18,29,33,35,46,55,65,66,93,98]
# bad = [14]
ROOT = '/data/vision/billf/object-properties/sound/sound/primitives/data/hmstudy/audio/'

for i in bad:
	cmd = 'python3 /data/vision/billf/object-properties/sound/sound/primitives/scripts/hmstudy/random_gen_sound.py %d %d' %(i,i)
	subprocess.call(cmd,shell=True)
	cmd = 'cp %s %s' %(ROOT+'%03d'%(i)+'/sound.wav', ROOT+'../prune/%03d.wav'%i)
	subprocess.call(cmd,shell=True)