from subprocess import call
import os

origin = '/data/vision/billf/object-properties/sound/sound/primitives/exp/random_v4/test_sound_prep'
dst = '/data/vision/billf/object-properties/sound/sound/primitives/exp/random_v4/test_sound'

good = [2,5,9,14,18,19,21,22,23,25,27,34,41,46,51,60,61,62,63,64,65,67,68,72,73,75,76,78,85,86,91,94,96,98,100,104,107,114,121,123,127,131,132,136,138,140,142,145,148,154]

for i, j in enumerate(good):
	cmd = 'cp -r %s %s' %(os.path.join(origin,str(j)), os.path.join(dst,str(i)))
	call(cmd,shell=True)
