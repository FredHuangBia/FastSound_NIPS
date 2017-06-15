import os, configparser
config = configparser.ConfigParser()

ROOT = '/data/vision/billf/object-properties/sound/sound/primitives/data/primV2d/'

first = []
second = []
third = []
fourth = []
fifth = []
sixth = []

collisionfile = open(ROOT+"../collisions.dat",'w')

for i in range(1, 100001):
	sound_dir = ROOT+'%06d' %i
	os.chdir(sound_dir)
	if not os.path.exists(sound_dir+'/sound.wav'):
		print(sound_dir)
		print("SOUND %06d is missing" %i)
	else:
		config.read(sound_dir+'/sim/click.ini')
		time = config['collisions']['time'].split()
		first.append(round(float(time[0])*60))
		second.append(round(float(time[1])*60))
		if len(time) > 2:
			third.append(round(float(time[2])*60))
		else:
			third.append(0)
		if len(time) > 3:
			fourth.append(round(float(time[2])*60))
		else:
			fourth.append(0)
		if len(time) > 4:
			fifth.append(round(float(time[2])*60))
		else:
			fifth.append(0)
		if len(time) > 5:
			sixth.append(round(float(time[2])*60))
		else:
			sixth.append(0)

collisionfile.write("[first]\nframe = ")
for i in first:
	collisionfile.write("%d " %i)
collisionfile.write("\n\n[second]\nframe = ")
for i in second:
	collisionfile.write("%d " %i)
collisionfile.write("\n\n[third]\nframe = ")
for i in third:
	collisionfile.write("%d " %i)
collisionfile.write("\n\n[fourth]\nframe = ")
for i in fourth:
	collisionfile.write("%d " %i)
collisionfile.write("\n\n[fifth]\nframe = ")
for i in fifth:
	collisionfile.write("%d " %i)
collisionfile.write("\n\n[sixth]\nframe = ")
for i in sixth:
	collisionfile.write("%d " %i)