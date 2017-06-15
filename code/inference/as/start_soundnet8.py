import sys
import subprocess
import os

GPU = int(sys.argv[1])
random = int(sys.argv[2])
split = int(sys.argv[3])

target_file = open('/data/vision/billf/object-properties/sound/sound/primitives/www/primV3b_cnnF_soundnet8_pretrainnone_mse1_LR0.001/151/targets.txt','r')
targets = [int(line.split()[0]) for line in target_file]

if random==1:
	path = '/data/vision/billf/object-properties/sound/sound/primitives/exp/primV3b_soundnet8_random/'
else:
	path = '/data/vision/billf/object-properties/sound/sound/primitives/exp/primV3b_soundnet8_prior/'

finished = []
try:
	finished_file = open(path+'finished.txt','r')
	for line in finished_file:
		finished.append(int(line))
	finished_file.close()
except IOError:
	finished_file = open(path+'finished.txt','w')
	finished_file.close()

for target in targets[8*split:8*(split+1)]:
	if not target in finished:
		cmd = 'CUDA_VISIBLE_DEVICES='+str(GPU)+' th '+' primV3b_soundnet8.lua '+str(target)+' 30 '+str(random)
		print(cmd)
		subprocess.call(cmd,shell=True)
		finished_file = open(path+'finished.txt','a+')
		finished_file.write(str(target)+'\n')
		finished_file.close()
		finished_file = open(path+'finished.txt','r')
		for line in finished_file:
			if not int(line) in finished:
				finished.append(int(line))
		finished_file.close()