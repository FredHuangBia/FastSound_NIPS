import sys
import subprocess
import os

GPU = int(sys.argv[1])
split = int(sys.argv[2])

target_file = open('/data/vision/billf/object-properties/sound/sound/primitives/www/primV3b_5000_cnnF_soundnet8_pretrainnone_mse1_LR0.001/50/targets.txt','r')
targets = [int(line.split()[0]) for line in target_file]

path = '/data/vision/billf/object-properties/sound/sound/primitives/exp/primV3b_5000_weight/'

finished = []
try:
	finished_file = open(path+'finished.txt','r')
	for line in finished_file:
		finished.append(int(line))
	finished_file.close()
except:
	finished_file = open(path+'finished.txt','w')
	finished_file.close()

for target in targets[6*split:6*(split+1)]:
	if not target in finished:
		cmd = 'CUDA_VISIBLE_DEVICES='+str(GPU)+' th '+' primV3b_5000_synthesis_weight.lua '+str(target)+' 30'
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