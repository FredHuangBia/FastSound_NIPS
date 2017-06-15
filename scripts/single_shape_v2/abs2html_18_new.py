import subprocess, os, sys

epoch = sys.argv[1]
split = int(sys.argv[2])
gpu = sys.argv[3]

ROOT = '/data/vision/billf/object-properties/sound/sound/primitives/exp/primV2d/'

fileIn = open('/data/vision/billf/object-properties/sound/sound/primitives/www/primV2d_cnnB_soundnet8_pretrainnone_mse1_LR0.001/%s/target-%02d.txt'%(epoch,split),'r')
fileOut = open('/data/vision/billf/object-properties/sound/sound/primitives/www/primV2d_cnnB_soundnet8_pretrainnone_mse1_LR0.001/%s/synth-%02d.txt'%(epoch,split),'w')

for line in fileIn:
	line = line.split()
	cmd = 'CUDA_VISIBLE_DEVICES=%s th /data/vision/billf/object-properties/sound/sound/primitives/code/inference/primV2d_synthesis_new.lua %s 5' %(gpu,line[0])
	print(cmd)
	subprocess.call(cmd, shell=True)
	os.chdir(ROOT+line[0])
	i = open('best.txt','r').readlines()[0]
	i = i.split()
	cmd = 'cp ./%s/sound.wav %s' %(i[0], line[1])
	subprocess.call(cmd, shell=True)
	fileOut.write("%s %s %s %s %s %s\n" %(line[0], i[1],i[2],i[3],i[4],i[5]))
	fileOut.flush()





