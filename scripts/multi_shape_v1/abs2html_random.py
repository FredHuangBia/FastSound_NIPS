import subprocess, os, sys

epoch = sys.argv[1]
split = int(sys.argv[2])
gpu = sys.argv[3]

ROOT = '/data/vision/billf/object-properties/sound/sound/primitives/exp/primV3b_random/'
FILE = '/data/vision/billf/object-properties/sound/sound/primitives/www/primV3b_cnnF_soundnet8_pretrainnone_mse1_LR0.001/'

fileIn = open(FILE+'%s/target-%02d.txt'%(epoch,split),'r')
fileOut = open(ROOT+'synth_random-%02d.txt'%(split),'w')

for line in fileIn:
	line = line.split()
	cmd = 'CUDA_VISIBLE_DEVICES=%s th /data/vision/billf/object-properties/sound/sound/primitives/code/inference/as/primV3b_synthesis_random.lua %s 30' %(gpu,line[0])
	print(cmd)
	subprocess.call(cmd, shell=True)
	os.chdir(ROOT+line[0])
	i = open('best.txt','r').readlines()[0]
	i = i.split()
	# cmd = 'cp ./%s/sound.wav %s' %(i[0], line[1])
	# subprocess.call(cmd, shell=True)
	fileOut.write("%s %s %s %s %s %s %s %s\n" %(line[0], i[1],i[2],i[3],i[4],i[5], i[6], i[7]))
	fileOut.flush()





