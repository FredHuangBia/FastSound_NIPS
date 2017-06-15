import os
import sys
import subprocess
import json

argv = sys.argv[1]

cmdList = []

fin = open("/data/vision/billf/object-properties/sound/sound/primitives/data/tet/split"+argv.zfill(2)+".txt","r")

for line in fin:
	cmdList=cmdList+line.split()


isbem = False
if len(sys.argv)>2:
	isbem = True
# cmdList = argv.split()

print cmdList

for x in range(len(cmdList)/2):
	obj_num = int(cmdList[x*2])
	mat_num = int(cmdList[x*2+1])
	print obj_num
	print mat_num
	if isbem:
		print 'BEM!!!!'
		subprocess.call('bash /data/vision/billf/object-properties/sound/sound/primitives/scripts/run_recalc_bem.sh %d %d'%(obj_num,mat_num),shell='True')
	else:
		print 'FEM!!!!'
		subprocess.call('bash /data/vision/billf/object-properties/sound/sound/primitives/scripts/run_precalc.sh %d %d &'%(obj_num,mat_num),shell='True')
	print 'bash /data/vision/billf/object-properties/sound/sound/primitives/scripts/run_precalc.sh %d %d &'%(obj_num,mat_num)