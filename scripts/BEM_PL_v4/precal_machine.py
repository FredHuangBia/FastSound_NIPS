''' 
	Parallel precalculation 
	Usage: python3 precal_machine.py task_num
'''
import os, sys, subprocess, json
from multiprocessing.dummy import Pool as ThreadPool
# import time

ROOT = '/data/vision/billf/object-properties/sound/sound/primitives/scripts/BEM_PL_v4/'
THREADNUM = 2
argv = sys.argv[1]

cmdList = []
fin = open(ROOT+"run4/input-"+argv.zfill(2)+".txt","r")

for line in fin:
	cmdList=cmdList+line.split()
print (cmdList)

for i in range(int(len(cmdList)/THREADNUM)):
	renew = 'kinit -R'
	subprocess.call(renew,shell=True)
	subcmd = cmdList[2*i*THREADNUM:2*(i+1)*THREADNUM]
	print(subcmd)
	pids = set()
	for x in range(int(len(subcmd)/2)):
		obj_num = int(subcmd[x*2])
		mat_num = int(subcmd[x*2+1])
		print ('BEM!!!!')
		cmd = 'bash %srun_recalc_bem.sh %02d %d'%(ROOT,obj_num,mat_num)
		print (cmd)
		p = subprocess.Popen(cmd, shell=True)
		pids.add(p.pid)
	while pids:
		pid, retval = os.wait()
		print('%s finished' %pid)
		pids.remove(pid)