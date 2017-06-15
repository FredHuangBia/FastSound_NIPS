''' Parallel precalculation '''
import os, sys, subprocess, json
from multiprocessing.dummy import Pool as ThreadPool
import time

ROOT = '/data/vision/billf/object-properties/sound/sound/primitives/scripts/BEM_PL/'
THREADNUM = 10
argv = sys.argv[1]

# def chunks(l, n):
#     """Yield successive n-sized chunks from l."""
#     for i in range(0, len(l), n):
#         yield l[i:i + n]

# def run_one_branch(cmdList):

cmdList = []
fin = open(ROOT+"run2/input-"+argv.zfill(2)+".txt","r")

for line in fin:
	cmdList=cmdList+line.split()
print (cmdList)

for i in range(int(len(cmdList)/THREADNUM)):
	renew = 'kinit -R'
	subprocess.call(renew,shell=True)
	subcmd = cmdList[4*i*THREADNUM:4*(i+1)*THREADNUM]
	print(subcmd)
	pids = set()
	for x in range(int(len(subcmd)/4)):
		obj_num = subcmd[x*4]
		sub_num = subcmd[x*4+1]
		mat_num0 = int(subcmd[x*4+2])
		mat_num1 = int(subcmd[x*4+3])
		print ('BEM!!!!')
		cmd = 'bash %srun_recalc_bem.sh %s %s %d %d'%(ROOT,obj_num,sub_num,mat_num0,mat_num1)
		print (cmd)
		# subprocess.call(cmd,shell='True')
		subprocess.call(renew,shell=True)
		p = subprocess.Popen('bash %s' %(ROOT+'run/sleep.sh'), shell=True)
		# p = subprocess.Popen(cmd, shell=True)
		pids.add(p.pid)
	while pids:
		pid, retval = os.wait()
		print('%s finished' %pid)
		pids.remove(pid)

# pool = ThreadPool(THREADNUM) 
# results = pool.map(run_one_branch, list(chunks(cmdList,int(len(cmdList)/THREADNUM))))
