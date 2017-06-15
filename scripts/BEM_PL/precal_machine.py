''' Parallel precalculation '''
import os, sys, subprocess, json
from multiprocessing.dummy import Pool as ThreadPool
# import time

ROOT = '/data/vision/billf/object-properties/sound/sound/primitives/scripts/BEM_PL/'
THREADNUM = 8
argv = sys.argv[1]

cmdList = []
fin = open(ROOT+"run8/input-"+argv.zfill(2)+".txt","r")

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
		p = subprocess.Popen(cmd, shell=True)
		pids.add(p.pid)
	while pids:
		pid, retval = os.wait()
		print('%s finished' %pid)
		pids.remove(pid)


# def chunks(l, n):
#     """Yield successive n-sized chunks from l."""
#     for i in range(0, len(l), n):
#         yield l[i:i + n]

# def run_one_branch(cmdList):
# 	print("CALL!!!")
# 	print(cmdList)
# 	for x in range(int(len(cmdList)/4)):
# 		obj_num = cmdList[x*4]
# 		sub_num = cmdList[x*4+1]
# 		mat_num0 = int(cmdList[x*4+2])
# 		mat_num1 = int(cmdList[x*4+3])

# 		print ('BEM!!!!')
# 		cmd = 'bash %srun_recalc_bem.sh %s %s %d %d'%(ROOT,obj_num,sub_num,mat_num0,mat_num1)
# 		print (cmd)
# 		# time.sleep(0.5)
# 		subprocess.call(cmd,shell='True')

# cmdList = []
# fin = open(ROOT+"run/input-"+argv.zfill(2)+".txt","r")

# for line in fin:
# 	cmdList=cmdList+line.split()
# print (cmdList)

# pool = ThreadPool(THREADNUM) 
# results = pool.map(run_one_branch, list(chunks(cmdList,int(len(cmdList)/THREADNUM))))
