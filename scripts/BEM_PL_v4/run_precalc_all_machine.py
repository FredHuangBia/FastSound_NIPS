''' 
	Task allocation across all machines. 
	Usage: python run_precalc_all_machine.py
'''
import os, sys, argparse, shutil, subprocess, glob
from functools import partial
from multiprocessing.dummy import Pool

# cpu_machine = [8,9,20,21,22,23,24,25,26,27,28,29,32,33,34,36,37,38]
# cpu_tasks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
# cpu_machine = [13,15,16,17,18,19,20,22,24,28]
# cpu_tasks = [0,1,2,3,4,5,6,7,8,9]
# cpu_machine = [1,2,3,5,14,16,17,22,37]
# cpu_tasks = [1,2,3,4,5,6,7,8,9]
cpu_machine = []
cpu_tasks = []
# gpu_machine = [4]
# gpu_tasks = [9]
# gpu_machine = [2,4,5,12,13,14,16,18,20]
# gpu_tasks =   [0,1,2, 3, 4, 5, 6, 7, 8]

gpu_machine = [12,13,14,16,18]
gpu_tasks = [0,1,2,3,4]
precal_path = '/data/vision/billf/object-properties/sound/sound/primitives/scripts/BEM_PL_v4/precal_machine.py'

assert(len(cpu_machine) >= len(cpu_tasks)), "more CPU machines required!"
assert(len(gpu_machine) >= len(gpu_tasks)), "more GPU machines required!" 

for x in range(0,len(cpu_tasks)):
    cmd = 'ssh -f vision%02d \'nohup python3 %s %d\' > vision%02d.txt' %(cpu_machine[x], precal_path, cpu_tasks[x], cpu_machine[x])
    print cmd
    subprocess.call(cmd,shell=True)
    print

for x in range(0,len(gpu_tasks)):
    cmd = 'ssh -f visiongpu%02d \'nohup python3 %s %d\' > visiongpu%02d.txt' %(gpu_machine[x], precal_path, gpu_tasks[x],gpu_machine[x])
    print cmd
    subprocess.call(cmd,shell=True)
    print
