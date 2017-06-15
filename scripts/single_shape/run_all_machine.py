''' Task allocation across all machines. '''
import os, sys, argparse, shutil, subprocess, glob
from functools import partial
from multiprocessing.dummy import Pool

# cpu_machine = [1, 2, 3, 5, 6, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
gpu_machine = [3, 3, 3, 3]
cpu_machine = []
# gpu_machine = [3,3,3,3,5,5,5,9,9,9,9,10,10,10,10,11,11,11,11,13,13,13,13,15,15,15,15,16,16,16,16]
# cpu_tasks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
cpu_tasks = []
# gpu_tasks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
# cpu_tasks = [1,2,3,4,5,6,7,8]
gpu_tasks = [0,1,2,3]

precal_path = '/data/vision/billf/object-properties/sound/sound/primitives/scripts/single_shape/gen_sound_machine.py'

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
