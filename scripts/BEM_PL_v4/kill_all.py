''' 
	kill all my tasks on listed machines 
	Usage: python kill_all.py
'''

import subprocess

# cpu_machine = []
cpu_machine = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38]
gpu_machine = [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20]
# gpu_machine = [4,15,16]
# gpu_machine = []
for x in cpu_machine:
	cmd = 'ssh -f vision%02d \'pkill -u qiujiali\'' %x
	print cmd
	subprocess.call(cmd,shell=True)
	print

for x in gpu_machine:
	cmd = 'ssh -f visiongpu%02d \'pkill -u qiujiali\'' %x
	print cmd
	subprocess.call(cmd,shell=True)
	print
