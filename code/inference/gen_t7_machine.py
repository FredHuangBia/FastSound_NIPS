import threading
import subprocess

def run(start, to):
	print('new thread spawned: ','th','gen_t7.lua',start,to)
	subprocess.call(['th','gen_t7.lua',str(start),str(to)])

threads = []

# for i in [8]:
# 	start = i*10000
# 	end = (i+1)*10000
# 	t = threading.Thread(target=run, args=(start,end))
# 	threads.append(t)

for i in range(10):
	start = 80000 + i*1000
	end = 80000 +(i+1)*1000
	t = threading.Thread(target=run, args=(start,end))
	threads.append(t)


for t in threads:
	t.start()

for t in threads:
	t.join()

