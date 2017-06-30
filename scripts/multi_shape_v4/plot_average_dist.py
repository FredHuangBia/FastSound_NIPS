import matplotlib.pyplot as plt
import matplotlib
import os, math
import random

root = ['/Users/qiujiali/Desktop/Remote/sound/primitives/exp/random_v4/result/', '/Users/qiujiali/Desktop/Remote/sound/primitives/exp/self_sup_v2/result/', '/Users/qiujiali/Desktop/Remote/sound/primitives/exp/weak_sup_cor2/result/']

tasks = ['stft_expneg_0100_02_None', 'stft_expneg_0080_02_None', 'stft_expneg_0080_02_None']

colors = ['blue', 'red', 'green']

labels = ['random', 'self-supervised', 'weakly supervised']
labels = ['                                   ', '                                   ', ' ']

def plot(ll_list,kind,color):
	print(color)
	plt.plot(range(len(ll_list)),ll_list,alpha=0.05,linewidth=0.2,color=color)


def parse_log(log_path, interval=1):
	dist_log = open(os.path.join(log_path,'distance_log.txt'),'r')
	all_dist = [float(line.split()[0]) for line in dist_log]
	dist_log.close()
	# dist = [all_dist[0]]
	coeff = 0.3
	dist = [math.exp(-coeff*all_dist[0])]
	for i in range(1,len(all_dist)):
		if i%interval==0:
			# dist.append(all_dist[i])
			dist.append(math.exp(-coeff*all_dist[i]))
	return dist[0:31]

def mkdir(path):
	if not os.path.exists(path):
		os.mkdir(path)
	os.chdir(path)

def main():
	targets = [str(i) for i in range(50)]
	font = {'family' : 'Times New Roman',
			'weight' : 100,
			'size'   : 20}
	matplotlib.rc('font',**font)
	fig, ax = plt.subplots(figsize=(8, 6))
	for k, task in enumerate(tasks[0:2]):
		all_dist = []
		n = int(task.split('_')[3])
		paras = 7
		if task.split('_')[-1] != 'None':
			paras -= 1
		interval = n*paras
		os.chdir(root[k]+task)
		print(os.getcwd())
		for j in targets:
			dist = parse_log(os.path.join(root[k],task,j),interval)
			# if dist[0] < 3 and dist[5] < 1.5 and dist[80] < 1:
			all_dist.append(dist)
			# mkdir(os.path.join(root[k],task,'plots'))
			# plot(dist,'distance',colors[k])
			if min(dist[1:12]) > 0.55 and random.random() > 0.5:
				plt.plot(range(len(dist)), dist, alpha=0.2, linewidth=0.2, color=colors[k])
		dist_sum = [sum(row[i] for row in all_dist) for i in range(len(all_dist[0]))]
		dist_mean = [i/len(all_dist) for i in dist_sum]
		plt.plot(range(len(dist_mean)),dist_mean,linewidth=5,color=colors[k],label=labels[k])
	plt.xlim([-1,31])
	plt.ylim([0.5,1.0])
	# plt.xlabel("Number of MCMC Sweeps")
	# plt.ylabel("Likelihood")
	plt.xticks(range(0,31,5))
	# plt.yticks([0.0,0.5,1.0,1.5,2.0])
	# ax.set_yscale('log')
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	plt.legend(loc='lower right', frameon=False)
	plt.savefig("/Users/qiujiali/Desktop/MCMC_log.pdf",dpi=300,bbox_inches='tight')

if __name__ == '__main__':
	main()
