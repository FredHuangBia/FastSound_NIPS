import matplotlib.pyplot as plt
import os

# root = '/data/vision/billf/object-properties/sound/sound/primitives/exp/'
root = '/Users/zhengjia/Desktop/sound/sound/primitives/exp/random_test/result/'
# root = '/data/vision/billf/object-properties/sound/sound/primitives/exp/random_test/result/'
# 
# tasks = ['stft_inverse_0075_03_rotation','stft_inverse_0100_02_rotation','stft_inverse_0100_02_None','stft_inverse_0075_03_None']
tasks = ['stft_inverse_0200_01_None','stft_inverse_0020_10_None']

def plot(ll_list,name,kind):
	plt.plot(range(len(ll_list)),ll_list)
	plt.xlabel("iterations")
	plt.ylabel(kind)
	plt.savefig("%s.%s.png"%(name,kind),dpi=300,bbox_inches='tight')
	plt.clf()


def parse_log(log_path, interval=1):
	dist_log = open(os.path.join(log_path,'distance_log.txt'),'r')
	all_dist = [line.split()[0] for line in dist_log]
	prob_log = open(os.path.join(log_path,'prob_log.txt'),'r')
	all_ll = [line.split()[0] for line in prob_log]
	prob_log.close()
	dist_log.close()
	dist = []
	ll = []
	for i in range(len(all_dist)):
		if i%interval==0:
			dist.append(all_dist[i])
			ll.append(all_ll[i])
	return ll, dist

def main():
	# target_file = open('/data/vision/billf/object-properties/sound/sound/primitives/www/primV3b_cnnF_soundnet8_pretrainnone_mse1_LR0.001/151/targets.txt','r')
	# target_file = open('/Users/qiujiali/Desktop/Remote/sound/primitives/www/primV3b_cnnF_soundnet8_pretrainnone_mse1_LR0.001/151/targets.txt','r')
	# targets = []
	# for line in target_file:
	# 	targets.append(line.split()[0])
	targets = [str(i) for i in range(20)]
	for task in tasks:
		n = int(task.split('_')[3])
		paras = 7
		if task.split('_')[-1] != 'None':
			paras -= 1
		interval = n*paras
		os.chdir(root+task)
		print(os.getcwd())
		for j in targets:
			ll, dist = parse_log(os.path.join(root,task,j),interval)
			if not os.path.exists(os.path.join(root,task,'plots')):
				os.mkdir(os.path.join(root,task,'plots'))
			os.chdir(os.path.join(root,task,'plots'))
			plot(ll,j,'ll')
			plot(dist,j,'dist')

if __name__ == '__main__':
	main()
