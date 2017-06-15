import matplotlib.pyplot as plt
import os

# root = '/data/vision/billf/object-properties/sound/sound/primitives/exp/'
root = '/Users/qiujiali/Desktop/Remote/sound/primitives/exp/'

# tasks = ['primV3b_nmh_corl_random_2','primV3b_nmh_corl_random_4','primV3b_nmh_corl_random_5', 'primV3b_nmh_stft_random_2', 'primV3b_nmh_stft_random_3', 'primV3b_nmh_stft_random_4', 'primV3b_nmh_stft_random_5']
# tasks = ['primV3b_nmh_corl_random_6','primV3b_nmh_stft_random_6']
tasks = ['primV3b_nmh_stft_random_7']
def plot(ll_list,name,kind):
	plt.plot(range(len(ll_list)),ll_list)
	plt.xlabel("iterations")
	plt.ylabel(kind)
	plt.savefig("%s.%s.png"%(name,kind),dpi=300,bbox_inches='tight')
	plt.clf()


def parse_log(log_path):
	log = open(log_path,'r')
	dist = []
	ll = []
	for line in log:
		line = line.split()
		dist.append(line[2])
		ll.append(line[1])
	return ll, dist

def main():
	# target_file = open('/data/vision/billf/object-properties/sound/sound/primitives/www/primV3b_cnnF_soundnet8_pretrainnone_mse1_LR0.001/151/targets.txt','r')
	target_file = open('/Users/qiujiali/Desktop/Remote/sound/primitives/www/primV3b_cnnF_soundnet8_pretrainnone_mse1_LR0.001/151/targets.txt','r')
	targets = []
	for line in target_file:
		targets.append(line.split()[0])
	for i in tasks:
		os.chdir(root+i)
		print(os.getcwd())
		for j in targets:
			ll, dist = parse_log(root+i+'/'+j+'/dist.log')
			os.chdir(root+'plots/'+i)
			plot(ll,j,'ll')
			plot(dist,j,'dist')

if __name__ == '__main__':
	main()
