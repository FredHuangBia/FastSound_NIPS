'''
	This script adds acceptance rate calculation functionality
	Usage: python3 best_mse.py
	Output: standard output on init and final MSE with overall acceptance rate
	Note: correponding directories need to be changed for use
'''

import os
from primV3b_mse import *

root = '/data/vision/billf/object-properties/sound/sound/primitives/exp/'

def parse_logs(path_to_logs):
	os.chdir(path_to_logs)
	dist_log = open('dist.log','r').readlines()
	init_dist = dist_log[0].split()[2]
	counter = 0
	for line in dist_log:
		cur_dist = line.split()[2]
		if cur_dist != init_dist:
			counter += 1
			init_dist = cur_dist
	final_dist = float(dist_log[-1].split()[2])
	return counter, final_dist

def compare_dist(task,case):
	counters = []
	final_dists = []
	# for i in range(1,6):
	for i in [1,2,4,5]:
		counter, final_dist = parse_logs(root+task+'_%d/'%i+case)
		counters.append(counter)
		final_dists.append(final_dist)
	best = min(enumerate(final_dists), key=lambda x: x[1])[0]
	init, final = call(task+'_%d/'%(best+1),case)
	print(final_dists, best, counters)
	return init, final, counters[best]

def main():
	nn_mse = []
	synth_mse = []
	counters = []
	tasks = ['primV3b_nmh_stft_random']
	targets_file = '/data/vision/billf/object-properties/sound/sound/primitives/www/primV3b_cnnF_soundnet8_pretrainnone_mse1_LR0.001/151/targets.txt'
	targets = get_targets(targets_file)
	for i in tasks:
		for j in targets:
			init_mse, final_mse, counter = compare_dist(i,j)
			nn_mse.append(init_mse)
			synth_mse.append(final_mse)
			counters.append(counter)
	nn_mse_mean = []
	synth_mse_mean = []
	for i in range(7):
		nn_mse_mean.append(sum([nn_mse[j][i] for j in range(len(nn_mse))]))
		synth_mse_mean.append(sum([synth_mse[j][i] for j in range(len(synth_mse))]))
	nn_mse_mean = [i/len(nn_mse) for i in nn_mse_mean]
	synth_mse_mean = [i/len(synth_mse) for i in synth_mse_mean]
	print("Recognition MSE is:")
	print(nn_mse_mean)
	print("Inference MSE is:")
	print(synth_mse_mean)
	print("Acceptance rate is")
	print([i/300 for i in counters])

if __name__ == '__main__':
	main()
