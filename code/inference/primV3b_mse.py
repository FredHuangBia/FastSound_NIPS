import math

def euler2quat(phi, theta, psi):
	a, b, c = (phi-psi)/2, (phi+psi)/2, theta/2
	i = math.cos(a)*math.sin(c)
	j = math.sin(a)*math.sin(c)
	k = math.sin(b)*math.cos(c)
	r = math.cos(b)*math.cos(c)
	return [i, j, k, r]

def get_quat(rot_label):
	rot_label -= 1
	pool = [-math.pi/2, -math.pi/4, 0, math.pi/4]
	phi = math.floor(rot_label/16)
	theta = math.floor(rot_label%16/4)
	psi = rot_label%16%4
	return euler2quat(pool[phi],pool[theta],pool[psi])

def MSE(label1, label2, kind):
	'''
		kind is a list indicating what kind of label we are dealing with:
		0: standard regression in continuous domain
		1: classification problem (shape)
		2: classification problem with special treatment (rotation)
		3: discrete sampling in continuous domain (specific modulus)
	'''
	assert(len(label1) == len(label2) and len(label1) == len(kind)), "wrong input for MSE!"
	error = [0]*len(label1)
	for i in range(len(label1)):
		if kind[i] == 0:
			error[i] += math.pow((label1[i]-label2[i])/2,2)
		elif kind[i] == 1:
			if label1[i] != label2[i]:
				error[i] += 1
		elif kind[i] == 2:
			quat1 = get_quat(label1[i])
			quat2 = get_quat(label2[i])
			if quat1 !=  quat2 and quat1 != [-i for i in quat2]:
				error[i] += 1
		elif kind[i] == 3:
			error[i] += math.pow((label1[i]-label2[i])/10,2)
	return error

def get_targets(pathtofile):
	file = open(pathtofile,'r')
	targets = []
	for line in file:
		line = line.split()
		targets.append(line[0])
	return targets

def get_ground_truth(pathtofile):
	file = open(pathtofile,'r')
	ground_truth = {}
	for line in file:
		line = line.split()
		if len(line) == 8:
			ground_truth[line[0]] = [int(line[1])+1,int(line[2])+1,int(line[3]),float(line[4]),float(line[5]),float(line[6]),float(line[7])]
	return ground_truth

def main():
	root = '/data/vision/billf/object-properties/sound/sound/primitives/exp/primV3b/'
	targets_file = '/data/vision/billf/object-properties/sound/sound/primitives/www/primV3b_cnnF_soundnet8_pretrainnone_mse1_LR0.001/93/targets.txt'
	targets = get_targets(targets_file)
	# targets = ['000280','101824']
	ground_truth_file = '/data/vision/billf/object-properties/sound/sound/primitives/data/primV3b.txt'
	ground_truth = get_ground_truth(ground_truth_file)
	kind = [1,3,2,0,0,0,0]
	nn_mse = []
	synth_mse = []
	sanity_mse = []
	for i in targets:
		mse_log = open(root+i+'/mse.log','r').readlines()
		sanity_log = open(root+i+'/sanity.log','r').readlines()
		nn_line = mse_log[0].split()
		synth_line = mse_log[1].split()
		sanity_line = sanity_log[0].split()
		# print(nn_line)
		nn_label = [int(nn_line[0]),int(nn_line[1]),int(nn_line[2]),float(nn_line[3]),float(nn_line[4]),float(nn_line[5]),float(nn_line[6])]
		synth_label = [int(synth_line[0]),int(synth_line[1]),int(synth_line[2]),float(synth_line[3]),float(synth_line[4]),float(synth_line[5]),float(synth_line[6])]
		sanity_label = [int(sanity_line[0]),int(sanity_line[1]),int(sanity_line[2]),float(sanity_line[3]),float(sanity_line[4]),float(sanity_line[5]),float(sanity_line[6])]
		ground_label = ground_truth[i]
		print("ground truth")
		print(ground_label)
		print("prediction")
		print(nn_label)
		print("synth")
		print(synth_label)
		print("sanity")
		print(sanity_label)
		print('\n')
		nn_mse.append(MSE(nn_label,ground_label,kind))
		synth_mse.append(MSE(synth_label,ground_label,kind))
		sanity_mse.append(MSE(sanity_label,ground_label,kind))
	# print(nn_mse)
	# print(synth_mse)
	nn_mse_mean = []
	synth_mse_mean = []
	sanity_mse_mean = []
	for i in range(len(kind)):
		nn_mse_mean.append(sum([nn_mse[j][i] for j in range(len(nn_mse))]))
		synth_mse_mean.append(sum([synth_mse[j][i] for j in range(len(synth_mse))]))
		sanity_mse_mean.append(sum([sanity_mse[j][i] for j in range(len(sanity_mse))]))
	nn_mse_mean = [i/len(nn_mse) for i in nn_mse_mean]
	synth_mse_mean = [i/len(synth_mse) for i in synth_mse_mean]
	sanity_mse_mean = [i/len(sanity_mse) for i in sanity_mse_mean]
	print("Recognition MSE is:")
	print(nn_mse_mean)
	print("Inference MSE is:")
	print(synth_mse_mean)
	print("Sanity check MSE is:")
	print(sanity_mse_mean)

if __name__ == '__main__':
	main()
