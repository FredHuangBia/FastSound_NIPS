from __future__ import division
import math, os, subprocess, sys
import argparse
import numpy

test = '/data/vision/billf/object-properties/sound/sound/primitives/exp/random_v4/test_sound/'
exp = '/data/vision/billf/object-properties/sound/sound/primitives/exp/random_v4/result/'

H_MAX = 2
H_MIN = 1
ALPHA_MAX = -5
ALPHA_MIN = -8
BETA_MAX = 5
BETA_MIN = 0
RES_MAX = 0.9
RES_MIN = 0.6


def quaternion_matrix(quaternion):
	# from the source of transformation.py,  http://www.lfd.uci.edu/~gohlke/code/transformations.py.html, and modified
	# 1. last column and last row removed
	# 2. transform from (x,y,z,w) to (w,x,y,z) at the first row
    """Return homogeneous rotation matrix from quaternion.
    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> numpy.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> numpy.allclose(M, numpy.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> numpy.allclose(M, numpy.diag([1, -1, -1, 1]))
    True
    """
    quaternion = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]
    q = numpy.array(quaternion, dtype=numpy.float64, copy=True)
    n = numpy.dot(q, q)
    q *= math.sqrt(2.0 / n)
    q = numpy.outer(q, q)
    return numpy.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0]],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0]],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2]] ])

def rot_disance(quaternion1,quaternion2):
	rot_m1 = quaternion_matrix(quaternion1)
	rot_m2 = quaternion_matrix(quaternion2)
	dif_m = numpy.identity(3) - (numpy.dot(rot_m1,numpy.transpose(rot_m2)))
	return numpy.linalg.norm(dif_m)
	# return numpy.sqrt(numpy.sum(dif_m[:,0]**2) + numpy.sum(dif_m[:,1]**2) + numpy.sum(dif_m[:,2]**2))


def reverse_mapping(label):
	new_label = label[:]
	new_label[3] = 2*(label[3]-(H_MIN+H_MAX)/2)/(H_MAX-H_MIN)
	new_label[4] = 2*(math.log(label[4],10)-(ALPHA_MIN+ALPHA_MAX)/2)/(ALPHA_MAX-ALPHA_MIN)
	new_label[5] = 2*(math.log(label[5],2)-(BETA_MIN+BETA_MAX)/2)/(BETA_MAX-BETA_MIN)
	new_label[6] = 2*(label[6]-(RES_MIN+RES_MAX)/2)/(RES_MAX-RES_MIN)
	return new_label

def read_label(string):
	string = string.split()
	label = [int(i) for i in string[0:2]]
	label.append([float(i) for i in string[2:6]])
	label = label + [float(i) for i in string[6:]]
	label = reverse_mapping(label)
	return label

def get_labels(path):
	label_list = []
	for line in open(path,'r').readlines():
		if line.strip() != '':
			label_list.append(read_label(line))
	return label_list

def get_gt_labels(path, start, end):
	os.chdir(path)
	cmd = '%s %d %d' %(os.path.join(path,'get_gt.sh'), start, end)
	subprocess.call(cmd, shell=True)
	gt_file = os.path.join(path, 'GT.txt')
	assert(os.path.exists(gt_file)),'Get GT Failed!'
	
	return get_labels(gt_file)

def get_init_labels(path, as_dir, start, end):
	os.chdir(path)
	cmd = '%s %s %d %d' %(os.path.join(path,'get_init.sh'), as_dir, start, end)
	subprocess.call(cmd, shell=True)
	label_file = os.path.join(path, as_dir, 'init_labels.txt')
	assert(os.path.exists(label_file)),'Get Init Labels Failed!'
	return get_labels(label_file)

def get_as_labels(path, as_dir, start, end):
	os.chdir(path)
	cmd = '%s %s %d %d' %(os.path.join(path,'get_label.sh'), as_dir, start, end)
	subprocess.call(cmd, shell=True)
	label_file = os.path.join(path, as_dir, 'final_labels.txt')
	assert(os.path.exists(label_file)),'Get AS Labels Failed!'
	return get_labels(label_file)

def get_kind(attr):
	default = [1,3,2,0,0,0,0] 
	if attr == 'rotation':
		default[2] = -1
	elif attr == 'height':
		default[3] = -1
	return default

def calc_mean(mse):
	mse_mean = [0]*len(mse[0])
	for i in range(len(mse)):
		for j in range(len(mse_mean)):
			if mse[i][j] != None:
				mse_mean[j] += mse[i][j]
			else:
				mse_mean[j] = None
	for i in range(len(mse_mean)):
		if mse_mean[i] != None:
			mse_mean[i] = mse_mean[i]/len(mse)
	return mse_mean

def MSE(label1, label2, kind):
	'''
		kind is a list indicating what kind of label we are dealing with:
		-1: ignore this attribute
		0: standard regression in continuous domain
		1: classification problem (shape)
		2: classification problem with special treatment (rotation)
		3: discrete sampling in continuous domain (specific modulus)
	'''
	assert(len(label1) == len(label2) and len(label1) == len(kind)), "wrong input for MSE!"
	error = [0]*len(label1)
	for i in range(len(label1)):
		if kind[i] == -1:
			error[i] = None
		elif kind[i] == 0:
			error[i] += math.pow((label1[i]-label2[i])/2,2)
		elif kind[i] == 1:
			if label1[i] != label2[i]:
				error[i] += 1
		elif kind[i] == 2:
			error[i] = rot_disance(label1[i],label2[i])
		elif kind[i] == 3:
			error[i] += math.pow((label1[i]-label2[i])/10,2)
	return error

def print_mse(mse, name):
	string = name
	for i in mse:
		if i != None:
			string += ' %.4f' %i
		else:
			string += ' ---'
	string += '\n'
	print(string)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('as_name', help='specify the name of the as directory / name of expriment')
	parser.add_argument('-s', '--start_id', type=int, help='starting test case',default=0)
	parser.add_argument('-e', '--end_id', type=int, help='ending test case', default=19)
	parser.add_argument('-f', '--fix', help='fix a certain varaible.', choices=['rotation','height'], default=None)
	parser.add_argument('-v', '--verbose', type=bool, help='show mse of each test case if on', default=True)
	args = parser.parse_args()

	gt_labels = get_gt_labels(test, args.start_id, args.end_id)
	init_labels = get_init_labels(exp, args.as_name, args.start_id, args.end_id)
	as_labels = get_as_labels(exp, args.as_name, args.start_id, args.end_id)
	kind = get_kind(args.fix)

	init_mse = []
	final_mse = []
	for i in range(len(gt_labels)):
		init_mse.append(MSE(gt_labels[i], init_labels[i], kind))
		final_mse.append(MSE(gt_labels[i], as_labels[i], kind))
	init_mse_mean = calc_mean(init_mse)
	final_mse_mean = calc_mean(final_mse)

	if args.verbose:
		for i in range(len(init_mse)):
			print_mse(init_mse[i], str(i)+' init ')
			print_mse(final_mse[i], str(i)+ ' final')
			print('')
	print('%s results:' %args.as_name)
	print('---------------------------------------')
	print_mse(init_mse_mean,'init_mean ')
	print_mse(final_mse_mean,'final_mean')


if __name__ == '__main__':
	main()
# python3 this -f rotation folder_rotation
# ptthon3 this folder_none
