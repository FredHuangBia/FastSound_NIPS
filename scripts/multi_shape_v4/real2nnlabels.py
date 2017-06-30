'''
	get GT label from each dir and scale into (-1, 1) for DL
	Usage: python3 real2nnlabels.py start end
'''
import math
import os
import sys

ROOT = '/data/vision/billf/object-properties/sound/sound/primitives/data/primV4a/' # where all sounds are located

H_MAX = 2
H_MIN = 1
ALPHA_MAX = -5
ALPHA_MIN = -8
BETA_MAX = 5
BETA_MIN = 0
RES_MAX = 0.9
RES_MIN = 0.6

def read_real_label(string):
	string = string.split()
	label = [int(i) for i in string[0:2]]
	label.append([float(i) for i in string[2:6]])
	label = label + [float(i) for i in string[6:]]
	return label

def get_nn_label(real_label):
	nn_label = real_label[0:3]
	nn_label.append(2*(real_label[3]-(H_MIN+H_MAX)/2)/(H_MAX-H_MIN))
	nn_label.append(2*(math.log(real_label[4],10)-(ALPHA_MIN+ALPHA_MAX)/2)/(ALPHA_MAX-ALPHA_MIN))
	nn_label.append(2*(math.log(real_label[5],2)-(BETA_MIN+BETA_MAX)/2)/(BETA_MAX-BETA_MIN))
	nn_label.append(2*(real_label[6]-(RES_MIN+RES_MAX)/2)/(RES_MAX-RES_MIN))
	return nn_label

def label_to_str(label):
    label_str = ''
    label_str = label_str+' '.join(str(e) for e in label[0:2])+' '
    label_str = label_str+' '.join(str(e) for e in label[2])+' '
    label_str = label_str+' '.join(str(e) for e in label[3:])
    return label_str

def main():
	start = int(sys.argv[1])
	end = int(sys.argv[2])

	datfile = open(os.path.join(ROOT, '../primV4a_%06d_%06d.dat'%(start,end)), 'w')
	txtfile = open(os.path.join(ROOT, '../primV4a_%06d_%06d.txt'%(start,end)), 'w')

	for i in range(start, end+1):
		labelfile = open(os.path.join(ROOT, '%03d' %(math.floor(i/100)+1), '%06d' %i, 'label.txt'),'r').readlines()[0].strip()
		datfile.write('%06d %s\n' %(i, labelfile))
		nn_label = get_nn_label(read_real_label(labelfile))
		txtfile.write('%06d %s\n' %(i, label_to_str(nn_label)))
		datfile.flush()
		txtfile.flush()

if __name__ == '__main__':
	main()

