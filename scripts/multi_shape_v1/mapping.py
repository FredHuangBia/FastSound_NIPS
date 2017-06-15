import os, sys

def from17480todirs():
	file = open('/data/vision/billf/object-properties/sound/sound/primitives/data/v2b/directories.txt','r')
	mapping = {}
	for line in file:
		line = line.split()
		mapping[line[0]] = line[1]
	return mapping

def fromdirtoxyz(parentdir):
	root = '/data/vision/billf/object-properties/sound/sound/primitives/data/v2.3/'
	file = open(root+parentdir+'/mapping.txt','r')
	mapping = {}
	for line in file:
		line = line.split()
		name = line[1]
		mapping[line[2]] = line[3]+line[4]+line[5]
	return name, mapping

def from17480toshapeinfo(astring):
	specific_modulus = [1.00,2.25,4.00,6.25,9.00,12.2,16.0,20.2,25.0,30.0]
	mapping1 = from17480todirs()

	directory = mapping1[astring].strip().split('/')
	name, mapping2 = fromdirtoxyz(directory[0])
	x, y, z  = mapping2[directory[1]].replace('(', '').replace(')', '').split(',')
	return name, "(%.1f,%.1f,%.1f)"%(float(x),float(y),float(z)), specific_modulus[int(directory[2][-1:])]

def main():
	shape_label = int(sys.argv[1])-1
	modulus_label = int(sys.argv[2])-1
	prim = [62,187,312,437,553,679,805,931,1057,1183,1309,1435,1593,1713]
	result = from17480toshapeinfo(str(prim[shape_label]*10+modulus_label))
	print(result[0], result[1], result[2])

if __name__ == '__main__':
	main()


