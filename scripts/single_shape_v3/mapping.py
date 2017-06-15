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
	prim = [29, 61, 145, 150, 173, 240, 258, 266, 267, 268, 276, 285, 300, 301, 361, 367, 380, 383, 398, 414, 438, 461, 472, 479, 482, 518, 525, 530, 536, 537, 538, 564, 573, 595, 600, 603, 642, 708, 728, 729, 745, 750, 760, 810, 879, 880, 881, 883, 894, 901, 908, 918, 923, 940, 996, 1005, 1070, 1088, 1097, 1125, 1126, 1135, 1155, 1157, 1182, 1217, 1240, 1246, 1251, 1297, 1307, 1312, 1335, 1341, 1364, 1373, 1386, 1400, 1401, 1403, 1415, 1436, 1492, 1503, 1525, 1532, 1546, 1625, 1626, 1634, 1642, 1650, 1658, 1678, 1692, 1708, 1715, 1732, 1739, 1740]
	result = from17480toshapeinfo(str(prim[shape_label]*10+modulus_label))
	print(result[0], result[1], result[2])

if __name__ == '__main__':
	main()


