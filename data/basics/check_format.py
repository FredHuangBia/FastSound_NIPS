import os

ROOT = '/data/vision/billf/object-properties/sound/sound/primitives/data/'
BASICS = ROOT+'basics/'

os.chdir(BASICS)
counter = 0
for eachfile in os.listdir():
	if eachfile[-3:] == 'ply':
		counter += 1
		fileIn = open(eachfile, 'r').readlines()
		assert(fileIn[3].split()[1] == "vertex"), "ERROR: incorrect file format!"
		num_vtx = int(fileIn[3].split()[2])
		assert(fileIn[9].strip() == "end_header"), "ERROR: incorrect file format!"
		print("%s done!" %eachfile)
print("\nTotal number of basic shapes is %d" %counter)