'''
	Given .node and .ele from unit size shape, scale it to 0.2 and save in .tet format
	Usage: python3 scale_tet.py
'''

import struct

UNIT = '/data/vision/billf/object-properties/sound/sound/primitives/data/v4/shapes/unit_shapes/'
ROOT = '/data/vision/billf/object-properties/sound/sound/primitives/data/v4/shapes/'


for i in range(14):
	node = open(UNIT+str(i)+'/%s.1.node'%str(i),'r').readlines()
	num_free_vtx = int(node[0].split()[0])
	data = [0,num_free_vtx]
	b_fvtx = struct.pack('i'*len(data), *data)

	ele = open(UNIT+str(i)+'/%s.1.ele'%str(i),'r').readlines()
	num_tet = int(ele[0].split()[0])
	b_tet = struct.pack('i', *[num_tet])

	tetFile = open(ROOT+str(i)+'/%s.tet'%str(i),'bw')

	tetFile.write(b_fvtx)
	scaling = 0.2
	# print(num_free_vtx)
	for line in node[1:num_free_vtx+1]:
		line = line.split()
		data = [scaling*float(line[1]),scaling*float(line[2]),scaling*float(line[3])]
		s = struct.pack('d'*len(data), *data)
		tetFile.write(s)

	tetFile.write(b_tet)
	# print(num_tet)
	for line in ele[1:num_tet+1]:
		line = line.split()
		data = [int(line[1]),int(line[2]),int(line[3]),int(line[4])]
		s = struct.pack('i'*len(data), *data)
		tetFile.write(s)