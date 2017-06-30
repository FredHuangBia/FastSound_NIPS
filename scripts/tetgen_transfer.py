import itertools
import os
import subprocess
import struct

ROOT = '/data/vision/billf/object-properties/sound/sound/primitives/data/v4s/shapes/'
def tetgen(ID):
	plydir = ROOT+'%d/'%ID
	# plypath = plydir+'%d-%d.ply'%(ID, plyID)
	# logfile = open(DST+'%d/tetgen.log'%ID,'a+')
	# subprocess.call([TETGEN,"-Fqa1e-3", plypath],stdout=logfile)
	tetFile = open(plydir+'%d.tet'%ID,'bw')

	node = open(plydir+'%d.1.node'%ID,'r').readlines()
	num_free_vtx = int(node[0].split()[0])
	data = [0,num_free_vtx]
	s = struct.pack('i'*len(data), *data)
	tetFile.write(s)
	# print(num_free_vtx)
	for line in node[1:num_free_vtx+1]:
		line = line.split()
		data = [float(line[1]),float(line[2]),float(line[3])]
		s = struct.pack('d'*len(data), *data)
		tetFile.write(s)

	ele = open(plydir+'%d.1.ele'%ID,'r').readlines()
	num_tet = int(ele[0].split()[0])
	s = struct.pack('i', *[num_tet])
	tetFile.write(s)
	# print(num_tet)
	for line in ele[1:num_tet+1]:
		line = line.split()
		data = [int(line[1]),int(line[2]),int(line[3]),int(line[4])]
		s = struct.pack('i'*len(data), *data)
		tetFile.write(s)

# tetgen(14)
# tetgen(15)
# tetgen(16)
tetgen(18)