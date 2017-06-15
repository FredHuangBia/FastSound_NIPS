import itertools
import os
import subprocess
import struct

ROOT = '/data/vision/billf/object-properties/sound/sound/primitives/data/'
BASICS = ROOT+'basics/'
DST = ROOT+'v2.2/'
TETGEN = '/data/vision/billf/object-properties/sound/software/tetgen1.5.0/build/tetgen'

def resize(fileName, outDir, fileOut, scale):
	fileIn = open(BASICS+fileName+'.ply', 'r').readlines()
	directory = DST+'%d/%d'%(outDir,fileOut)
	if not os.path.exists(directory):
		os.makedirs(directory)
	fileOut = open(DST+'%d/%d/%d-%d.ply'%(outDir,fileOut,outDir,fileOut),'w')
	assert(fileIn[3].split()[1] == "vertex"), "ERROR: incorrect file format!"
	num_vtx = int(fileIn[3].split()[2])
	assert(fileIn[9].strip() == "end_header"), "ERROR: incorrect file format!"
	for i in range(10,10+num_vtx):
		x,y,z = fileIn[i].split()[:]
		x = float(x)*scale[0]
		y = float(y)*scale[1]
		z = float(z)*scale[2]
		fileIn[i] = '%f %f %f\n' %(x,y,z)
	fileOut.writelines(fileIn)
	print("%s %s done!" %(fileName,scale))

def threesym():
	listofscale = [round(x*0.25,3) for x in range(1,9)]
	return list(itertools.combinations_with_replacement(listofscale,3))

def twosym():
	listofxy = [x*1/3 for x in range(1,7)]
	listofz = [x*1/3 for x in range(1,7)]
	firstpair = list(itertools.combinations_with_replacement(listofxy,2))
	output = []
	for i in listofz:
		for j in firstpair:
			tmp = list(j)
			tmp.append(i)
			output.append(tuple(tmp))
	return output

def onesym():
	listofx = [x*1/3 for x in range(1,6)]
	listofy = [x*1/3 for x in range(1,6)]
	listofz = [x*1/3 for x in range(1,6)]
	output = []
	for i in listofx:
		for j in listofy:
			for k in listofz:
				output.append((i,j,k))
	return output

def tetgen(counter, plycounter, scale):
	plydir = DST+'%d/%d/'%(counter, plycounter)
	plypath = plydir+'%d-%d.ply'%(counter, plycounter)
	logfile = open(DST+'%d/tetgen.log'%counter,'a+')

	volumn = scale[0]*scale[1]*scale[2]
	limit = volumn*10**(-3)

	# cmd = TETGEN + ' -Fqa%.2g ' %limit + plypath
	cmd = TETGEN + ' -Fq ' + plypath
	# print(cmd)
	subprocess.call(cmd,shell=True,stdout=logfile)
	tetFile = open(plydir+'%d-%d.tet'%(counter, plycounter),'bw')

	node = open(plydir+'%d-%d.1.node'%(counter, plycounter),'r').readlines()
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

	ele = open(plydir+'%d-%d.1.ele'%(counter, plycounter),'r').readlines()
	num_tet = int(ele[0].split()[0])
	s = struct.pack('i', *[num_tet])
	tetFile.write(s)
	# print(num_tet)
	for line in ele[1:num_tet+1]:
		line = line.split()
		data = [int(line[1]),int(line[2]),int(line[3]),int(line[4])]
		s = struct.pack('i'*len(data), *data)
		tetFile.write(s)

def main():
	threeaxis = ['cube0','cube1','cube2','sphere0']
	twoaxis = ['sphere1','sphere2','torus0','torus1','torus2','cylinder0','cylinder1','cylinder2','cone0','cone1','cone2','cone3','cone4','cone5','pyramid0','pyramid1','pyramid2','pyramid3','pyramid4','pyramid5','wedge0','wedge1','wedge2']
	oneaxis = ['tet0','tet1','tet2','tet3','tet4','tet5','wedge3','wedge4','wedge5','cylinder3','cylinder4','cylinder5']
	allbasics = {'threeaxis': threeaxis, 'twoaxis': twoaxis, 'oneaxis': oneaxis}
	threelist = threesym()
	twolist = twosym()
	onelist = onesym()
	counter = 0
	basic2dir = open(DST+'basic2dir.txt','w')
	for key, value in allbasics.items():
		for name in value:
			directory = DST+'%d'%counter
			if not os.path.exists(directory):
				os.makedirs(directory)
			basic2dir.write('%s %d\n' %(name, counter))
			plycounter = 0
			mapping = open(DST+'%d/mapping.txt'%counter,'w')
			if key == 'threeaxis':
				for scale in threelist:
					resize(name,counter,plycounter,scale)
					mapping.write('%d %s %d %s\n' %(counter, name, plycounter, scale))
					tetgen(counter, plycounter, scale)
					plycounter += 1
			elif key == 'twoaxis':
				for scale in twolist:
					resize(name,counter,plycounter,scale)
					mapping.write('%d %s %d %s\n' %(counter, name, plycounter, scale))
					tetgen(counter, plycounter, scale)
					plycounter += 1
			elif key == 'oneaxis':
				for scale in onelist:
					resize(name,counter,plycounter,scale)
					mapping.write('%d %s %d %s\n' %(counter, name, plycounter, scale))
					tetgen(counter, plycounter, scale)
					plycounter += 1
			counter += 1



if __name__ == "__main__":
	main()
