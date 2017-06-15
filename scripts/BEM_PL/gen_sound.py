import os, subprocess
import numpy as np
from decimal import Decimal

ROOTDIR = '/data/vision/billf/object-properties/sound/sound/primitives/data/v2.3/'
CLICKSYNTH = '/data/vision/billf/object-properties/sound/sound/code/ModalSound/build/bin/click_synth'

def normal(a,b,c):
	x = np.array(a) - np.array(b)
	y = np.array(b) - np.array(c)
	normalVec = np.cross(x,y)
	return normalVec/np.linalg.norm(normalVec)

def load(objName):
	vtx = []
	face = []
	for line in open(objName+'.obj','r').readlines():
		if line[0] == 'v':
			line = line.split()
			vtx.append([float(line[1]),float(line[2]),float(line[3])])
		if line[0] == 'f':
			line = line.split()
			face.append([int(line[1]),int(line[2]),int(line[3])])
	return vtx, face

def prep_click(vtx,face,num_clicks):
	num_face = len(face)
	sample_points = range(0,num_face,int(num_face/num_clicks))
	normals = []
	for triangleId in sample_points:
		vtxId = face[triangleId]
		a,b,c = vtxId[0]-1, vtxId[1]-1, vtxId[2]-1
		normalVec = normal(vtx[a],vtx[b],vtx[c])
		normals.append(list(normalVec))
	return sample_points, normals

def write_ini(filePath,objName, sample_points, normals):
	alpha0 = 1*10**(-6)
	beta0 = 4
	rho0 = 1000
	separation = 0.3
	amp = 5
	camera = [3,3,3]
	iniFile = open(filePath,'w')
	iniFile.write("[mesh]\nsurface_mesh = %s.obj\nvertex_mapping = adddd\n\n[audio]\nuse_audio_device = false\ndevice = \nTS = 1.0\namplitude = 2.0\ncontinuous = true\n\n[gui]\ngui=false\n\n[transfer]\nmoments = moments/moments.pbuf\n\n[modal]\nshape = %s.ev\ndensity = %.0f\nalpha = %.1E\nbeta = %.0f\nvtx_map = %s.vmap\n\n[camera]\nx = 0\ny = 0\nz = 1\n\n[collisions]" %(objName,objName,rho0,Decimal(alpha0),beta0,objName))
	time = [x*separation for x in range(len(sample_points))]
	ID = sample_points
	amplitude = [amp]*len(sample_points)
	norm1 = [x[0] for x in normals]
	norm2 = [x[1] for x in normals]
	norm3 = [x[2] for x in normals]
	iniFile.write("\ntime =")
	for i in time:
		iniFile.write(" %f" %i)
	iniFile.write("\nID =")
	for i in ID:
		iniFile.write(" %f" %i)
	iniFile.write("\namplitude =")
	for i in amplitude:
		iniFile.write(" %f" %i)
	iniFile.write("\nnorm1 =")
	for i in norm1:
		iniFile.write(" %f" %i)
	iniFile.write("\nnorm2 =")
	for i in norm2:
		iniFile.write(" %f" %i)
	iniFile.write("\nnorm3 =")
	for i in norm3:
		iniFile.write(" %f" %i)
	iniFile.write("\ncamX =")
	for i in [camera[0]]*len(sample_points):
		iniFile.write(" %f" %i)
	iniFile.write("\ncamY =")
	for i in [camera[1]]*len(sample_points):
		iniFile.write(" %f" %i)
	iniFile.write("\ncamZ =")
	for i in [camera[2]]*len(sample_points):
		iniFile.write(" %f" %i)

def main():
	exp = '0'
	objName = '0-36'
	os.chdir(ROOTDIR+objName)
	allDir = os.listdir()
	soundDir = []
	for i in allDir:
		if i[0] == 'm':
			if i[4] == exp:
				soundDir.append(i)
	for eachDir in sorted(soundDir):
		os.chdir(ROOTDIR+objName+'/'+eachDir)
		print(os.getcwd())
		vtx,face = load(objName)
		filePath = 'click-true.ini'
		sample_points, normals = prep_click(vtx,face,10)
		write_ini(filePath,objName,sample_points,normals)
		subprocess.call([CLICKSYNTH, filePath])
		print('%s sound has generated!' %objName)

if __name__ == '__main__':
	main()
