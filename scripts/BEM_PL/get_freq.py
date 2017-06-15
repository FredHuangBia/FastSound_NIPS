import os, subprocess

ROOT_DIR = '/data/vision/billf/object-properties/sound/sound/primitives/data/v2.3/'
objName = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

def CreateDir(path):
    if not os.path.exists(path):
        os.makedirs(path)

matcfg = sorted(os.listdir(ROOT_DIR+'materials'))
material = []
for i in matcfg:
	if i[-3:] == 'cfg':
		material.append('mat-%s-%s'%(i.split('.')[0].split('-')[1], i.split('.')[0].split('-')[2]))

for i in objName:
	CreateDir(ROOT_DIR+'shapes/%s/freq'%i)
	for j in material:
		cmd = 'matlab -nodisplay -nodesktop -nosplash -r "write_frequencies \'%s/%s\' \'%s/freq/%s.txt\'; quit"' %(i,j,i,j)
		subprocess.call(cmd,shell=True)

