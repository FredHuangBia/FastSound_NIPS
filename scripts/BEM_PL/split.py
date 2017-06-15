import os

ROOT_DIR = '/data/vision/billf/object-properties/sound/sound/primitives/data/v2.3/'
objName = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
subNum = [125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 126, 120, 120]
matcfg = sorted(os.listdir(ROOT_DIR+'materials'))
material = []
for i in matcfg:
	if i[-3:] == 'cfg':# and i[9] == '3':
		material.append((i.split('.')[0].split('-')[1], i.split('.')[0].split('-')[2]))

fileOut = open(ROOT_DIR+'../../scripts/BEM_PL/run/input.txt','w')
for i in objName:
	for j in range(subNum[i]):
		for k in material:
			fileOut.write('%s %d %s %s\n' %(i, j, k[0],k[1]))
