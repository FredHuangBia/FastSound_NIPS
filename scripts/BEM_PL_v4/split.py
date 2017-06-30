'''
	write taks in txt file
	Usage: python3 split.py
'''

import os

ROOT_DIR = '/data/vision/billf/object-properties/sound/sound/primitives/data/v4/'
# objName = list(range(14))
objName = [18]
material = list(range(10))

fileOut = open(ROOT_DIR+'../../scripts/BEM_PL_v4/run4/input.txt','w')
for i in objName:
	for k in material:
		fileOut.write('%d %d\n' %(i, k))
