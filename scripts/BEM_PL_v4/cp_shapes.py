''' 
	copy multiple items for each unit primitives from v2.3
	usage: python3 cp_shapes.py
'''

import os
from subprocess import call

root = '/data/vision/billf/object-properties/sound/sound/primitives/data/v2.3/'
new_dst = '/data/vision/billf/object-properties/sound/sound/primitives/data/v4/shapes/unit_shapes/'

def CreateDir(path):
    if not os.path.exists(path):
        os.makedirs(path)

for i in range(14):
	CreateDir(new_dst+str(i))
	os.chdir(root+str(i))
	file = open('mapping.txt','r')
	parent_dir, child_dir = None, None
	for line in file:
		line = line.split()
		if line[3]+line[4]+line[5] == '(1.0,1.0,1.0)':
			parent_dir, child_dir = line[0], line[2]
	assert(parent_dir == str(i))
	os.chdir(os.path.join(root,str(i),child_dir))
	name = parent_dir+'-'+child_dir
	cmd = 'cp %s.orig.obj %s' %(name, os.path.join(new_dst,str(i),str(i)+'.orig.obj'))
	call(cmd, shell=True)
	cmd = 'cp %s.ply %s' %(name, os.path.join(new_dst,str(i),str(i)+'.ply'))
	call(cmd, shell=True)
	cmd = 'cp %s.tet %s' %(name, os.path.join(new_dst,str(i),str(i)+'.tet'))
	call(cmd, shell=True)
	cmd = 'cp volume.txt %s' %(os.path.join(new_dst,str(i),'volume.txt'))
	call(cmd, shell=True)
	cmd = 'cp %s.1.node %s' %(name, os.path.join(new_dst,str(i),str(i)+'.1.node'))
	call(cmd, shell=True)
	cmd = 'cp %s.1.ele %s' %(name, os.path.join(new_dst,str(i),str(i)+'.1.ele'))
	call(cmd, shell=True)

