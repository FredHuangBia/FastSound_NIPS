import os

ROOT = '/data/vision/billf/object-properties/sound/sound/primitives/config/single_shape/scene/'

obj_num = 2
obj_0_pose0 = 100
obj_0_pose1 = 100

obj_1_pose0 = [i for i in range(6)]
obj_1_pose1 = [i for i in range(7)]

obj_0_bullet_setup = 0
obj_1_bullet_setup = 0
obj_0_is_active = 0
obj_1_is_active = 1

lighting = 1000
camera = 1000

def template(obj_1_pose0,obj_1_pose1):
	return ['[Objects]\n','obj_num = %d\n' %obj_num,'obj_0_pose0 = %d\n' %obj_0_pose0, 'obj_0_pose1 = %d\n' %obj_0_pose1,\
	'obj_1_pose0 = %d\n' %obj_1_pose0, 'obj_1_pose1 = %d\n' %obj_1_pose1, 'obj_0_bullet_setup = 0\n', 'obj_1_bullet_setup = 0\n'\
	'obj_0_is_active = 0\n', 'obj_1_is_active = 1\n\n',\
	'[Lighting]\n', 'lighting = 1000\n\n', '[Camera]\n','camera = 1000']

if __name__ == '__main__':
	for i in obj_1_pose0:
		for j in obj_1_pose1:
			name = "scene-"+str(i)+"-"+str(j)
			f = open(ROOT+name+".cfg", "w")
			a = template(i, j)
			for line in a:
				# print(line)
				f.write(line)
			f.close()
