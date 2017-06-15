import os, subprocess, random, math, sys
from decimal import Decimal

ROOT = '/data/vision/billf/object-properties/sound/sound/primitives/data/primV2e/'
scripts = '/data/vision/billf/object-properties/sound/sound/primitives/scripts/single_shape_v3/'

alpha_range = (-5, -8)
beta_range = (0, 5)
restitution_range = (0.6, 0.9)
height_range = (3, 5)
total_bem = 17480

def euler2quat(phi, theta, psi):
	a, b, c = (phi-psi)/2, (phi+psi)/2, theta/2
	i = math.cos(a)*math.sin(c)
	j = math.sin(a)*math.sin(c)
	k = math.sin(b)*math.cos(c)
	r = math.cos(b)*math.cos(c)
	return [i, j, k, r]

def get_label(v_range, value):
	mean = sum(v_range)/2
	width = abs(v_range[1]-v_range[0])
	return (value-mean)*2/width

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def write_bullet_setup(config_dst):
	config_dir = os.path.join(config_dst, "bullet_setup")
	create_dir(config_dir)
	os.chdir(config_dir)
	file_out = open("bullet_0.cfg",'w')
	file_out.write("[DEFAULT]\nlinearDamping = 0.0\nangularDamping = 0.0\ncollisionMargin = 0.04")
	file_out.close()

def write_camera(config_dst):
	config_dir = os.path.join(config_dst, "camera")
	create_dir(config_dir)
	os.chdir(config_dir)
	file_out = open("camera_1000.cfg",'w')
	file_out.write("[Camera]\nlook_at = [0.5,0.5,2.5]\nr = 5\ntheta = 45\nphi = 25\nfocal_length = 24\nsensor_width = 40")
	file_out.close()

def write_lighting(config_dst):
	config_dir = os.path.join(config_dst, "lighting")
	create_dir(config_dir)
	os.chdir(config_dir)
	file_out = open("lighting_1000.cfg",'w')
	file_out.write("[Lights]\nlight_num=4\nstyle = custom\n\n[Light_0]\ntype = POINT\npoint_at=[0,0,0]\nr = 5\ntheta = 0\nphi = 60\nenergy = 5\n\n[Light_1]\ntype = POINT\npoint_at=[0,0,0]\nr = 5\ntheta = 90\nphi = 60\nenergy = 5\n\n[Light_2]\ntype = POINT\npoint_at=[0,0,0]\nr = 5\ntheta = 180\nphi = 60\nenergy = 5\n\n[Light_3]\ntype = POINT\npoint_at=[0,0,0]\nr = 5\ntheta = 270\nphi = 60\nenergy = 5")
	file_out.close()

def sample_material():
	alpha = random.uniform(alpha_range[0], alpha_range[1])
	beta = random.uniform(beta_range[0], beta_range[1])
	restitution = random.uniform(restitution_range[0], restitution_range[1])
	return alpha, beta, restitution

def write_material(config_dst):
	config_dir = os.path.join(config_dst, "material")
	create_dir(config_dir)
	os.chdir(config_dir)
	file_out = open("mat-10-10-10.cfg",'w')
	file_out.write("[DEFAULT]\nname = material-10-10-10\nyoungs = 2.02E+10\npoison = 0.25\ndensity = 1000\nalpha = 1.0E-09\nbeta = 32\nfriction = 0.22\nrollingFriction = 0.05\nspinningFriction = 0.05\nrestitution = 1")
	file_out.close()

	alpha, beta, restitution = sample_material()
	file_out = open("mat-0-0-0.cfg",'w')
	file_out.write("[DEFAULT]\nname = material-0-0-0\nyoungs = 2.02E+10\npoison = 0.25\ndensity = 1000\nalpha = %.3E\nbeta = %.3f\nfriction = 0.22\nrollingFriction = 0.05\nspinningFriction = 0.05\nrestitution = %.3f" %(Decimal(math.pow(10,alpha)), math.pow(2,beta), restitution))
	file_out.close()
	return alpha, beta, restitution

def sample_pose():
	pool = [-math.pi/2, -math.pi/4, 0, math.pi/4]
	phi, theta, psi = random.randrange(len(pool)), random.randrange(len(pool)), random.randrange(len(pool))
	quat = euler2quat(pool[phi], pool[theta], pool[psi])
	rotation_label = 16*phi + 4*theta + psi + 1

	height = random.uniform(height_range[0], height_range[1])
	return height, quat, rotation_label

def write_pose(config_dst):
	config_dir = os.path.join(config_dst, "pose")
	create_dir(config_dir)
	os.chdir(config_dir)
	file_out = open("pose-100-100.cfg",'w')
	file_out.write("[DEFAULT]\ncenter = [0,0,0]\nrotation = [0,0,0,1]\nlinear_velocity = [0,0,0]\nangular_velocity = [0,0,0]")
	file_out.close()

	height, rotation, rotation_label = sample_pose()
	file_out = open("pose-0-0.cfg",'w')
	file_out.write("[DEFAULT]\ncenter = [0.0,%.3f,0.0]\nrotation = [%.4f,%.4f,%.4f,%.4f]\nlinear_velocity = [0,0,0]\nangular_velocity = [0,0,0]" %(height, rotation[0], rotation[1], rotation[2], rotation[3]))
	file_out.close()
	return height, rotation, rotation_label

def write_scene(config_dst):
	config_dir = os.path.join(config_dst, "scene")
	create_dir(config_dir)
	os.chdir(config_dir)
	file_out = open("scene-0-0.cfg",'w')
	file_out.write("[Objects]\nobj_num = 2\nobj_0_pose0 = 100\nobj_0_pose1 = 100\nobj_1_pose0 = 0\nobj_1_pose1 = 0\nobj_0_bullet_setup = 0\nobj_1_bullet_setup = 0\nobj_0_is_active = 0\nobj_1_is_active = 1\n\n[Lighting]\nlighting = 1000\n\n[Camera]\ncamera = 1000")
	file_out.close()

def write_simulation(config_dst):
	config_dir = os.path.join(config_dst, "simulation")
	create_dir(config_dir)
	os.chdir(config_dir)
	file_out = open("sim_0.dat",'w')
	file_out.write("[DEFAULT]\nduration = 3.0\nFPS = 60\nGUI = 0")
	file_out.close()

def sample_bem(prim=None):
	if prim is None:
		pool = list(range(total_bem))
	else:
		# assert(isinstance(prim, list)), "input must be a list"
		pool = []
		for i in range(prim):
			for j in range(10):
				pool.append(10*i+j)
	return random.choice(pool)

def main():
	prim = [29, 61, 145, 150, 173, 240, 258, 266, 267, 268, 276, 285, 300, 301, 361, 367, 380, 383, 398, 414, 438, 461, 472, 479, 482, 518, 525, 530, 536, 537, 538, 564, 573, 595, 600, 603, 642, 708, 728, 729, 745, 750, 760, 810, 879, 880, 881, 883, 894, 901, 908, 918, 923, 940, 996, 1005, 1070, 1088, 1097, 1125, 1126, 1135, 1155, 1157, 1182, 1217, 1240, 1246, 1251, 1297, 1307, 1312, 1335, 1341, 1364, 1373, 1386, 1400, 1401, 1403, 1415, 1436, 1492, 1503, 1525, 1532, 1546, 1625, 1626, 1634, 1642, 1650, 1658, 1678, 1692, 1708, 1715, 1732, 1739, 1740] # random.seed(13)
	# prim = None
	assert(len(sys.argv) == 3), "WRONG ARGS!"
	start, end = int(sys.argv[1]), int(sys.argv[2])
	assert(start <= end), "INVALID INPUT"
	data_label = open(ROOT+'../labels2e/%06d-%06d.txt' %(start, end),'w')
	data_value = open(ROOT+'../labels2e/%06d-%06d.dat' %(start, end),'w')
	for i in range(start, end+1):
		output_dir = os.path.join(ROOT, '%06d' %i)
		create_dir(output_dir)
		os.chdir(output_dir)
		config_dst = os.path.join(output_dir,"config")
		create_dir(config_dst)
		write_bullet_setup(config_dst)
		write_camera(config_dst)
		write_lighting(config_dst)
		alpha, beta, restitution = write_material(config_dst)
		height, rotation, rotation_label = write_pose(config_dst)
		write_scene(config_dst)
		write_simulation(config_dst)
		if prim is not None:
			bem_id = sample_bem(len(prim))
			real_bem_id = prim[math.floor(bem_id/10)]*10 + int(bem_id%10)
		else:
			bem_id = sample_bem()
			real_bem_id = bem_id
		cmd = 'python ' + scripts + 'gen_sound.py ' + output_dir + ' 0 0 100000 10 10 10 ' + '%d'%(real_bem_id) + ' 0 0 0 > log.txt'
		print(cmd)
		subprocess.call('kinit -R', shell=True)
		subprocess.call(cmd, shell=True)
		data_label.write("%06d %d %d %d %.4f %.4f %.4f %.4f\n" %(i, math.floor(bem_id/10), int(bem_id%10), rotation_label, get_label(height_range, height), get_label(alpha_range, alpha), get_label(beta_range, beta), get_label(restitution_range, restitution)))
		data_value.write("%06d %d [%.3f,%.3f,%.3f,%.3f] %.3f %.3e %.3f %.3f\n" %(i, real_bem_id, rotation[0], rotation[1], rotation[2], rotation[3], height, Decimal(math.pow(10,alpha)), math.pow(2,beta), restitution))
		data_label.flush()
		data_value.flush()
		print("%06d has generated!" %i)

if __name__ == '__main__':
	main()
