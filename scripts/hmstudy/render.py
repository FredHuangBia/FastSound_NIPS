import sys, ConfigParser, os, subprocess
import numpy, math

scripts = '/data/vision/billf/object-properties/sound/sound/primitives/scripts/hmstudy/'
config_dir = '/data/vision/billf/object-properties/sound/sound/primitives/scripts/hmstudy/config/'
shape_dir = '/data/vision/billf/object-properties/sound/sound/primitives/data/v2b/'
BLENDER = '/data/vision/billf/object-properties/sound/software/blender/blender'

def quaternion_matrix(quaternion):
	quat = []
	for i in [3,0,1,2]:
		quat.append(quaternion[i])
	quaternion = quat
	q = numpy.array(quaternion, dtype=numpy.float64, copy=True)
	n = numpy.dot(q, q)
	q *= math.sqrt(2.0 / n)
	q = numpy.outer(q, q)
	return numpy.array([[1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0]],
						[    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0]],
						[    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2]]])

def write_motion(dst_dir,height,rotation):
	file = open(dst_dir+'/motion.dat','w')
	file.write('0 0 0 0 0 1 0 0 0 1 0 0 0 1\n')
	rot_matrix = quaternion_matrix(rotation)
	file.write('1 0 0 %f 0 %f %f %f %f %f %f %f %f %f'%(height,\
		rot_matrix[0,0],rot_matrix[0,1],rot_matrix[0,2],
		rot_matrix[1,0],rot_matrix[1,1],rot_matrix[1,2],
		rot_matrix[2,0],rot_matrix[2,1],rot_matrix[2,2]))
	file.close()

def render(obj_id,height,rotation,dst_dir,skip_factor=1):
	# rotation is defined in quaterion 
	obj_ids = [100000,obj_id]
	rotations = [[0.000000,0.000000,0.000000,1.000000],rotation]
	objnum = len(obj_ids)
	subprocess.call('sh %sunset.sh'%scripts,shell=True)
	blendercfg = ConfigParser.ConfigParser()
	blendercfg.add_section('Objects')
	blendercfg.set('Objects','obj_num','%d'%objnum)
	for i in range(objnum):
		blendercfg.set('Objects','obj_%d'%i,'%s'%(shape_dir+'%d/%d.orig.obj'%(obj_ids[i],obj_ids[i])))
		blendercfg.set('Objects','obj_%d_rot'%i,'[%f,%f,%f,%f]'%tuple(rotations[i]))

	blendercfg.add_section('OutPath')
	blendercfg.set('OutPath','outpath','%s'%dst_dir)
	blendercfg.add_section('Camera')
	blendercfg.set('Camera','path','%s'%os.path.join(config_dir,'camera.cfg'))
	blendercfg.add_section('Lighting')
	blendercfg.set('Lighting','path','%s'%os.path.join(config_dir,'lighting.cfg'))
	write_motion(dst_dir,height,rotation)
	blendercfg.add_section('MotionFile')
	blendercfg.set('MotionFile','path','%s'%os.path.join(dst_dir,'motion.dat'))
	with open(dst_dir+'/blender_render.cfg', 'w+') as configfile:
		blendercfg.write(configfile)
	blank = os.path.join(scripts,'blank.blend')
	blenderScript = os.path.join(scripts,'blender_render_scene.py')
	subprocess.call('unset PYTHONPATH',shell = True)
	subprocess.call('%s %s --background --python %s %s %d'%(BLENDER,blank,blenderScript,\
					os.path.join(dst_dir,'blender_render.cfg'),skip_factor),shell=True)
	cmd = 'mv %s/0.00000.png %s/pose.png' %(dst_dir,dst_dir)
	subprocess.call(cmd, shell=True)

def main():
	sound_dir = '/data/vision/billf/object-properties/sound/sound/primitives/data/hmstudy/audio/'
	sound_id = int(sys.argv[1])
	os.chdir(sound_dir+'%03d'%sound_id)
	clickcfg = ConfigParser.ConfigParser()
	clickcfg.read('./sim/click.ini')
	obj_id = int(clickcfg.get('mesh','surface_mesh')[:-4])
	print(obj_id)
	posecfg = ConfigParser.ConfigParser()
	posecfg.read('./config/pose/pose-0-0.cfg')
	rotation_str = posecfg.get('DEFAULT','rotation').strip('[').strip(']').split(',')
	rotation = [float(i) for i in rotation_str]
	height = float(posecfg.get('DEFAULT','center').split(',')[1])
	print(rotation)
	print(height)
	render(obj_id,height,rotation,sound_dir+'%03d'%sound_id)
if __name__ == '__main__':
	main()
	  