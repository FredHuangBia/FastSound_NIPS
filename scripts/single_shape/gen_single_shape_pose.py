import os

ROOT = '/data/vision/billf/object-properties/sound/sound/primitives/config/single_shape/pose/'

s = 0.707
t = 0.5

quaternion = [[0,0,0,1],[0,0,s,s],[0,s,0,s],[t,t,t,t],[s,0,0,s],[t,-t,t,t],[s,0,s,0]]
x = 0
y = [3 + 0.3*i for i in range(6)]
z = 0
linear_velocity = [0,0,0]
angular_velocity = [0,0,0]

def template(y,quaternion):
	return ['[DEFAULT]\n','center = [%.1f,%.1f,%.1f]\n' %(x,y,z),'rotation = [%.3f,%.3f,%.3f,%.3f]\n' %(quaternion[0],quaternion[1],quaternion[2],quaternion[3]), 'linear_velocity = [0,0,0]\n', 'angular_velocity = [0,0,0]']

if __name__ == '__main__':
	for i in range(6):
		for j in range(7):
			name = "pose-"+str(i)+"-"+str(j)
			f = open(ROOT+name+".cfg", "w")
			a = template(y[i], quaternion[j])
			for line in a:
				f.write(line)
			f.close()
