import os
from decimal import Decimal
# import numpy as np

ROOT = '/data/vision/billf/object-properties/sound/sound/primitives/config/single_shape/material/'

youngs = 2.02*10**(10)
poison = 0.25
density = 1000

alpha_tmp = [100000,10000,1000,100,10,1]
alpha = [i*10**(-9) for i in alpha_tmp]
beta = [1,2,4,8,16,32]

friction = 0.3
rollingFriction = 0.05
spinningFriction = 0.05

restitution = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85]


def template(name,alpha,beta,restitution):
	return ['[DEFAULT]\n','name = %s\n' %name,'youngs = 2.02E+10\n', 'poison = 0.25\n', 'density = 1000\n', 'alpha = %.1E\n' %Decimal(alpha), 'beta = %d\n' %beta, 'friction = %.2f\n' %friction, 'rollingFriction = %.2f\n' %rollingFriction, 'spinningFriction = %.2f\n' %spinningFriction, 'restitution = %.2f' %restitution]

if __name__ == '__main__':
	for i in range(6):
		for j in range(6):
			for k in range(6):
				name = "material-"+str(i)+"-"+str(j)+"-"+str(k)
				f = open(ROOT+name+".cfg", "w")
				a = template(name, alpha[i], beta[j], restitution[k])
				for line in a:
					f.write(line)
				f.close()

