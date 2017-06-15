import os
from decimal import Decimal
import numpy as np

ROOT = '/data/vision/billf/object-properties/sound/sound/primitives/data/v2.3/'

E0 = 20*10**9
rho0 = 1000
nu0 = 0.25
alpha0 = 1*10**(-6)
beta0 = 4

# E_tmp = [1,5,10,15,20,25,30]
E_tmp = [1.0, 2.25, 4.0, 6.25, 9.0, 12.25, 16.0, 20.25, 25.0, 30.0]
E = [i*10**9 for i in E_tmp]
nu = [i/3 for i in range(3,13)]

def template(E,nu,rho,alpha,beta):
	return ['[DEFAULT]\n','name = test\n','youngs = %.2E\n' %Decimal(E), 'poison = %.2f\n' %nu, 'density = %d\n' %rho, 'alpha = %.1E\n' %Decimal(alpha), 'beta = %d\n' %beta, 'friction = 0.25\n', 'rollingFriction = 0.00\n', 'spinningFriction = 0.00\n', 'restitution = 0.6']

def Youngs():
	os.chdir(ROOT + 'materials')
	for counter, value in enumerate(E):
		fileOut = open('material-0-%d.cfg' %counter, 'w')
		fileOut.writelines(template(value,nu0,rho0,alpha0,beta0))
	print("Young's modulus done!")

def Poisson():
	os.chdir(ROOT + 'materials')
	for counter, value in enumerate(nu):
		fileOut = open('material-1-%d.cfg' %counter, 'w')
		fileOut.writelines(template(E0,value,rho0,alpha0,beta0))
	print("poisson ratio done!")

def main():
	Youngs()
	# Poisson()

if __name__ == '__main__':
	main()
