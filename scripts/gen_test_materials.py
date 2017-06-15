import os
from decimal import Decimal
import numpy as np

ROOT = '/data/vision/billf/object-properties/sound/sound/primitives/'
ACCEL = ROOT + 'config/accel/'

E0 = 20*10**9
rho0 = 1000
nu0 = 0.25
alpha0 = 1*10**(-6)
beta0 = 4

E_tmp = [1,5,10,15,20,25,30]
E = [i*10**9 for i in E_tmp]
nu = [0.1,0.15,0.2,0.25,0.3,0.35,0.4]
alpha_tmp = [10000,1000,100,10,1]
alpha = [i*10**(-9) for i in alpha_tmp]
beta = [0,1,2,4,8,16,32,64]

def template(E,nu,rho,alpha,beta):
	return ['[DEFAULT]\n','name = test\n','youngs = %.1E\n' %Decimal(E), 'poison = %.2f\n' %nu, 'density = %d\n' %rho, 'alpha = %.1E\n' %Decimal(alpha), 'beta = %d\n' %beta, 'friction = 0.25\n', 'rollingFriction = 0.00\n', 'spinningFriction = 0.00\n', 'restitution = 0.6']

def Youngs():
	os.chdir(ACCEL + 'materials')
	for counter, value in enumerate(E):
		fileOut = open('material-0-%d.cfg' %counter, 'w')
		fileOut.writelines(template(value,nu0,rho0,alpha0,beta0))
	print("Young's modulus done!")

def Poisson():
	os.chdir(ACCEL + 'materials')
	for counter, value in enumerate(nu):
		fileOut = open('material-1-%d.cfg' %counter, 'w')
		fileOut.writelines(template(E0,value,rho0,alpha0,beta0))
	print("poisson ratio done!")

def Alpha():
	os.chdir(ACCEL + 'materials')
	for counter, value in enumerate(alpha):
		fileOut = open('material-2-%d.cfg' %counter, 'w')
		fileOut.writelines(template(E0,nu0,rho0,value,beta0))
	print("alpha done!")

def Beta():
	os.chdir(ACCEL + 'materials')
	for counter, value in enumerate(beta):
		fileOut = open('material-3-%d.cfg' %counter, 'w')
		fileOut.writelines(template(E0,nu0,rho0,alpha0,value))
	print("beta done!")

def main():
#	Youngs()
#	Poisson()
#	Alpha()
	Beta()

if __name__ == '__main__':
	main()
