import os

ROOT = '/data/vision/billf/object-properties/sound/sound/primitives/data/v2.3/'
objName = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
subNum = [125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 126, 120, 120]
dirs = ['mat-0-0', 'mat-0-1', 'mat-0-2', 'mat-0-3', 'mat-0-4', 'mat-0-5', 'mat-0-6', 'mat-0-7', 'mat-0-8', 'mat-0-9',]

for i in objName:
	print(i)
	for k in range(subNum[i]):
		os.chdir(ROOT+'%d/%d'%(i,k))
		files = os.listdir()
		matdirs = []

		for j in dirs:
			if os.path.exists(j):
				matdirs.append(j)
			else:
				print("%s" %ROOT+'%d/%d/%s'%(i,k,j))
				# print("%s: no such dir" %ROOT+'%d/%d/%s'%(i,k,j))
		# for j in files:
		# 	if j[:3] == 'mat':
		# 		matdirs.append(j)
		# if len(matdirs) != 10:
		# 	print("something wrong! %d/%d"%(i,k))

		for j in matdirs:
			curPath = os.path.join(ROOT,'%d/%d'%(i,k),j)
			os.chdir(curPath)

			objPath = os.path.join(curPath,'%d-%d.obj' %(i,k))
			beminput = os.path.join(curPath,'bem_input','init_bem.mat')
			bemoutput = os.path.join(curPath,'bem_result','output-59.dat')
			genmoments = os.path.join(curPath,'moments','moments.pbuf')

			if os.path.exists(objPath) is False:
				print("%s" %curPath)
				# print("%s: extmat failed" %curPath)
				# pass
			elif os.path.exists(beminput) is False:
				print("%s" %curPath)
				# print("%s: BEM init failed" %curPath)
				# pass
			elif os.path.exists(bemoutput) is False:
				print("%s" %curPath)
				# print("%s: BEM solving failed" %curPath)
				# pass
			elif os.path.exists(genmoments) is False:
				print("%s" %curPath)
				# print("%s: gen_moments failed" %curPath)
				# pass
			# else:
				# print("%s: pipeline completed!" %curPath)
