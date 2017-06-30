import os

ROOT = '/data/vision/billf/object-properties/sound/sound/primitives/data/v4s/BEMs/'
# objName = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
objName = [18]
matName = list(range(10))

for i in objName:
	for j in matName:
		curPath = os.path.join(ROOT,'%02d%d'%(i,j))
		os.chdir(curPath)

		objPath = os.path.join(curPath,'%02d%d.obj' %(i,j))
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
