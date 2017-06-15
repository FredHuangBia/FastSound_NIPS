import os, time
import progressbar as pb

ROOT = '/data/vision/billf/object-properties/sound/sound/primitives/data/v2.3/'
objName = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
subNum = [125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 126, 120, 120]
dirs = ['mat-0-0', 'mat-0-1', 'mat-0-2', 'mat-0-3', 'mat-0-4', 'mat-0-5', 'mat-0-6', 'mat-0-7', 'mat-0-8', 'mat-0-9',]

successFile = open('finished.txt','w')
success = []
total = 731
def check_all():
	new_success = 5
	time.sleep(0.1)
	# for i in objName:
	# 	for k in range(subNum[i]):
	# 		os.chdir(ROOT+'%d/%d'%(i,k))
	# 		files = os.listdir()
	# 		matdirs = []

	# 		for j in dirs:
	# 			if os.path.exists(j):
	# 				matdirs.append(j)
	# 			# else:
	# 			# 	print("%s" %ROOT+'%d/%d/%s'%(i,k,j))

	# 		for j in matdirs:
	# 			curPath = os.path.join(ROOT,'%d/%d'%(i,k),j)
	# 			os.chdir(curPath)

	# 			objPath = os.path.join(curPath,'%d-%d.obj' %(i,k))
	# 			beminput = os.path.join(curPath,'bem_input','init_bem.mat')
	# 			bemoutput = os.path.join(curPath,'bem_result','output-59.dat')
	# 			genmoments = os.path.join(curPath,'moments','moments.pbuf')

	# 			if os.path.exists(objPath) is True and \
	# 				os.path.exists(beminput) is True and \
	# 				os.path.exists(bemoutput) is True and \
	# 				os.path.exists(gen_moments) is True:
	# 				yeah = '%d/%d/%s'%(i,k,j)
	# 				if yeah not in success:
	# 					successFile.write("%s" %curPath)
	# 					success.append(yeah)
	# 					new_success += 1
	return new_success

progress = pb.ProgressBar(widgets=_widgets, maxval=total).start()
progvar = 0
for i in range(500):
	increase = check_all()
	progvar += increase
	progress.update(progvar)
	if progvar > total:
		break
