root = '/data/vision/billf/object-properties/sound/sound/primitives/result/single_shape/v1/'

mean = 0
max_v = 0
total_prev = 0
total = 0

for scene0 in range(6):
	for scene1 in range(7):
		for mat0 in range(6):
			print('%.2f' % (100/float(6)*scene0 + 100/float(6)/float(7)*scene1 + 100/float(6)/float(7)/float(6)*mat0) + "%" )
			for mat1 in range(6):
				for mat2 in range(6):
					path = root + 'scene-' + str(scene0) +'-' + str(scene1) + '/mat-10-10-10-' + str(mat0) + '-' + str(mat1) + '-' + str(mat2) + '/sound.raw' 
					f = open(path, "r")
					for line in f:
						val = float(line)
						if abs(val)>max_v:
							max_v = abs(val)
						mean = (mean*total+val)/float(total+1)
						total_prev = total
						total = total + 1
						if total <= total_prev:
							print('ERROR: exceed the upper bound of int')
							break
					f.close()

print('mean: ',mean)
print('max: ',max_v)



std = 0
total_prev = 0
total = 0

for scene0 in range(6):
	for scene1 in range(7):
		for mat0 in range(6):
			print('%.2f' % (100/float(6)*scene0 + 100/float(6)/float(7)*scene1 + 100/float(6)/float(7)/float(6)*mat0) + "%" )
			for mat1 in range(6):
				for mat2 in range(6):
					path = root + 'scene-' + str(scene0) +'-' + str(scene1) + '/mat-10-10-10-' + str(mat0) + '-' + str(mat1) + '-' + str(mat2) + '/sound.raw' 
					f = open(path, "r")
					for line in f:
						val = float(line)
						std = ((std**2*total + (val - mean)**2)/float(total+1))**(1/2)
						total_prev = total
						total = total + 1
						if total <= total_prev:
							print('ERROR: exceed the upper bound of int')
							break
					f.close()
print(std)