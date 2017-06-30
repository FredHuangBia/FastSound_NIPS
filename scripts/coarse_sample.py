orig_dataset = open('/data/vision/billf/object-properties/sound/sound/primitives/data/primV3b.txt','r')
new_dataset = open('/data/vision/billf/object-properties/sound/sound/primitives/data/primV3b_coarse.txt','w')

coarse_shapes = [5,6,7,12,13]

cor2id={'5':0,'6':1,'7':2,'12':3,'13':4}

def mapping(value):
	if value<=0:
		return 0
	else:
		return 1

def mapping_h(value):
	if value<=-0.5:
		return 0
	elif value <=0:
		return 1
	elif value <= 0.5:
		return 2
	else:
		return 3

total = 0
for line in orig_dataset:
	pieces = line.split()
	if int(pieces[1]) in coarse_shapes:
		new_dataset.write( pieces[0]+' ')
		new_dataset.write( str(cor2id[pieces[1]])+' ')
		new_dataset.write( str(mapping_h(float(pieces[4])))+' ')
		new_dataset.write( str(mapping(float(pieces[5])))+' ')
		new_dataset.write( str(mapping(float(pieces[7])))+'\n')
		total+=1

print(total)
