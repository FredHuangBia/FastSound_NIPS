root = '/data/vision/billf/object-properties/sound/sound/primitives/www/amt_fastsound/fileList/'

current = 0
for i in range(5):
	fo = open(root + 'fileList-' + str(i) + '.txt', 'w')
	for j in range(10):
		fo.write('data/sound/' + str(current) + '.m4a\n')
		if j == 9:
			fo.write('data/sound/' + str(current) + '.ogg')
		else:
			fo.write('data/sound/' + str(current) + '.ogg\n')
		current += 1
	fo.close()
		