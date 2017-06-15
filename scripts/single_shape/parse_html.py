root = '/data/vision/billf/object-properties/sound/sound/primitives/www/primV2d_cnnB_soundnet8_pretrainnone_mse1_LR0.001/192/index.html'
synth = '/data/vision/billf/object-properties/sound/sound/primitives/www/primV2d_cnnB_soundnet8_pretrainnone_mse1_LR0.001/192/synth.txt'
f = '/data/vision/billf/object-properties/sound/sound/primitives/www/primV2d_cnnB_soundnet8_pretrainnone_mse1_LR0.001/192/final.html'


# to get synth infered value between -1 and 1

html = open(root,'r')
final = open(f, 'w')

out = open(synth, 'r')
outs = out.readlines()

current_index = 0
for line in html:
	line = line.split()
	if len(line) >=13 and line[13]=='N/A':
		for i in range(8):
			current = outs[current_index]
			current = current.split()
			line[13+18*i] = current[0]
			line[14+18*i] = current[1]
			line[15+18*i] = current[2]
			line[16+18*i] = current[3]
			line[17+18*i] = current[4]
			current_index += 1
	for item in line:
		final.write(item)
		final.write(' ')
	final.write('\n')

html.close()
final.close()
out.close()


