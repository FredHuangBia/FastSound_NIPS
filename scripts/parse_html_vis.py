index = open('/data/vision/billf/object-properties/sound/sound/primitives/www/primV2e_cnnE_soundnet8_pretrainnone_mse1_LR0.001/131/index.html','r')
vis = open('/data/vision/billf/object-properties/sound/sound/primitives/www/primV2e_cnnE_soundnet8_pretrainnone_mse1_LR0.001/131/vis.html','w')
target_value = open('/data/vision/billf/object-properties/sound/sound/primitives/www/primV2e_cnnE_soundnet8_pretrainnone_mse1_LR0.001/131/target_value.txt','r')
output_value=open('/data/vision/billf/object-properties/sound/sound/primitives/www/primV2e_cnnE_soundnet8_pretrainnone_mse1_LR0.001/131/output_value.txt','r')

target_values = [value for value in target_value]
output_values = [value for value in output_value]

current = 0
for line in index:
	pieces=line.split()
	if len(pieces)>0 and pieces[-1] == '</td></tr>':
		vis.write('<tr><td></td><td nowrap>target: ')
		for i in range(len(pieces)):
			if i>0 and i%24==0:
				vis.write('<br/>target: '+target_values[current][0:-1]+' ')
				vis.write('<br/>output: '+output_values[current][0:-1]+' ')
				current+=1
				vis.write('</td><td nowrap>target: ')
			elif i>0:
				vis.write(pieces[i]+' ')

	else:
		vis.write(line)