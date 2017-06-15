import os
import subprocess

visualize = True

www_path = '/data/vision/billf/object-properties/sound/sound/primitives/www/primV3b_5000_cnnF_soundnet8_pretrainnone_mse1_LR0.001/50/'
audioDir = '/data/vision/billf/object-properties/sound/sound/primitives/www/primV3b_5000_cnnF_soundnet8_pretrainnone_mse1_LR0.001/50/audio/'
synth_folders = ['primV3b_5000_weight','primV3b_5000_stft_recog','primV3b_5000_innerP_recog']
exp_path = '/data/vision/billf/object-properties/sound/sound/primitives/exp/'

target_file = open(www_path+'targets.txt','r')
targets = [line.split()[0] for line in target_file]

index = open(www_path+'index.html','r')
if visualize:
	vis = open(www_path+'vis.html','w')
else:
	vis = open(www_path+'dif.html','w')




def get_labels_and_move(audioDir, synth_folder, move):
	best_labels = []
	current_sound_ind = 1
	for target in targets:
		best_sound_add = ''
		best_file_path = os.path.join(exp_path,synth_folder,target,'best.txt')
		best_file = open( best_file_path ,'r')
		for line in best_file:
			best_labels.append( line.split()[1:] )
			best_sound_add = os.path.join(exp_path,synth_folder,target,line.split()[0],'sound.wav')
			break
		cmd = 'cp '+best_sound_add+' '+os.path.join(audioDir,synth_folder+str(current_sound_ind)+'.wav')
		if move:
			subprocess.call(cmd,shell=True)
		current_sound_ind+=1
		best_file.close()
	return best_labels

def get_init_labels(synth_folder):
	init_labels = []
	for target in targets:
		init_file_path = os.path.join(exp_path,synth_folder,target,'mse.log')
		init_file = open( init_file_path ,'r')
		for line in init_file:
			init_labels.append( line.split()[0:] )
			break
		init_file.close()
	return init_labels


weight_labels =	get_labels_and_move(audioDir,synth_folders[0], False)
stft_prior_labels = get_labels_and_move(audioDir,synth_folders[1], False)
innerP_prior_labels = get_labels_and_move(audioDir,synth_folders[2], False)
# init_weight_random_labels = get_init_labels(synth_folders[1])
# init_raw_random_labels =	get_init_labels(synth_folders[3])
# init_sound8_random_labels =	get_init_labels(synth_folders[5])


target_labels = []
inference_labels = []

for line in index:
	line = line.split()
	if line[-1] == '</td></tr>':
		for i in range(8):
			target_labels.append(line[25*i+2:25*i+9])
			inference_labels.append(line[25*i+10:25*i+17])

vis.write('<html>\n<table border="1">\n')

current_sound = 0
for i in range(8):
	vis.write('<tr>'+'\n')

	for j in range(6):
		if not (len(target_labels[current_sound])>1):
			break
		vis.write('<td>&nbsp'+str(current_sound+1)+'\n')
		vis.write('</td>'+'\n')


		vis.write('<td>'+'\n')
		vis.write('<br>')
		vis.write('target '+'<br><br><br>')
		vis.write('recognition '+'<br><br><br>')
		vis.write('ourfeature_recog '+'<br><br><br>')
		vis.write('stft_recog '+'<br><br><br>')	
		vis.write('innerP_recog '+'<br><br><br>')	

		
		vis.write('</td>'+'\n')

		vis.write('<td nowrap>'+'\n')

		if visualize :
			vis.write('''<audio controls="controls" preload="none"><source src='./audio/%s.wav' type='audio/wav'>Your browser does not support the audio tag.</audio><br>\n''' % ('target'+str(current_sound+1) ))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(target_labels[current_sound][0]), int(target_labels[current_sound][1]), int(target_labels[current_sound][2]), float(target_labels[current_sound][3]), float(target_labels[current_sound][4]), float(target_labels[current_sound][5]), float(target_labels[current_sound][6])   ))
			vis.write('''<audio controls="controls" preload="none"><source src='./audio/%s.wav' type='audio/wav'>Your browser does not support the audio tag.</audio><br>\n''' % ('output'+str(current_sound+1) ))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(inference_labels[current_sound][0]), int(inference_labels[current_sound][1]), int(inference_labels[current_sound][2]), float(inference_labels[current_sound][3]), float(inference_labels[current_sound][4]), float(inference_labels[current_sound][5]), float(inference_labels[current_sound][6])   ))
			vis.write('''<audio controls="controls" preload="none"><source src='./audio/%s.wav' type='audio/wav'>Your browser does not support the audio tag.</audio><br>\n''' % ('primV3b_5000_weight'+str(current_sound+1) ))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(weight_labels[current_sound][0]), int(weight_labels[current_sound][1]), int(weight_labels[current_sound][2]), float(weight_labels[current_sound][3]), float(weight_labels[current_sound][4]), float(weight_labels[current_sound][5]), float(weight_labels[current_sound][6])   ))
			vis.write('''<audio controls="controls" preload="none"><source src='./audio/%s.wav' type='audio/wav'>Your browser does not support the audio tag.</audio><br>\n''' % ('primV3b_5000_stft_recog'+str(current_sound+1) ))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(stft_prior_labels[current_sound][0]), int(stft_prior_labels[current_sound][1]), int(stft_prior_labels[current_sound][2]), float(stft_prior_labels[current_sound][3]), float(stft_prior_labels[current_sound][4]), float(stft_prior_labels[current_sound][5]), float(stft_prior_labels[current_sound][6]) ))			
			vis.write('''<audio controls="controls" preload="none"><source src='./audio/%s.wav' type='audio/wav'>Your browser does not support the audio tag.</audio><br>\n''' % ('primV3b_5000_innerP_recog'+str(current_sound+1) ))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(innerP_prior_labels[current_sound][0]), int(innerP_prior_labels[current_sound][1]), int(innerP_prior_labels[current_sound][2]), float(innerP_prior_labels[current_sound][3]), float(innerP_prior_labels[current_sound][4]), float(innerP_prior_labels[current_sound][5]), float(innerP_prior_labels[current_sound][6]) ))			

		else:
			vis.write('''<audio controls="controls" preload="none"><source src='./audio/%s.wav' type='audio/wav'>Your browser does not support the audio tag.</audio><br>\n''' % ('target'+str(current_sound+1) ))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(target_labels[current_sound][0]), int(target_labels[current_sound][1]), int(target_labels[current_sound][2]), float(target_labels[current_sound][3]), float(target_labels[current_sound][4]), float(target_labels[current_sound][5]), float(target_labels[current_sound][6])   ))
			vis.write('''<audio controls="controls" preload="none"><source src='./audio/%s.wav' type='audio/wav'>Your browser does not support the audio tag.</audio><br>\n''' % ('output'+str(current_sound+1) ))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(inference_labels[current_sound][0]), int(inference_labels[current_sound][1]), int(inference_labels[current_sound][2]), float(inference_labels[current_sound][3])-float(target_labels[current_sound][3]), float(inference_labels[current_sound][4])-float(target_labels[current_sound][4]), float(inference_labels[current_sound][5])-float(target_labels[current_sound][5]), float(inference_labels[current_sound][6])-float(target_labels[current_sound][6])   ))
			vis.write('''<audio controls="controls" preload="none"><source src='./audio/%s.wav' type='audio/wav'>Your browser does not support the audio tag.</audio><br>\n''' % ('primV3b_5000_weight'+str(current_sound+1) ))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(weight_labels[current_sound][0]), int(weight_labels[current_sound][1]), int(weight_labels[current_sound][2]), float(weight_labels[current_sound][3])-float(target_labels[current_sound][3]), float(weight_labels[current_sound][4])-float(target_labels[current_sound][4]), float(weight_labels[current_sound][5])-float(target_labels[current_sound][5]), float(weight_labels[current_sound][6])-float(target_labels[current_sound][6])   ))
			vis.write('''<audio controls="controls" preload="none"><source src='./audio/%s.wav' type='audio/wav'>Your browser does not support the audio tag.</audio><br>\n''' % ('primV3b_5000_stft_recog'+str(current_sound+1) ))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(stft_prior_labels[current_sound][0]), int(stft_prior_labels[current_sound][1]), int(stft_prior_labels[current_sound][2]), float(stft_prior_labels[current_sound][3])-float(target_labels[current_sound][3]), float(stft_prior_labels[current_sound][4])-float(target_labels[current_sound][4]), float(stft_prior_labels[current_sound][5])-float(target_labels[current_sound][5]), float(stft_prior_labels[current_sound][6]) -float(target_labels[current_sound][6])))
			vis.write('''<audio controls="controls" preload="none"><source src='./audio/%s.wav' type='audio/wav'>Your browser does not support the audio tag.</audio><br>\n''' % ('primV3b_innerP_recog'+str(current_sound+1) ))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(innerP_prior_labels[current_sound][0]), int(innerP_prior_labels[current_sound][1]), int(innerP_prior_labels[current_sound][2]), float(innerP_prior_labels[current_sound][3])-float(target_labels[current_sound][3]), float(innerP_prior_labels[current_sound][4])-float(target_labels[current_sound][4]), float(innerP_prior_labels[current_sound][5])-float(target_labels[current_sound][5]), float(innerP_prior_labels[current_sound][6]) -float(target_labels[current_sound][6])))

		vis.write('</td>'+'\n')
		current_sound +=1
	vis.write('</tr>'+'\n')

vis.write('</table>\n</html>\n')