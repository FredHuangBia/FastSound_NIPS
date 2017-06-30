import os
import subprocess

visualization = True

www_path = '/data/vision/billf/object-properties/sound/sound/primitives/www/primV3b_cnnF_soundnet8_pretrainnone_mse1_LR0.001/151/'
audioDir = '/data/vision/billf/object-properties/sound/sound/primitives/www/primV3b_cnnF_soundnet8_pretrainnone_mse1_LR0.001/151/audio/'
synth_folders = ['primV3b_weight','primV3b_random','primV3b_raw_recog','primV3b_raw_random','primV3b_soundnet8_prior','primV3b_soundnet8_random','primV3b_stft_recog','primV3b_corl_recog','primV3b_corl_random','primV3b_mh']
exp_path = '/data/vision/billf/object-properties/sound/sound/primitives/exp/'

target_file = open(www_path+'targets.txt','r')
targets = [line.split()[0] for line in target_file]

index = open(www_path+'index.html','r')
if visualization:
	vis = open(www_path+'vis.html','w')
else:
	vis = open(www_path+'dif.html','w')


def get_labels_and_move(audioDir, synth_folder, move):
	best_labels = []
	current_sound_ind = 1
	for target in targets:
		find = 1
		try:
			best_sound_add = ''
			best_file_path = os.path.join(exp_path,synth_folder,target,'best.txt')
			best_file = open( best_file_path ,'r')
			for line in best_file:
				best_labels.append( line.split()[1:] )
				best_sound_add = os.path.join(exp_path,synth_folder,target,line.split()[0]+'-2','sound.wav')
				break
			cmd = 'cp '+best_sound_add+' '+os.path.join(audioDir,synth_folder+str(current_sound_ind)+'.wav')
			best_file.close()
		except:
			find = 0
			best_labels.append( [0,0,0,0,0,0,0] )
			print('Not found corresponding to target: '+str(target))
		if move and find==1:
			subprocess.call(cmd,shell=True)
		current_sound_ind+=1			
	return best_labels

def get_labels_and_move_tmp(audioDir, synth_folder, move):
	best_labels = []
	current_sound_ind = 1
	for target in targets:
		find = 1
		try:
			best_sound_add = ''
			best_file_path = os.path.join(exp_path,synth_folder,target,'dist.log')
			best_file = open( best_file_path ,'r')
			line = best_file.readlines()[-1]
			best_labels.append( line.split()[4:11] )
			best_sound_add = os.path.join(exp_path,synth_folder,target,line.split()[2],'sound.wav')
			cmd = 'cp '+best_sound_add+' '+os.path.join(audioDir,synth_folder+str(current_sound_ind)+'.wav')
			best_file.close()
		except:
			find = 0
			best_labels.append( [0,0,0,0,0,0,0] )
			print('Not found corresponding to target: '+str(target))
		if move and find==1:
			subprocess.call(cmd,shell=True)
		current_sound_ind+=1			
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
weight_random_labels =	get_labels_and_move(audioDir,synth_folders[1], False)
raw_prior_labels = get_labels_and_move(audioDir,synth_folders[2], False)
raw_random_labels =	get_labels_and_move(audioDir,synth_folders[3], False)
sound8_prior_labels =	get_labels_and_move(audioDir,synth_folders[4], False)
sound8_random_labels =	get_labels_and_move(audioDir,synth_folders[5], False)
stft_recog_labels = get_labels_and_move(audioDir,synth_folders[6], False)
corl_recog_labels = get_labels_and_move(audioDir,synth_folders[7], False)
corl_random_labels = get_labels_and_move(audioDir,synth_folders[8], False)
mh_recog_labels = get_labels_and_move_tmp(audioDir,synth_folders[9], True)

init_weight_random_labels = get_init_labels(synth_folders[1])
init_raw_random_labels =	get_init_labels(synth_folders[3])
init_sound8_random_labels =	get_init_labels(synth_folders[5])
init_corl_random_labels = get_init_labels(synth_folders[8])

target_labels = []
inference_labels = []

for line in index:
	line = line.split()
	if line[-1] == '</td></tr>':
		for i in range(8):
			target_labels.append(line[24*i+1:24*i+8])
			inference_labels.append(line[24*i+9:24*i+16])

vis.write('<html>\n<table border="1">\n')

current_sound = 0
for i in range(8):
	vis.write('<tr>'+'\n')

	for j in range(6):
		vis.write('<td>&nbsp'+str(current_sound+1)+'\n')
		vis.write('</td>'+'\n')


		vis.write('<td>'+'\n')
		vis.write('<br>')
		vis.write('target '+'<br><br>')
		vis.write('recognition '+'<br><br><br>')
		vis.write('ourfeature_recog '+'<br><br><br>')
		vis.write('ourfeature_random '+'<br><br><br><br>')
		vis.write('raw_recog '+'<br><br><br>')
		vis.write('raw_random '+'<br><br><br>')
		vis.write('sound8_recog '+'<br><br><br>')
		vis.write('sound8_random '+'<br><br><br>')		
		vis.write('stft_recog '+'<br><br><br>')
		vis.write('corl_recog '+'<br><br><br>')		
		vis.write('corl_random '+'<br><br><br>')	
		vis.write('mh_recog '+'<br><br>')
		
		vis.write('</td>'+'\n')

		vis.write('<td nowrap>'+'\n')

		if visualization :
			vis.write('''<audio controls="controls" preload="none"><source src='./audio/%s.wav' type='audio/wav'>Your browser does not support the audio tag.</audio><br>\n''' % ('target'+str(current_sound+1) ))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(target_labels[current_sound][0]), int(target_labels[current_sound][1]), int(target_labels[current_sound][2]), float(target_labels[current_sound][3]), float(target_labels[current_sound][4]), float(target_labels[current_sound][5]), float(target_labels[current_sound][6])   ))
			vis.write('''<audio controls="controls" preload="none"><source src='./audio/%s.wav' type='audio/wav'>Your browser does not support the audio tag.</audio><br>\n''' % ('output'+str(current_sound+1) ))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(inference_labels[current_sound][0]), int(inference_labels[current_sound][1]), int(inference_labels[current_sound][2]), float(inference_labels[current_sound][3]), float(inference_labels[current_sound][4]), float(inference_labels[current_sound][5]), float(inference_labels[current_sound][6])   ))
			vis.write('''<audio controls="controls" preload="none"><source src='./audio/%s.wav' type='audio/wav'>Your browser does not support the audio tag.</audio><br>\n''' % ('primV3b_weight'+str(current_sound+1) ))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(weight_labels[current_sound][0]), int(weight_labels[current_sound][1]), int(weight_labels[current_sound][2]), float(weight_labels[current_sound][3]), float(weight_labels[current_sound][4]), float(weight_labels[current_sound][5]), float(weight_labels[current_sound][6])   ))
			vis.write('''<audio controls="controls" preload="none"><source src='./audio/%s.wav' type='audio/wav'>Your browser does not support the audio tag.</audio><br>\n''' % ('primV3b_random'+str(current_sound+1) ))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(init_weight_random_labels[current_sound][0]), int(init_weight_random_labels[current_sound][1]), int(init_weight_random_labels[current_sound][2]), float(init_weight_random_labels[current_sound][3]), float(init_weight_random_labels[current_sound][4]), float(init_weight_random_labels[current_sound][5]), float(init_weight_random_labels[current_sound][6]) ))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(weight_random_labels[current_sound][0]), int(weight_random_labels[current_sound][1]), int(weight_random_labels[current_sound][2]), float(weight_random_labels[current_sound][3]), float(weight_random_labels[current_sound][4]), float(weight_random_labels[current_sound][5]), float(weight_random_labels[current_sound][6]) ))
			vis.write('''<audio controls="controls" preload="none"><source src='./audio/%s.wav' type='audio/wav'>Your browser does not support the audio tag.</audio><br>\n''' % ('primV3b_raw_recog'+str(current_sound+1) ))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(raw_prior_labels[current_sound][0]), int(raw_prior_labels[current_sound][1]), int(raw_prior_labels[current_sound][2]), float(raw_prior_labels[current_sound][3]), float(raw_prior_labels[current_sound][4]), float(raw_prior_labels[current_sound][5]), float(raw_prior_labels[current_sound][6]) ))			
			vis.write('''<audio controls="controls" preload="none"><source src='./audio/%s.wav' type='audio/wav'>Your browser does not support the audio tag.</audio><br>\n''' % ('primV3b_raw_random'+str(current_sound+1) ))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(init_raw_random_labels[current_sound][0]), int(init_raw_random_labels[current_sound][1]), int(init_raw_random_labels[current_sound][2]), float(init_raw_random_labels[current_sound][3]), float(init_raw_random_labels[current_sound][4]), float(init_raw_random_labels[current_sound][5]), float(init_raw_random_labels[current_sound][6])   ))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(raw_random_labels[current_sound][0]), int(raw_random_labels[current_sound][1]), int(raw_random_labels[current_sound][2]), float(raw_random_labels[current_sound][3]), float(raw_random_labels[current_sound][4]), float(raw_random_labels[current_sound][5]), float(raw_random_labels[current_sound][6])   ))
			vis.write('''<audio controls="controls" preload="none"><source src='./audio/%s.wav' type='audio/wav'>Your browser does not support the audio tag.</audio><br>\n''' % ('primV3b_soundnet8_prior'+str(current_sound+1) ))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(sound8_prior_labels[current_sound][0]), int(sound8_prior_labels[current_sound][1]), int(sound8_prior_labels[current_sound][2]), float(sound8_prior_labels[current_sound][3]), float(sound8_prior_labels[current_sound][4]), float(sound8_prior_labels[current_sound][5]), float(sound8_prior_labels[current_sound][6])   ))
			vis.write('''<audio controls="controls" preload="none"><source src='./audio/%s.wav' type='audio/wav'>Your browser does not support the audio tag.</audio><br>\n''' % ('primV3b_soundnet8_random'+str(current_sound+1) ))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(init_sound8_random_labels[current_sound][0]), int(init_sound8_random_labels[current_sound][1]), int(init_sound8_random_labels[current_sound][2]), float(init_sound8_random_labels[current_sound][3]), float(init_sound8_random_labels[current_sound][4]), float(init_sound8_random_labels[current_sound][5]), float(init_sound8_random_labels[current_sound][6])   ))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(sound8_random_labels[current_sound][0]), int(sound8_random_labels[current_sound][1]), int(sound8_random_labels[current_sound][2]), float(sound8_random_labels[current_sound][3]), float(sound8_random_labels[current_sound][4]), float(sound8_random_labels[current_sound][5]), float(sound8_random_labels[current_sound][6])   ))
			vis.write('''<audio controls="controls" preload="none"><source src='./audio/%s.wav' type='audio/wav'>Your browser does not support the audio tag.</audio><br>\n''' % ('primV3b_stft_recog'+str(current_sound+1) ))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(stft_recog_labels[current_sound][0]), int(stft_recog_labels[current_sound][1]), int(stft_recog_labels[current_sound][2]), float(stft_recog_labels[current_sound][3]), float(stft_recog_labels[current_sound][4]), float(stft_recog_labels[current_sound][5]), float(stft_recog_labels[current_sound][6])   ))
			vis.write('''<audio controls="controls" preload="none"><source src='./audio/%s.wav' type='audio/wav'>Your browser does not support the audio tag.</audio><br>\n''' % ('primV3b_corl_recog'+str(current_sound+1) ))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(corl_recog_labels[current_sound][0]), int(corl_recog_labels[current_sound][1]), int(corl_recog_labels[current_sound][2]), float(corl_recog_labels[current_sound][3]), float(corl_recog_labels[current_sound][4]), float(corl_recog_labels[current_sound][5]), float(corl_recog_labels[current_sound][6])   ))
			vis.write('''<audio controls="controls" preload="none"><source src='./audio/%s.wav' type='audio/wav'>Your browser does not support the audio tag.</audio><br>\n''' % ('primV3b_corl_random'+str(current_sound+1) ))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(init_corl_random_labels[current_sound][0]), int(init_corl_random_labels[current_sound][1]), int(init_corl_random_labels[current_sound][2]), float(init_corl_random_labels[current_sound][3]), float(init_corl_random_labels[current_sound][4]), float(init_corl_random_labels[current_sound][5]), float(init_corl_random_labels[current_sound][6])   ))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(corl_random_labels[current_sound][0]), int(corl_random_labels[current_sound][1]), int(corl_random_labels[current_sound][2]), float(corl_random_labels[current_sound][3]), float(corl_random_labels[current_sound][4]), float(corl_random_labels[current_sound][5]), float(corl_random_labels[current_sound][6])   ))
			vis.write('''<audio controls="controls" preload="none"><source src='./audio/%s.wav' type='audio/wav'>Your browser does not support the audio tag.</audio><br>\n''' % ('primV3b_mh'+str(current_sound+1) ))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(mh_recog_labels[current_sound][0]), int(mh_recog_labels[current_sound][1]), int(mh_recog_labels[current_sound][2]), float(mh_recog_labels[current_sound][3]), float(mh_recog_labels[current_sound][4]), float(mh_recog_labels[current_sound][5]), float(mh_recog_labels[current_sound][6])   ))

		else:
			vis.write('''<audio controls="controls" preload="none"><source src='./audio/%s.wav' type='audio/wav'>Your browser does not support the audio tag.</audio><br>\n''' % ('target'+str(current_sound+1) ))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(target_labels[current_sound][0]), int(target_labels[current_sound][1]), int(target_labels[current_sound][2]), float(target_labels[current_sound][3]), float(target_labels[current_sound][4]), float(target_labels[current_sound][5]), float(target_labels[current_sound][6])   ))
			vis.write('''<audio controls="controls" preload="none"><source src='./audio/%s.wav' type='audio/wav'>Your browser does not support the audio tag.</audio><br>\n''' % ('output'+str(current_sound+1) ))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(inference_labels[current_sound][0]), int(inference_labels[current_sound][1]), int(inference_labels[current_sound][2]), float(inference_labels[current_sound][3])-float(target_labels[current_sound][3]), float(inference_labels[current_sound][4])-float(target_labels[current_sound][4]), float(inference_labels[current_sound][5])-float(target_labels[current_sound][5]), float(inference_labels[current_sound][6])-float(target_labels[current_sound][6])   ))
			vis.write('''<audio controls="controls" preload="none"><source src='./audio/%s.wav' type='audio/wav'>Your browser does not support the audio tag.</audio><br>\n''' % ('primV3b_weight'+str(current_sound+1) ))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(weight_labels[current_sound][0]), int(weight_labels[current_sound][1]), int(weight_labels[current_sound][2]), float(weight_labels[current_sound][3])-float(target_labels[current_sound][3]), float(weight_labels[current_sound][4])-float(target_labels[current_sound][4]), float(weight_labels[current_sound][5])-float(target_labels[current_sound][5]), float(weight_labels[current_sound][6])-float(target_labels[current_sound][6])   ))
			vis.write('''<audio controls="controls" preload="none"><source src='./audio/%s.wav' type='audio/wav'>Your browser does not support the audio tag.</audio><br>\n''' % ('primV3b_random'+str(current_sound+1) ))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(init_weight_random_labels[current_sound][0]), int(init_weight_random_labels[current_sound][1]), int(init_weight_random_labels[current_sound][2]), float(init_weight_random_labels[current_sound][3])-float(target_labels[current_sound][3]), float(init_weight_random_labels[current_sound][4])-float(target_labels[current_sound][4]), float(init_weight_random_labels[current_sound][5])-float(target_labels[current_sound][5]), float(init_weight_random_labels[current_sound][6]) -float(target_labels[current_sound][6])))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(weight_random_labels[current_sound][0]), int(weight_random_labels[current_sound][1]), int(weight_random_labels[current_sound][2]), float(weight_random_labels[current_sound][3])-float(target_labels[current_sound][3]), float(weight_random_labels[current_sound][4])-float(target_labels[current_sound][4]), float(weight_random_labels[current_sound][5])-float(target_labels[current_sound][5]), float(weight_random_labels[current_sound][6])-float(target_labels[current_sound][6])   ))
			vis.write('''<audio controls="controls" preload="none"><source src='./audio/%s.wav' type='audio/wav'>Your browser does not support the audio tag.</audio><br>\n''' % ('primV3b_raw_recog'+str(current_sound+1) ))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(raw_prior_labels[current_sound][0]), int(raw_prior_labels[current_sound][1]), int(raw_prior_labels[current_sound][2]), float(raw_prior_labels[current_sound][3])-float(target_labels[current_sound][3]), float(raw_prior_labels[current_sound][4])-float(target_labels[current_sound][4]), float(raw_prior_labels[current_sound][5])-float(target_labels[current_sound][5]), float(raw_prior_labels[current_sound][6]) -float(target_labels[current_sound][6])))			
			vis.write('''<audio controls="controls" preload="none"><source src='./audio/%s.wav' type='audio/wav'>Your browser does not support the audio tag.</audio><br>\n''' % ('primV3b_raw_random'+str(current_sound+1) ))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(init_raw_random_labels[current_sound][0]), int(init_raw_random_labels[current_sound][1]), int(init_raw_random_labels[current_sound][2]), float(init_raw_random_labels[current_sound][3])-float(target_labels[current_sound][3]), float(init_raw_random_labels[current_sound][4])-float(target_labels[current_sound][4]), float(init_raw_random_labels[current_sound][5])-float(target_labels[current_sound][5]), float(init_raw_random_labels[current_sound][6])-float(target_labels[current_sound][6])   ))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(raw_random_labels[current_sound][0]), int(raw_random_labels[current_sound][1]), int(raw_random_labels[current_sound][2]), float(raw_random_labels[current_sound][3])-float(target_labels[current_sound][3]), float(raw_random_labels[current_sound][4])-float(target_labels[current_sound][4]), float(raw_random_labels[current_sound][5])-float(target_labels[current_sound][5]), float(raw_random_labels[current_sound][6])-float(target_labels[current_sound][6])   ))
			vis.write('''<audio controls="controls" preload="none"><source src='./audio/%s.wav' type='audio/wav'>Your browser does not support the audio tag.</audio><br>\n''' % ('primV3b_soundnet8_prior'+str(current_sound+1) ))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(sound8_prior_labels[current_sound][0]), int(sound8_prior_labels[current_sound][1]), int(sound8_prior_labels[current_sound][2]), float(sound8_prior_labels[current_sound][3])-float(target_labels[current_sound][3]), float(sound8_prior_labels[current_sound][4])-float(target_labels[current_sound][4]), float(sound8_prior_labels[current_sound][5])-float(target_labels[current_sound][5]), float(sound8_prior_labels[current_sound][6])-float(target_labels[current_sound][6])   ))
			vis.write('''<audio controls="controls" preload="none"><source src='./audio/%s.wav' type='audio/wav'>Your browser does not support the audio tag.</audio><br>\n''' % ('primV3b_soundnet8_random'+str(current_sound+1) ))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(init_sound8_random_labels[current_sound][0]), int(init_sound8_random_labels[current_sound][1]), int(init_sound8_random_labels[current_sound][2]), float(init_sound8_random_labels[current_sound][3])-float(target_labels[current_sound][3]), float(init_sound8_random_labels[current_sound][4])-float(target_labels[current_sound][4]), float(init_sound8_random_labels[current_sound][5])-float(target_labels[current_sound][5]), float(init_sound8_random_labels[current_sound][6])-float(target_labels[current_sound][6])   ))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(sound8_random_labels[current_sound][0]), int(sound8_random_labels[current_sound][1]), int(sound8_random_labels[current_sound][2]), float(sound8_random_labels[current_sound][3])-float(target_labels[current_sound][3]), float(sound8_random_labels[current_sound][4])-float(target_labels[current_sound][4]), float(sound8_random_labels[current_sound][5])-float(target_labels[current_sound][5]), float(sound8_random_labels[current_sound][6])-float(target_labels[current_sound][6])   ))
			vis.write('''<audio controls="controls" preload="none"><source src='./audio/%s.wav' type='audio/wav'>Your browser does not support the audio tag.</audio><br>\n''' % ('primV3b_stft_random'+str(current_sound+1) ))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(stft_recog_labels[current_sound][0]), int(stft_recog_labels[current_sound][1]), int(stft_recog_labels[current_sound][2]), float(stft_recog_labels[current_sound][3])-float(target_labels[current_sound][3]), float(stft_recog_labels[current_sound][4])-float(target_labels[current_sound][4]), float(stft_recog_labels[current_sound][5])-float(target_labels[current_sound][5]), float(stft_recog_labels[current_sound][6])-float(target_labels[current_sound][6])   ))
			vis.write('''<audio controls="controls" preload="none"><source src='./audio/%s.wav' type='audio/wav'>Your browser does not support the audio tag.</audio><br>\n''' % ('primV3b_corl_recog'+str(current_sound+1) ))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(corl_recog_labels[current_sound][0]), int(corl_recog_labels[current_sound][1]), int(corl_recog_labels[current_sound][2]), float(corl_recog_labels[current_sound][3])-float(target_labels[current_sound][3]), float(corl_recog_labels[current_sound][4])-float(target_labels[current_sound][4]), float(corl_recog_labels[current_sound][5])-float(target_labels[current_sound][5]), float(corl_recog_labels[current_sound][6])-float(target_labels[current_sound][6])   ))
			vis.write('''<audio controls="controls" preload="none"><source src='./audio/%s.wav' type='audio/wav'>Your browser does not support the audio tag.</audio><br>\n''' % ('primV3b_corl_random'+str(current_sound+1) ))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(init_corl_random_labels[current_sound][0]), int(init_corl_random_labels[current_sound][1]), int(init_corl_random_labels[current_sound][2]), float(init_corl_random_labels[current_sound][3])-float(target_labels[current_sound][3]), float(init_corl_random_labels[current_sound][4])-float(target_labels[current_sound][4]), float(init_corl_random_labels[current_sound][5])-float(target_labels[current_sound][5]), float(init_corl_random_labels[current_sound][6])-float(target_labels[current_sound][6])   ))
			vis.write('''______&nbsp %02d &nbsp %02d &nbsp %02d &nbsp %.2f &nbsp %.2f &nbsp %.2f &nbsp %.2f <br>\n''' % ( int(corl_random_labels[current_sound][0]), int(corl_random_labels[current_sound][1]), int(corl_random_labels[current_sound][2]), float(corl_random_labels[current_sound][3])-float(target_labels[current_sound][3]), float(corl_random_labels[current_sound][4])-float(target_labels[current_sound][4]), float(corl_random_labels[current_sound][5])-float(target_labels[current_sound][5]), float(corl_random_labels[current_sound][6])-float(target_labels[current_sound][6])   ))

		vis.write('</td>'+'\n')
		current_sound +=1
	vis.write('</tr>'+'\n')

vis.write('</table>\n</html>\n')