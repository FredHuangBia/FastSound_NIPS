import matplotlib.pyplot as plt

activations_file = open('./activation_last_layer.txt','r')
activations = [line for line in activations_file]
activations_file.close()

value_file = open('../../../data/primV3b.txt')
values = [value for value in value_file]
value_file.close()

mapping = {1:1, 2:4, 3:7, 4:10, 5:13, 6:2, 7:5, 8:8, 9:11, 10:14, 11:3, 12:6, 13:9, 14:12, 15:15}

def activate(start,to):
	plt.figure()
	for j in range(start,to,1):
		current_plot = 0
		for activation in [activations[j*3],activations[j*3+1],activations[j*3+2]]:
			pieces=activation.split()
			for i in range(5):
				current_plot+=1
				raw_file = open('../../../data/primV3b/001/%06d/sound.raw'%int(pieces[i+1]),'r')
				value = [float(line) for line in raw_file]
				plt.subplot(5, 3, mapping[current_plot])
				plt.tight_layout(pad=0, w_pad=0, h_pad=0)
				plt.plot(value)
				plt.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
				plt.tick_params(axis='y',which='both',bottom='off',top='off',labelbottom='off',left='off',labelleft='off')
				string = values[int(pieces[i+1])-1][7:-1]
				string = string.split()
				label = '%02d  %02d  %02d\n%.02f  %.02f  %.02f  %.02f'%(int(string[0]),int(string[1]),int(string[2]),float(string[3]),float(string[4]),float(string[5]),float(string[6]))
				plt.xlabel(label,fontsize=9)
				raw_file.close()
		plt.savefig('./imgs/%d-%d-%d.png'%(j*3,j*3+1,j*3+2),dpi=300)
		plt.close()
		print(j)

# activate(0,341)

def filters():
	plt.figure()
	current_plot = 0
	for i in range(16):
		current_plot+=1
		filter_file = open('./filter%d.txt'%int(i+1),'r')
		value = [float(line) for line in filter_file]
		plt.subplot(2, 8, current_plot)
		plt.plot(value)
		filter_file.close()		
	plt.show()
	# plt.savefig('./imgs/%d-%d-%d.png'%(j*3,j*3+1,j*3+2),dpi=300,bbox_inches='tight')
	# plt.clf()

def activate_analyze(start,to):
	for j in range(start,to,1):
		for activation in [activations[j*3],activations[j*3+1],activations[j*3+2]]:
			pieces=activation.split()
			alpha_small = 1
			alpha_big = 1
			height_small = 1
			height_big = 1
			rest_small = 1
			rest_big = 1
			for i in range(5):
				string = values[int(pieces[i+1])-1][7:-1]
				string = string.split()
				label = (int(string[0]),int(string[1]),int(string[2]),float(string[3]),float(string[4]),float(string[5]),float(string[6]))
				if label[3]<0.7:
					height_big -=1
				if label[3]>-0.7:
					height_small -=1
				if label[4]<0.7:
					alpha_big -=1
				if label[4]>-0.7:
					alpha_small -=1
				if label[6]<0.7:
					rest_big -=1
				if label[6]>-0.7:
					rest_small -=1
				
			if alpha_small == 1 or alpha_big == 1 or alpha_big == 1 or height_small == 1 or height_big == 1 or rest_small == 1 or rest_big == 1:
				print(activation[0:-1])
				print(label)



for i in range(0,341):
	activate_analyze(i,i+1)
