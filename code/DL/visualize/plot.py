import matplotlib.pyplot as plt
import math
import numpy.fft as nft
import numpy as np
from scipy import signal
import scipy.fftpack

activations_file = open('./weak/activation_last_layer.txt','r')
activations = [line for line in activations_file]
activations_file.close()

value_file = open('../../../data/primV4a.txt')
values = [value for value in value_file]
value_file.close()

mapping = {1:1, 2:4, 3:7, 4:10, 5:13, 6:2, 7:5, 8:8, 9:11, 10:14, 11:3, 12:6, 13:9, 14:12, 15:15}
fs = 44100
N = 1e4

def activate(start,to,typee):
	plt.figure()
	for j in range(start,to,1):
		current_plot = 0
		for activation in [activations[j*3],activations[j*3+1],activations[j*3+2]]:
			pieces=activation.split()
			for i in range(5):
				current_plot+=1
				raw_file = open('../../../data/primV4a/%03d/%06d/sound.raw'% (  math.floor(int(pieces[i+1])/100.0) +1, int(pieces[i+1]) ),'r')
				value = [float(line) for line in raw_file]
				plt.subplot(5, 3, mapping[current_plot])
				plt.tight_layout(pad=0, w_pad=0, h_pad=0)
				# value = scipy.fftpack.fft(value)
				if typee == 'fft':
					value = np.abs(nft.rfft(value))
					# ids = range(len(value))
					# down_value = [sum(value[e:e+50]) for e in ids if e%50==0]
					plt.plot(value)
				elif typee == 'time':
					plt.plot(value)
				# f, t, Sxx = signal.spectrogram(np.asarray(value), fs)
				# plt.pcolormesh(t, f, Sxx)
				plt.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
				plt.tick_params(axis='y',which='both',bottom='off',top='off',labelbottom='off',left='off',labelleft='off')
				string = values[int(pieces[i+1])-1][7:-1]
				string = string.split()
				label = '%02d  %02d\n%.02f  %.02f  %.02f  %.02f'%(int(string[0]),int(string[1]),float(string[6]),float(string[7]),float(string[8]),float(string[9]))
				plt.xlabel(label,fontsize=9)
				raw_file.close()
		if typee == 'fft':
			plt.savefig('./weak/activation_fft/%d-%d-%d.png'%(j*3,j*3+1,j*3+2),dpi=300)
		elif typee == 'time':
			plt.savefig('./weak/activations/%d-%d-%d.png'%(j*3,j*3+1,j*3+2),dpi=300)
		plt.close()
		print(j)

# activate(0,341)

def filters():
	plt.figure(figsize=(40,5))
	current_plot = 0
	for i in range(16):
		current_plot+=1
		filter_file = open('./weak/filters/filter%d.txt'%int(i+1),'r')
		value = [float(line) for line in filter_file]
		i = 0
		value_down = []
		for i in range(64):
			if i> 0 and i <63:
				value_down.append((0.5*value[i-1]+value[i]+0.5*value[i+1])/2.0)
		else:
			value_down.append(value[i])
		plt.subplot(2, 8, current_plot)
		plt.tick_params(
	    axis='x',          # changes apply to the x-axis
	    which='both',      # both major and minor ticks are affected
	    bottom='off',      # ticks along the bottom edge are off
	    top='off',         # ticks along the top edge are off
	    labelbottom='off') # labels along the bottom edge are off
		plt.tick_params(
	    axis='y',          # changes apply to the x-axis
	    which='both',      # both major and minor ticks are affected
	    left='off',      # ticks along the bottom edge are off
	    top='off',         # ticks along the top edge are off
	    labelleft='off') # labels along the bottom edge are off
		plt.axis('off')		
		plt.plot(value_down,color='black')
		filter_file.close()		
	# plt.show()

	plt.savefig('/Users/zhengjia/Desktop/firstLayerSmooth.pdf',dpi=300,bbox_inches='tight')
	# plt.clf()

def activate_one(ID, subIDs, typee, color_hex):
	
	activation = activations[ID]
	pieces=activation.split()
	for i in subIDs:
		print('start ploting ',str(i))
		plt.figure(figsize=(16,6))
		print('../../../data/primV4a/%03d/%06d/sound.raw'% (  math.floor(int(pieces[i+1])/100.0) +1, int(pieces[i+1]) ))
		raw_file = open('../../../data/primV4a/%03d/%06d/sound.raw'% (  math.floor(int(pieces[i+1])/100.0) +1, int(pieces[i+1]) ),'r' )
		value = [float(line) for line in raw_file]
		raw_file.close()
		if typee == 'fft':
			value = np.abs(nft.rfft(value))
			ids = range(len(value))
			down_value = [sum(value[e:e+20]) for e in ids if e%20 == 0]
			plt.plot(down_value, linewidth = 12.0, color = color_hex)
		elif typee == 'time':
			plt.plot(value, color = color_hex, linewidth = 12.0)
		elif typee == 'spec':
			f, t, Sxx = signal.spectrogram(np.asarray(value), fs)
			plt.pcolormesh(t, f, Sxx)
		plt.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
		plt.tick_params(axis='y',which='both',bottom='off',top='off',labelbottom='off',left='off',labelleft='off')
		plt.axis('off')	
		# string = values[int(pieces[i+1])-1][7:-1]
		# string = string.split()
		# label = '%02d  %02d\n%.02f  %.02f  %.02f  %.02f'%(int(string[0]),int(string[1]),float(string[6]),float(string[7]),float(string[8]),float(string[9]))
		# plt.xlabel(label,fontsize=9)
		plt.savefig('/Users/zhengjia/Desktop/sound/zhengjia/plots/primitives/activation_weak/'+typee+'/%d-%d.png'%(ID, i),dpi=300,bbox_inches='tight')
		plt.close()


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

# activate(240,341,'fft')
# activate(250,341,'time')
# filters()

#weak
# activate_one(164,[0,1],'time','#36383a')
# activate_one(55,[0,1],'time','#36383a')
# activate_one(592,[0,1],'fft','#36383a')
# activate_one(992,[0,1],'fft','#36383a')
# activate_one(884,[0,1],'fft','#36383a')

# weak
activate_one(340,[0,1],'time','#36383a')
activate_one(124,[0,1],'time','#36383a')
activate_one(423,[0,1],'fft','#36383a')
activate_one(428,[0,1],'fft','#36383a')
activate_one(35,[0,1],'fft','#36383a')


# for i in range(0,341):
# 	activate_analyze(i,i+1)
