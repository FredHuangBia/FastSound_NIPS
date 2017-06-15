import os
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = '/Users/qiujiali/Desktop/Remote/sound/primitives/data/v2.3/'
mapping = {'mat-0-0.txt':'E/p=1','mat-0-1.txt':'E/p=5','mat-0-2.txt':'E/p=10','mat-0-3.txt':'E/p=15','mat-0-4.txt':'E/p=20','mat-0-5.txt':'E/p=25','mat-0-6.txt':'E/p=30','mat-1-0.txt':'v=0.10','mat-1-1.txt':'v=0.15','mat-1-2.txt':'v=0.20','mat-1-3.txt':'v=0.25','mat-1-4.txt':'v=0.30','mat-1-5.txt':'v=0.35','mat-1-6.txt':'v=0.40'}

def load(filepath, flag=0):
	assert(flag < 2), "Nothing to load!"
	freqDir = ROOT_DIR + filepath + '/freq/'
	fileList = sorted(os.listdir(freqDir))
	fileIn = []
	for i in fileList:
		if i[-3:]=='txt':
			if flag == 0 and i[4] == '0':
				fileIn.append(i)
			elif flag == 1 and i[4] == '1':
				fileIn.append(i)
	print(fileIn)
	xticks = []
	output = np.zeros((60,len(fileIn)))
	for counter,eachFile in enumerate(fileIn):
		currentFile = open(freqDir+eachFile,'r').readlines()
		for i in range(len(currentFile)):
			output[i][counter] = float(currentFile[i])
		xticks.append(mapping[eachFile])
	return xticks, output

def load_single(name):
	freqPath = ROOT_DIR + name + '/freq/mat-0-0.txt'
	output = np.zeros((60,1))
	fileIn = open(freqPath,'r').readlines()
	for i in range(len(fileIn)):
		output[i] = float(fileIn[i])
	return output

def plot(xticks, nparray, line=[], square=False):
	plt.rcParams["font.family"] = "Times New Roman"
	num_col = nparray.shape[1]
	for i in range(num_col):
		if square is True:
			plt.scatter([i]*60,np.power(nparray[:,i],2),marker='_',s=5)
		else:
			plt.scatter([i]*60,nparray[:,i],marker='_',s=5)
	print(xticks)
	plt.xticks(range(num_col), xticks)
	ax = plt.gca()
	ax.grid(True, which='x')
	for mode in line:
		if square is True:
			plt.plot(range(num_col),np.power(nparray[mode,:],2),color='grey',linewidth=0.5,alpha=0.5)
		else:
			plt.plot(range(num_col),nparray[mode,:],color='grey',linewidth=0.5,alpha=0.5)

def plot_one_shape(name):
	print("Ploting E/p ...")
	xticks,output = load(name,0)
	plot(xticks,output,[0,9,19,29,39,49,59])
	plt.xlabel("specific modulus")
	plt.ylabel("modal frequencies (Hz)")
	plt.savefig('plots/%s-youngs.pdf' %name, bbox_inches='tight', dpi=300)
	plt.clf()

	print("Ploting E/p ...")
	xticks,output = load(name,0)
	plot(xticks,output,[0,9,19,29,39,49,59],True)
	plt.xlabel("specific modulus")
	plt.ylabel("modal frequencies squared")
	plt.savefig('plots/%s-youngs2.pdf' %name, bbox_inches='tight', dpi=300)
	plt.clf()

	print("Ploting v ...")
	xticks,output = load(name,1)
	plot(xticks,output,[0,9,19,29,39,49,59])
	plt.xlabel("poisson ratio")
	plt.ylabel("modal frequencies")
	plt.savefig('plots/%s-poisson.pdf' %name, bbox_inches='tight', dpi=300)
	plt.clf()

def main():
	# plot_one_shape('0-85')
	# plot_one_shape('24-53')
	# plot_one_shape('3-85')

if __name__ == '__main__':
	main()
