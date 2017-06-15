import configparser

ROOT = '/afs/csail.mit.edu/u/q/qiujiali/sound/sound/primitives/code/inference/'
config = configparser.ConfigParser()
config.read(ROOT+'all_mse.cfg')

content = ['Recognition_MSE', 'Inference_MSE']
print('')
for i in config.sections():
	print(i)
	print('------------------------------')
	for j in content:
		line = config[i][j].strip('[').strip(']').split(', ')
		print('%s\t%.4f %.4f %.4f %.4f %.4f %.4f %.4f'%(j,float(line[0]),float(line[1]),float(line[2]),float(line[3]),float(line[4]),float(line[5]),float(line[6])))
	print('')