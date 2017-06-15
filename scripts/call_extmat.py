import subprocess
import os

extmat = '/data/vision/billf/object-properties/sound/sound/code/ModalSound/build/bin/extmat'

files = os.listdir()
for filename in files:
	if filename[-3:] == 'tet':
		subprocess.call([extmat,'-f',filename[:-4],'-y','7e+9','-p','0.3','-s'])
		print("%s done!" %filename)
