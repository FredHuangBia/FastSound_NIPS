import subprocess,sys

ROOT = '/data/vision/billf/object-properties/sound/sound/primitives/scripts/single_shape/'
GENSOUND = ROOT+'gen_sound.py'
argv = sys.argv[1]

fin = open(ROOT+"run6/input-"+argv.zfill(2)+".txt","r")

for line in fin:
	renew = 'kinit -R'
	subprocess.call(renew,shell=True)
	cmd = 'python %s %s'%(GENSOUND,line.strip())
	print(cmd)
	subprocess.call(cmd,shell=True)



