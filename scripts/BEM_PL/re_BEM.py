import subprocess, os, sys

split = int(sys.argv[1])

bad = open('/data/vision/billf/object-properties/sound/sound/primitives/scripts/BEM_PL/bad-%02d.txt' %split,'r').readlines()

for line in bad:
	line = line.strip()
	os.chdir(line+'/moments')
	cmd = '/data/vision/billf/object-properties/sound/sound/code/ModalSound/build/bin/gen_moments ../fastbem/input-%d.dat ../bem_result/output-%d.dat 0 59'
	subprocess.call(cmd, shell=True)