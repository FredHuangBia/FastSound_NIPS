from subprocess import call
import os

metric = 'stft'
# N = [(100,2),(75,3)]
N = [(100,2)]
prob_type = 'expneg'

root = '/data/vision/billf/object-properties/sound/sound/primitives/scripts/multi_shape_v4/inference/'

cmdFile = open(root+'cmd/cmd2.sh','w')

# for i in N:
# 	for j in range(20):
# 		cmd = 'python ' + root + 'as_gibbs.py' + ' -m %s -n %d -N %d -t %s -f rotation %d &\n' %(metric,i[0],i[1],prob_type,j)
# 		cmdFile.write(cmd)

for i in N:
	for j in range(50):
		cmd = 'python ' + root + 'as_gibbs.py' + ' -m %s -n %d -N %d -t %s %d &\n' %(metric,i[0],i[1],prob_type,j)
		cmdFile.write(cmd)

# os.chdir(root+'cmd')
# cmd = 'split -dl 10 --additional-suffix=.sh cmd.sh cmd-'
# call(cmd,shell=True)
# cmd = 'chmod +x cmd-*'
# call(cmd,shell=True)
