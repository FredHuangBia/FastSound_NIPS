import os

target_file = open('/data/vision/billf/object-properties/sound/sound/primitives/www/primV3b_cnnF_soundnet8_pretrainnone_mse1_LR0.001/93/targets.txt','r')
targets = []
for line in target_file:
	targets.append(line.split()[0])
print(targets)

root = '/data/vision/billf/object-properties/sound/sound/primitives/exp/primV3b/'
best_list = []
for i in range(len(targets)):
	os.chdir(root+targets[i])
	best = open('best.txt','r').readlines()[0].split()[0]
	best_list.append(best)

print(best_list)

