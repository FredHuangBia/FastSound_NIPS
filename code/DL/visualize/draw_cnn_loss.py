import matplotlib.pyplot as plt

root = '/data/vision/billf/object-properties/sound/sound/primitives/models/primV4a_cnnAll_soundnet8_pretrainnone_mse1_LR0.001/'

train_log = open(root+'train.log','r')
train_loss = [ float(line.split()[-1]) for line in train_log if (len(line.split())>0 and line.split()[0]=='*')  ]
train_log.close()

test_log = open(root+'test.log','r')
test_loss = [ float(line.split()[-1]) for line in test_log if (len(line.split())>0 and line.split()[0]=='*')  ]
test_log.close()

