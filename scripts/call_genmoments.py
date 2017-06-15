import os
import sys
import argparse
import shutil
from functools import partial
from multiprocessing.dummy import Pool
from subprocess import call
import glob

thnum =20


# obj_list_path= '/data/vision/billf/object-properties/sound/sound/data/final100/small_stats.txt'
# mat_matrix_path  = '/data/vision/billf/object-properties/sound/sound/data/material_matrix.dat'

# obj_list_file = open(obj_list_path,'r')
# mat_matrix_file = open(mat_matrix_path,'r')

obj_list = os.listdir("/data/vision/billf/object-properties/sound/sound/primitives/shapes/all/")

task_list = dict()
for obj in obj_list:
    task_list[obj] = [0,1,2,3,4,6,7]
                
args=[]

for k in task_list.keys():
    for v in task_list[k]:
        if os.path.exists('/data/vision/billf/object-properties/sound/sound/primitives/data/%d/models/mat-%d/bem_result/output-59.dat'%(k,v)):
            if not os.path.exists('/data/vision/billf/object-properties/sound/sound/data/final100/%d/models/mat-%d/moments/moments.pbuf'%(k,v)):
        		args.append([k,v])
print len(args)

cmd = []
for k in args:
	path = '/data/vision/billf/object-properties/sound/sound/data/final100/%d/models/mat-%d/'%(k[0],k[1])
	cmd.append('bash /data/vision/billf/object-properties/sound/sound/script/call_genmoments.sh "%s"'%(path))
print len(cmd)




pool = Pool(thnum)
failedCmdCnt = 0
for i, returnCode in enumerate(pool.imap(partial(call, shell = True), cmd)):
    if returnCode != 0:
        failedCmdCnt += 1
print 'failed cmd: ', failedCmdCnt
