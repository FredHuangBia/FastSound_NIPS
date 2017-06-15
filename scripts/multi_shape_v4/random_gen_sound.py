'''
MH sampling for infereing latent variables of a given sound.
'''
from __future__ import division
import math
import os
import sys
from subprocess import call
import argparse
import numpy as np
import scipy as sp
import scipy.signal
from copy import copy
from tqdm import tqdm


colorHeader = '\x1b[6;30;42m '
colorTail = ' \x1b[0m '

SOUND_SCRIPTS = '/data/vision/billf/object-properties/sound/sound/primitives/scripts/multi_shape_v4/'
DATASET = '/data/vision/billf/object-properties/sound/sound/primitives/data/primV4a/'

PI = np.pi

PRIM = list(range(14))
H_MAX = 2.0
H_MIN = 1.0
ALPHA_MAX = -5
ALPHA_MIN = -8
BETA_MAX = 5
BETA_MIN = 0
RES_MAX = 0.9
RES_MIN = 0.6
VAR = 1/3.0

def mkdir(directory):
    if not os.path.exists(directory):
            os.makedirs(directory)

def sample_quat_uniform():
    '''uniformly sample quaternion
    x,y,z,w, bullet convention '''

    # Pick axis uniformly over S2
    axis = np.random.randn(3)
    axis = axis / float(np.linalg.norm(axis))
    # rotation
    angle = PI*np.random.rand(1)
    q_x = axis[0]*np.sin(angle/2.0)
    q_y = axis[1]*np.sin(angle/2.0)
    q_z = axis[2]*np.sin(angle/2.0)
    q_w = np.cos(angle/2.0)
    return [q_x[0], q_y[0], q_z[0], q_w[0]]

def linear_random(min_val, max_val):
    '''linear random in [min,max)'''
    return (max_val+min_val)/2.0 + (np.random.random(1)-0.5)*(max_val-min_val)

def gen_sound(label, path, prefix):
    '''Generate Sound using label'''
    output_dir = os.path.join(path, '%03d' %(math.floor(prefix/100)+1), '%06d' %prefix)
    mkdir(output_dir)
    gen_config_script = os.path.join(SOUND_SCRIPTS, 'gen_config.sh')
    config_arguments = [output_dir, str(label[3]), \
                        '[%.4f, %.4f, %.4f, %.4f]'%tuple(label[2]), \
                        str(label[4]), str(label[5]), str(label[6])]
    call(['sh', gen_config_script]+config_arguments)
    gen_sound_script = os.path.join(SOUND_SCRIPTS, 'gen_sound.py')
    sound_arguments = "%s 0 0 100000 10 10 10 %02d%d 0 0 0 > %s" \
        %(output_dir, label[0], label[1], os.path.join(output_dir, 'gen_sound.log'))
    call('python %s %s'%(gen_sound_script, sound_arguments), shell=True)

def gen_random_label():
    '''generate random label for sound synthesis'''
    shape = np.random.randint(14, size=1)
    prim_type = PRIM[shape[0]]
    youngs = np.random.randint(10, size=1)[0]
    quat = sample_quat_uniform()
    height = linear_random(H_MIN, H_MAX)[0]
    alpha = 10**linear_random(ALPHA_MIN, ALPHA_MAX)[0]
    beta = 2**linear_random(BETA_MIN, BETA_MAX)[0]
    res = linear_random(RES_MIN, RES_MAX)[0]
    return [prim_type, youngs, quat, height, alpha, beta, res]

def save_label(label, out_dir, prefix):
    '''save label for generated sound'''
    file_name = os.path.join(out_dir,'%03d' %(math.floor(prefix/100)+1), '%06d' %prefix,'label.txt')
    with open(file_name, 'w') as label_file:
        label_file.write(' '.join(str(e) for e in label[0:2])+' ')
        label_file.write(' '.join(str(e) for e in label[2])+' ')
        label_file.write(' '.join(str(e) for e in label[3:]))

def gen_random_sound(start, end):
    ''' Generate random sound '''
    
    sounddir = DATASET
    mkdir(sounddir)
    for k in tqdm(range(start, end)):
    # for k in range(num):
        # print '----------Generating sound %06d----------'%k
        label = gen_random_label()
        gen_sound(label, sounddir, k)
        save_label(label, sounddir, k)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--start', help='Starting ID', type=int, default=0)
    parser.add_argument('-e', '--end', help='Ending ID', type=int, default=0)

    args = parser.parse_args()
    
    gen_random_sound(args.start, args.end+1)
    
