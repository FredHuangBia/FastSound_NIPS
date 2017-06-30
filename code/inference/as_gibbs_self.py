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
from utils import cd, mkdir


colorHeader = '\x1b[6;30;42m '
colorTail = ' \x1b[0m '

ROOT = '/data/vision/billf/object-properties/sound/sound/primitives/exp/self_sup_v2/'
# ROOT = '/data/vision/billf/object-properties/sound/sound/primitives/exp/hmstudy/'
SOUND_SCRIPTS = '/data/vision/billf/object-properties/sound/sound/primitives/scripts/multi_shape_v4/'

PI = np.pi

PRIM = list(range(14))
# PRIM = [0, 2, 5, 9, 12 ,13]
H_MAX = 2.0
H_MIN = 1.0
ALPHA_MAX = -5
ALPHA_MIN = -8
BETA_MAX = 5
BETA_MIN = 0
RES_MAX = 0.9
RES_MIN = 0.6
VAR = 1/3.0
CONST = 30   # accept rate offset

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

def feature_raw(sound):
    '''raw'''
    return np.array(sound)

def feature_mfcc(sound):
    '''mfcc'''
    return mfcc

def feature_stft(sound):
    '''stft distance'''
    freq, time, stft = scipy.signal.spectrogram(np.array(sound), 44100, scaling='spectrum', nperseg=5001, noverlap=2000, nfft=44100, mode='magnitude')
    return stft

# def feature_corl(sound):
#     '''cross correlation of two signals'''
#     pass


def linear_random(min_val, max_val):
    '''linear random in [min,max)'''
    return (max_val+min_val)/2.0 + (np.random.random(1)-0.5)*(max_val-min_val)

def linear_mapping(value, min_val, max_val):
    '''linear random in [min,max)'''
    return (max_val+min_val)/2.0 + (value/2.0)*(max_val-min_val)

def sample_height(label, n1, n2):
    '''hack height, n1 is target, n2 is new resulting from label'''
    f = 44100.0
    dh = 0.5*10*((n2/f)**2-(n1/f)**2)
    dh_label = dh/float(H_MAX-H_MIN)
    delta = 0.05
    center_val = label - dh_label
    return linear_random(center_val-delta, center_val+delta)[0]

def gen_sound(label, path, prefix=""):
    '''Generate Sound using label'''
    output_dir = os.path.join(path, prefix)
    mkdir(output_dir)
    gen_config_script = os.path.join(SOUND_SCRIPTS, 'gen_config.sh')
    config_arguments = [output_dir, str(label[3]), \
                        '[%.4f, %.4f, %.4f, %.4f]'%tuple(label[2]), \
                        str(label[4]), str(label[5]), str(label[6])]
    call(['sh', gen_config_script]+config_arguments)
    gen_sound_script = os.path.join(SOUND_SCRIPTS, 'gen_sound.py')
    sound_arguments = "%s 0 0 100000 10 10 10 %02d%d 0 0 0 > %s" \
        %(output_dir, label[0], label[1], os.path.join(path, prefix, 'gen_sound.log'))
    call('python %s %s'%(gen_sound_script, sound_arguments), shell=True)

def gen_random_label():
    '''generate random label for sound synthesis'''
    shape = np.random.randint(len(PRIM), size=1)
    prim_type = PRIM[shape[0]]
    youngs = np.random.randint(10, size=1)[0]
    quat = sample_quat_uniform()
    height = linear_random(H_MIN, H_MAX)[0]
    alpha = 10**linear_random(ALPHA_MIN, ALPHA_MAX)[0]
    beta = 2**linear_random(BETA_MIN, BETA_MAX)[0]
    res = linear_random(RES_MIN, RES_MAX)[0]
    return [prim_type, youngs, quat, height, alpha, beta, res]

# def gen_init_label(attr,path):
#     '''generate init labels with one attr fixed'''
#     file_name = os.path.join(path, 'label.txt')
#     label_line = open(file_name,'r').readlines()[0].split()
#     gt_label = [int(i) for i in label_line[0:2]]
#     gt_label.append([float(i) for i in label_line[2:6]])
#     gt_label = gt_label + [float(i) for i in label_line[6:]]
#     assert(len(gt_label) == 7), "Wrong Label File!"
#     # print 'GT label: %s' % label_to_str(gt_label)
#     init_label = gen_random_label()
#     if attr == 'rotation':
#         init_label[2] = gt_label[2]
#     elif attr == 'height':
#         init_label[3] = gt_label[3]
#     return init_label

def nn2real(nnList):
    '''height, alpha, beta, res'''
    height = linear_mapping(nnList[0],H_MIN, H_MAX)
    alpha = 10**linear_mapping(nnList[1],ALPHA_MIN, ALPHA_MAX)
    beta = 2**linear_mapping(nnList[2],BETA_MIN, BETA_MAX)
    res = linear_mapping(nnList[3],RES_MIN, RES_MAX)
    return [height, alpha, beta, res]

def get_init_label(path, test_id):
    label_line = open(path,'r').readlines()[test_id].split()
    self_sup_label = [int(i) for i in label_line[0:2]]
    self_sup_label.append(sample_quat_uniform())
    self_sup_label = self_sup_label + nn2real([float(i) for i in label_line[6:10]])
    return self_sup_label

# def gen_init_label(attr,path):
#     '''generate init labels with one attr fixed'''
#     file_name = os.path.join(path, 'label.txt')
#     label_line = open(file_name,'r').readlines()[0].split()
#     gt_label = [int(i) for i in label_line[0:2]]
#     gt_label.append([float(i) for i in label_line[2:6]])
#     gt_label = gt_label + [float(i) for i in label_line[6:]]
#     assert(len(gt_label) == 7), "Wrong Label File!"
#     print 'GT label: %s' % label_to_str(gt_label)
#     init_label = gen_random_label()
#     gt_label[3] = init_label[3]
#     return gt_label

def get_sample_attr(attr):
    all_attr = list(range(7))
    if attr == 'rotation':
        del all_attr[2]
    elif attr == 'height':
        del all_attr[3]
    return all_attr

def save_label(label, out_dir, prefix=""):
    '''save label for generated sound'''
    file_name = os.path.join(out_dir, prefix,'label.txt')
    with open(file_name, 'w') as label_file:
        label_file.write(' '.join(str(e) for e in label[0:2])+' ')
        label_file.write(' '.join(str(e) for e in label[2])+' ')
        label_file.write(' '.join(str(e) for e in label[3:]))

def calc_distance(feature_1, feature_2):
    '''Calculate Feature Space distance'''
    if feature_1.size != feature_2.size:
        raise ValueError('Feature Size Mismatch!')
    return np.sum(np.power(feature_1-feature_2, 2))


def gen_random_sound(num):
    ''' Generate random sound '''
    num = num+100
    print 'Generating %d random sounds'%num
    sounddir = os.path.join(ROOT, 'test_sound')
    mkdir(sounddir)
    for k in tqdm(range(num)):
    # for k in range(num):
        label = gen_random_label()
        gen_sound(label, sounddir, str(k))
        save_label(label, sounddir, str(k))

def read_raw_wav(sound_file_path):
    ''' read raw wav file '''
    target_sound = []
    with open(sound_file_path, 'r') as raw_wav_file:
        for line in raw_wav_file.readlines():
            target_sound.append(float(line))
    return target_sound

def read_gen_sound_raw(label, dir, prefix=''):
    '''load a sound generated by label'''
    gen_sound(label, dir, prefix)
    return read_raw_wav(os.path.join(dir, prefix, 'sound.raw'))

def find_first_click(raw_wav):
    '''find the first non-zero index of the sound vector'''
    return next((i for i, x in enumerate(raw_wav) if x), None)

def calc_feature_distance(sound_1, sound_2, feature_func):
    '''calc feature space distace of 2 sounds'''
    return calc_distance(feature_func(sound_1), feature_func(sound_2))

def label_to_str(label):
    label_str = ''
    label_str = label_str+' '.join(str(e) for e in label[0:2])+' '
    label_str = label_str+' '.join(str(e) for e in label[2])+' '
    label_str = label_str+' '.join(str(e) for e in label[3:])
    return label_str

class Logger:
    def __init__(self,path):
        self.distance_log = open(os.path.join(path, 'distance_log.txt'), 'a+', 0)
        self.label_log = open(os.path.join(path, 'label_log.txt'), 'a+', 0)
        self.ar_log = open(os.path.join(path, 'ar_log.txt'), 'a+', 0)
        self.prob_log = open(os.path.join(path, 'prob_log.txt'), 'a+', 0)
        self.record_log = open(os.path.join(path, 'record_log.txt'), 'a+', 0)

    def log_distance(self, distance):
        self.distance_log.write('%.4f\n'%distance)
    def log_label(self, label):
        self.label_log.write('%s\n'%label_to_str(label))
    def log_ar(self, status):
        self.ar_log.write('%s\n'%status)
    def log_prob(self, prob):
        self.prob_log.write('%.3E\n'%prob)
    def log_record(self, record):
        self.record_log.write('%f %s' %(record['distance'], label_to_str(record['label'])))

    def log(self, distance, label, status, prob, record):
        self.log_distance(distance)
        self.log_label(label)
        self.log_ar(status)
        self.log_prob(prob)
        self.log_record(record)
    def close(self):
        self.distance_log.close()
        self.label_log.close()
        self.ar_log.close()
        self.prob_log.close()
        self.record_log.close()

# def map_to_prob(distance):
#     return 1/(distance)

def map_to_prob(distance, time):
    t = float(time) + CONST
    return np.exp(-distance * (0.5+t*t/30.0))

def forward_mapping(realvalue, attrid):
    if attrid == 4:
        return math.log(realvalue,10)
    elif attrid == 5:
        return math.log(realvalue,2)

def gaussian_random(max_val,min_val):
    return np.random.randn(1)*VAR*(max_val-min_val)/2.0

def update_label(label, Id, attr=None):
    newlabel = copy(label)
    if Id == 0:
        newlabel[0] = PRIM[np.random.randint(len(PRIM), size=1)[0]]
    if Id == 1:
        newlabel[1] = np.random.randint(10, size=1)[0]
    if Id == 2:
        if attr != 'rotation':
            newlabel[2] = sample_quat_uniform()
    if Id == 3:
        if attr != 'height':
            newlabel[3] = gaussian_random(H_MAX, H_MIN)[0]+ label[3]
    if Id == 4:
        power = gaussian_random(ALPHA_MAX, ALPHA_MIN)[0] + forward_mapping(label[4],Id)
        while power<-8 or power>-5:
            power = gaussian_random(ALPHA_MAX, ALPHA_MIN)[0] + forward_mapping(label[4],Id)
        newlabel[4] = 10**(power)
    if Id == 5:
        power = gaussian_random(BETA_MAX, BETA_MIN)[0] + forward_mapping(label[5],Id)
        while power<0 or power>5:
            power = gaussian_random(BETA_MAX, BETA_MIN)[0] + forward_mapping(label[5],Id)
        newlabel[5] = 2**(power)
    if Id == 6:
        rest = gaussian_random(RES_MAX, RES_MIN)[0]+label[6]
        while rest<0.6 or rest > 0.9:
            rest = gaussian_random(RES_MAX, RES_MIN)[0]+label[6]
        newlabel[6] = rest
    return newlabel

def load_label(path):
    with open(path,'r') as f:
        raw_label = f.readline()

def remove(path):
    os.chdir(path)
    cmd = 'rm -r %s' %path
    call(cmd,shell=True)

def read_label(string):
    string = string.split()
    label = [int(i) for i in string[0:2]]
    label.append([float(i) for i in string[2:6]])
    label = label + [float(i) for i in string[6:]]
    return label

def restore_latest_label(path):
    last_line = open(os.path.join(path, 'label_log.txt'), 'r').readlines()[-1].strip()
    return read_label(last_line)

def restore_counter(path):
    acc_file = open(os.path.join(path, 'ar_log.txt'), 'r').readlines()
    counter = len(acc_file) - 1
    acc_counter = 0
    for line in acc_file:
        if line.split()[0] == 'accept':
            acc_counter += 1
    return counter, acc_counter

def restore_record(path):
    best_line = open(os.path.join(path, 'label.txt'), 'r').readlines()[0].strip()
    return read_label(best_line)


METRIC_DICT = {'raw':feature_raw, 'mfcc': feature_mfcc, 'stft':feature_stft}
#DIST_DICT = {'inverse':inverse_distance, 'expneg':expneg}
# pylint: disable=C0103
if __name__ == '__main__':
    #INPUT: ID
    #OPTIONAL FLAG: root, #iter, #sampling, metric, is_gen
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--generate', help='Generate Random sound.', type=int, default=0)
    parser.add_argument('test_id', type=int, help='specify the id of wavefile to be tested')
    parser.add_argument('-m', '--metric', help='specify metric for sound: raw, stft, mfcc.', choices=['raw', 'mfcc', 'stft'], default='stft')
    parser.add_argument('-n', '--niter', help='specify #iterations of MH sampling for conditional prob.', type=int, default=30)
    parser.add_argument('-N', '--nsample', help='Specify #samples for gibbs sampling.', type=int, default=10)
    parser.add_argument('-t', '--type', help='mapping l2 distance to probability: inverse, expneg', choices=['inverse', 'expneg'], default='expneg')
    parser.add_argument('-T', '--temp', help='factor for expneg. default = 1000', type=float, default=1000.0)
    parser.add_argument('-f', '--fix', help='fix a certain varaible.', choices=['rotation','height'], default=None)
    parser.add_argument('-r', '--resume', help='resume previous ieterations', type=int, default=0)

    args = parser.parse_args()
    if args.generate != 0:
        gen_random_sound(args.generate)
    else:
        test_id = args.test_id
        target_dir = os.path.join(ROOT, 'test_sound', str(test_id))
        sound_file_path = os.path.join(target_dir, 'sound.raw')
        target_sound = read_raw_wav(sound_file_path)
        target_height = find_first_click(target_sound)
        feature_function = METRIC_DICT[args.metric]
        dir_name = '%s_%s_%04d_%02d_%s' %(args.metric,args.type,args.niter,args.nsample, str(args.fix))
        result_dir = os.path.join(ROOT, 'result', dir_name ,str(test_id))
        mkdir(result_dir)
        self_sup_label_file = '/data/vision/billf/object-properties/sound/sound/primitives/data/self_sup/self_label.txt'
        
        if args.resume != 0:
            init_label = restore_latest_label(result_dir)
        else:
            if args.fix == None:
                # init_label = gen_random_label()
                init_label = get_init_label(self_sup_label_file, args.test_id)
            else:
                init_label = gen_init_label(args.fix, target_dir)

        if args.resume == 0:
            init_sound = read_gen_sound_raw(init_label, result_dir, 'init')
        else:
            init_sound = read_gen_sound_raw(init_label, result_dir, 'restore')

        init_distance = calc_feature_distance(target_sound, init_sound, feature_function)
        init_height = find_first_click(init_sound)
        # print 'INIT label: %s' % label_to_str(init_label)
        # print 'INIT distance: %.4f' % init_distance
        # print 'INIT prob: %.5f' % map_to_prob(init_distance,0.0)
        log = Logger(result_dir)
        #mapping = DIST_DICT[args.type]
        label = init_label
        distance = init_distance
        height = init_height
        
        # resume counters
        if args.resume != 0:
            counter, acc_counter = restore_counter(result_dir)
        else:
            counter = 0
            acc_counter = 0

        # resume record
        if args.resume == 0:
            record = {'label':init_label, 'distance':init_distance}
            log.log(init_distance, init_label, 'init', map_to_prob(init_distance,0.0), record)
            
        else:
            record_label = restore_record(result_dir)
            record_sound = read_gen_sound_raw(record_label, result_dir, 'record')
            record_distance = calc_feature_distance(target_sound, record_sound, feature_function)
            record = {'label': record_label, 'distance': record_distance}

        print(init_distance)
        all_attr = get_sample_attr(args.fix)
        # all_attr = [3]
        ## Gibbs + MH

        # resume iteration number
        if args.resume == 0:
            looping = range(args.niter)
        else:
            looping = range(args.niter, args.niter+args.resume)

        for t in tqdm(looping):
            for label_id in all_attr:
                if label_id == 3:
                    for sample in tqdm(range(args.nsample)):
                        status = 'reject'
                        print 'customized height sampling'
                        newlabel = copy(label)
                        newlabel[3] = sample_height(label[3],target_height, height)
                        # print 'NEW LABEL %d: %s' % (label_id, label_to_str(newlabel))
                        newsound = read_gen_sound_raw(newlabel, os.path.join(result_dir, str(t)), str(counter))
                        newdistance = calc_feature_distance(target_sound, newsound, feature_function)
                        newheight = find_first_click(newsound)
                        # print(target_height, newheight, height)
                        if abs(newheight - target_height) <= abs(height - target_height):
                            status = 'accept'
                            acc_counter += 1
                            label = newlabel
                            distance = newdistance
                            height = newheight
                            if newdistance < record['distance']:
                                record['label'] = newlabel
                                record['distance'] = newdistance
                                print '***Current BEST!***'
                        log.log(distance, label, status, map_to_prob(distance, t), record)
                        counter += 1
                        print 'sample %d: distance %.4f %s'%(counter, newdistance, status)
                        print 'Current Acc Rate: %.4f'%(float(acc_counter)/float(counter))

                else:
                    for sample in tqdm(range(args.nsample)):
                        status = 'reject'
                        newlabel = update_label(label, label_id, args.fix)
                        # print(colorHeader+str(newlabel)+colorTail)
                        
                        # print 'NEW LABEL %d: %s' % (label_id, label_to_str(newlabel))
                        newsound = read_gen_sound_raw(newlabel, os.path.join(result_dir, str(t)), str(counter))
                        newdistance = calc_feature_distance(target_sound, newsound, feature_function)
                        newheight = find_first_click(newsound)
                        # print(distance)
                        alpha = map_to_prob(newdistance, t)/map_to_prob(distance, t)
                        if alpha >= 1:
                            status = 'accept'
                            acc_counter += 1
                            label = newlabel
                            distance = newdistance
                            height = newheight
                            if newdistance < record['distance']:
                                record['label'] = newlabel
                                record['distance'] = newdistance
                                print '***Current BEST!***'
                        elif alpha > 1e-2:
                            if np.random.binomial(1, alpha, 1) == 1:
                                label = newlabel
                                distance = newdistance
                                height = newheight
                                status = 'accept %.3f' %alpha
                                acc_counter += 1
                        log.log(distance, label, status, map_to_prob(distance, t), record)
                        counter += 1
                        print 'sample %d: distance %.4f %s'%(counter, newdistance, status)
                        print 'Current Acc Rate: %.4f'%(float(acc_counter)/float(counter))
                #label = record['label']
                #print '-------------------------label %d sampling finished--------------------------- '
                #print 'best:  dist %.4f'%record['distance']
            cmd = 'rm -r %s' %(os.path.join(ROOT, 'result', dir_name, str(args.test_id), str(t-1)))
            # print(cmd)
            call(cmd, shell=True)
        log.close()
        save_label(record['label'],result_dir)
        # new_dir_name = '%s_%s_%04d_%02d_%s' %(args.metric, args.type, (args.niter+args.resume), args.nsample, str(args.fix))
        # cmd = 'mv %s %s' %(os.path.join(ROOT, dir_name), os.path.join(ROOT, new_dir_name))
        # call(cmd, shell=True)
